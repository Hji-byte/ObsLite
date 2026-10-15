"""Explicit runtime adapters for text-only Qwen3.5 and vLLM 0.27.1.

These do not implement rewards, advantages or PPO. GPU synchronization remains
unverified until a real two-iteration smoke succeeds.
"""


class VLLMHijack:
    @staticmethod
    def hijack():
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        from vllm.lora.peft_helper import PEFTHelper
        from verl.utils.vllm_utils import TensorLoRARequest

        original = LRUCacheWorkerLoRAManager._load_adapter
        if getattr(original, '_webshop_tensor_lora', False):
            return

        def load_adapter(self, request):
            if not isinstance(request, TensorLoRARequest):
                return original(self, request)
            helper = PEFTHelper.from_dict(request.peft_config)
            helper.validate_legal(self.lora_config)
            model = self._adapter_manager.model
            mapper = getattr(model, 'hf_to_vllm_mapper', None)
            if mapper is not None:
                mapper = mapper.get_unstacked_mapper()
            lora = self._lora_model_cls.from_lora_tensors(
                lora_model_id=request.lora_int_id, tensors=request.lora_tensors,
                peft_helper=helper, device='cpu', dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size, weights_mapper=mapper,
                skip_prefixes=getattr(model, 'lora_skip_prefixes', None))
            lora.is_3d_lora_weight = request.is_3d_lora_weight
            import os
            audit_dir = os.environ.get('WEBSHOP_RL_AUDIT_DIR')
            if audit_dir:
                import hashlib
                import json
                from pathlib import Path
                import torch
                from vllm.lora.utils import parse_fine_tuned_lora_name
                digest = hashlib.sha256()
                for name, tensor in sorted(request.lora_tensors.items()):
                    module_name, is_a = parse_fine_tuned_lora_name(name, mapper)
                    layer = lora.get_lora(module_name)
                    if layer is None:
                        raise ValueError(f'LoRA tensor was not loaded: {name}')
                    loaded = layer.lora_a if is_a else layer.lora_b
                    expected = tensor.detach().to(device='cpu', dtype=loaded.dtype)
                    if not torch.equal(loaded, expected):
                        raise ValueError(f'LoRA tensor conversion mismatch: {name}')
                    digest.update(name.encode())
                    digest.update(str(tensor.dtype).encode())
                    digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                path = Path(audit_dir)/'lora_loads.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as stream:
                    stream.write(json.dumps({'adapter_id': request.lora_int_id,
                        'source_sha256': digest.hexdigest(), 'tensors': len(request.lora_tensors),
                        'loaded_values_match_cast_source': True}) + '\n')
            return lora

        load_adapter._webshop_tensor_lora = True
        LRUCacheWorkerLoRAManager._load_adapter = load_adapter


def check_tensor_lora():
    """Exercise the real vLLM CPU loader with known tensors, without a model."""
    from types import SimpleNamespace
    import torch
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from verl.utils.vllm_utils import TensorLoRARequest
    VLLMHijack.hijack()
    name = 'base_model.model.model.layers.0.self_attn.q_proj'
    tensors = {name + '.lora_A.weight': torch.arange(8, dtype=torch.float32).reshape(2, 4),
               name + '.lora_B.weight': torch.arange(8, dtype=torch.float32).reshape(4, 2)}
    request = TensorLoRARequest(lora_name='cpu_probe', lora_int_id=1, lora_path='unused',
        peft_config={'r': 2, 'lora_alpha': 2, 'target_modules': ['q_proj'], 'bias': 'none'},
        lora_tensors=tensors)
    manager = SimpleNamespace(
        _adapter_manager=SimpleNamespace(model=SimpleNamespace()),
        _lora_model_cls=LoRAModel, vocab_size=16,
        lora_config=SimpleNamespace(max_lora_rank=16, lora_dtype=torch.float32))
    # vLLM enables pinned host buffers even on this driverless machine. This
    # CPU-only probe checks tensor conversion; actual pinning is a GPU-smoke check.
    import vllm.lora.lora_model as lora_module
    pinned = lora_module.PIN_MEMORY
    try:
        lora_module.PIN_MEMORY = False
        result = LRUCacheWorkerLoRAManager._load_adapter(manager, request)
    finally:
        lora_module.PIN_MEMORY = pinned
    layer = result.get_lora('model.layers.0.self_attn.q_proj')
    assert layer is not None and len(result.loras) == 1
    assert torch.equal(layer.lora_a, tensors[name + '.lora_A.weight'])
    assert torch.equal(layer.lora_b, tensors[name + '.lora_B.weight'])
