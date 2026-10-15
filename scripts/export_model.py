"""Export merged WebShop SFT models in BF16 and verify source file signatures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_signature(path):
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('*.json')) | set(path.glob('*.jinja')))
    if not any(f.suffix == '.safetensors' for f in files):
        raise ValueError(f'No safetensors weights: {path}')
    return {f.name: file_hash(f) for f in files}


def export_model(model_path, adapter, destination):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    signature = dict(base=source_signature(model_path),
        adapter=source_signature(adapter) if adapter else None, export_protocol='text_bf16_v1')
    marker = destination/'export_manifest.json'
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError('Incomplete export directory; inspect it rather than overwrite')
        old = json.loads(marker.read_text())
        if old['source'] != signature or old['weights_sha256'] != file_hash(destination/'model.safetensors'):
            raise RuntimeError('Export mismatch; use a new output directory')
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                device_map='cpu', local_files_only=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    # Export text-model names directly, avoiding save_pretrained's multimodal
    # namespace prefix. Work only on a CPU copy; never write source weights.
    state = {}
    for name, tensor in model.state_dict().items():
        key = name.replace('model.language_model.', 'model.', 1) if name.startswith('model.language_model.') else name
        if key in state or not key.startswith(('model.layers.', 'model.embed_tokens.', 'model.norm.', 'lm_head.')):
            raise RuntimeError(f'Unexpected/duplicate text weight {key}')
        state[key] = tensor.detach().contiguous().clone()
    config = model.config.get_text_config().to_dict()
    config.update(model_type='qwen3_5_text', architectures=['Qwen3_5ForCausalLM'],
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    destination.mkdir(parents=True)
    (destination/'config.json').write_text(json.dumps(config, indent=2))
    tokenizer.save_pretrained(destination)
    save_file(state, str(destination/'model.safetensors'), metadata={'format':'pt'})
    marker.write_text(json.dumps(dict(source=signature,
        weights_sha256=file_hash(destination/'model.safetensors')), indent=2))

