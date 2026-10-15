"""Create an auditable compatibility copy; never edit the official checkout."""
import argparse
import ast
from pathlib import Path
import shutil

from .common import sha256, write_json
from .prepare_rl import COMMIT

PATCHED_FILES = ['verl/workers/fsdp_workers.py', 'verl/utils/vllm_utils.py',
                 'verl/workers/actor/dp_actor.py', 'verl/models/transformers/monkey_patch.py']


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one pinned compatibility target: {old[:80]}')
    return source.replace(old, new)


def adapt_source(name, source, spear=False, pods=False):
    if spear or pods:
        raise ValueError('This public runtime supports plain GRPO only')
    if name == 'verl/workers/fsdp_workers.py':
        source = replace_once(source, 'import os\n', 'import os\n'
            'if os.environ.get("WEBSHOP_RL_FLA") == "1":\n'
            '    import fla.ops.gated_delta_rule\n')
        source = replace_once(source,
            'from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq',
            'from transformers import AutoConfig, AutoModelForCausalLM')
        source = replace_once(source,
            '            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():\n'
            '                actor_module_class = AutoModelForVision2Seq\n'
            '            else:\n'
            '                actor_module_class = AutoModelForCausalLM',
            '            if actor_model_config.architectures != ["Qwen3_5ForCausalLM"]:\n'
            '                raise ValueError("WebShop runtime requires the text-only SFT export")\n'
            '            actor_module_class = AutoModelForCausalLM')
        source = replace_once(source,
            'actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")',
            'actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="sdpa")')
        source = replace_once(source, '        assert role in ["actor", "ref"]\n',
            '        assert role in ["actor", "ref"]\n'
            '        if os.environ.get("WEBSHOP_RL_FLA") == "1":\n'
            '            from scripts.rl_fast_path import select_backend\n'
            '            print({"webshop_fast_kernel": select_backend("fla")}, flush=True)\n')
    elif name == 'verl/utils/vllm_utils.py':
        source = replace_once(source, 'from vllm.lora.models import LoRAModel',
                              'from vllm.lora.lora_model import LoRAModel')
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLLMHijack')
        lines = source.splitlines(keepends=True)
        source = ''.join(lines[:node.lineno-1]) + (
            'from scripts.rl_runtime_compat import VLLMHijack\n') + ''.join(lines[node.end_lineno:])
    elif name == 'verl/workers/actor/dp_actor.py':
        source = replace_once(source,
            '        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):\n',
            '        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16, enabled=self.config.get("forward_autocast", True)):\n')
        imports = ('if is_cuda_available:\n'
                   '    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n'
                   'elif is_npu_available:\n'
                   '    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input\n')
        source = replace_once(source, imports, '')
        needle = '        self.use_remove_padding = self.config.get("use_remove_padding", False)\n'
        source = replace_once(source, needle, needle +
            '        if self.use_remove_padding:\n'
            '            global index_first_axis, pad_input, rearrange, unpad_input\n' +
            ''.join('            ' + line + '\n' for line in imports.splitlines()))
        source = replace_once(source, 'from verl.workers.actor import BasePPOActor\n',
            'from verl.workers.actor import BasePPOActor\nfrom scripts.rl_padding import trim_masked_prompt\n')
        needle = '                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)\n'
        source = replace_once(source, needle, needle +
            '            if not self.use_remove_padding:\n'
            '                input_ids, attention_mask, position_ids = trim_masked_prompt(\n'
            '                    input_ids, attention_mask, position_ids, response_length)\n'
            '                batch_size, seqlen = input_ids.shape\n')
        needle = '            else:  # not using rmpad and no ulysses sp\n                extra_args = {}\n'
        source = replace_once(source, needle,
            '            else:  # not using rmpad and no ulysses sp\n'
            '                extra_args = {"logits_to_keep": response_length + 1}\n')
    elif name == 'verl/models/transformers/monkey_patch.py':
        source = replace_once(source, '    if is_trl_available():\n',
            '    if is_trl_available() and hasattr(__import__("trl"), "AutoModelForCausalLMWithValueHead"):\n')
    else:
        raise ValueError(name)
    compile(source, name, 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.set_defaults(spear_self_imitation=False)
    parser.set_defaults(pods=False)
    args = parser.parse_args()
    if args.pods and args.spear_self_imitation:
        raise ValueError('This public runtime supports plain GRPO only')
    if args.output.exists():
        raise ValueError('Use a new compatibility directory')
    shutil.copytree(args.upstream, args.output,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    changes = {}
    names = PATCHED_FILES + (['verl/trainer/ppo/ray_trainer.py'] if args.spear_self_imitation or args.pods else [])
    for name in names:
        original, target = args.upstream/name, args.output/name
        target.write_text(adapt_source(name, original.read_text(encoding='utf-8'), spear=args.spear_self_imitation, pods=args.pods), encoding='utf-8')
        changes[name] = {'original_sha256': sha256(original), 'adapted_sha256': sha256(target)}
    write_json(args.output/'webshop_compatibility.json', {
        'upstream_commit': COMMIT, 'changes': changes,
        'runtime_versions': {'transformers': '5.15.0', 'vllm': '0.27.1'},
        'changes_to_algorithm': args.spear_self_imitation or args.pods, 'spear_self_imitation': args.spear_self_imitation,
        'pods': args.pods,
        'gpu_smoke_passed': False,
        'scope': 'text-only model loading with SDPA; vLLM tensor LoRA loader API; lazy padding imports'})
    print(args.output)


if __name__ == '__main__':
    main()
