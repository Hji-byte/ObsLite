"""Verify a prepared run; launch only with --run."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, read_json, write_json, sha256
from .prepare_rl import COMMIT


def verify_sft_export(export, base, adapter):
    from scripts.export_model import source_signature
    manifest = read_json(export/'export_manifest.json')
    if manifest['source']['base'] != source_signature(base):
        raise ValueError('SFT export base does not match')
    if manifest['source']['adapter'] != source_signature(adapter):
        raise ValueError('SFT export adapter does not match')
    if manifest['weights_sha256'] != sha256(export/'model.safetensors'):
        raise ValueError('SFT export weight hash mismatch')
    return manifest


def verify_runtime(runtime, upstream):
    from .prepare_rl_runtime import adapt_source, PATCHED_FILES
    record = read_json(runtime/'webshop_compatibility.json')
    if record['upstream_commit'] != COMMIT:
        raise ValueError('Runtime compatibility revision mismatch')
    expected_changes = set(PATCHED_FILES)
    spear = record.get('spear_self_imitation', False)
    pods = record.get('pods', False)
    if spear or pods:
        expected_changes.add('verl/trainer/ppo/ray_trainer.py')
    if set(record['changes']) != expected_changes:
        raise ValueError('Unexpected runtime patch scope')
    for original in upstream.rglob('*'):
        name = original.relative_to(upstream).as_posix()
        if '.git' in original.relative_to(upstream).parts or original.suffix not in {'.py', '.yaml', '.yml'}:
            continue
        target = runtime/name
        expected = original.read_text(encoding='utf-8')
        if name in expected_changes:
            expected = adapt_source(name, expected, spear=spear, pods=pods)
        if target.read_text(encoding='utf-8') != expected:
            raise ValueError(f'Runtime source differs from reviewed adaptation: {name}')
    for package, expected in record['runtime_versions'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Unverified runtime version for {package}')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--runtime', type=Path, help='Verified compatibility copy for Qwen3.5/vLLM 0.27.1')
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    import yaml
    prepared, upstream = args.prepared.resolve(), args.upstream.resolve()
    manifest = read_json(prepared/'manifest.json')
    if manifest['upstream_commit'] != COMMIT:
        raise ValueError('Prepared run has a different upstream revision')
    for name, digest in manifest['prepared_files'].items():
        if sha256(prepared/name) != digest:
            raise ValueError(f'Prepared run file changed: {name}')
    for name, digest in manifest['integration_files'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Integration source changed after preparation: {name}')
    config = yaml.safe_load((prepared/'config/ppo_trainer.yaml').read_text())
    if config['webshop_adapter'].get('spear') or config['webshop_adapter'].get('pods') or config['algorithm']['adv_estimator'] != 'grpo':
        raise ValueError('Archived comparison config: use its local historical code')
    if manifest.get('validation_enabled') is False and (config['trainer']['val_before_train'] or config['trainer']['test_freq'] > 0 or config['trainer'].get('val_only', False)):
        raise ValueError('This prepared run has no development split; validation must remain disabled')
    runtime = args.runtime.resolve() if args.runtime else upstream
    compatibility = verify_runtime(runtime, upstream) if args.runtime else None
    environment_python = config['webshop_adapter'].get('environment_python')
    if not environment_python or not Path(environment_python).is_file():
        raise ValueError('Missing isolated WebShop interpreter; reprepare with --environment-python')
    model = Path(config['actor_rollout_ref']['model']['path'])
    verify_sft_export(model, args.base, args.adapter)
    if not (model/'generation_config.json').is_file():
        raise ValueError('SFT export needs generation_config.json for upstream checkpoint saving')
    fast = config['webshop_adapter'].get('fast_fp32')
    if fast:
        if sha256(Path(fast['proof'])) != fast['sha256']:
            raise ValueError('Fast-kernel proof changed')
        if importlib.metadata.version('fla-core') != '0.5.2':
            raise ValueError('Unverified FLA version')
    recovery = config['webshop_adapter'].get('recovery')
    if recovery:
        resume = Path(config['trainer']['resume_from_path'])
        for name, digest in recovery['checkpoint_files'].items():
            if sha256(resume/name) != digest:
                raise ValueError(f'Recovery checkpoint changed: {name}')
    from .no_demo_prompt import no_demo_prefix
    from .generate_teacher import load_prompt
    snapshot = Path(config['webshop_adapter']['snapshot'])
    prefix = no_demo_prefix(load_prompt(None), snapshot)
    report = {'upstream_commit': COMMIT, 'model': str(model), 'compatibility': compatibility,
              'export_sha256': sha256(model/'export_manifest.json'),
              'sft_data_sha256': sha256(snapshot/'teacher_sft.jsonl'),
              'prompt': prefix, 'gpu_smoke_passed': False,
              'versions': {name: importlib.metadata.version(name) for name in
                           ['torch', 'transformers', 'peft', 'vllm', 'ray', 'tensordict']}}
    write_json(prepared/'launch_preflight.json', report)
    if not args.run:
        print('Source/model/prompt checks passed. No GPU work started.')
        return
    if compatibility is None:
        raise RuntimeError('Qwen3.5 training requires the reviewed --runtime compatibility copy')
    if sys.platform != 'linux':
        raise RuntimeError('Run official veRL/Ray/vLLM on the Linux GPU server')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(runtime), str(ROOT), env.get('PYTHONPATH', '')])
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONUNBUFFERED'] = '1'
    # Match the working SFT evaluation backend on this RTX 6000D runtime.
    env['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    env['WEBSHOP_RL_AUDIT_DIR'] = str(prepared/'checkpoints')
    env['WEBSHOP_RL_FLA'] = '1' if fast else '0'
    # Official trainer is retained, including its checkpoint and weight-sync paths.
    subprocess.run([sys.executable, str(prepared/'main_webshop.py')], env=env,
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
