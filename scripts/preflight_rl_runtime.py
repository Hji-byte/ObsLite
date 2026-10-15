"""CPU-only import checks for the pinned veRL integration (never loads weights)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, write_json


PROBES = {
    'tensor_runtime': 'import torch; from tensordict import TensorDict; '
                      'd=TensorDict({"x":torch.ones(2,3)},batch_size=[2]); '
                      'assert d[0]["x"].sum().item()==3',
    'qwen35_model': 'from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM',
    'trainer': 'import verl.trainer.main_ppo',
    'actor_worker': 'import verl.workers.fsdp_workers',
    'rollout': 'import verl.workers.rollout.vllm_rollout.vllm_rollout_spmd',
    'weight_sync': 'import verl.workers.sharding_manager.fsdp_vllm',
    'collector_hook': 'from scripts.rl_bridge import install; install()',
    'tensor_lora': 'from scripts.rl_runtime_compat import check_tensor_lora; check_tensor_lora()',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', choices=list(PROBES), action='append')
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join([str(args.upstream.resolve()), str(ROOT)]))
    results = {}
    for name, code in PROBES.items():
        if args.probe and name not in args.probe:
            continue
        try:
            run = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                                 text=True, capture_output=True, timeout=90)
            results[name] = {'passed': run.returncode == 0, 'returncode': run.returncode,
                             'stdout': run.stdout[-12000:], 'stderr': run.stderr[-18000:]}
        except subprocess.TimeoutExpired:
            results[name] = {'passed': False, 'error': '90-second import timeout'}
        write_json(args.output, {'cpu_only': True, 'gpu_smoke_passed': False, 'probes': results})
        print(json.dumps({'probe': name, **results[name]}), flush=True)
    if not all(r['passed'] for r in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
