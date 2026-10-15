"""Prepare the pinned GiGPO launch files without loading a model or using a GPU."""
import argparse
import json
from pathlib import Path
import random
import shutil

from .common import ROOT, read_json, write_json, sha256

COMMIT = '20bd331bdbc9026a5668e11362178e10ab7400c8'
CONFIG = ROOT / 'configs/grpo.yaml'


def put(config, dotted, value):
    parts = dotted.split('.')
    node = config
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def build_config(model_path, output, snapshot, eto_root, train_file, dev_file=None, smoke=False, algorithm='grpo'):
    import yaml
    config = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    adaptations = {
        'algorithm.adv_estimator': algorithm,
        'actor_rollout_ref.model.path': str(model_path),
        'data.train_files': str(train_file), 'data.val_files': str(train_file),
        'trainer.default_local_dir': str(output),
        'trainer.experiment_name': algorithm,
        'webshop_adapter.eto_root': str(eto_root),
        'webshop_adapter.snapshot': str(snapshot),
    }
    if smoke:
        adaptations.update({'data.train_batch_size': 2, 'data.val_batch_size': 2,
                            'trainer.total_training_steps': 2, 'trainer.save_freq': 1})
    for key, value in adaptations.items():
        put(config, key, value)
    return config, {}, adaptations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='SFT merged export, not the raw base')
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--eto-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--task-list', type=Path, help='Frozen task-list directory; required for formal training')
    p.add_argument('--training-steps', type=int, help='Explicit trainer iteration budget')
    p.add_argument('--save-every', type=int, help='Checkpoint interval, including a deliberate early pause')
    p.add_argument('--train-batch-size', type=int, choices=[2, 16], help='Use 16 for full-batch smoke')
    p.add_argument('--algorithm', choices=['grpo'], default='grpo')
    p.add_argument('--fast-fp32-proof', type=Path, help='Passing FP32 FLA component comparison report')
    p.add_argument('--vllm-max-num-seqs', type=int, choices=[8, 16, 32],
                   help='Explicit measured generation concurrency; does not change rollout group size')
    p.add_argument('--resume-smoke', type=Path,
                   help='Explicitly continue a verified step-1 smoke checkpoint; not exact dataloader replay')
    p.add_argument('--environment-python', type=Path,
                   help='Separate CPU WebShop interpreter retaining NumPy 1.x')
    p.add_argument('--context-policy', choices=['full_history', 'mask_old_observations'],
                   default='full_history')
    p.add_argument('--keep-recent-history', type=int, default=4,
                   help='Prior observations kept in addition to the task and current observation')
    p.add_argument('--skip-initial-validation', action='store_true',
                   help='Disable the pre-training dev rollout for a matched fixed-step pilot')
    p.add_argument('--test-every', type=int,
                   help='Compatibility option: only -1 is supported (validation disabled)')
    p.add_argument('--max-checkpoints', type=int,
                   help='Number of actor checkpoints to retain; omit to use the default')
    args = p.parse_args()
    if not args.smoke and not args.task_list:
        raise ValueError('Formal training requires --task-list; refusing the default full task pool')
    if args.training_steps is not None and args.training_steps < 1:
        raise ValueError('training-steps must be positive')
    if args.test_every not in (None, -1):
        raise ValueError('No development split is distributed; training validation must remain disabled')
    if args.max_checkpoints is not None and args.max_checkpoints < 1:
        raise ValueError('max-checkpoints must be positive')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    import yaml
    import pyarrow as pa
    import pyarrow.parquet as pq
    out.mkdir(parents=True, exist_ok=True)
    task_ids = list(range(1500, 12087))
    random.Random(42).shuffle(task_ids)
    if args.smoke:
        task_ids = task_ids[:2]
    task_list_record = None
    if args.task_list:
        from .common import canonical, read_jsonl
        source = args.task_list.resolve()
        split_rows = {s: read_jsonl(source/f'{s}_tasks.jsonl') for s in ['train', 'test']}
        task_list_record = {'counts': {s: len(rows) for s, rows in split_rows.items()}}
        for s, rows in split_rows.items():
            ids = [r['task_id'] for r in rows]
            if not ids or len(ids) != len(set(ids)) or any(r['split'] != s for r in rows):
                raise ValueError(f'Invalid {s} task list')
            for other, others in split_rows.items():
                if s != other and ({canonical(r['instruction_reference']) for r in rows} &
                                   {canonical(r['instruction_reference']) for r in others}):
                    raise ValueError('Task lists contain cross-split text overlap')
        (out/'task_list').mkdir()
        for split in ['train', 'test']:
            shutil.copy2(source/f'{split}_tasks.jsonl', out/'task_list'/f'{split}_tasks.jsonl')
        task_ids = [r['task_id'] for r in split_rows['train']]
        if args.smoke:
            task_ids = task_ids[:args.train_batch_size or 2]
    # IDs carry the dataset selection; environment construction audits materialized goals.
    for split, ids in [('train', task_ids)]:
        rows = [{'data_source': 'webshop', 'prompt': [{'role': 'user', 'content': 'WebShop task'}],
                 'env_kwargs': {'task_id': i}, 'reward_model': {'style': 'rule', 'ground_truth': ''},
                 'extra_info': {'task_id': i, 'split': split}} for i in ids]
        pq.write_table(pa.Table.from_pylist(rows), out / f'{split}.parquet')
    config, official, adaptations = build_config(args.model, out / 'checkpoints', args.snapshot,
        args.eto_root, out/'train.parquet', None, args.smoke, args.algorithm)
    config['webshop_adapter']['environment_python'] = str(args.environment_python) if args.environment_python else None
    config['webshop_adapter']['token_rollout_limit'] = -1 if args.smoke else 2
    if args.context_policy == 'mask_old_observations':
        if args.keep_recent_history < 0:
            raise ValueError('keep-recent-history must be non-negative')
        config['webshop_adapter']['context_policy'] = {
            'type': 'mask_old_observations',
            'keep_recent_history': args.keep_recent_history,
            'preserve_error_observations': True,
        }
    if task_list_record:
        config['webshop_adapter']['fixed_task_list'] = str(out/'task_list')
    for key, value in [('trainer.total_training_steps', args.training_steps),
                       ('data.train_batch_size', args.train_batch_size),
                       ('trainer.save_freq', args.save_every),
                       ('actor_rollout_ref.rollout.max_num_seqs', args.vllm_max_num_seqs),
                       ('trainer.val_before_train', False if args.skip_initial_validation else None),
                       ('trainer.test_freq', args.test_every),
                       ('trainer.max_actor_ckpt_to_keep', args.max_checkpoints)]:
        if value is not None:
            put(config, key, value)
            adaptations[key] = value
    if args.fast_fp32_proof:
        proof_path = args.fast_fp32_proof.resolve()
        proof = json.loads(proof_path.read_text())
        if not proof['all_passed'] or proof['precision'] != 'fp32':
            raise ValueError('FP32 FLA comparison must pass before preparing the candidate')
        for role in ['reference', 'candidate']:
            if sha256(Path(proof[f'{role}_report'])) != proof[f'{role}_report_sha256']:
                raise ValueError('Benchmark report changed')
        for role in ['actor', 'ref']:
            for suffix, value in {'forward_autocast': False, 'fsdp_config.model_dtype': 'fp32',
                'fsdp_config.mixed_precision.param_dtype': 'fp32',
                'fsdp_config.mixed_precision.reduce_dtype': 'fp32',
                'fsdp_config.mixed_precision.buffer_dtype': 'fp32'}.items():
                key = f'actor_rollout_ref.{role}.{suffix}'
                put(config, key, value)
                adaptations[key] = value
        config['webshop_adapter']['fast_fp32'] = {'proof': str(proof_path), 'sha256': sha256(proof_path)}
    if args.resume_smoke:
        resume = args.resume_smoke.resolve()
        if not args.smoke or resume.name != 'global_step_1':
            raise ValueError('Recovery is restricted to step-1 of the two-iteration smoke')
        recovery = json.loads((resume/'recovery_audit.json').read_text())
        if not recovery['verified']:
            raise ValueError('Checkpoint recovery audit failed')
        for name, digest in recovery['checkpoint_files'].items():
            if sha256(resume/name) != digest:
                raise ValueError(f'Recovery checkpoint changed: {name}')
        for key, value in {'trainer.resume_mode': 'resume_path',
                           'trainer.resume_from_path': str(resume)}.items():
            put(config, key, value)
            adaptations[key] = value
        config['webshop_adapter']['recovery'] = recovery
    # The adaptation dictionary shares this mapping, so the manifest records it too.
    (out/'config').mkdir()
    (out/'config/ppo_trainer.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (out/'main_webshop.py').write_text('from scripts.run_rl import run\nrun(globals())\n', encoding='utf-8')
    write_json(out/'manifest.json', {'upstream_commit': COMMIT,
        'official_overrides': official, 'adaptations': adaptations,
        'config_source': {'path': 'configs/grpo.yaml', 'sha256': sha256(CONFIG)},
        'train_ids': task_ids, 'validation_enabled': False,
        'task_list_summary': task_list_record,
        'prepared_files': {str(f.relative_to(out).as_posix()): sha256(f) for f in
                           [out/'main_webshop.py', out/'config/ppo_trainer.yaml',
                            out/'train.parquet'] +
                           (sorted(p for p in (out/'task_list').rglob('*') if p.is_file()) if task_list_record else [])},
        'integration_files': {f'scripts/{name}': sha256(Path(__file__).with_name(name))
                              for name in ['rl_protocol.py', 'rl_bridge.py', 'prepare_rl.py', 'run_rl.py',
                                           'context_policy.py',
                                           'rl_environment_process.py', 'rl_runtime_compat.py',
                                           'rl_padding.py', 'rl_fast_path.py',
                                           'prepare_rl_runtime.py',
                                           'launch_rl.py', 'common.py', 'generate_teacher.py',
                                           'no_demo_prompt.py', 'train_student.py']},
        'validation_status': 'prepared_only; GPU and environment integration pending',
        'budget_unit': 'official trainer iteration, not necessarily one optimizer update'})
    print(json.dumps({'prepared': str(out), 'train_tasks': len(task_ids),
                      'validation_enabled': False, 'gpu_used': False}))


if __name__ == '__main__':
    main()
