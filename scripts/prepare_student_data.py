"""Freeze completed teacher records; provisional snapshots cannot start training."""
import argparse
from pathlib import Path
from .common import read_json, read_jsonl, write_jsonl, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose an empty snapshot directory')
    tasks = read_jsonl(args.dataset)
    assert len(tasks) == 2000 and len({r['task_id'] for r in tasks}) == 2000
    config = read_json(args.teacher_run / 'manifest.json')
    assert config['temperature'] == 0 and config['dataset_sha256'] == sha256(args.dataset)
    from .generate_teacher import load_prompt
    if config['prompt'] != load_prompt(None):
        raise ValueError('Teacher run differs from the shared prompt configuration')
    rows, completed = [], []
    for task in tasks:
        path = args.teacher_run / 'episodes' / f"{task['task_id']}.json"
        if not path.exists():
            continue
        e = read_json(path)
        assert e['task_id'] == task['task_id'] and e['status'] == 'completed'
        completed.append(task['task_id'])
        if e['trainable']:
            assert e['success'] and e['done'] and abs(e['reward'] - 1) < 1e-8
            assert all(not s.get('error') and s['finish_reason'] == 'stop' for s in e['steps'])
            assert e['sft_messages'][0] == config['prompt'][0]
            assert e['sft_messages'][-1]['role'] == 'assistant'
            rows.append({'task_id': e['task_id'], 'messages': e['sft_messages']})
    if len(completed) != len(tasks):
        raise ValueError('Teacher generation is incomplete; finish all tasks before exporting SFT data')
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / 'teacher_sft.jsonl', rows)
    print(f'Snapshot: {len(completed)}/2000 completed; {len(rows)} clean trajectories')


if __name__ == '__main__':
    main()
