"""Build teacher-based component cases and run the existing FP32/FLA gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .common import sha256, write_json
from .context_policy import ObservationMaskingPolicy
from .train_student import template


def prepare_cases(model, data, output, method, tokenizer=None):
    import torch
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if output.exists():
        raise ValueError('Use a fresh check directory')
    policy = ObservationMaskingPolicy() if method == 'mask4' else None
    samples, skipped = [], 0
    for line in data.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        # The fixed evaluation tasks are 0-499; never use them for this check.
        if int(row['task_id']) < 500:
            raise ValueError('Evaluation task found in teacher check data')
        messages = row['messages']
        if messages[-1]['role'] != 'assistant':
            raise ValueError('Teacher sample must end with an assistant action')
        prompt = messages[:-1]
        if policy:
            prompt, _ = policy.apply(prompt, 1)
        ids = template(tokenizer, prompt, generation=True)
        full = template(tokenizer, prompt + [messages[-1]])
        if full[:len(ids)] != ids:
            raise ValueError('Teacher response does not extend generation prompt')
        response = full[len(ids):]
        if not ids or not response or len(ids) > 31744 or len(response) > 1024:
            skipped += 1
            continue
        samples.append((len(full), row['id'], ids, response))
    if len(samples) < 3:
        raise ValueError('Need at least three eligible teacher samples')
    samples.sort(key=lambda x: (x[0], x[1]))
    selected = [samples[0], samples[len(samples)//2], samples[-1]]
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError('Tokenizer requires a pad token')
    cases, records = [], []
    for label, (length, sample_id, ids, response) in zip(['short','median','long'], selected):
        left, right = 31744-len(ids), 1024-len(response)
        mask = torch.tensor([[0]*left + [1]*(len(ids)+len(response)) + [0]*right], dtype=torch.long)
        cases.append({'label': label, 'batch': {
            'input_ids': torch.tensor([[pad]*left + ids + response + [pad]*right], dtype=torch.long),
            'attention_mask': mask,
            'position_ids': (mask.cumsum(-1)-1).clamp(min=0),
            'responses': torch.tensor([response + [pad]*right], dtype=torch.long)}})
        records.append({'label':label, 'sample_id':sample_id, 'active_tokens':length,
                        'response_tokens':len(response)})
    output.mkdir(parents=True)
    torch.save(cases, output/'cases.pt')
    write_json(output/'manifest.json', {'source':str(data.resolve()), 'source_sha256':sha256(data),
        'cases_sha256':sha256(output/'cases.pt'), 'cases':records, 'method':method,
        'eligible_samples':len(samples), 'out_of_budget_check_samples':skipped,
        'source_kind':'teacher SFT data; not on-policy rollout', 'gpu_used':False,
        'scope':'startup component check, not full PPO validation or a throughput measurement'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['mask4','full'],default='mask4')
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh check directory')
    prepare_cases(a.model,a.data,a.output/'cases',a.method)
    for backend, name in [('torch','reference'),('fla','candidate')]:
        subprocess.run([sys.executable,'-m','scripts.benchmark_rl_fast','--cases',str(a.output/'cases'),
            '--model',str(a.model),'--backend',backend,'--precision','fp32',
            '--output',str(a.output/name),'--run-gpu'],check=True)
    subprocess.run([sys.executable,'-m','scripts.compare_rl_fast',
        '--reference',str(a.output/'reference'),'--candidate',str(a.output/'candidate'),
        '--output',str(a.output/'proof.json')],check=True)

if __name__ == '__main__':
    main()
