"""Prepare real rollout cases on CPU; explicitly gated GPU forward/backward benchmark.

The masked NLL+entropy objective exercises LoRA training, without an optimizer step.
This is a component benchmark, not an FSDP/PPO throughput or policy-quality claim.
"""
import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

from .common import sha256, write_json, read_json


def prepare(rollout, output):
    import torch
    if output.exists():
        raise ValueError('Use a fresh case directory')
    batch = torch.load(rollout, map_location='cpu', weights_only=False)['batch']
    lengths = batch['attention_mask'].sum(-1)
    order = lengths.argsort()
    chosen = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    labels = ['short', 'median', 'long']
    output.mkdir(parents=True)
    cases, rows = [], []
    for label, index in zip(labels, chosen):
        sample = {key: batch[key][index:index+1].clone() for key in
                  ['input_ids', 'attention_mask', 'position_ids', 'responses']}
        cases.append({'label': label, 'batch': sample})
        width = sample['responses'].shape[-1]
        rows.append({'label': label, 'index': index, 'active_tokens': int(lengths[index]),
                     'response_tokens': int(sample['attention_mask'][:, -width:].sum())})
    torch.save(cases, output/'cases.pt')
    report = {'source': str(rollout), 'source_sha256': sha256(rollout),
              'cases_sha256': sha256(output/'cases.pt'), 'cases': rows, 'gpu_used': False}
    write_json(output/'manifest.json', report)
    print(json.dumps(report))


def benchmark(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required for --run-gpu; CPU preparation is already sufficient')
    from .rl_fast_path import select_backend, crop_response_padding
    backend = select_backend(args.backend)
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    from .train_student import TARGETS
    from .rl_padding import trim_masked_prompt
    manifest = read_json(args.cases/'manifest.json')
    assert sha256(args.cases/'cases.pt') == manifest['cases_sha256']
    if args.output.exists():
        raise ValueError('Use a fresh benchmark output directory')
    args.output.mkdir(parents=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                attn_implementation='sdpa', device_map='cuda')
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS,
                                               lora_dropout=0.0, task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    cases = torch.load(args.cases/'cases.pt', map_location='cpu', weights_only=False)
    reports = []
    for case in cases:
        batch = {k: v.to('cuda') for k, v in case['batch'].items()}
        if args.trim_response:
            batch = crop_response_padding(batch)
        width = batch['responses'].shape[-1]
        ids, mask, positions = trim_masked_prompt(batch['input_ids'], batch['attention_mask'],
                                                batch['position_ids'], width)
        selected = mask[:, -width:].bool()
        times, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(args.repeats + 1):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                               use_cache=False, logits_to_keep=width+1).logits[:, :-1].float()
                # Keep response-padding logits in the untrimmed variant so the
                # benchmark exercises their real output/entropy memory overhead.
                log_probs = logits.log_softmax(-1)
                nll = -log_probs.gather(-1, batch['responses'].unsqueeze(-1)).squeeze(-1)
                entropy = -(log_probs.exp()*log_probs).sum(-1)
                loss = (nll[selected] - .001*entropy[selected]).mean()
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            assert torch.isfinite(loss)
            if repeat:
                times.append(elapsed)
                losses.append(float(loss.detach()))
        gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        torch.save({'gradients': gradients,
                    'selected_logprobs': log_probs.gather(-1, batch['responses'].unsqueeze(-1)).squeeze(-1)[selected].detach().cpu(),
                    'loss': losses[-1]}, args.output/f'{case["label"]}.pt')
        reports.append({'label': case['label'], 'input_tokens': ids.shape[-1],
                        'response_slots': width, 'response_tokens': int(selected.sum()),
                        'seconds': times, 'median_seconds': statistics.median(times),
                        'losses': losses, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        del logits, log_probs, nll, entropy, loss, gradients
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(json.dumps(reports[-1]), flush=True)
    assert backend['counters']['chunk_calls'] > 0
    write_json(args.output/'report.json', {'backend': backend, 'trim_response': args.trim_response,
        'precision': args.precision,
        'cases_sha256': manifest['cases_sha256'], 'model': str(args.model), 'results': reports,
        'model_sha256': sha256(args.model/'model.safetensors'),
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft', 'triton']},
        'adapter_sha256': sha256(args.adapter/'adapter_model.safetensors') if args.adapter else None,
        'gpu': torch.cuda.get_device_name(), 'optimizer_steps': 0,
        'scope': 'single-process LoRA masked NLL+entropy forward/backward; not full FSDP/PPO'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-rollout', type=Path)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--adapter', type=Path, help='Saved RL adapter, loaded without updating it')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['torch', 'fla'], default='torch')
    parser.add_argument('--trim-response', action='store_true')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.prepare_rollout:
        if args.run_gpu:
            parser.error('Preparation and GPU execution are separate modes')
        prepare(args.prepare_rollout, args.output)
    elif args.run_gpu and args.cases and args.model and args.repeats >= 1:
        benchmark(args)
    else:
        parser.error('Use --prepare-rollout, or --run-gpu with --cases and --model')


if __name__ == '__main__':
    main()
