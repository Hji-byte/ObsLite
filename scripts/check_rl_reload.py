"""Reload a trusted smoke adapter and generate from an actual collected prompt."""
import argparse
from pathlib import Path

from .common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-export', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    batch = torch.load(args.rollout, map_location='cpu', weights_only=False)['batch']
    prompt_width = batch['input_ids'].shape[-1] - batch['responses'].shape[-1]
    prompt = batch['input_ids'][0, :prompt_width]
    prompt = prompt[batch['attention_mask'][0, :prompt_width].bool()][None].cuda()
    assert prompt.numel() > 0
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_export, dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.sft_export)
    with torch.inference_mode():
        result = model.generate(
            input_ids=prompt, attention_mask=torch.ones_like(prompt),
            do_sample=False, max_new_tokens=128, return_dict_in_generate=True,
            output_scores=True, pad_token_id=tokenizer.pad_token_id)
    assert result.scores and all(torch.isfinite(scores).all() for scores in result.scores)
    generated = result.sequences[0, prompt.shape[-1]:].tolist()
    assert generated
    write_json(args.output, {
        'generation_reload_verified': True, 'adapter': str(args.adapter),
        'prompt_tokens': prompt.numel(), 'generated_tokens': len(generated),
        'finite_generation_scores': True, 'text': tokenizer.decode(generated),
        'scope': 'checkpoint reload and generation only; not a success-rate evaluation'})
    print(args.output)


if __name__ == '__main__':
    main()
