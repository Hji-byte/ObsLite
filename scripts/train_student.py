"""WebShop assistant-only QLoRA; uses teacher trajectories directly without truncation."""
import argparse
from pathlib import Path
from collections.abc import Mapping
from .common import read_json, read_jsonl, write_json, sha256

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
           'in_proj_a', 'in_proj_b', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']


def template(tokenizer, messages, generation=False):
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation,
                                          enable_thinking=False, return_dict=False, truncation=False)
    return list(value['input_ids'] if isinstance(value, Mapping) else value)


def encode(tokenizer, row, last_only=False):
    messages = row['messages']
    assert [m['role'] for m in messages] == ['system'] + ['user', 'assistant'] * ((len(messages)-1)//2)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False,
                                             add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    full = encoded['input_ids']
    assert full == template(tokenizer, messages)
    labels = [-100] * len(full)
    cursor = 0
    for message_index, message in enumerate(messages):
        if '<|im_start|>' in message['content'] or '<|im_end|>' in message['content']:
            raise ValueError('Message contains chat boundary tokens')
        header = '<|im_start|>' + message['role'] + '\n'
        start = rendered.index(header, cursor) + len(header)
        stop = rendered.index('<|im_end|>', start)
        body = rendered[start:stop]
        empty_think = '<think>\n\n</think>\n\n'
        if message['role'] == 'assistant' and body == empty_think + message['content']:
            start += len(empty_think)
            body = rendered[start:stop]
        if body != message['content']:
            raise ValueError(f"Chat template altered message content: {row['task_id']}")
        cursor = stop + len('<|im_end|>')
        if message['role'] != 'assistant' or (last_only and message_index != len(messages)-1):
            continue
        for j,(left,right) in enumerate(encoded['offset_mapping']):
            if left >= start and right <= cursor and right > left:
                labels[j] = full[j]
            elif right > start and left < cursor:
                raise ValueError('Token straddles assistant supervision boundary')
    assert any(t != -100 for t in labels)
    return {'input_ids': full, 'attention_mask': [1]*len(full), 'labels': labels}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=10240)
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    data_hash = sha256(args.data/'teacher_sft.jsonl')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.data/'teacher_sft.jsonl')
    assert len({r['task_id'] for r in rows}) == len(rows)
    assert all(1500 <= r['task_id'] < 12087 for r in rows)
    encoded = [encode(tokenizer, row) for row in rows]
    selected = [(r,e) for r,e in zip(rows,encoded) if len(e['input_ids']) <= args.max_length]
    assert selected, 'No training rows fit the length limit'
    report = {'source_rows':len(rows), 'kept':len(selected),
              'max_length':args.max_length, 'largest_source':max(len(e['input_ids']) for e in encoded),
              'dropped_task_ids':[r['task_id'] for r,e in zip(rows,encoded) if len(e['input_ids'])>args.max_length],
              'supervised_tokens':sum(sum(t!=-100 for t in e['labels']) for _,e in selected),
              'mask':'all assistant turns and end markers; system/user/generation prefix masked',
              'truncation':False}
    write_json(args.output.with_name(args.output.name+'.preflight.json'), report)
    print(report, flush=True)
    if args.preflight:
        return
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to one GPU')
    config = {'model':args.model, 'data_sha256':data_hash, 'epochs':args.epochs,
              'max_length':args.max_length, 'lr':2e-4, 'batch_size':2, 'gradient_accumulation':4,
              'lora_r':64, 'lora_alpha':128, 'lora_dropout':0.05, 'targets':TARGETS,
              'seed':42, 'quantization':'nf4_double', 'smoke':args.smoke,
              'script_sha256':sha256(Path(__file__)), 'selected_task_ids':[r['task_id'] for r,e in selected]}
    checkpoint = None
    if args.resume:
        assert read_json(args.output/'training_manifest.json') == config
        checkpoint = get_last_checkpoint(str(args.output))
        assert checkpoint, 'No optimizer checkpoint'
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output exists; resume explicitly')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'training_manifest.json',config)
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),device_map={'':0})
    model.config.use_cache=False
    if hasattr(model.config,'text_config'): model.config.text_config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=.05,target_modules=TARGETS,
                                        bias='none',task_type='CAUSAL_LM'))
    train_args=TrainingArguments(output_dir=str(args.output),per_device_train_batch_size=2,
        gradient_accumulation_steps=4,num_train_epochs=args.epochs,max_steps=2 if args.smoke else -1,
        learning_rate=2e-4,bf16=bf16,fp16=not bf16,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=1,save_steps=25,
        save_total_limit=2,optim='adamw_torch_fused',report_to='none',seed=42,data_seed=42)
    training_rows=[e for _,e in selected]
    if args.smoke: training_rows=sorted(training_rows,key=lambda e:-len(e['input_ids']))[:16]
    trainer=Trainer(model=model,args=train_args,train_dataset=Dataset.from_list(training_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100))
    result=trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(args.output/'final'))
    tokenizer.save_pretrained(str(args.output/'final'))
    trainer.save_state()
    write_json(args.output/'result.json',{'global_step':trainer.state.global_step,'metrics':result.metrics})


if __name__ == '__main__':
    main()
