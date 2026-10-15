"""Adapt environment rewards/prompts; keep upstream penalties, advantages and loss."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

from .common import canonical, read_json, write_json
from .generate_teacher import create_environment, load_prompt, verify_upstream
from .no_demo_prompt import no_demo_prefix
from .rl_protocol import WebshopManager


def make_envs(config):
    from .rl_environment_process import EnvironmentProcess, RemoteManager
    python = config.webshop_adapter.get('environment_python')
    if not python:
        raise ValueError('Prepare with --environment-python pointing to the isolated WebShop runtime')
    owner = EnvironmentProcess(python, config)
    return RemoteManager(owner, 'train'), RemoteManager(owner, 'dev')


def _make_envs_local(config):
    cfg = config.webshop_adapter
    if cfg.reward.type != 'native_score':
        raise ValueError('Expected native_score reward mapping')
    reward_scale = float(cfg.reward.scale)
    eto, snapshot = Path(cfg.eto_root), Path(cfg.snapshot)
    verify_upstream(eto)
    prefix = no_demo_prefix(load_prompt(eto), snapshot)
    root, assets = create_environment(eto, [], config.env.seed)
    forbidden = {canonical(g['instruction_text']) for g in root.server.goals[:1500]}
    fixed = cfg.get('fixed_task_list')
    if fixed:
        from .common import read_jsonl
        fixed = Path(fixed)
        for split in ['train', 'test']:
            for row in read_jsonl(fixed/f'{split}_tasks.jsonl'):
                if canonical(root.server.goals[row['task_id']]['instruction_text']) != canonical(row['instruction_reference']):
                    raise ValueError(f'Fixed task list instruction mismatch: {row["task_id"]}')

    # The official ID split can contain text duplicates after normalization.
    # Filter by the fully materialized catalog, before the upstream dataset loads.
    import pyarrow as pa
    import pyarrow.parquet as pq
    train_rows = pq.read_table(str(config.data.train_files)).to_pylist()
    kept, excluded = [], []
    for row in train_rows:
        task_id = int(row['env_kwargs']['task_id'])
        if not 1500 <= task_id < 12087:
            raise ValueError('Non-training task in train parquet')
        if canonical(root.server.goals[task_id]['instruction_text']) in forbidden:
            excluded.append(task_id)
        else:
            kept.append(row)
    if len(kept) < config.data.train_batch_size:
        raise ValueError('Not enough non-overlapping training tasks for the configured batch')
    if fixed and excluded:
        raise ValueError('Frozen task list unexpectedly lost tasks during environment audit')
    out = Path(config.trainer.default_local_dir)
    out.mkdir(parents=True, exist_ok=True)
    filtered = out/'train_verified.parquet'
    pq.write_table(pa.Table.from_pylist(kept), filtered)
    config.data.train_files = str(filtered)
    write_json(out/'task_pool_audit.json', {'kept_ids': [r['env_kwargs']['task_id'] for r in kept],
               'excluded_heldout_text_overlap': excluded, 'reason': 'SFT evaluation canonical normalization'})

    def factory():
        env = type(root)(observation_mode='text_rich', server=root.server,
                         session_prefix='rl_' + uuid.uuid4().hex + '_')
        original_reset = env.reset
        def reset(task_id):
            if task_id >= 1500 and canonical(root.server.goals[task_id]['instruction_text']) in forbidden:
                raise ValueError(f'Training instruction overlaps held-out set: {task_id}')
            result = original_reset(int(task_id))
            expected_goal = canonical(root.server.goals[task_id]['instruction_text'])
            if canonical(env.instruction_text.removeprefix('Instruction:').strip()) != expected_goal:
                raise ValueError('Environment reset selected a different task')
            return result
        env.reset = reset
        return env

    write_json(out / 'environment_manifest.json', assets)
    from omegaconf import OmegaConf
    write_json(out / 'effective_config.json', OmegaConf.to_container(config, resolve=True))
    train = WebshopManager(factory, prefix[0], 'train', config.env.rollout.n,
                           config.env.max_steps, out / 'train_trajectories.jsonl',
                           reward_scale=reward_scale)
    dev = WebshopManager(factory, prefix[0], 'dev', 1, config.env.max_steps,
                         out / 'dev_trajectories.jsonl', reward_scale=reward_scale)
    # Retain the owner for the lifetime of the shared server.
    train.server_owner = dev.server_owner = root
    return train, dev


def preprocess_single_sample(self, item, gen_batch, obs):
    import torch
    from .context_policy import context_policy_from_config, prepare_context
    from .train_student import template
    messages = deepcopy(obs['messages'][item])
    adapter = getattr(self.config, 'webshop_adapter', None)
    policy_config = adapter.get('context_policy') if adapter is not None else None
    policy = context_policy_from_config(policy_config)
    # WebshopSession receives one system message; its trajectory begins at index 1.
    messages, _ = prepare_context(messages, policy, trajectory_start=1)
    for msg in messages:
        if '<|im_start|>' in msg['content'] or '<|im_end|>' in msg['content']:
            raise ValueError('Chat boundary token in message content')
    ids = template(self.tokenizer, messages, generation=True)
    limit = self.config.data.max_prompt_length
    if not ids or len(ids) > limit:
        # Do not silently delete history or selectively drop trajectories.
        raise ValueError(f'RL context budget exceeded: {len(ids)} > {limit}; inspect rollout logs')
    pad = self.tokenizer.pad_token_id
    if pad is None:
        raise ValueError('Tokenizer needs a pad token')
    n_pad = limit - len(ids)
    mask = torch.tensor([0] * n_pad + [1] * len(ids), dtype=torch.long)
    positions = (mask.cumsum(0) - 1).clamp(min=0)
    return {'input_ids': torch.tensor([pad] * n_pad + ids),
            'attention_mask': mask, 'position_ids': positions,
            'raw_prompt_ids': ids, 'raw_prompt': messages,
            'anchor_obs': obs['anchor'][item], 'index': item,
            'data_source': gen_batch.non_tensor_batch['data_source'][item]}


def install():
    """Called inside the official Ray TaskRunner, not just in its parent process."""
    import agent_system.environments as environments
    from agent_system.multi_turn_rollout import TrajectoryCollector
    environments.make_envs = make_envs
    TrajectoryCollector.preprocess_single_sample = preprocess_single_sample

    import verl.trainer.ppo.ray_trainer as trainer_module
    original_advantage = trainer_module.compute_advantage
    if not getattr(original_advantage, '_webshop_audit', False):
        def advantage(*args, **kwargs):
            data = original_advantage(*args, **kwargs)
            mask = data.batch['response_mask'].bool()
            values = data.batch['advantages']
            import torch
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite GiGPO/GRPO advantages')
            # This reports available policy signal, not actual gradient magnitude.
            print(json.dumps({'webshop_signal': {
                'step_samples': len(values), 'response_tokens': int(mask.sum()),
                'nonzero_advantage_tokens': int(((values != 0) & mask).sum()),
                'format_valid_ratio': float(data.non_tensor_batch['is_action_valid'].mean())}}), flush=True)
            return data
        advantage._webshop_audit = True
        trainer_module.compute_advantage = advantage

    # Preserve exact emitted tokens in addition to decoded environment transcripts.
    original = TrajectoryCollector.gather_rollout_data
    if getattr(original, '_webshop_audit', False):
        return
    def gather(self, *args, **kwargs):
        import torch
        data = original(self, *args, **kwargs)
        out = Path(self.config.trainer.default_local_dir) / 'token_rollouts'
        out.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex
        limit = self.config.webshop_adapter.get('token_rollout_limit', -1)
        save_tokens = limit < 0 or len(list(out.glob('*.pt'))) < limit
        if save_tokens:
            torch.save({'batch': {k: v.cpu() for k, v in data.batch.items()},
                        'non_tensor_batch': data.non_tensor_batch}, out / (name + '.pt'))
        responses = data.batch['responses']
        eos = self.tokenizer.eos_token_id
        mask = data.batch['attention_mask'][:, -responses.shape[1]:]
        lengths = mask.sum(-1).long()
        last = responses.gather(1, (lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        eos_ids = [eos] if isinstance(eos, int) else (eos or [])
        ended = torch.zeros_like(last, dtype=torch.bool)
        for token_id in eos_ids:
            ended |= (last == token_id) & (lengths > 0)
        with (out / 'index.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'file': name + '.pt' if save_tokens else None, 'step_samples': len(responses),
                                'responses_without_eos': int((~ended).sum())}) + '\n')
        return data
    gather._webshop_audit = True
    TrajectoryCollector.gather_rollout_data = gather
