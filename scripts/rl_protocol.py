"""SFT-compatible WebShop transitions for the pinned official GiGPO collector.

No model/GPU imports here. Environment operations are serialized; each rollout
owns a separate session even when it shares the immutable product server.
"""
from copy import deepcopy
import json
import math
import uuid

from .common import observation_message
from .generate_teacher import parse_action, is_full_success


class WebshopSession:
    def __init__(self, env, system_message, max_steps=20, *, reward_scale=10.0):
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('reward_scale must be finite and positive')
        self.reward_scale = float(reward_scale)
        self.env = env
        self.system = deepcopy(system_message)
        self.max_steps = max_steps
        self.done = True

    def reset(self, task_id, split):
        lower, upper = (1500, 12087) if split == 'train' else (500, 1500)
        if type(task_id) is not int or not lower <= task_id < upper:
            raise ValueError(f'Task {task_id} is outside the {split} split')
        self.task_id, self.split = task_id, split
        result = self.env.reset(task_id)
        self.observation = result[0] if isinstance(result, tuple) else result
        self.turn = 0
        self.done = False
        self.task_score = 0.0
        self.reason = None
        self.messages = [deepcopy(self.system)]
        self.records = []
        self._append_observation()
        return self.view()

    def _append_observation(self):
        clicks = [] if self.done else self.env.get_available_actions()['clickables']
        self.messages.append({'role': 'user', 'content': observation_message(
            self.observation, clicks, self.done)})

    def view(self):
        # Official grouping uses the current observation, not the entire history.
        # Task identity is also retained; no cross-task grouping is possible.
        anchor = json.dumps([self.task_id, self.messages[-1]['content']], ensure_ascii=False)
        return {'text': self.messages[-1]['content'], 'messages': deepcopy(self.messages),
                'anchor': anchor}

    def step(self, text):
        if self.done:
            return self.view(), 0.0, True, self.info(True, None)
        self.turn += 1
        before = self.view()
        self.messages.append({'role': 'assistant', 'content': text})
        reward, error, action = 0.0, None, None
        try:
            action = parse_action(text)
            format_valid = True
        except ValueError:
            format_valid = False
            error = 'invalid_format'
            self.observation = ('Invalid format. Use Thought: ... followed by one '
                                'Action: search[...] or Action: click[...].')
        if action is not None:
            name, argument = action.split('[', 1)
            argument = argument[:-1].lower()
            available = self.env.get_available_actions()
            allowed = ((name == 'search' and bool(argument.strip())) or
                       (name == 'click' and argument in available['clickables'] and argument != 'search'))
            if not allowed:
                error = 'invalid_action'
                self.observation = 'Invalid action!'
            else:
                try:
                    self.observation, value, terminated, _ = self.env.step(action)
                except AssertionError:
                    error = 'invalid_action'
                    self.observation = 'Invalid action!'
                else:
                    reward = float(value)
                    if not math.isfinite(reward) or not 0 <= reward <= 1:
                        raise ValueError(f'Unexpected environment reward: {reward}')
                    if reward and not terminated:
                        raise ValueError('Expected terminal-only WebShop reward')
                    if terminated:
                        self.done = True
                        self.reason = 'purchase'
                        self.task_score = reward
        if not self.done and self.turn >= self.max_steps:
            self.done = True
            self.reason = 'max_steps'
        self._append_observation()
        info = self.info(format_valid, error)
        # Preserve native scores for evaluation; upstream applies the format
        # penalty separately, after computing returns from these scaled rewards.
        raw_reward = reward
        reward = raw_reward * self.reward_scale
        self.records.append({'task_id': self.task_id, 'turn': self.turn, 'action': action,
                             'response': text, 'anchor': before['anchor'], 'reward': reward,
                             'raw_reward': raw_reward, 'reward_scale': self.reward_scale,
                             **info})
        return self.view(), reward, self.done, info

    def info(self, valid, error):
        return {'is_action_valid': valid, 'execution_error': error,
                'task_score': self.task_score, 'won': is_full_success(
                    self.reason == 'purchase', self.task_score),
                'terminated': self.reason == 'purchase',
                'truncated': self.reason == 'max_steps', 'terminate_reason': self.reason}


class WebshopManager:
    """Duck-typed official EnvironmentManagerBase, with explicit task IDs."""
    def __init__(self, env_factory, system_message, split, group_size, max_steps=20, log_path=None,
                 *, reward_scale=10.0):
        self.factory, self.system = env_factory, system_message
        self.split, self.group_size, self.max_steps = split, group_size, max_steps
        self.sessions = []
        self.log_path = log_path
        self.batch_id = None
        self.reward_scale = reward_scale

    def reset(self, kwargs):
        if kwargs is None or len(kwargs) == 0:
            raise ValueError('Explicit env_kwargs.task_id is required; no random fallback')
        ids = [int(k['task_id']) for k in kwargs]
        if len(ids) % self.group_size:
            raise ValueError('Incomplete rollout group')
        for start in range(0, len(ids), self.group_size):
            if len(set(ids[start:start+self.group_size])) != 1:
                raise ValueError('Each group must reset to the same task')
        while len(self.sessions) < len(ids):
            self.sessions.append(WebshopSession(self.factory(), self.system, self.max_steps,
                                                reward_scale=self.reward_scale))
        self.active = self.sessions[:len(ids)]
        self.batch_id = uuid.uuid4().hex
        views = [s.reset(task_id, self.split) for s, task_id in zip(self.active, ids)]
        return self._views(views), [s.info(True, None) for s in self.active]

    @staticmethod
    def _views(views):
        return {'text': [v['text'] for v in views], 'image': None,
                'anchor': [v['anchor'] for v in views],
                'messages': [v['messages'] for v in views]}

    def step(self, texts):
        import numpy as np
        if len(texts) != len(self.active):
            raise ValueError('Action/session count mismatch')
        was_active = [not s.done for s in self.active]
        results = [s.step(t) for s, t in zip(self.active, texts)]
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.with_suffix('.events.jsonl').open('a', encoding='utf-8') as f:
                for i, (s, active) in enumerate(zip(self.active, was_active)):
                    if active:
                        f.write(json.dumps({'batch_id': self.batch_id, 'trajectory': i,
                                            **s.records[-1]}, ensure_ascii=False) + '\n')
        views, rewards, dones, infos = zip(*results)
        return self._views(views), np.array(rewards), np.array(dones), list(infos)

    def success_evaluator(self, **kwargs):
        import numpy as np
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as f:
                for i, s in enumerate(self.active):
                    f.write(json.dumps({'batch_id': self.batch_id, 'trajectory': i,
                        'task_id': s.task_id, 'split': s.split,
                        'reward': s.task_score * s.reward_scale,
                        'raw_reward': s.task_score, 'reward_scale': s.reward_scale,
                        'steps': s.records, 'messages': s.messages,
                        **s.info(True, None)}, ensure_ascii=False) + '\n')
        return {'success_rate': np.array([s.info(True, None)['won'] for s in self.active], dtype=float),
                'webshop_task_score': np.array([s.task_score for s in self.active]),
                'webshop_timeout_rate': np.array([s.reason == 'max_steps' for s in self.active], dtype=float)}

    def close(self):
        for session in self.sessions:
            session.env.close()
