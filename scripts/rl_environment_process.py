"""Local JSON pipe bridge to the frozen NumPy 1.x WebShop environment."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


class EnvironmentProcess:
    def __init__(self, python, config):
        from omegaconf import OmegaConf
        from .common import ROOT
        log_path = Path(config.trainer.default_local_dir) / 'environment_process.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open('a', encoding='utf-8')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['PYTHONPATH'] = str(ROOT)  # Do not leak the model runtime's dependencies.
        self.process = subprocess.Popen(
            [str(python), '-u', '-m', 'scripts.rl_environment_process'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', cwd=ROOT, env=env)
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.remaining = {'train', 'dev'}

        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=read, daemon=True).start()
        try:
            result = self.call('init', config=OmegaConf.to_container(config, resolve=True))
            config.data.train_files = result['train_files']
        except BaseException:
            self.close()
            raise

    def call(self, method, **kwargs):
        with self.lock:
            if self.closed:
                raise RuntimeError('WebShop process is closed')
            self.process.stdin.write(json.dumps({'method': method, **kwargs}) + '\n')
            self.process.stdin.flush()
            try:
                line = self.responses.get(timeout=300)
            except queue.Empty as exc:
                raise RuntimeError('WebShop process timed out; inspect environment_process.log') from exc
            if line is None:
                raise RuntimeError('WebShop process exited; inspect environment_process.log')
            result = json.loads(line)
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result['result']

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()


class RemoteManager:
    def __init__(self, owner, split):
        self.owner, self.split = owner, split

    def reset(self, kwargs):
        # Parquet/NumPy integer scalars must be regular JSON integers.
        ids = [{'task_id': int(k['task_id'])} for k in kwargs]
        return self.owner.call('reset', split=self.split, kwargs=ids)

    def step(self, texts):
        import numpy as np
        views, rewards, dones, infos = self.owner.call('step', split=self.split, texts=list(texts))
        return views, np.asarray(rewards), np.asarray(dones, dtype=bool), infos

    def success_evaluator(self, **kwargs):
        import numpy as np
        return {key: np.asarray(value) for key, value in
                self.owner.call('success_evaluator', split=self.split).items()}

    def close(self):
        self.owner.remaining.discard(self.split)
        if not self.owner.remaining:
            self.owner.close()


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Environment imports/search progress cannot corrupt JSON.
    managers = {}
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                method = request['method']
                if method == 'init':
                    from omegaconf import OmegaConf
                    from .rl_bridge import _make_envs_local
                    config = OmegaConf.create(request['config'])
                    managers['train'], managers['dev'] = _make_envs_local(config)
                    result = {'train_files': config.data.train_files}
                elif method == 'reset':
                    result = managers[request['split']].reset(request['kwargs'])
                elif method == 'step':
                    result = managers[request['split']].step(request['texts'])
                elif method == 'success_evaluator':
                    result = managers[request['split']].success_evaluator()
                else:
                    raise ValueError(f'Unknown environment method: {method}')
                response = {'result': json_value(result)}
            except Exception:
                import traceback
                response = {'error': traceback.format_exc()}
            protocol.write(json.dumps(response, ensure_ascii=False) + '\n')
            protocol.flush()
    finally:
        for manager in managers.values():
            manager.close()


if __name__ == '__main__':
    main()
