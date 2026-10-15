"""Use the installed verl-agent entry with our environment hook inside Ray."""
from pathlib import Path
from importlib.util import find_spec


def patched_entry(source):
    needle = '    def run(self, config):\n'
    if source.count(needle) != 1:
        raise ValueError('Unsupported verl-agent TaskRunner; install the version in README.md')
    return source.replace(needle, needle +
        '        from scripts.rl_bridge import install\n        install()\n')


def run(namespace):
    spec = find_spec('verl.trainer.main_ppo')
    if spec is None or not spec.origin:
        raise RuntimeError('Install verl-agent and set PYTHONPATH as documented')
    source = patched_entry(Path(spec.origin).read_text(encoding='utf-8'))
    # Preserve the prepared entry file location so Hydra finds its local config/.
    # Execute in the actual __main__ namespace, retaining Ray's serialization behavior.
    exec(compile(source, namespace['__file__'], 'exec'), namespace)
