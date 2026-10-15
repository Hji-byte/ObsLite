"""CPU-only import/binding probe. Does not certify CUDA execution or speed."""
import argparse
import importlib.metadata
from pathlib import Path

from .common import write_json
from .rl_fast_path import select_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch', 'fla'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = select_backend(args.backend)
    import torch
    versions = {}
    for name in ['torch', 'transformers', 'peft', 'triton', 'fla-core', 'einops']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    assert versions['torch'] == '2.13.0' and versions['transformers'] == '5.15.0'
    if args.backend == 'fla':
        assert versions['fla-core'] == '0.5.2'
    write_json(args.output, {'import_and_binding_passed': True, 'selected': selected,
                            'versions': versions, 'torch_cuda_build': torch.version.cuda,
                            'cuda_initialized': torch.cuda.is_initialized(),
                            'gpu_execution_verified': False, 'speedup_verified': False})
    print(args.output)


if __name__ == '__main__':
    main()
