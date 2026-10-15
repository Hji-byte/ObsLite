"""Explicit, benchmark-only Qwen3.5 kernel selection and response-padding crop."""
import inspect


def function_bindings(function):
    found, seen = [], set()
    def visit(fn):
        if id(fn) in seen or not inspect.isfunction(fn):
            return
        seen.add(id(fn))
        found.append(f'{fn.__module__}.{fn.__name__}')
        for cell in fn.__closure__ or ():
            visit(cell.cell_contents)
    visit(function)
    return found


def select_backend(backend):
    if backend == 'fla':
        # transformers5.15 resolves attributes of fla, without importing nested
        # modules. Import before modeling_qwen3_5 evaluates its decorators.
        import fla.ops.gated_delta_rule  # noqa: F401
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    if backend == 'torch':
        module.torch_chunk_gated_delta_rule = inspect.unwrap(module.torch_chunk_gated_delta_rule)
    elif backend != 'fla':
        raise ValueError(backend)
    bindings = function_bindings(module.torch_chunk_gated_delta_rule)
    actual_fla = any(name.startswith('fla.ops.gated_delta_rule.') for name in bindings)
    if actual_fla != (backend == 'fla'):
        raise RuntimeError(f'Unexpected kernel binding: {bindings}; use a fresh Python process')
    original = module.torch_chunk_gated_delta_rule
    counters = {'chunk_calls': 0}
    def counted(*args, **kwargs):
        counters['chunk_calls'] += 1
        if counters['chunk_calls'] == 1:
            print({'webshop_kernel_first_call': backend}, flush=True)
        return original(*args, **kwargs)
    module.torch_chunk_gated_delta_rule = counted
    return {'backend': backend, 'bindings': bindings, 'counters': counters}


def crop_response_padding(batch):
    """Trim a common all-masked response suffix, keeping all real response tokens."""
    width = batch['responses'].shape[-1]
    mask = batch['attention_mask'][:, -width:]
    active = mask.bool().any(dim=0).nonzero()
    if not active.numel():
        raise ValueError('No active response tokens')
    keep = int(active[-1, 0]) + 1
    drop = width - keep
    output = dict(batch)
    if drop:
        for name in ['input_ids', 'attention_mask', 'position_ids']:
            output[name] = batch[name][..., :-drop]
        output['responses'] = batch['responses'][..., :keep]
    return output
