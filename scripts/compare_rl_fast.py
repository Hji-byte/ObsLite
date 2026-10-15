"""Compare saved component benchmarks; fail when configured numerical gates fail."""
import argparse
import math
from pathlib import Path

from .common import read_json, write_json, sha256


def compare_artifacts(reference, candidate):
    import torch
    reference_logs, candidate_logs = reference['selected_logprobs'], candidate['selected_logprobs']
    assert reference_logs.shape == candidate_logs.shape
    error = (reference_logs.double()-candidate_logs.double()).abs()
    rgrads, cgrads = reference['gradients'], candidate['gradients']
    assert rgrads.keys() == cgrads.keys()
    norm_r = norm_c = dot = squared_error = 0.
    for name in rgrads:
        r, c = rgrads[name].double(), cgrads[name].double()
        assert r.shape == c.shape and torch.isfinite(r).all() and torch.isfinite(c).all()
        norm_r += float(r.square().sum())
        norm_c += float(c.square().sum())
        dot += float((r*c).sum())
        squared_error += float((r-c).square().sum())
    assert norm_r > 0 and norm_c > 0
    return {'loss_abs_error': abs(reference['loss']-candidate['loss']),
            'logprob_max_abs_error': float(error.max()), 'logprob_mean_abs_error': float(error.mean()),
            'gradient_relative_l2': math.sqrt(squared_error/norm_r),
            'gradient_cosine': dot/math.sqrt(norm_r*norm_c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-logprob-error', type=float, default=.05)
    parser.add_argument('--max-mean-logprob-error', type=float, default=.005)
    parser.add_argument('--max-gradient-relative-l2', type=float, default=.03)
    parser.add_argument('--min-gradient-cosine', type=float, default=.999)
    parser.add_argument('--max-loss-error', type=float, default=.005)
    args = parser.parse_args()
    import torch
    ref, candidate = read_json(args.reference/'report.json'), read_json(args.candidate/'report.json')
    assert ref.get('precision', 'bf16') == candidate.get('precision', 'bf16'), 'Compare kernels at the same precision'
    for key in ['cases_sha256', 'model', 'model_sha256', 'adapter_sha256', 'versions']:
        assert ref[key] == candidate[key], f'Mismatched benchmark input: {key}'
    assert [r['label'] for r in ref['results']] == [r['label'] for r in candidate['results']]
    rows = []
    for r, c in zip(ref['results'], candidate['results']):
        # Only artifacts produced by the local benchmark are loaded.
        stats = compare_artifacts(torch.load(args.reference/f'{r["label"]}.pt', weights_only=True),
                                  torch.load(args.candidate/f'{c["label"]}.pt', weights_only=True))
        passed = (stats['logprob_max_abs_error'] <= args.max_logprob_error
                  and stats['logprob_mean_abs_error'] <= args.max_mean_logprob_error
                  and stats['gradient_relative_l2'] <= args.max_gradient_relative_l2
                  and stats['gradient_cosine'] >= args.min_gradient_cosine
                  and stats['loss_abs_error'] <= args.max_loss_error)
        rows.append({'label': r['label'], **stats, 'numerical_gate_passed': passed,
                     'speedup': r['median_seconds']/c['median_seconds']})
    write_json(args.output, {'results': rows, 'all_passed': all(r['numerical_gate_passed'] for r in rows),
                            'precision': ref.get('precision', 'bf16'),
                            'reference_report': str((args.reference/'report.json').resolve()),
                            'candidate_report': str((args.candidate/'report.json').resolve()),
                            'reference_report_sha256': sha256(args.reference/'report.json'),
                            'candidate_report_sha256': sha256(args.candidate/'report.json'),
                            'thresholds': {k:v for k,v in vars(args).items() if isinstance(v, float)},
                            'scope': 'same-precision component comparison; full PPO smoke still required'})
    if not all(r['numerical_gate_passed'] for r in rows):
        raise SystemExit('Numerical gate failed; do not enable this candidate in training')


if __name__ == '__main__':
    main()
