#!/usr/bin/env python3
"""Compare identical 2- or 100-step continuations; fail on numerical divergence."""
import argparse
import json
from pathlib import Path

import torch
from analyze_verified_instructions import exact_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-exact', action='store_true',
                        help='Require identical weights, AdamW/RNG state and recorded numerical metrics')
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference, candidate = Path(args.reference), Path(args.candidate)
    summaries = [json.loads((root / 'summary.json').read_text()) for root in (reference, candidate)]
    if summaries[0]['batch_sha256'] != summaries[1]['batch_sha256']:
        raise AssertionError('Batch identity mismatch')
    metrics = [[json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
               for root in (reference, candidate)]
    if len(metrics[0]) != len(metrics[1]) or any(a['step'] != b['step'] for a, b in zip(*metrics)):
        raise AssertionError('Training steps differ')
    max_loss_delta = max(abs(a['loss'] - b['loss']) for a, b in zip(*metrics))
    validation_delta = abs(summaries[0]['final_validation']['mean'] - summaries[1]['final_validation']['mean'])
    source = torch.load(args.source_checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    reference_state = torch.load(reference / 'last.pt', map_location='cpu', weights_only=False)
    candidate_state = torch.load(candidate / 'last.pt', map_location='cpu', weights_only=False)
    reference_weights = reference_state['model_state_dict']
    candidate_weights = candidate_state['model_state_dict']
    exact = {key: exact_state(reference_state[key], candidate_state[key])
             for key in ('model_state_dict', 'optimizer_states', 'rng_states')}
    fields = ('step', 'loss', 'grad_norm', 'learning_rate', 'tokens')
    exact['recorded_metrics'] = exact_state(*[[{key: row[key] for key in fields} for row in rows]
                                            for rows in metrics])
    exact['validation'] = exact_state(summaries[0]['final_validation'], summaries[1]['final_validation'])
    if source.keys() != reference_weights.keys() or source.keys() != candidate_weights.keys():
        raise AssertionError('Checkpoint tensor inventories differ')
    error, update, worst = 0.0, 0.0, 0.0
    changed = 0
    for name in source:
        a, b, original = reference_weights[name].double(), candidate_weights[name].double(), source[name].double()
        if not torch.isfinite(b).all():
            raise AssertionError(f'Nonfinite updated tensor {name}')
        error += (a - b).square().sum().item()
        update += (a - original).square().sum().item()
        worst = max(worst, (a - b).abs().max().item())
        changed += int(not torch.equal(b, original))
    relative = (error / max(update, 1e-30)) ** .5
    # BF16 GEMM/reduction order differs across DDP vs sequential microbatches.
    # Thresholds are fixed before launch and never silently widened by the runner.
    passed = max_loss_delta <= .03 and validation_delta <= .02 and relative <= .10 and changed == len(source)
    if args.require_exact:
        passed = passed and all(exact.values())
    result = dict(status='passed' if passed else 'failed', reference=str(reference), candidate=str(candidate),
                  paired_throughput_ratio=summaries[1]['tokens_per_second'] / summaries[0]['tokens_per_second'],
                  historical_throughput_ratio=summaries[1]['tokens_per_second'] /
                      summaries[1]['historical_baseline']['historical_tokens_per_second'],
                  historical_ratio_scope='Descriptive only: separate initialization and instrumentation',
                  max_training_loss_delta=max_loss_delta, final_validation_delta=validation_delta,
                  relative_update_difference=relative, max_weight_difference=worst,
                  updated_tensors=changed, total_tensors=len(source),
                  exact=exact, require_exact=args.require_exact,
                  limits=dict(training_loss=.03, validation_loss=.02, relative_update_difference=.10))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise AssertionError('Continuation failed the numerical or required exact-state acceptance check')


if __name__ == '__main__':
    main()
