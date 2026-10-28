#!/usr/bin/env python3
"""Audit paired SFT weights/batches and summarize all saved strict evaluations."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def exact_state(left, right):
    """Compare nested optimizer/RNG state, including dtype and nonfinite values."""
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and bool(torch.isfinite(left).all())
                and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_state(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and not math.isfinite(left):
        return False
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Completed instruction experiment directory')
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--candidate', default='native', help='Candidate subdirectory; reference stays fixed')
    parser.add_argument('--reference', help='Optional reference run directory for a numerical-only audit')
    parser.add_argument('--numerical-only', action='store_true', help='Audit a diagnostic repetition without new evaluations')
    parser.add_argument('--require-exact', action='store_true',
                        help='Require identical weights, optimizer/RNG state and recorded losses')
    args = parser.parse_args()
    output = Path(args.output)
    if not args.numerical_only and not (output / 'results.json').exists():
        raise ValueError('Require generation results before analysis')
    if args.reference and not args.numerical_only:
        parser.error('--reference is only supported with --numerical-only')
    source_path = Path(args.source_checkpoint).resolve()
    expected_hash = read(output / 'inputs.json')['hashes'].get(str(source_path))
    with source_path.open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if expected_hash != source_hash:
        raise ValueError('Source checkpoint is not the recorded initialization')
    import torch
    torch.set_num_threads(2)
    initial = torch.load(args.source_checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    reference_root = Path(args.reference) if args.reference else output / 'reference'
    roots = (reference_root, output / args.candidate)
    checkpoints = [torch.load(root / 'last.pt', map_location='cpu', weights_only=False) for root in roots]
    weights = [checkpoint['model_state_dict'] for checkpoint in checkpoints]
    optimizer_equal = exact_state(checkpoints[0]['optimizer_states'], checkpoints[1]['optimizer_states'])
    rng_equal = exact_state(checkpoints[0]['rng_states'], checkpoints[1]['rng_states'])
    if not initial.keys() == weights[0].keys() == weights[1].keys():
        raise ValueError('Checkpoint inventories differ')
    error, update, maximum, changed = 0., 0., 0., 0
    for name, original in initial.items():
        a, b, original = weights[0][name].double(), weights[1][name].double(), original.double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite updated weight')
        error += (a - b).square().sum().item()
        update += (a - original).square().sum().item()
        maximum = max(maximum, (a - b).abs().max().item())
        changed += int(not torch.equal(b, original))
    metrics = [[json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
               for root in roots]
    fields = ('event', 'step', 'loss', 'grad_norm', 'learning_rate')
    recorded = [[{key: row[key] for key in fields if key in row} for row in rows] for rows in metrics]
    recorded_equal = exact_state(*recorded)
    training = [[row for row in rows if row['event'] == 'train'] for rows in metrics]
    if [row['step'] for row in training[0]] != [row['step'] for row in training[1]]:
        raise ValueError('Steps differ')
    max_loss = max(abs(a['loss'] - b['loss']) for a, b in zip(*training))
    summaries = [read(root / 'summary.json') for root in roots]
    validation_delta = abs(summaries[0]['final_validation']['mean'] - summaries[1]['final_validation']['mean'])
    relative = (error / max(update, 1e-30)) ** .5
    batch_match = all(read(reference_root / f'batches-rank{rank}.json') ==
                      read(output / args.candidate / f'batches-rank{rank}.json') for rank in (0, 1))
    # Reuse the acceptance limits fixed for the preceding 100-step experiment.
    passed = batch_match and max_loss <= .03 and validation_delta <= .02 and relative <= .1 and changed == len(initial)
    exact = (batch_match and maximum == 0 and max_loss == 0 and validation_delta == 0
             and optimizer_equal and rng_equal and recorded_equal)
    if args.require_exact:
        passed = passed and exact
    comparison = dict(status='passed' if passed else 'failed', candidate=args.candidate,
                      batch_hashes_match=batch_match,
                      max_training_loss_delta=max_loss, final_validation_delta=validation_delta,
                      relative_update_difference=relative, max_weight_difference=maximum,
                      updated_tensors=changed, total_tensors=len(initial),
                      exact_state_and_recorded_losses=exact, require_exact=args.require_exact,
                      optimizer_state_exact=optimizer_equal, rng_state_exact=rng_equal,
                      recorded_metrics_exact=recorded_equal,
                      limits=dict(training_loss=.03, validation_loss=.02, relative_update_difference=.1))
    if args.numerical_only:
        (output / f'comparison-{args.candidate}.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps(comparison, indent=2))
        if not passed:
            raise AssertionError('Diagnostic failed the numerical or required exact-state acceptance check')
        return
    if args.candidate != 'native':
        parser.error('Full evaluation analysis currently requires the native candidate')
    agreements = {}
    for suite in ('transformations', 'original'):
        reports = [read(output / f'{mode}-{suite}.json') for mode in ('before', 'reference', 'native')]
        identities = [[(case['id'], case['expected'], case.get('prompt'), case.get('messages'))
                       for case in report['cases']] for report in reports]
        if not identities[0] == identities[1] == identities[2]:
            raise ValueError('Evaluation cases differ across checkpoints')
        if not reports[0]['decoding'] == reports[1]['decoding'] == reports[2]['decoding']:
            raise ValueError('Decoding differs across checkpoints')
        agreements[suite] = dict(
            identical_responses=sum(a['response'] == b['response'] and a['stopped'] == b['stopped']
                                    for a, b in zip(reports[1]['cases'], reports[2]['cases'])),
            identical_correctness=sum(a['correct'] == b['correct']
                                      for a, b in zip(reports[1]['cases'], reports[2]['cases'])),
            total=len(reports[0]['cases']))
    original_path = source_path.parent / 'instructions.json'
    original = read(original_path)
    before_original = read(output / 'before-original.json')
    if [(case['id'], case['prompt'], case['expected']) for case in original['cases']] != [
            (case['id'], case['prompt'], case['expected']) for case in before_original['cases']]:
        raise ValueError('The original 60-case diagnostic was changed')
    agreements['baseline_reproduction'] = dict(
        original_file=str(original_path), original_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        identical_responses=sum(a['response'] == b['response'] and a['stopped'] == b['stopped']
                                for a, b in zip(original['cases'], before_original['cases'])),
        original_correct=sum(case['correct'] for case in original['cases']),
        repeated_correct=sum(case['correct'] for case in before_original['cases']),
        total=len(original['cases']))
    result = dict(comparison=comparison, agreements=agreements,
                  experiment=read(output / 'results.json'))
    (output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(comparison=comparison, agreements=agreements), indent=2))
    if not passed:
        raise AssertionError('Paired SFT failed the numerical or required exact-state acceptance check')


if __name__ == '__main__':
    main()
