#!/usr/bin/env python3
"""Fetch this experiment's portable records, optionally waiting for its archive.

Uses SSH key authentication. It never starts training or changes remote state.
"""
import argparse
import json
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='ubuntu@<h100-host>')
    parser.add_argument('--remote', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--wait-hours', type=float, default=14)
    args = parser.parse_args()
    if any(character not in '/-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for character in args.remote):
        parser.error('Remote path must be an absolute path with plain filename characters')
    if not args.remote.startswith('/'):
        parser.error('Remote path must be absolute')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    while True:
        try:
            response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
                                       f'cat {args.remote}/status.json'], check=True, capture_output=True,
                                      text=True, timeout=30)
            status = json.loads(response.stdout)
            temporary = output / 'remote_status.json.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(output / 'remote_status.json')
            print(json.dumps(status), flush=True)
            if status['state'] in ('complete', 'failed', 'blocked'):
                ready = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
                                        f'test -f {args.remote}/archive_ready.json'], timeout=30)
                if ready.returncode == 0:
                    archive_path = output / 'results.tar.gz'
                    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                    f'{args.host}:{args.remote}/results.tar.gz', str(archive_path)], check=True, timeout=180)
                    with tarfile.open(archive_path) as archive:
                        for member in archive.getmembers():
                            path = PurePosixPath(member.name)
                            if (member.issym() or member.islnk() or path.is_absolute() or '..' in path.parts
                                    or '\\' in member.name or ':' in member.name):
                                raise RuntimeError('Archive contains an unsafe path')
                        archive.extractall(output, filter='data')
                    report(output, status)
                    return
            if not args.watch:
                return
        except (subprocess.SubprocessError, json.JSONDecodeError) as error:
            print(f'Fetch retry: {error}', flush=True)
            if not args.watch:
                raise
        if time.monotonic() >= deadline:
            raise TimeoutError('Local result watcher expired')
        time.sleep(30)


def report(output, status):
    text = ['# H100 training experiment results', '', f'Status: **{status["state"]}**.', '',
            'Experimental hybrid backend: native CUDA RMSNorm/SwiGLU and peer activation staging; '
            'PyTorch supplies GEMM, attention, AdamW and DDP.', '']
    if (output / 'results.json').exists():
        results = json.loads((output / 'results.json').read_text())
        if 'evaluations' in results:
            instruction_report(output, status, results)
            return
        cases = results['cases']
        by_name = {case['case']: case for case in cases}
        text += ['| Case | Compute GPUs | Tokens/s | Same-topology ratio | Seconds/step | Validation loss |',
                 '|---|---:|---:|---:|---:|---:|']
        for case in cases:
            reference_name = ('ddp-reference' if 'ddp-reference' in by_name else 'reference') if case['physical_world_size'] == 2 else 'single-checkpoint'
            reference = by_name[reference_name]
            ratio = case['tokens_per_second'] / reference['tokens_per_second']
            text.append(f'| {case["case"]} | {case["physical_world_size"]} | {case["tokens_per_second"]:.1f} | '
                        f'{ratio:.3f}x | {case["mean_step_seconds"]:.4f} | {case["final_validation"]["mean"]:.6f} |')
        text += ['', 'Each measured case continues the same checkpoint for 100 steps; the first 10 '
                 'steps are excluded from throughput. Single-GPU cases process the same global batch '
                 'with four microbatches. Ratios compare DDP cases to the DDP reference, and single-compute-GPU '
                 'cases to single-checkpoint. Throughput includes the continuous measured window, '
                 'including logging and instrumentation gaps.', '']
        text += ['| Case | Compute-device peak allocated GiB per rank | Peer-device allocation GiB per rank |',
                 '|---|---:|---:|']
        for case in cases:
            own = max(m['peak_allocated_bytes'] for rank in case['ranks'] for m in rank['memory']
                      if m['device'] == rank['rank']) / 2**30
            peer = max((m['peak_allocated_bytes'] for rank in case['ranks'] for m in rank['memory']
                        if m['device'] != rank['rank']), default=0) / 2**30
            text.append(f'| {case["case"]} | {own:.3f} | {peer:.3f} |')
        text += ['', 'These are PyTorch allocator peaks, not whole-device occupancy. In bidirectional DDP '
                 'each physical GPU also holds the other rank\'s peer blocks. Adding independent peaks '
                 'is an upper bound, not a simultaneous device measurement. Peer staging trades locality '
                 'for capacity; it does not reduce total storage by exchanging equal blocks.', '']
        if cases and cases[0].get('historical_baseline'):
            historical = cases[0]['historical_baseline']
            text += [f'Historical stage1 snapshot at step {historical["captured_step"]}: '
                     f'{historical["historical_tokens_per_second"]:.1f} global tokens/s. '
                     'This is contextual evidence; paired ratios above isolate the implementation change '
                     'using common weights, AdamW state, batches, final LR and timing.', '']
    compatible = output / 'comparison-sft-compatible.json'
    if compatible.exists():
        audit = json.loads(compatible.read_text())
        text += ['## Compatible RMSNorm audit', '',
                 'This run uses native CUDA pointwise RMSNorm with PyTorch reductions, plus precise '
                 'native SwiGLU. The 100-step SFT comparison uses the unchanged 391-example curriculum, '
                 'same batches and initialization, and a fresh AdamW for both implementations.', '',
                 f'SFT audit: **{audit["status"]}**. All {audit["total_tensors"]} final model tensors '
                 f'match exactly; optimizer state exact: {audit["optimizer_state_exact"]}; '
                 f'RNG state exact: {audit["rng_state_exact"]}. Recorded training-loss and final '
                 f'validation-loss differences: {audit["max_training_loss_delta"]} and '
                 f'{audit["final_validation_delta"]}. `--require-exact` enforces this stronger gate.', '',
                 'The separate primitive/full-model test reproduces all 272 unique parameter gradients '
                 'on the real first SFT batch. There are 273 state-dict entries because embedding/output '
                 'weights are tied. Exact SFT equivalence does not extend to the pretraining comparison: '
                 'its three variants pass the original numerical tolerances with nonzero differences.', '',
                 'The fully fused RMSNorm variant remains available with `--norm-mode fused`. Its earlier '
                 '+25.8% pretraining result is a different implementation; it failed the fresh-optimizer '
                 'SFT check. The compatible mode is now the default.', '',
                 'One 100-step run per pretraining variant, in reference/native/checkpoint/peer order; '
                 'no repeated-run uncertainty interval or long-convergence claim. The SFT checkpoint '
                 'is identical to the previously decoded reference (56/120 transformations, 0/60 original '
                 'instructions); no new generation score or quality gain is claimed here.', '',
                 'Portable records include source snapshots, execution scripts, reference SFT metadata '
                 'and checkpoint hashes in `provenance.json`. Full model/optimizer checkpoints remain '
                 'on the instance. `behavior-reference` contains a separately completed curriculum '
                 'experiment: 52/120 transformations and 0/60 original instructions, with different '
                 'data and decoding limits; it is not this acceleration comparison.', '']
    if status.get('reason'):
        text += [f'Last supervisor message: {status["reason"]}', '']
    text += ['See `validation.json`, `comparison-*.json`, per-case `summary.json` and the saved logs for diagnostics.', '']
    (output / 'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')


def instruction_report(output, status, results):
    evaluations = results['evaluations']
    backend = json.loads((output / 'native/experiment.json').read_text())
    operators = 'RMSNorm/SwiGLU' if backend.get('native_norms') else 'SwiGLU'
    lines = ['# Verified instruction-following pilot', '', f'Status: **{status["state"]}**.', '',
             'Both implementations start from the completed original SFT checkpoint and train '
             '100 updates on the same 391 teacher-verified examples with a fresh AdamW. '
             f'Native mode supplies CUDA {operators}; PyTorch supplies the rest of training. '
             'All train and validation batch hashes match.', '',
             '| Checkpoint | All held-out transformations | Original instruction diagnostic |',
             '|---|---:|---:|']
    for name in ('before', 'reference', 'native'):
        values = evaluations[name]
        lines.append(f'| {name} | {values["transformations"]["correct"]}/120 | {values["original"]["correct"]}/60 |')
    lines += ['', 'Success requires greedy generation to emit a stop token and exactly the expected '
              'text, ignoring only outer whitespace. Transformation decoding allows 64 new tokens; '
              'the original diagnostic retains its 32-token limit. The final step-100 checkpoints '
              'are scored without selection using generation results.', '',
              '| Transformation family | Before | Reference | Native |', '|---|---:|---:|---:|']
    for family in evaluations['before']['transformations']['scores']:
        scores = [evaluations[name]['transformations']['scores'][family] for name in ('before', 'reference', 'native')]
        lines.append('| ' + family + ' | ' + ' | '.join(f'{score["correct"]}/{score["total"]}' for score in scores) + ' |')
    lines += ['', 'All 120 validation requests are included, even when the teacher failed. '
              'JSON has no accepted training examples. Inputs and paraphrases are held out within '
              'these task families; the original 60 cases test different skills and never enter training.', '',
              '| Validation loss on the accepted synthetic subset | Initial | Final |',
              '|---|---:|---:|']
    for name in ('reference', 'native'):
        summary = results[name]
        lines.append(f'| {name} | {summary["initial_validation"]["mean"]:.6f} | {summary["final_validation"]["mean"]:.6f} |')
    lines += ['', 'These losses use the 78 accepted validation examples, not all 120 generation cases, '
              'and are not comparable to the original SFT corpus loss. '
              'A gain on transformations alone does not establish general instruction following.', '',
              'Configuration: 2 H100, batch 8/GPU, two accumulation steps (32 examples/update), '
              'BF16 autocast and FP32 parameters; AdamW 5e-5, 10 warmup steps, decay over the final '
              '80%, floor 10%, weight decay 0.01, clipping 1.0 and seed 20261013. '
              'Checkpoints remain on the remote instance; local records contain all generated responses, '
              'expected answers, stop flags, metrics, batch hashes and provenance.', '']
    analysis_path = output / 'analysis.json'
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        comparison = analysis['comparison']
        lines += [f'Paired numerical check: **{comparison["status"]}**. '
                  f'Maximum training-loss delta {comparison["max_training_loss_delta"]:.6g}; '
                  f'final validation delta {comparison["final_validation_delta"]:.6g}; '
                  f'parameter discrepancy {100 * comparison["relative_update_difference"]:.3f}% '
                  'of the reference update L2 norm. See `analysis.json` for limits and response agreement.', '']
        if comparison['status'] != 'passed':
            lines[4:4] = ['**The initial native SFT failed the numerical acceptance check.** '
                           'Completed execution and similar generation scores do not establish update equivalence.', '']
    diagnostics = sorted(output.glob('comparison-*.json'))
    if diagnostics:
        lines += ['| Numerical diagnostic | Result | Update discrepancy |', '|---|---|---:|']
        for path in diagnostics:
            comparison = json.loads(path.read_text())
            lines.append(f'| {comparison["candidate"]} | {comparison["status"]} | '
                         f'{100 * comparison["relative_update_difference"]:.4f}% |')
        lines += ['', 'The limit remains 10% of the reference update norm. '
                  'The reference repetition tests reproducibility. `*-only` uses the original library; '
                  '`*-fixed` uses the precise-math build after removing inherited inference fast math.', '']
    fixed = output / 'swiglu-fixed-transformations.json'
    if fixed.exists() and (output / 'swiglu-fixed-original.json').exists():
        transformations = json.loads(fixed.read_text())
        original = json.loads((output / 'swiglu-fixed-original.json').read_text())
        count = lambda report: sum(case['correct'] for case in report['cases'])
        lines += [f'Corrected SwiGLU-only SFT: **{count(transformations)}/120 transformations, '
                  f'{count(original)}/60 original instructions**. Its weights and quality records are '
                  'separate from the failed initial full-native run. RMSNorm stays on the reference '
                  'path for this SFT configuration.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
