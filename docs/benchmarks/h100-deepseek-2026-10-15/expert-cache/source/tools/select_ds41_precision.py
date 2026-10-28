#!/usr/bin/env python3
"""Calibrate routed MXFP4 arithmetic from fixed-token DS41 benchmark artifacts.

The policy is scoped to the measured model, binary, GPU placement, cache and
workloads. It is numerical evidence, not an instruction-following certificate.
FP32 is the fallback. No CPU decision or calibration runs inside token decode.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_case(path):
    path = Path(path)
    rows = [json.loads(line) for line in (path / 'measurements.jsonl').read_text().splitlines()]
    setup = next(row for row in rows if row['kind'] == 'setup')
    if not setup.get('teacher_forcing'):
        raise ValueError(f'{path}: fixed teacher-forced tokens required')
    prompt = json.loads((path / 'prompt-ids.json').read_text())
    targets = json.loads((path / 'target-ids.json').read_text())
    if len(prompt) != setup['prompt_tokens'] or len(targets) != setup['decode_steps'] + 1:
        raise ValueError(f'{path}: inconsistent token counts')
    vocab = setup['vocab']
    if any(type(t) is not int or not 0 <= t < vocab for t in prompt + targets):
        raise ValueError(f'{path}: invalid token ID')
    raw = np.fromfile(path / 'all-logits.f32', dtype='<f4')
    if raw.size != len(targets) * vocab or not np.isfinite(raw).all():
        raise ValueError(f'{path}: incomplete/nonfinite logits')
    decode = [row for row in rows if row.get('phase') == 'decode' and row['round'] > 0]
    if len(decode) < 3 or len({row['round'] for row in decode}) != len(decode):
        raise ValueError(f'{path}: at least three distinct warm repeats required')
    for row in decode:
        if row['tokens'] != setup['decode_steps'] or row['seconds'] <= 0 or not math.isfinite(row['seconds']):
            raise ValueError(f'{path}: invalid timing')
        if row.get('routed_cpu_calls', 0) or row.get('peer_fallback_calls', 0):
            raise ValueError(f'{path}: unexpected runtime fallback')
    completions = [row for row in rows if row['kind'] == 'completion']
    if {r['round'] for r in completions} != {0, *[r['round'] for r in decode]} or not all(r['repeat_bitwise_equal'] for r in completions):
        raise ValueError(f'{path}: repeat validation missing')
    command = json.loads((path.parent / (path.name + '-command.json')).read_text())
    if command['returncode'] != 0:
        raise ValueError(f'{path}: failed measurement process')
    mode = command['environment'].get('DS41_MXFP4_COMPUTE', 'fp32')
    if mode not in ['fp32', 'f16', 'q8']:
        raise ValueError(f'{path}: unknown arithmetic mode')
    for row in decode:
        if mode != 'fp32' and (row.get(mode + '_launches', 0) <= 0 or
                              row[mode + '_launches'] != row.get('cooperative_mxfp4_launches')):
            raise ValueError(f'{path}: reduced arithmetic coverage missing')
        if command['environment'].get('DS41_PEER_MODE') in ['copy', 'execute'] and row.get('peer_calls', 0) != row.get('routed_gpu_calls'):
            raise ValueError(f'{path}: peer coverage missing')
    environment = json.loads((path.parent / 'environment.json').read_text())
    return dict(path=str(path), setup=setup, prompt=prompt, targets=targets, logits=raw.reshape(-1, vocab),
                times=[r['seconds'] for r in decode], command=command, environment=environment,
                logits_sha256=sha(path / 'all-logits.f32'))


def compare(reference, candidate):
    if reference['prompt'] != candidate['prompt'] or reference['targets'] != candidate['targets']:
        raise ValueError('candidate used different tokens')
    for key in ['vocab', 'layers', 'shards', 'model_file_bytes', 'tokenizer_pre']:
        if reference['setup'][key] != candidate['setup'][key]:
            raise ValueError(f'model mismatch: {key}')
    for key in ['binary_sha256', 'gpu']:
        if reference['environment'][key] != candidate['environment'][key]:
            raise ValueError(f'execution identity mismatch: {key}')
    # Precision selection must not quietly select a larger cache or more GPUs.
    def execution_env(case):
        return {k: v for k, v in case['command']['environment'].items() if k != 'DS41_MXFP4_COMPUTE'}
    if execution_env(reference) != execution_env(candidate):
        raise ValueError('precision candidates used different runtime settings')
    # The benchmark's last eight argv entries are executable and its seven
    # fixed arguments. Output directory/label may differ, semantic work may not.
    if reference['command'].get('command', [])[-7:-2] != candidate['command'].get('command', [])[-7:-2]:
        raise ValueError('precision candidates used different model/workload arguments')
    kl, nll_delta, reference_nll, rel_l2, argmax = [], [], [], [], []
    for i, (a, b) in enumerate(zip(reference['logits'], candidate['logits'])):
        a, b = a.astype(np.float64), b.astype(np.float64)
        la, lb = a - a.max(), b - b.max()
        la -= np.log(np.exp(la).sum()); lb -= np.log(np.exp(lb).sum())
        kl.append(max(0., float(np.sum(np.exp(la) * (la - lb)))))
        target = reference['targets'][i]
        reference_nll.append(float(-la[target]))
        nll_delta.append(float(la[target] - lb[target]))
        rel_l2.append(float(np.sqrt(np.sum((a - b) ** 2)) / max(np.sqrt(np.sum(a ** 2)), 1e-30)))
        argmax.append(int(np.argmax(a)) == int(np.argmax(b)))
    speedup = statistics.median(reference['times']) / statistics.median(candidate['times'])
    return dict(mean_kl=float(np.mean(kl)), p99_kl=float(np.quantile(kl, .99)), max_kl=max(kl),
                reference_nll=float(np.mean(reference_nll)), candidate_nll=float(np.mean(reference_nll) + np.mean(nll_delta)),
                nll_increase=float(np.mean(nll_delta)), perplexity_ratio=float(np.exp(np.mean(nll_delta))),
                max_relative_l2=max(rel_l2), argmax_agreement=float(np.mean(argmax)),
                bitwise_equal=reference['logits_sha256'] == candidate['logits_sha256'],
                speedup=speedup, conservative_speedup=min(reference['times']) / max(candidate['times']),
                reference_tokens_per_second=reference['setup']['decode_steps'] / statistics.median(reference['times']),
                tokens_per_second=candidate['setup']['decode_steps'] / statistics.median(candidate['times']))


def choose(evaluations, limits):
    verdicts = {}
    for mode, metrics in evaluations.items():
        reasons = []
        if len(metrics) < 2:
            reasons.append('independent validation workload required')
        for index, m in enumerate(metrics):
            for key, limit in [('mean_kl', limits['mean_kl']), ('p99_kl', limits['p99_kl']),
                               ('nll_increase', limits['nll_increase']), ('max_relative_l2', limits['relative_l2'])]:
                if not math.isfinite(m[key]) or (limit is not None and m[key] > limit):
                    reasons.append(f'workload {index}: {key} exceeds {limit}')
            if m['speedup'] < limits['speedup'] or m['conservative_speedup'] <= 1:
                reasons.append(f'workload {index}: speed improvement not established')
        geometric_speedup = math.exp(statistics.mean(math.log(m['speedup']) for m in metrics)) if metrics else 0
        verdicts[mode] = dict(eligible=not reasons, reasons=reasons, geometric_speedup=geometric_speedup, workloads=metrics)
    eligible = [mode for mode, verdict in verdicts.items() if verdict['eligible']]
    selected = max(eligible, key=lambda mode: verdicts[mode]['geometric_speedup']) if eligible else 'fp32'
    return selected, verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--contexts', type=int, nargs='+', default=[32, 128])
    parser.add_argument('--placement', choices=['single', 'copy', 'execute'], default='execute')
    parser.add_argument('--max-mean-kl', type=float, default=.005)
    parser.add_argument('--max-p99-kl', type=float, default=.05)
    parser.add_argument('--max-nll-increase', type=float, default=.01)
    parser.add_argument('--max-relative-l2', type=float, default=None,
                        help='optional raw-logit gate; disabled by default because softmax ignores common offsets')
    parser.add_argument('--min-speedup', type=float, default=1.03)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', nargs=argparse.REMAINDER, help='optional calibrated benchmark command, verified by SHA256')
    args = parser.parse_args()
    if len(set(args.contexts)) < 2 or len(set(args.contexts)) != len(args.contexts):
        parser.error('use at least two distinct calibration/validation contexts')
    limits = dict(mean_kl=args.max_mean_kl, p99_kl=args.max_p99_kl, nll_increase=args.max_nll_increase,
                  relative_l2=args.max_relative_l2, speedup=args.min_speedup)
    if any(not math.isfinite(v) or v < 0 for v in limits.values() if v is not None) or limits['speedup'] < 1:
        parser.error('invalid acceptance limits')
    evaluations = {mode: [] for mode in ['f16', 'q8']}
    evidence = []
    reference_settings = None
    for context in args.contexts:
        reference = read_case(args.root / f'{args.placement}-fp32-p{context}')
        if reference['setup']['prompt_tokens'] != context or reference['command']['environment'].get('DS41_MXFP4_COMPUTE', 'fp32') != 'fp32':
            raise ValueError('reference label does not match execution')
        settings = (reference['environment']['binary_sha256'], reference['environment']['gpu'], reference['command']['environment'])
        if reference_settings is not None and settings != reference_settings:
            raise ValueError('calibration/validation used different execution settings')
        reference_settings = settings
        for mode in evaluations:
            candidate = read_case(args.root / f'{args.placement}-{mode}-p{context}')
            if candidate['command']['environment'].get('DS41_MXFP4_COMPUTE', 'fp32') != mode:
                raise ValueError('candidate precision label does not match execution')
            evaluations[mode].append(compare(reference, candidate))
        evidence.append(dict(context=context, prompt_sha256=sha(Path(reference['path']) / 'prompt-ids.json'),
                             target_sha256=sha(Path(reference['path']) / 'target-ids.json')))
    selected, verdicts = choose(evaluations, limits)
    env = reference['command']['environment'].copy()
    env['DS41_MXFP4_COMPUTE'] = selected
    original = reference['command']['command'][-8:]
    policy = dict(schema=1, scope='routed MXFP4 dot products; FP32 accumulation', selected=selected,
                  limits=limits, candidates=verdicts, evidence=evidence, environment=env,
                  binary_sha256=reference['environment']['binary_sha256'], gpu=reference['environment']['gpu'],
                  model=original[1], corpus_sha256=sha(args.root / 'corpus.txt'),
                  assurance='local numerical calibration; task quality and broader corpora remain separate evaluations')
    output = args.output or args.root / 'precision-policy.json'
    output.write_text(json.dumps(policy, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(selected=selected, policy=str(output), candidates=verdicts), indent=2))
    if args.run:
        if sha(args.run[0]) != policy['binary_sha256']:
            raise ValueError('launcher binary differs from calibrated binary')
        current_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'], text=True)
        if current_gpu != policy['gpu']:
            raise ValueError('launcher GPU/driver identity differs from calibration')
        # Restrict this launcher to the measured model/corpus/benchmark shape.
        original = reference['command']['command'][-8:]
        if (len(args.run) != 8 or args.run[1:3] != original[1:3] or
            int(args.run[3]) not in args.contexts or args.run[4:6] != original[4:6]):
            raise ValueError('launcher workload lies outside calibrated model/contexts')
        if sha(args.run[2]) != policy['corpus_sha256']:
            raise ValueError('launcher corpus changed since calibration')
        runtime = {k: v for k, v in os.environ.items() if not k.startswith('DS41_')}
        runtime.update(env)
        raise SystemExit(subprocess.call(args.run, env=runtime))


if __name__ == '__main__':
    main()
