import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', ROOT / 'select_ds41_precision.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def main():
    summary = {}
    for context in [32, 128]:
        cases = {}
        for label in ['single-fp32', 'copy-fp32', 'execute-fp32', 'execute-f16', 'execute-q8']:
            name = label + '-p' + str(context)
            case = s.read_case(ROOT / name)
            rows = [json.loads(line) for line in (ROOT / name / 'measurements.jsonl').read_text().splitlines()]
            warm = [r for r in rows if r.get('phase') == 'decode' and r['round'] > 0]
            prefill = [r for r in rows if r.get('phase') == 'prefill' and r['round'] > 0]
            tokens = sum(r['tokens'] for r in warm)
            counts = {key: sum(r[key] for r in warm) / tokens for key in [
                'routed_upload_bytes', 'peer_weight_copy_bytes', 'peer_activation_copy_bytes', 'peer_output_copy_bytes',
                'peer_experts', 'routed_kernel_launches', 'f16_launches', 'q8_launches', 'routed_gpu_calls', 'peer_calls', 'peer_fallback_calls']}
            counts['stage_ms_per_token'] = {key: 1000 * sum(r[key] for r in warm) / tokens
                for key in ['moe', 'attention', 'engram', 'hc', 'other', 'qkv', 'compression', 'attention_loop', 'attention_tail']}
            assert counts['routed_gpu_calls'] == 40 and counts['peer_fallback_calls'] == 0
            assert counts['f16_launches'] > 0 if label.endswith('f16') else counts['f16_launches'] == 0
            assert counts['q8_launches'] > 0 if label.endswith('q8') else counts['q8_launches'] == 0
            counts.update(tokens_per_second=64 / statistics.median(case['times']),
                prefill_tokens_per_second=context / statistics.median(r['seconds'] for r in prefill),
                warm_decode_seconds=case['times'], logits_sha256=case['logits_sha256'])
            command = case['command']
            start = datetime.datetime.fromisoformat(command['utc_start']).replace(tzinfo=None)
            end = start + datetime.timedelta(seconds=command['wall_seconds'])
            peaks = {}
            with (ROOT / 'telemetry.csv').open() as stream:
                for row in csv.DictReader(stream, skipinitialspace=True):
                    stamp = datetime.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
                    if start <= stamp <= end:
                        device = row['index']
                        peaks[device] = max(peaks.get(device, 0), float(row['memory.used [MiB]'].split()[0]))
            counts['sampled_peak_vram_mib'] = peaks
            for line in (ROOT / (name + '-resources.txt')).read_text().splitlines():
                if 'Maximum resident set size (kbytes)' in line:
                    counts['peak_rss_mib'] = float(line.split(':')[-1].strip()) / 1024
            cases[label] = counts
        assert cases['single-fp32']['logits_sha256'] == cases['copy-fp32']['logits_sha256'] == cases['execute-fp32']['logits_sha256']
        for label, case in cases.items():
            case['speedup_vs_single'] = case['tokens_per_second'] / cases['single-fp32']['tokens_per_second']
            case['speedup_vs_copy'] = case['tokens_per_second'] / cases['copy-fp32']['tokens_per_second']
        summary[str(context)] = cases
    policy = json.loads((ROOT / 'precision-policy.json').read_text())
    lines = ['# DeepSeek: peer expert execution and precision calibration', '',
        'Two H100 80 GB SXM5, NV18. Logical GPU 0 is physical GPU 1. Owner = expert ID modulo 2.',
        'Single GPU: 62 GiB expert cache. Copy/execute: 62 GiB per GPU (124 GiB total); copy adds bounded transient slots.',
        'Same MXFP4 weight bytes in every arithmetic mode. FP16/Q8 change routed dot products; scales, nonlinearities, accumulation, shared expert, attention and final reduction remain FP32.',
        'Fixed corpus tokens, contexts 32/128, 64 decode steps, three warm repeats after one cold request. These are teacher-forced throughput measurements, not the earlier autoregressive workload.', '',
        '| Context | Mode | Decode tok/s | Prefill tok/s | vs single | vs peer-copy | Host uploads MiB/token | P2P weights MiB/token | P2P activations+outputs MiB/token |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for context, cases in summary.items():
        for name, m in cases.items():
            lines.append(f"| {context} | {name} | {m['tokens_per_second']:.4f} | {m['prefill_tokens_per_second']:.4f} | {m['speedup_vs_single']:.4f}x | {m['speedup_vs_copy']:.4f}x | {m['routed_upload_bytes']/2**20:.2f} | {m['peer_weight_copy_bytes']/2**20:.2f} | {(m['peer_activation_copy_bytes']+m['peer_output_copy_bytes'])/2**20:.3f} |")
    lines += ['', 'FP32 single/copy/execute logits are bitwise equal on both contexts. Each candidate also repeated bitwise within its own arithmetic mode.', '',
        '## Automatic precision decision', '', f"Selected: **{policy['selected']}**. Acceptance limits: `{json.dumps(policy['limits'], sort_keys=True)}`.", '',
        '| Mode | Context | Mean KL | P99 KL | NLL increase | Perplexity ratio | Max relative L2 | Argmax agreement | Speed vs same-placement FP32 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode, decision in policy['candidates'].items():
        for context, m in zip([32, 128], decision['workloads']):
            lines.append(f"| {mode} | {context} | {m['mean_kl']:.6g} | {m['p99_kl']:.6g} | {m['nll_increase']:.6g} | {m['perplexity_ratio']:.6g} | {m['max_relative_l2']:.6g} | {m['argmax_agreement']:.4%} | {m['speedup']:.4f}x |")
    for mode, decision in policy['candidates'].items():
        lines += ['', f"{mode}: " + ('eligible' if decision['eligible'] else '; '.join(decision['reasons'])), '']
    lines += ['Raw logit L2 is diagnostic by default; a common logit offset does not change softmax probabilities. The optional L2 gate is disabled for the final selection.', '']
    lines += ['## Memory', '', '| Context | Mode | Physical GPU 0 peak MiB | Physical GPU 1 peak MiB | Peak RSS MiB |',
        '|---:|---|---:|---:|---:|']
    for context, cases in summary.items():
        for name, m in cases.items():
            peaks = m['sampled_peak_vram_mib']
            lines.append(f"| {context} | {name} | {peaks.get('0', 0):.0f} | {peaks.get('1', 0):.0f} | {m['peak_rss_mib']:.0f} |")
    lines += ['', 'VRAM values are maxima of one-second telemetry samples, not allocation high-water counters. RSS is the process high-water value reported by time(1).', '']
    lines += ['## Remaining pipeline time', '', '| Context | FP32 placement | MoE ms/token | Attention ms/token | HC ms/token | Engram ms/token | Other ms/token |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for context, cases in summary.items():
        for name in ['single-fp32', 'copy-fp32', 'execute-fp32']:
            m = cases[name]['stage_ms_per_token']
            lines.append(f"| {context} | {name} | {m['moe']:.3f} | {m['attention']:.3f} | {m['hc']:.3f} | {m['engram']:.3f} | {m['other']:.3f} |")
    lines += ['', 'These are existing host-side Session stage measurements, not isolated device busy time. Substage counters and overlapping work must not be added as independent costs.', '']
    lines += ['Numerical calibration covers 130 prediction positions of one corpus. It does not establish general task accuracy or instruction following. Token equality is diagnostic, not an acceptance requirement. Limits are explicit experimental defaults and are configurable.', '',
        '## Ownership and execution', '',
        'Both GPU caches reserve all selected slices before submitting kernels. With at least 64 slots and at most 24 selected slices, this synchronous call keeps them alive until both streams finish. Cold misses still stage through CPU pinned memory; resident peer weights no longer traverse the CPU.',
        'Execute mode uses cross-device input/ready events, P2P input/output copies and an ordered primary-GPU reduction. Callback exceptions drain both streams and invalidate both caches before reuse. Copy mode uses the same ownership/cache capacity, transferring the selected peer weight slices to temporary primary slots.',
        'GPU placement and precision are independent opt-in controls. The default remains single-GPU FP32. Calibration happens outside decode, with no per-token CPU selector.', '',
        '## Validation and artifacts', '',
        '`ctest.log`, `numerical-tests.json`, per-case `*-command.json`, raw `all-logits.f32`, token IDs, timing JSONL, source snapshots, `environment.json`, `telemetry.csv`, and `precision-policy.json` preserve reproduction evidence.',
        'Native CUDA arithmetic and peer ordering references: [half2 arithmetic](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html), [integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html), [multi-GPU programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html).', '']
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
