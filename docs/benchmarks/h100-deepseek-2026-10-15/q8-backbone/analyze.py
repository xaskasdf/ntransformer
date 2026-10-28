import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', ROOT / 'select_ds41_precision.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

def main():
    summary = {}
    for context in [32, 128, 512]:
        cases = {}
        evidence = {}
        for label in ['legacy', 'cooperative']:
            name = f'{label}-p{context}'
            case = s.read_case(ROOT / name)
            evidence[label] = case
            rows = [json.loads(line) for line in (ROOT / name / 'measurements.jsonl').read_text().splitlines()]
            warm = [r for r in rows if r.get('phase') == 'decode' and r['round'] > 0]
            prefill = [r for r in rows if r.get('phase') == 'prefill' and r['round'] > 0]
            tokens = sum(r['tokens'] for r in warm)
            counts = {key: sum(r[key] for r in warm) / tokens for key in [
                'routed_upload_bytes', 'peer_weight_copy_bytes', 'peer_activation_copy_bytes', 'peer_output_copy_bytes',
                'q8_projection_launches', 'q8_cooperative_launches', 'prologue_readback_waits', 'routed_gpu_calls',
                'resident_attention_calls', 'peer_fallback_calls']}
            assert counts['routed_gpu_calls'] == 40 and counts['peer_fallback_calls'] == 0
            assert counts['q8_projection_launches'] >= 200
            assert (counts['q8_cooperative_launches'] > 0) == (label == 'cooperative')
            counts['stage_ms_per_token'] = {key: 1000 * sum(r[key] for r in warm) / tokens
                for key in ['moe', 'attention', 'engram', 'hc', 'other', 'qkv', 'compression', 'resident_attention']}
            counts.update(tokens_per_second=64 / statistics.median(case['times']),
                prefill_tokens_per_second=context / statistics.median(r['seconds'] for r in prefill),
                warm_decode_seconds=case['times'], logits_sha256=case['logits_sha256'],
                logit_values=int(case['logits'].size))
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
            resource = (ROOT / (name + '-resources.txt')).read_text()
            counts['peak_rss_mib'] = float(re.search(r'Maximum resident set size \(kbytes\): (\d+)', resource)[1]) / 1024
            cases[label] = counts
        a, b = evidence['legacy'], evidence['cooperative']
        assert a['prompt'] == b['prompt'] and a['targets'] == b['targets']
        assert a['logits_sha256'] == b['logits_sha256']
        for case in [a, b]:
            env = dict(case['command']['environment'])
            flag = env.pop('DS41_Q8_COOPERATIVE')
            assert flag == ('0' if case is a else '1')
            if case is a: reference_env = env
            else: assert env == reference_env
        summary[str(context)] = dict(cases=cases, bitwise_equal=True,
            speedup=cases['cooperative']['tokens_per_second'] / cases['legacy']['tokens_per_second'],
            separated_warm_ranges=max(b['times']) < min(a['times']))
    micro = []
    for line in (ROOT / 'q8-microbenchmark.log').read_text().splitlines():
        if line.startswith('BENCH '):
            values = dict(item.split('=') for item in line.split()[1:])
            micro.append({key: int(value) if key in ['rows', 'cols', 'groups'] else float(value)
                          for key, value in values.items()})
    assert len(micro) == 5
    assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 48', (ROOT / 'ctest.log').read_text())
    total = sum(v['cases']['cooperative']['logit_values'] for v in summary.values())
    result = dict(contexts=summary, microbenchmarks=micro, bitwise_logit_values=total)
    lines = ['# H100 DeepSeek: cooperative Q8 backbone loads', '',
        'Controlled comparison on two H100 80 GB SXM5 GPUs, NVLink, DeepSeek-V4.1-Flash-Q8_0 (10 shards). Both variants use owner-GPU expert execution, 62 GiB expert cache per GPU and FP32 routed arithmetic over MXFP4 weights.', '',
        'The only runtime difference is `DS41_Q8_COOPERATIVE=0|1`. Q8_0 weights, FP32 activations, block dot-product order and warp reduction are unchanged. This is a memory-access optimization, independent of the routed FP32/F16/Q8 selector.', '',
        '## End-to-end results', '',
        '64 teacher-forced continuation steps, identical corpus token IDs, one cold request and three warm repeats. Decode throughput is 64 divided by median warm decode time. Context order is 32, 128, 512; candidate/control order reverses at 128.', '',
        '| Prompt tokens | Legacy decode tok/s | Cooperative decode tok/s | Gain | Legacy prefill tok/s | Cooperative prefill tok/s |',
        '|---:|---:|---:|---:|---:|---:|']
    for context, m in summary.items():
        a, b = m['cases']['legacy'], m['cases']['cooperative']
        lines.append(f"| {context} | {a['tokens_per_second']:.4f} | {b['tokens_per_second']:.4f} | {(m['speedup']-1)*100:.2f}% | {a['prefill_tokens_per_second']:.4f} | {b['prefill_tokens_per_second']:.4f} |")
    lines += ['', f'All {total:,} logits match bitwise across variants; every cold/warm repetition also matches within its variant. This proves equivalence on these 195 prediction positions, not general task quality. Exact logits remain diagnostic, rather than a requirement for future optimizations.', '',
        '## Stage measurements', '',
        '| Context | Variant | Attention ms/token | QKV ms/token | Resident attention + tail ms/token | Compression ms/token | MoE ms/token |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for context, m in summary.items():
        for label, case in m['cases'].items():
            t = case['stage_ms_per_token']
            lines.append(f"| {context} | {label} | {t['attention']:.3f} | {t['qkv']:.3f} | {t['resident_attention']:.3f} | {t['compression']:.3f} | {t['moe']:.3f} |")
    lines += ['', 'Host-side Session timers include submission and waits; QKV, resident attention/tail and compression are attention substages and must not be added again to total attention. These are not isolated GPU busy times.', '',
        '## Kernel microbenchmarks', '',
        'CUDA-event spans, median of seven batches of 20 launches, alternating variant order, five warmups. Operands are reused and may hit device cache: these are warm isolated projections, not a full-model bandwidth measurement.', '',
        '| Rows | Columns | Groups | Legacy ms | Cooperative ms | Speedup |', '|---:|---:|---:|---:|---:|---:|']
    for m in micro:
        lines.append(f"| {m['rows']} | {m['cols']} | {m['groups']} | {m['legacy_ms']:.6f} | {m['cooperative_ms']:.6f} | {m['speedup']:.3f}x |")
    lines += ['', '## Memory and remaining host work', '',
        '| Context | Variant | GPU 0 peak MiB | GPU 1 peak MiB | RSS peak MiB | Warm weight H2D MiB/token |',
        '|---:|---|---:|---:|---:|---:|']
    for context, m in summary.items():
        for label, case in m['cases'].items():
            peaks = case['sampled_peak_vram_mib']
            lines.append(f"| {context} | {label} | {peaks.get('0',0):.0f} | {peaks.get('1',0):.0f} | {case['peak_rss_mib']:.0f} | {case['routed_upload_bytes']/2**20:.3f} |")
    lines += ['', 'VRAM peaks are one-second telemetry samples. RSS is time(1) high-water memory. The cooperative kernel uses 12,928 bytes of shared memory per CTA, with no additional global weight/activation allocation or repacking.', '',
        'Each warp loads contiguous words for 32 Q8 blocks; a CTA loads one activation tile shared by eight rows. The arithmetic follows the same lane/block sequence. Columns not divisible by 64 retain the legacy path, avoiding unaligned row reads. Partial rows still participate in CTA barriers.', '',
        'The change does not remove CPU scheduling, QKV mirror readback, compressor/indexer host work or tail readback. Expert cache misses still stage through pinned CPU memory. At context 512, even warm requests upload about 68.9 MiB/token because of cache churn. P2P expert ownership and cache budgets are unchanged. Further progress should target those remaining boundaries and attention reduction costs with separate evidence.', '',
        '## Validation and reproduction', '',
        '48 CTests passed without skips/failures. The new direct CUDA test covers zero/random/large activations, varied scales, partial CTAs, grouped matrices, aligned tiles and odd-block fallback, comparing both kernels bitwise and against an independent double-precision CPU dot. Eight cross-process prologue/tail fixture outputs also match within their paired controls.', '',
        '`prepare.py`, `run.py`, `analyze.py`, source snapshots, command/environment hashes, corpus/token IDs, raw logits, timing JSONL, telemetry and resource logs are included. `DS41_Q8_COOPERATIVE=1` is opt-in; `0` is the control. Recalibrate precision policies after changing the binary; old policies are intentionally bound to their recorded binary hash.', '',
        'The previous llama.cpp numbers used another execution/token protocol. This experiment does not establish an updated cross-engine performance gap.', '']
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({c: dict(speedup=m['speedup'], separated=m['separated_warm_ranges']) for c, m in summary.items()}))

if __name__ == '__main__': main()
