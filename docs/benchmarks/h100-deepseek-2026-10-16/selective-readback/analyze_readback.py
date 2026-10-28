import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    result = dict(cases={}, quality={}, monitoring={})
    for case in ['p32', 'p512', 'p128', 'heldout-p512']:
        entry = {}
        for selective in [0, 1]:
            runs = sorted(ROOT.glob(f'nt-selective{selective}-{case}-r[01]'))
            if not runs: continue
            phases = []
            for run in runs:
                validation = json.loads((ROOT / (run.name + '-validation.json')).read_text())
                assert validation['exact_hc_reference'] and sha(run / 'all-logits.f32') == validation['logits_sha256']
                rows = [json.loads(line) for line in (run / 'measurements.jsonl').read_text().splitlines()]
                for row in rows:
                    if row['kind'] == 'phase': phases.append(dict(run=run.name, **row))
                result['quality'][case] = dict(logits_sha256=validation['logits_sha256'],
                    prediction_positions=validation['prediction_positions'], logits_per_position=validation['logits_per_position'])
            warm = [row for row in phases if row['round'] >= 2]
            data = dict(launches=len(runs), phases={})
            for phase in ['prefill', 'decode']:
                values = [r for r in warm if r['phase'] == phase]
                latencies = [r['seconds'] for r in values]
                fields = ['prologue_calls', 'prologue_readback_waits', 'prologue_readback_bytes',
                    'prologue_deferred_calls', 'prologue_staging_waits', 'raw_recovery_calls', 'raw_recovery_bytes',
                    'attention', 'qkv', 'compression', 'resident_attention', 'moe', 'hc', 'hc_projection',
                    'engram', 'other', 'routed_upload_bytes', 'peer_activation_copy_bytes', 'peer_output_copy_bytes']
                data['phases'][phase] = dict(samples=len(values), seconds_median=statistics.median(latencies),
                    tokens_per_second=values[0]['tokens'] / statistics.median(latencies),
                    seconds_min=min(latencies), seconds_max=max(latencies),
                    launch_medians=[statistics.median(r['seconds'] for r in values if r['run'] == run.name) for run in runs],
                    counters_median={f: statistics.median(r[f] for r in values) for f in fields},
                    major_faults_sum=sum(r['major_page_faults'] for r in values),
                    input_blocks_sum=sum(r['input_blocks'] for r in values))
            pairs = {(r['run'], r['round']): {} for r in warm}
            for r in warm: pairs[r['run'], r['round']][r['phase']] = r['seconds']
            data['work_seconds_median'] = statistics.median(sum(p.values()) for p in pairs.values())
            data['work_launch_medians'] = [statistics.median(sum(p.values()) for (name, _), p in pairs.items() if name == run.name) for run in runs]
            entry[str(selective)] = data
        if len(entry) == 2:
            entry['comparison'] = {}
            for phase in ['prefill', 'decode']:
                before, after = [entry[str(mode)]['phases'][phase] for mode in [0, 1]]
                entry['comparison'][phase] = dict(speedup=before['seconds_median'] / after['seconds_median'],
                    launch_speedups=[a / b for a, b in zip(before['launch_medians'], after['launch_medians'])])
            entry['comparison']['whole_work'] = dict(speedup=entry['0']['work_seconds_median'] / entry['1']['work_seconds_median'],
                launch_speedups=[a / b for a, b in zip(entry['0']['work_launch_medians'], entry['1']['work_launch_medians'])])
        result['cases'][case] = entry
    observations = [json.loads(line) for line in (ROOT / 'compute-processes.jsonl').read_text().splitlines()]
    assert observations and all(not r.get('foreign') and 'error' not in r for r in observations)
    result['monitoring']['gpu_process_observations'] = len(observations)
    result['monitoring']['foreign_gpu_processes'] = 0
    with (ROOT / 'telemetry.csv').open() as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    result['monitoring']['sampled_peak_vram_mib'] = {
        index: max(float(r['memory.used [MiB]'].split()[0]) for r in rows if r['index'] == index)
        for index in sorted({r['index'] for r in rows})}
    result['unique_prediction_positions'] = sum(r['prediction_positions'] for r in result['quality'].values())
    result['unique_logit_values'] = sum(r['prediction_positions'] * r['logits_per_position'] for r in result['quality'].values())
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    report(result)
    return result

def report(result):
    lines = ['# Selective QKV readback — two H100s', '',
        'DeepSeek-V4.1-Flash on the corrected hyper-connection graph. The same executable runs both modes: '
        '`DS41_QKV_SELECTIVE_READBACK=0` restores the former host mirrors; the default (`1`) returns only the query latent '
        'needed by indexer-source layers. Q and the sliding-window KV ring remain on device. No weight format or precision changes.', '',
        '## Controlled performance', '',
        'Two independent launches per mode and paired workload, with reversed mode order. Each process runs two warmups '
        'and three measured requests. The same corpus tokens, 64 teacher-forced continuation tokens, '
        '62 GiB expert cache per GPU, SLRU 80%, FP32 routed compute, cooperative Q8 backbone and P2P owner execution are preserved. '
        'Builds and initial correctness tests finished before the performance cases. '
        'A strengthened isolation test was compiled and rerun only after all measurements completed.', '',
        '| Workload | Phase | Baseline tok/s | Selective tok/s | Throughput gain | Gain by launch |',
        '|---|---|---:|---:|---:|---|']
    for case in ['p32', 'p512']:
        entry = result['cases'][case]
        for phase in ['prefill', 'decode']:
            before, after = [entry[str(mode)]['phases'][phase] for mode in [0, 1]]
            comparison = entry['comparison'][phase]
            gains = ', '.join(f'{(s - 1) * 100:+.2f}%' for s in comparison['launch_speedups'])
            lines.append(f'| {case} | {phase} | {before["tokens_per_second"]:.4f} | {after["tokens_per_second"]:.4f} | '
                f'{(comparison["speedup"] - 1) * 100:+.2f}% | {gains} |')
    lines += ['', 'Throughput uses tokens divided by median warm phase time; each paired cell has six measurements. '
        'The repeated requests within a process are not independent hardware runs. Whole-work time sums prefill and decode compute intervals; '
        'startup, logging and validation are outside these intervals. Initial requests remain diagnostic because the OS file cache was not reset.', '',
        '| Workload | Baseline warm work s | Selective warm work s | Whole-work throughput gain |',
        '|---|---:|---:|---:|']
    for case in ['p32', 'p512']:
        e = result['cases'][case]
        lines.append(f'| {case} | {e["0"]["work_seconds_median"]:.6f} | {e["1"]["work_seconds_median"]:.6f} | '
            f'{(e["comparison"]["whole_work"]["speedup"] - 1) * 100:+.2f}% |')
    lines += ['', '## Removed dependencies', '',
        '| p512 prefill counter | Baseline | Selective |', '|---|---:|---:|']
    e = result['cases']['p512']
    for field in ['prologue_calls', 'prologue_readback_waits', 'prologue_readback_bytes', 'prologue_deferred_calls',
                  'prologue_staging_waits', 'raw_recovery_calls']:
        lines.append(f'| {field} | {e["0"]["phases"]["prefill"]["counters_median"][field]:,.0f} | '
            f'{e["1"]["phases"]["prefill"]["counters_median"][field]:,.0f} |')
    lines += ['', 'An asynchronous prologue owns a separate pinned input buffer and a CUDA event protecting its reuse. '
        'The buffer holds input plus RoPE tables (20,736 bytes for this model). No new device tensor is allocated by this change. '
        'The normal tail completes the stream before the next layer. If GPU attention fails, the host recovers Q and all valid raw ring slots, '
        'including history omitted by prior calls. Sequence checks prevent stale device history from overwriting newer CPU state.', '',
        'QKV host intervals now stop earlier, so some pending GPU work is charged to later compressor or attention waits. '
        'Do not interpret the QKV timer reduction as faster projection kernels or add nested timers to parent totals. '
        'The global throughput change, rather than the wait-count reduction, measures the speed benefit.', '',
        '| p512 prefill host interval | Baseline s | Selective s |', '|---|---:|---:|']
    for field in ['attention', 'qkv', 'compression', 'resident_attention']:
        lines.append(f'| {field} | {e["0"]["phases"]["prefill"]["counters_median"][field]:.6f} | '
            f'{e["1"]["phases"]["prefill"]["counters_median"][field]:.6f} |')
    lines += ['', '`qkv`, `compression` and `resident_attention` are nested inside `attention`; '
        'their intervals include CPU work, stream submissions and waits, and are not isolated GPU kernel times.', '',
        '## Correctness and scope', '',
        f'All {result["unique_logit_values"]:,} unique logits across {result["unique_prediction_positions"]} prediction positions '
        'match the saved fixed-HC reference bitwise. Every process also matches across its five requests. '
        'The two additional quality cases (p128 and heldout-p512) have one candidate launch each; their timings are diagnostics, not paired speedups.', '',
        'Four prologue test configurations cover legacy/cooperative Q8 kernels and batched/serial readback. '
        'They check exact selective Q/QR/KV, caller-input mutation, global scratch reuse, deferred submissions, window wrap, '
        'isolated states/layers, failed-tail recovery, reset/destruction and stale-read rejection. '
        'Attention, tail, compressor, indexer and HC tests also pass. The final isolation test uses distinct inputs '
        'for each layer and session; GPU-disabled behavior is checked separately. Test assertions are enabled. '
        'Compute Sanitizer was unavailable in the configured toolchain, so this run does not claim a CUDA memory-sanitizer pass.', '',
        f'GPU monitoring recorded {result["monitoring"]["gpu_process_observations"]} observations without foreign compute processes. '
        f'Peak sampled VRAM by physical GPU (MiB): {result["monitoring"]["sampled_peak_vram_mib"]}. '
        'Monitoring samples are not allocation peaks and do not detect unrelated CPU-only work. Warm page faults and I/O are recorded in summary.json.', '',
        'The arithmetic and validated outputs are unchanged; these corpus checks do not extend task-quality coverage. '
        'Prefill remains token-serial, and HC, router, compression/indexer and attention-output boundaries still involve the CPU. '
        'The next substantial prefill target is a causal layer-wise token-tile path with batched projections and expert grouping. '
        'The smaller follow-up is to avoid allocating/zeroing host Q/KV buffers that the resident path never consumes.', '',
        '## Reproduction', '',
        'The bundle retains per-round measurements, full logits, validation hashes, command environments, source snapshots, '
        'test logs, telemetry and this analyzer. The library differs from the fixed-HC library only in ds41_gpu.cu.o and '
        'ds41_forward.cpp.o; CUDA compiler flags and architecture targets are preserved. The failed first harness compile '
        '(missing benchmark_resources.h) and successful retry are both recorded. It occurred before testing and measurement.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

if __name__ == '__main__': main()
