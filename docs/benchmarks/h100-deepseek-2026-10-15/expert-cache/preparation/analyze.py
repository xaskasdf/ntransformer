import csv
import datetime
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', ROOT / 'select_ds41_precision.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

def main():
    tests = (ROOT / 'ctest.log').read_text()
    assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 52', tests)
    assert not any(word in tests for word in ['FAIL', 'SKIP', 'Not Run'])
    replay = json.loads((ROOT / 'replay-summary.json').read_text())
    traces = [json.loads(line) for line in (ROOT / 'trace-lru-p512/measurements.jsonl').read_text().splitlines()]
    trace_phases = {(r['round'], r['phase']): r for r in traces if r['kind'] == 'phase'}
    predicted = {}
    for percent, records in replay.items():
        phases = {}
        for r in records:
            match = re.search(r'cache-(\d+)-(prefill|decode)-gpu(\d+)\.u64$', r['file'])
            round, phase, gpu = int(match[1]), match[2], int(match[3])
            assert gpu == r['gpu']
            assert r['recorded_misses'] == trace_phases[round, phase][f'cache{gpu}_misses']
            if percent == '0': assert r['hit_mismatches'] == 0
            data = phases.setdefault((round, phase), dict(misses=0, bytes=0, first_use=0))
            data['misses'] += r['misses']; data['bytes'] += r['misses'] * r['slice_bytes']
            data['first_use'] += r['first_use']
        predicted[percent] = phases
    summary = {}
    comparisons = {}
    prompts = {}
    total_logits = 0
    for workload in ['p32', 'p128', 'p512', 'heldout-p512']:
        cases = {}; evidence = {}
        for policy in ['lru', 'slru']:
            name = policy + '-' + workload
            case = s.read_case(ROOT / name); evidence[policy] = case
            rows = [json.loads(line) for line in (ROOT / name / 'measurements.jsonl').read_text().splitlines()]
            phases = {(r['round'], r['phase']): r for r in rows if r['kind'] == 'phase'}
            warm = [phases[i, 'decode'] for i in range(1, 4)]
            prefill = [phases[i, 'prefill'] for i in range(1, 4)]
            tokens = sum(r['tokens'] for r in warm)
            counts = {key: sum(r[key] for r in warm) / tokens for key in [
                'routed_upload_bytes', 'staged_slices', 'transient_copy_threads', 'peer_upload_bytes',
                'cache0_hits', 'cache1_hits', 'cache0_misses', 'cache1_misses', 'cache0_evictions', 'cache1_evictions',
                'cache0_promotions', 'cache1_promotions', 'cache0_demotions', 'cache1_demotions']}
            counts['stage_ms_per_token'] = {key: 1000 * sum(r[key] for r in warm) / tokens
                for key in ['moe', 'attention', 'engram', 'hc', 'other', 'routed']}
            counts.update(tokens_per_second=64 / statistics.median(case['times']),
                prefill_tokens_per_second=case['setup']['prompt_tokens'] / statistics.median(r['seconds'] for r in prefill),
                warm_decode_seconds=case['times'],
                warm_request_seconds=[phases[i,'prefill']['seconds'] + phases[i,'decode']['seconds'] for i in range(1,4)],
                cold_request_seconds=phases[0,'prefill']['seconds'] + phases[0,'decode']['seconds'],
                cold_request_upload_bytes=phases[0,'prefill']['routed_upload_bytes'] + phases[0,'decode']['routed_upload_bytes'],
                warm_request_upload_bytes=statistics.mean(phases[i,'prefill']['routed_upload_bytes'] + phases[i,'decode']['routed_upload_bytes'] for i in range(1,4)),
                logits_sha256=case['logits_sha256'], logit_values=int(case['logits'].size))
            assert sum(counts[f'cache{i}_misses'] for i in range(2)) == counts['staged_slices']
            if workload == 'p512':
                expected = predicted['0' if policy == 'lru' else '80']
                for key, row in phases.items():
                    assert row['routed_upload_bytes'] == expected[key]['bytes'], (name, key, 'replay mismatch')
            start = datetime.datetime.fromisoformat(case['command']['utc_start']).replace(tzinfo=None)
            end = start + datetime.timedelta(seconds=case['command']['wall_seconds'])
            peaks = {}
            with (ROOT / 'telemetry.csv').open() as stream:
                for row in csv.DictReader(stream, skipinitialspace=True):
                    stamp = datetime.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
                    if start <= stamp <= end:
                        peaks[row['index']] = max(peaks.get(row['index'], 0), float(row['memory.used [MiB]'].split()[0]))
            counts['sampled_peak_vram_mib'] = peaks
            counts['peak_rss_mib'] = float(re.search(r'Maximum resident set size \(kbytes\): (\d+)',
                (ROOT / (name + '-resources.txt')).read_text())[1]) / 1024
            cases[policy] = counts
        a,b = evidence['lru'], evidence['slru']
        assert a['prompt'] == b['prompt'] and a['targets'] == b['targets']
        assert a['logits_sha256'] == b['logits_sha256']
        a_env = dict(a['command']['environment']); b_env = dict(b['command']['environment'])
        assert a_env.pop('DS41_CACHE_POLICY') == 'lru' and b_env.pop('DS41_CACHE_POLICY') == 'slru'
        assert a_env == b_env and a_env['DS41_CACHE_TRACE'] == '0'
        left, right = cases['lru'], cases['slru']
        comparisons[workload] = dict(decode_speedup=right['tokens_per_second']/left['tokens_per_second'],
            request_speedup=statistics.median(left['warm_request_seconds'])/statistics.median(right['warm_request_seconds']),
            warm_decode_upload_reduction=1-right['routed_upload_bytes']/left['routed_upload_bytes'] if left['routed_upload_bytes'] else None,
            separated_warm_decode_ranges=max(b['times']) < min(a['times']),
            separated_warm_request_ranges=max(right['warm_request_seconds']) < min(left['warm_request_seconds']))
        total_logits += right['logit_values']; summary[workload] = cases
        prompts[workload] = a['prompt']
    assert prompts['p512'] != prompts['heldout-p512'], 'held-out corpus reused calibration prompt'
    assert s.sha(ROOT/'trace-lru-p512/all-logits.f32') == s.sha(ROOT/'lru-p512/all-logits.f32')
    # A predeclared candidate, not a policy chosen after inspecting holdout.
    accepted = all(comparisons[w]['request_speedup'] >= 1.03 and
        comparisons[w]['decode_speedup'] >= 1.03 and comparisons[w]['separated_warm_decode_ranges'] and
        comparisons[w]['separated_warm_request_ranges'] and
        summary[w]['slru']['routed_upload_bytes'] < summary[w]['lru']['routed_upload_bytes']
        for w in ['p512','heldout-p512']) and all(comparisons[w]['request_speedup'] >= .98 for w in ['p32','p128'])
    replay_summary = {}
    for percent, phases in predicted.items():
        warm = [phases[i,'decode'] for i in range(1,4)]
        replay_summary[percent] = dict(warm_decode_mib_per_token=sum(r['bytes'] for r in warm)/192/2**20,
            first_use_misses=sum(r['first_use'] for r in warm), recurrent_misses=sum(r['misses']-r['first_use'] for r in warm))
    result = dict(workloads=summary, comparisons=comparisons, replay=replay_summary,
        accepted_on_measured_workloads=accepted, bitwise_logit_values=total_logits,
        candidate_protected_percent=80, default_policy='lru')
    lines = ['# DeepSeek expert-cache replacement experiment', '',
        'Two H100 80 GB SXM5 GPUs, owner-GPU expert execution over NVLink, 62 GiB expert cache per GPU, cooperative Q8 backbone, FP32 routed arithmetic. The candidate is SLRU with 80% protected capacity, declared before the access trace and held-out workload were measured. The default stays LRU.', '',
        ('The candidate meets the predeclared performance/transfer criteria on these workloads.' if accepted else
         'The candidate does not meet all predeclared performance/transfer criteria; it is not accepted as a general replacement for LRU.'), '',
        'Acceptance requires at least 3% improvement in both decode and whole warm-request time, plus fewer warm decode uploads and separated warm timing ranges, on both context-512 corpora. Contexts 32/128 must not regress by more than 2% in whole-request time. These are experimental thresholds, not a general guarantee.', '',
        '## Controlled inference measurements', '',
        'Identical token IDs and arithmetic; 64 teacher-forced continuation steps; one cold request and three warm repeats. Variant order alternates between workloads. Trace recording is disabled for throughput measurements. The held-out corpus is a separate original narrative prepared before the trace was inspected.', '',
        '| Workload | LRU decode tok/s | SLRU decode tok/s | Decode gain | Whole warm-request gain | H2D MiB/token LRU -> SLRU |',
        '|---|---:|---:|---:|---:|---:|']
    for w, cases in summary.items():
        a,b=cases['lru'],cases['slru']; m=comparisons[w]
        lines.append(f"| {w} | {a['tokens_per_second']:.4f} | {b['tokens_per_second']:.4f} | {(m['decode_speedup']-1)*100:.2f}% | {(m['request_speedup']-1)*100:.2f}% | {a['routed_upload_bytes']/2**20:.3f} -> {b['routed_upload_bytes']/2**20:.3f} |")
    lines += ['', f'All {total_logits:,} logits match bitwise between policies, with finite and exact repeats within each process. This covers 260 prediction positions; it is not a task-quality certification.', '',
        '## Cold requests, prefill and memory', '',
        '| Workload | Policy | Warm prefill tok/s | Cold request s | Cold H2D GiB | Warm request H2D GiB | GPU 0 / GPU 1 peak MiB | RSS peak MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for w, cases in summary.items():
        for policy,m in cases.items():
            peaks=m['sampled_peak_vram_mib']
            lines.append(f"| {w} | {policy} | {m['prefill_tokens_per_second']:.4f} | {m['cold_request_seconds']:.3f} | {m['cold_request_upload_bytes']/2**30:.3f} | {m['warm_request_upload_bytes']/2**30:.3f} | {peaks.get('0',0):.0f} / {peaks.get('1',0):.0f} | {m['peak_rss_mib']:.0f} |")
    lines += ['', 'VRAM values are maxima of one-second samples; RSS is time(1) high-water memory. Whole-request time adds measured prefill and decode phases, excluding setup, file output and validation. This exposes any transfer savings purchased by slower prefill.', '',
        '## Offline replay of actual ordered accesses', '',
        '| Protected capacity | Warm decode H2D MiB/token | First-use misses | Recurrent misses |', '|---:|---:|---:|---:|']
    for percent, m in replay_summary.items():
        lines.append(f"| {percent}% | {m['warm_decode_mib_per_token']:.3f} | {m['first_use_misses']} | {m['recurrent_misses']} |")
    lines += ['', 'The 0% replay reproduces every recorded LRU hit/miss. Replay uses the same C++ metadata implementation as inference; both measured policies reproduce the corresponding predicted phase upload bytes exactly. The 50% and 90% rows are diagnostic simulations, not measured speedups or tuning on the held-out corpus.', '',
        '## Ownership and limits', '',
        'New entries enter probation. A hit in a later operation promotes a slice to the protected queue; excess protected entries return to probation. Same-operation duplicates do not count as reuse. Eviction prefers unreserved probation entries, then unreserved protected entries. Every returned slot is reserved by an operation epoch until the next synchronous operation, after both CUDA streams have drained. No future routing decisions or token identities enter the policy.', '',
        'GPU weights remain MXFP4, cache capacities and peer ownership are unchanged, and no extra GPU weight buffer is allocated. Replacement metadata remains on the CPU. Avoided misses remove their pageable-to-pinned staging and H2D copy; remaining misses and CPU dispatch are still present. This is not complete CPU bypass.', '',
        '## Validation and reproduction', '',
        '52 CTests passed without skips/failures. CPU tests check an independent LRU model, slot contents, reservations across demotions, all-reserved rejection, duplicates, scans, changing working sets and invalidation. CUDA integration includes both peer placements, arithmetic variants and callback recovery. Raw trace files contain a four-u64 header (magic, slots, slice bytes, record count) followed by key/batch/hit u64 triples.', '',
        '`prepare.py`, `trace.py`, `run.py`, `queue.py`, `analyze.py`, `frozen-inputs.json`, build/source identity, command logs, token IDs, logits, telemetry and trace/replay evidence are archived. `DS41_CACHE_POLICY=slru` opts in; `DS41_CACHE_PROTECTED_PERCENT` defaults to 80. LRU remains the default and rollback control.', '']
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(accepted=accepted,comparisons=comparisons)))

if __name__ == '__main__': main()
