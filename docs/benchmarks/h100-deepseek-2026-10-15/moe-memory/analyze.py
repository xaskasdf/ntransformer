import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def compare(name, reference, case, ref):
    a = np.fromfile(ROOT / reference / 'all-logits.f32', dtype='<f4').reshape(-1, case['vocab'])
    b = np.fromfile(ROOT / name / 'all-logits.f32', dtype='<f4').reshape(a.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    differing = [i for i, (x, y) in enumerate(zip(case['predicted_ids'], ref['predicted_ids'])) if x != y]
    aligned = len(a) if not differing else differing[0] + 1
    if case['consumed_ids'] == ref['consumed_ids']: aligned = len(a)
    square = norm = max_abs = 0.
    argmax = 0
    for x, y in zip(a[:aligned], b[:aligned]):
        x, y = x.astype(np.float64), y.astype(np.float64)
        delta = y - x
        square += float(delta @ delta); norm += float(x @ x)
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        argmax += int(x.argmax() == y.argmax())
    return dict(reference=reference, aligned_vectors=aligned, total_vectors=len(a),
        max_abs=max_abs, relative_l2=(square/norm)**.5 if norm else 0., argmax_matches=argmax,
        all_logits_bitwise_equal=case['all_logits_sha256'] == ref['all_logits_sha256'],
        consumed_ids_equal=case['consumed_ids'] == ref['consumed_ids'],
        expert_upload_bytes_equal=all(case[p]['per_token']['routed_upload_bytes'] == ref[p]['per_token']['routed_upload_bytes'] for p in ['prefill', 'decode']),
        prefill_speedup=case['prefill']['tokens_per_second']/ref['prefill']['tokens_per_second'],
        decode_speedup=case['decode']['tokens_per_second']/ref['decode']['tokens_per_second'])

def main():
    telemetry = []
    for path in ROOT.glob('*telemetry.csv'):
        with path.open() as f:
            for r in csv.DictReader(f, skipinitialspace=True):
                if r.get('index') == '1':
                    telemetry.append((datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc),
                                      int(r['memory.used [MiB]'].split()[0])/1024))
    cases = {}
    for path in sorted(ROOT.glob('*/measurements.jsonl')):
        name = path.parent.name
        command_path = ROOT / (name + '-command.json')
        if '--partial' in sys.argv and not command_path.exists(): continue
        command = json.loads(command_path.read_text())
        records = [json.loads(line) for line in path.read_text().splitlines()]
        setup = next(r for r in records if r['kind'] == 'setup')
        done = [r for r in records if r['kind'] == 'completion']
        assert command['returncode'] == 0 and len(done) == 4 and all(r['repeat_bitwise_equal'] for r in done), name
        case = dict(prompt_tokens=setup['prompt_tokens'], decode_steps=setup['decode_steps'], vocab=setup['vocab'],
                    all_logits_sha256=sha(path.parent/'all-logits.f32'), predicted_ids=done[0]['predicted_ids'],
                    consumed_ids=done[0].get('consumed_ids', done[0]['predicted_ids'][:-1]),
                    all_repeats_bitwise_equal=True, environment=command['environment'])
        case['logit_values'] = (path.parent/'all-logits.f32').stat().st_size//4
        assert case['logit_values'] == (case['decode_steps']+1)*case['vocab']
        for phase in ['prefill', 'decode']:
            selected = [r for r in records if r.get('phase') == phase and r['round'] > 0]
            assert len(selected) == 3
            count = sum(r['tokens'] for r in selected)
            seconds = sum(r['seconds'] for r in selected)
            skip = {'kind', 'round', 'phase', 'tokens', 'seconds', 'token_seconds'}
            metrics = {k: sum(r[k] for r in selected)/count for k in selected[0] if k not in skip}
            assert metrics['routed_gpu_calls'] == metrics['shared_gpu_calls'] == metrics['resident_attention_calls'] == 40
            assert metrics['routed_cpu_calls'] == metrics['shared_cpu_calls'] == metrics['moe_trace_failures'] == 0
            traced = case['environment'].get('DS41_MOE_TRACE', '0') != '0'
            assert metrics['moe_traced_calls'] == (40 if traced else 0)
            default = '1' if name.startswith('defaults-') else '0'
            register = case['environment'].get('DS41_MXFP4_REGISTER_LUT', default) != '0'
            assert metrics['register_lut_launches'] == (80 if register else 0)
            if 'cooperative_mxfp4_launches' in metrics:
                cooperative = case['environment'].get('DS41_MXFP4_COOPERATIVE', default) != '0'
                assert metrics['cooperative_mxfp4_launches'] == (80 if cooperative else 0)
            case[phase] = dict(tokens_per_second=count/seconds, ms_per_token=seconds/count*1000,
                               warm_requests=3, per_token=metrics,
                               request_tokens_per_second=[r['tokens']/r['seconds'] for r in selected])
        for line in (ROOT/(name+'-resources.txt')).read_text().splitlines():
            if 'Maximum resident set size (kbytes)' in line: case['peak_rss_gib'] = int(line.split(':')[-1])/1024**2
        start = datetime.fromisoformat(command['utc_start'])
        end = start + timedelta(seconds=command['wall_seconds'])
        samples = [v for t,v in telemetry if start <= t <= end]
        case['sampled_peak_gpu_gib'] = max(samples) if samples else None
        cases[name] = case
    for name, case in cases.items():
        reference = 'control-p'+str(case['prompt_tokens'])
        if reference not in cases: reference = 'pilot-control'
        if reference in cases and cases[reference]['prompt_tokens'] == case['prompt_tokens']:
            case['comparison'] = compare(name, reference, case, cases[reference])
    (ROOT/'summary.json').write_text(json.dumps(dict(cases=cases), indent=2)+'\n', encoding='utf-8', newline='\n')
    lines = ['# DeepSeek MXFP4 memory-access experiment', '',
        'One physical H100 (GPU 1), the same ten GGUF shards, FP32 activations and the unchanged 62 GiB expert cache. Each process runs one initial request and three warm repeats with 64 decode steps. EOS is recorded without stopping fixed work. Throughput divides total warm work by total warm time; loading, validation and output files are excluded. Warm repeats do not establish independent process-level variance.', '',
        '| Case | Prefill tok/s | Decode tok/s | Decode ratio | Matched argmax | Relative L2 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, c in cases.items():
        d = c.get('comparison', {})
        lines.append(f"| {name} | {c['prefill']['tokens_per_second']:.4f} | {c['decode']['tokens_per_second']:.4f} | {d.get('decode_speedup', float('nan')):.4f} | {d.get('argmax_matches')} / {d.get('aligned_vectors')} | {d.get('relative_l2', float('nan')):.9g} |")
    lines += ['', '## MoE phase measurements', '',
        'Trace-enabled cases record host intervals and adjacent events on the existing CUDA stream. Host staging overlaps device work; these columns must not be added. Preparation includes the early shared expert and possible host submission gaps. Compute spans include launches and any stream idle gaps, not isolated GPU busy time. Trace overhead is measured separately; throughput claims use untraced cases.', '',
        '| Traced case | Host staging ms/token | Upload stream ms/token | Routed compute stream ms/token | Host wait ms/token |',
        '|---|---:|---:|---:|---:|']
    for name,c in cases.items():
        m = c['decode']['per_token']
        if not m['moe_traced_calls']: continue
        lines.append(f"| {name} | {1000*m['moe_host_staging']:.4f} | {1000*m['moe_stream_upload']:.4f} | {1000*m['moe_stream_compute']:.4f} | {1000*m['moe_host_wait']:.4f} |")
    lines += ['', '| Case | MoE wall ms/token | Routed upload MiB/token | Cooperative launches/token | Sampled peak GPU GiB | Peak RSS GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for name,c in cases.items():
        m = c['decode']['per_token']
        lines.append(f"| {name} | {1000*m['moe']:.4f} | {m['routed_upload_bytes']/1024**2:.4f} | {m.get('cooperative_mxfp4_launches', 0):.0f} | {c['sampled_peak_gpu_gib']} | {c['peak_rss_gib']:.4f} |")
    lines += ['', '## Implementation and validation', '',
        'The cooperative kernel keeps one warp per output row and the existing ordered FP32 accumulation. Warps load contiguous words for each 32-block MXFP4 tile, then decode from shared memory. The CTA loads the activation tile once, with a 33-float pitch to avoid shared-memory bank conflicts. Shared storage is 8,576 bytes per CTA; the weight cache and model representation do not change. Rows require columns divisible by 128 for aligned word loads; other shapes use the original kernel. Partial rows participate in all barriers and cannot write past output bounds.', '',
        'Tests include all 16 quantized codes, changing activations, cache hits/misses, partial CTAs/tiles, alignment fallback, actual and maximum expert dimensions, routing duplicates, signed weights, callback ownership and exception recovery. Independent FP64 GEMV references and cross-process binary comparisons complement full-model finite/repeat checks. Identical logits are a regression result for these inputs, not an independent language-quality evaluation.', '',
        'The register-only, cooperative-only and combined probes separate the interaction between the two changes. NVIDIA documents serialization when warp lanes read distinct constant-memory addresses and the benefit of coalesced global accesses in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#constant-memory). The observed interaction is consistent with one memory-access cost hiding another; no hardware-counter trace was obtained to prove individual stall causes.', '',
        'The pilot and isolated register-LUT probes use benchmark_pilot.cpp with the prior attention library. Candidate cases use the new harness with its additional cooperative-launch counter. Compiler flags, binary/library hashes, environments and source snapshots are archived. Model assets stay external.', '',
        ('Final source enables both flags by default. Set DS41_MXFP4_COOPERATIVE=0 and DS41_MXFP4_REGISTER_LUT=0 for the original control; set just one flag to zero to isolate its contribution. The final build passes 28 selected DS41 CTest cases, 18 cross-process numerical cases with both flags unset in the default fixtures. These final correctness checks use a 2 GiB expert-cache budget alongside Brandon training; no throughput is measured concurrently. The isolated full-model timings come from the candidate build with both flags explicitly enabled. A source check confirms that final production changes are only the two flag defaults and a size comment. candidate-source and candidate-environment.json preserve the opt-in build used for the factorial and paired-context cases; initial-source preserves the prior attention source used by the pilot library. The compiled harness is the root benchmark_ds41.cpp (benchmark_pilot.cpp for the pilot), whose hash is recorded separately. The final source/tools copy is synchronized with that harness.' if (ROOT/'validation-final.json').exists() else 'The opt-in candidate is under validation; final default validation has not completed.'), '',
        'Memory peaks are one-second samples. Byte counters are successful logical payloads, not physical bus transactions. Only one GPU is used; peer ownership/cache and copy/compute overlap remain separate experiments.', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({n:dict(decode=c['decode']['tokens_per_second'], comparison=c.get('comparison')) for n,c in cases.items()}, indent=2))

if __name__ == '__main__': main()
