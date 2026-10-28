import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = {32: 'eb15407ed93c1cb3e6ed845b32f7bad62d6cd4fe459fccc0e236c4b00e0c76ce',
            128: '6f2df19adc6049c7d7e2d62aefd07de33fe71d93835d5b05571b6f62f4de3840'}

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def numerical(name, ref_name, c, ref):
    a = np.fromfile(ROOT / ref_name / 'all-logits.f32', dtype='<f4').reshape(-1, c['vocab'])
    b = np.fromfile(ROOT / name / 'all-logits.f32', dtype='<f4').reshape(a.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    # Greedy runs cease to share a context after consuming a differing prediction.
    differing = [i for i, (x, y) in enumerate(zip(c['predicted_ids'], ref['predicted_ids'])) if x != y]
    aligned = len(a) if not differing else differing[0] + 1
    if c.get('consumed_ids') == ref.get('consumed_ids') and c.get('consumed_ids') is not None:
        aligned = len(a)
    max_abs = 0.; square = 0.; norm = 0.; kl = []; argmax = 0
    for x, y in zip(a[:aligned], b[:aligned]):
        x, y = x.astype(np.float64), y.astype(np.float64)
        delta = y - x
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        square += float(delta @ delta); norm += float(x @ x)
        argmax += int(x.argmax() == y.argmax())
        lx = x - x.max(); lx -= np.log(np.exp(lx).sum())
        ly = y - y.max(); ly -= np.log(np.exp(ly).sum())
        kl.append(max(0., float(np.exp(lx) @ (lx - ly))))
    return dict(aligned_vectors=aligned, total_vectors=len(a), max_abs=max_abs,
                relative_l2=(square / norm)**.5 if norm else 0., rmse=(square / (aligned*c['vocab']))**.5,
                argmax_matches=argmax, mean_kl=float(np.mean(kl)), max_kl=max(kl),
                all_logits_bitwise_equal=c['all_logits_sha256'] == ref['all_logits_sha256'])

def main():
    telemetry = []
    for p in ROOT.glob('*telemetry.csv'):
        with p.open() as f:
            for r in csv.DictReader(f, skipinitialspace=True):
                if r.get('index') == '1':
                    telemetry.append((datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc),
                                      int(r['memory.used [MiB]'].split()[0]) / 1024))
    cases = {}
    for p in sorted(ROOT.glob('*/measurements.jsonl')):
        if '--partial' in sys.argv and not (ROOT / (p.parent.name + '-command.json')).exists():
            continue
        records = [json.loads(line) for line in p.read_text().splitlines()]
        setup = next(r for r in records if r['kind'] == 'setup')
        done = [r for r in records if r['kind'] == 'completion']
        name = p.parent.name
        command = json.loads((ROOT / (name + '-command.json')).read_text())
        assert command['returncode'] == 0 and len(done) >= 2 and all(r['repeat_bitwise_equal'] for r in done), name
        c = dict(prompt_tokens=setup['prompt_tokens'], decode_steps=setup['decode_steps'], vocab=setup['vocab'],
                 all_logits_sha256=sha(p.parent / 'all-logits.f32'), final_logits_sha256=sha(p.parent / 'final-logits.f32'),
                 predicted_ids=done[0]['predicted_ids'], consumed_ids=done[0].get('consumed_ids', done[0]['predicted_ids'][:-1]),
                 all_repeats_bitwise_equal=True, environment=command['environment'])
        c['logit_values'] = (p.parent / 'all-logits.f32').stat().st_size // 4
        assert c['logit_values'] == (c['decode_steps'] + 1) * c['vocab']
        for phase in ['prefill', 'decode']:
            selected = [r for r in records if r.get('phase') == phase and r['round'] > 0]
            assert len(selected) == len(done) - 1
            count, seconds = sum(r['tokens'] for r in selected), sum(r['seconds'] for r in selected)
            skip = {'kind', 'round', 'phase', 'tokens', 'seconds', 'token_seconds'}
            d = {k: sum(r[k] for r in selected) / count for k in selected[0] if k not in skip}
            assert d['hc_gpu_calls'] == 80 and d['routed_gpu_calls'] == d['shared_gpu_calls'] == d['router_gpu_calls'] == 40
            assert d['routed_cpu_calls'] == d['shared_cpu_calls'] == 0
            resident = c['environment'].get('DS41_ATTENTION_GPU', '1') != '0'
            assert d['resident_attention_calls'] == (40 if resident else 0), (name, phase, d['resident_attention_calls'])
            c[phase] = dict(tokens_per_second=count/seconds, ms_per_token=seconds/count*1000,
                            warm_requests=len(selected), per_token=d,
                            request_tokens_per_second=[r['tokens']/r['seconds'] for r in selected])
        for line in (ROOT / (name + '-resources.txt')).read_text().splitlines():
            if 'Maximum resident set size (kbytes)' in line: c['peak_rss_gib'] = int(line.split(':')[-1]) / 1024**2
        start = datetime.fromisoformat(command['utc_start']); end = start + timedelta(seconds=command['wall_seconds'])
        samples = [v for t, v in telemetry if start <= t <= end]
        c['sampled_peak_gpu_gib'] = max(samples) if samples else None
        c['matches_previous_hc_default'] = c['all_logits_sha256'] == PREVIOUS.get(c['prompt_tokens'])
        cases[name] = c
    for name, c in cases.items():
        ref_name = 'control-p' + str(c['prompt_tokens'])
        if ref_name in cases:
            ref = cases[ref_name]
            c['comparison'] = numerical(name, ref_name, c, ref)
            c['comparison'].update(prefill_speedup=c['prefill']['tokens_per_second']/ref['prefill']['tokens_per_second'],
                                    decode_speedup=c['decode']['tokens_per_second']/ref['decode']['tokens_per_second'],
                                    consumed_ids_equal=c['consumed_ids'] == ref['consumed_ids'],
                                    expert_upload_bytes_equal=all(c[p]['per_token']['routed_upload_bytes'] == ref[p]['per_token']['routed_upload_bytes'] for p in ['prefill','decode']))
    summary = dict(cases=cases)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# DeepSeek resident attention experiment', '',
             'One physical H100 (GPU 1), the same ten GGUF shards, FP32 activations and a 62 GiB expert cache. Each process runs one initial request and three warm repeats, with 64 decode steps. EOS is recorded but does not stop this fixed-work benchmark. Rates divide total warm work by total warm time. Initial uploads, validation and output files are outside token timings. No CUPTI tracing.', '',
             '| Case | Prefill tok/s | Decode tok/s | Decode ratio | Matched argmax / aligned vectors | Relative L2 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, c in cases.items():
        d = c.get('comparison', {})
        lines.append(f"| {name} | {c['prefill']['tokens_per_second']:.4f} | {c['decode']['tokens_per_second']:.4f} | {d.get('decode_speedup', float('nan')):.4f} | {d.get('argmax_matches')} / {d.get('aligned_vectors')} | {d.get('relative_l2', float('nan')):.9g} |")
    lines += ['', '| Case | Attention total ms/token | Prologue readback + tail input MiB/token | New compressed + metadata KiB/token | Resident calls/token | Sampled peak GPU GiB |',
              '|---|---:|---:|---:|---:|---:|']
    for name, c in cases.items():
        d = c['decode']['per_token']
        lines.append(f"| {name} | {1000*d['attention']:.4f} | {(d['prologue_readback_bytes'] + d['tail_input_upload_bytes'])/1024**2:.6f} | {(d['compressed_upload_bytes']+d['attention_metadata_upload_bytes'])/1024:.4f} | {d['resident_attention_calls']:.0f} | {c['sampled_peak_gpu_gib']} |")
    lines += ['', '## Dataflow', '',
              'The session owns a device raw KV ring (10 MiB for 40 layers, window 128, dimension 512), independent query/output scratch, RoPE tables and growing compressed-row buffers. Prologue writes Q directly into that session and appends its new KV row with a device copy. It still reads back the 1,280-value query latent for CPU indexing and a 512-value KV mirror for CPU fallback. It removes the 32,768-value Q readback and the equally sized attention-output upload at every layer: 10 MiB/token of logical transfer payload. Host prologue/tail waits remain; removing these also requires moving compression/indexing and residual state.', '',
              'Compressed rows retain their CPU computation and selection. Each owner uploads its newly visible immutable prefix; capacity growth recopies that prefix and reset invalidates it. Selection order, repeated keys and layer sharing are preserved. Copies use separate offsets in a session-owned pinned staging allocation, which remains alive through the tail completion wait.', '',
              'The score kernel cooperatively stages sixteen keys into shared memory and computes ordered FP32 products/sums. The second kernel includes attention sinks, computes exponentials in double then rounds them to FP32, accumulates the denominator and values in key order, and applies inverse RoPE. The output projection consumes its device result directly. Tensor precision and model architecture are unchanged; library/FMA rounding can differ from CPU, so output differences are measured rather than assumed harmless.', '',
              '## Validation and interpretation', '',
              'Tests cover the actual 64-head, 512-dimensional shape, raw ring wrap, changing queries, sinks, compressed-prefix growth and sharing, selected order/duplicates, two independent owners, reset, query recovery and invalid bounds. Prologue tests compare resident Q and host mirrors against ordinary prologue bitwise, and count avoided readback bytes. Twenty registered tests include these cases and prior HC/router/shared/backbone/MoE/prefill checks.', '',
              'The benchmark checks finite values and full-logit repeat identity. Numerical comparison uses only vectors whose input contexts are aligned; it reports the aligned count explicitly if greedy paths diverge. KL, argmax agreement and logit norms are regression diagnostics, not an independent language-quality benchmark. Identical logits are useful but not mandatory.', '',
              'The existing CPU attention_loop/tail intervals describe the control. The resident path reports a combined resident_attention host interval that includes GPU attention, inverse RoPE, output projection and final wait; it is nested inside attention total. Do not compare that interval with the CPU loop alone or add nested phases together. Transfer counters are successful logical payloads, not PCIe transactions. Memory peaks are one-second samples, not allocator peaks. Warm repeats do not establish independent process-level variance.', '',
              'Resident attention is enabled by default; `DS41_ATTENTION_GPU=0` selects the CPU attention control. Detailed stage dumping retains the CPU attention path. Allocation/shape limits retain ordinary prologue or recover Q for CPU attention; failure to recover an in-flight query fails loudly. Reset clears visibility metadata without exposing stale rows. The existing single-stream engine remains non-reentrant.', '',
              'The six control/resident cases use initial-source and initial-environment.json. The final source additionally enables the default, invalidates stale query metadata before replacement, hardens probe failure recovery, and extends tests to zero RoPE and fallback. It preserves the successful-path arithmetic. defaults-p32 and ctest-final validate that final build with the attention flag unset. Both source snapshots and their hashes are retained.', '',
              'Raw logs, outputs, telemetry, source/build hashes and this analysis accompany the report. Model weights remain external.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({n: dict(decode=c['decode']['tokens_per_second'], prefill=c['prefill']['tokens_per_second'], comparison=c.get('comparison')) for n,c in cases.items()}, indent=2))

if __name__ == '__main__': main()
