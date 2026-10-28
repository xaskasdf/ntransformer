import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = {32: 'eb15407ed93c1cb3e6ed845b32f7bad62d6cd4fe459fccc0e236c4b00e0c76ce',
            128: '6f2df19adc6049c7d7e2d62aefd07de33fe71d93835d5b05571b6f62f4de3840'}

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    cases = {}
    telemetry = []
    for p in ROOT.glob('*telemetry.csv'):
        with p.open() as f:
            for r in csv.DictReader(f, skipinitialspace=True):
                if r.get('index') == '1':
                    telemetry.append((datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc),
                                      int(r['memory.used [MiB]'].split()[0]) / 1024))
    for p in sorted(ROOT.glob('*/measurements.jsonl')):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        setup = next(r for r in rows if r['kind'] == 'setup')
        done = [r for r in rows if r['kind'] == 'completion']
        name = p.parent.name
        command = json.loads((ROOT / (name + '-command.json')).read_text())
        assert command['returncode'] == 0 and len(done) >= 2, name
        assert all(r['repeat_bitwise_equal'] for r in done), name
        c = dict(prompt_tokens=setup['prompt_tokens'], decode_steps=setup['decode_steps'],
                 all_logits_sha256=sha(p.parent / 'all-logits.f32'),
                 final_logits_sha256=sha(p.parent / 'final-logits.f32'),
                 predicted_ids=done[0]['predicted_ids'], all_repeats_bitwise_equal=True,
                 environment=command['environment'])
        c['logit_values'] = (p.parent / 'all-logits.f32').stat().st_size // 4
        assert c['logit_values'] == (setup['decode_steps'] + 1) * setup['vocab']
        for phase in ['prefill', 'decode']:
            selected = [r for r in rows if r.get('phase') == phase and r['round'] > 0]
            assert len(selected) == len(done) - 1
            count, seconds = sum(r['tokens'] for r in selected), sum(r['seconds'] for r in selected)
            skip = {'kind', 'round', 'phase', 'tokens', 'seconds', 'token_seconds'}
            detail = {k: sum(r[k] for r in selected) / count for k in selected[0] if k not in skip}
            assert detail['hc_gpu_calls'] + detail['hc_cpu_pool_calls'] + detail['hc_cpu_serial_calls'] == 80
            assert detail['routed_cpu_calls'] == detail['shared_cpu_calls'] == 0
            assert detail['routed_gpu_calls'] == detail['shared_gpu_calls'] == detail['router_gpu_calls'] == 40
            c[phase] = dict(tokens_per_second=count / seconds, ms_per_token=seconds / count * 1000,
                            warm_requests=len(selected), per_token=detail,
                            request_tokens_per_second=[r['tokens'] / r['seconds'] for r in selected])
        for line in (ROOT / (name + '-resources.txt')).read_text().splitlines():
            if 'Maximum resident set size (kbytes)' in line:
                c['peak_rss_gib'] = int(line.split(':')[-1]) / 1024**2
        start = datetime.fromisoformat(command['utc_start'])
        end = start + timedelta(seconds=command['wall_seconds'])
        samples = [v for t, v in telemetry if start <= t <= end]
        c['sampled_peak_gpu_gib'] = max(samples) if samples else None
        c['matches_previous_router_default'] = c['all_logits_sha256'] == PREVIOUS.get(c['prompt_tokens'])
        cases[name] = c
    for name, c in cases.items():
        ref = cases.get('control-p' + str(c['prompt_tokens']))
        if ref:
            c['comparison'] = dict(all_logits_bitwise_equal=c['all_logits_sha256'] == ref['all_logits_sha256'],
                                   predicted_ids_equal=c['predicted_ids'] == ref['predicted_ids'],
                                   expert_upload_bytes_equal=all(c[p]['per_token']['routed_upload_bytes'] == ref[p]['per_token']['routed_upload_bytes'] for p in ['prefill', 'decode']),
                                   prefill_speedup=c['prefill']['tokens_per_second'] / ref['prefill']['tokens_per_second'],
                                   decode_speedup=c['decode']['tokens_per_second'] / ref['decode']['tokens_per_second'])
    summary = dict(cases=cases)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# DeepSeek hyper-connection projection experiment', '',
             'One physical H100 (GPU 1), the same 10 GGUF shards, FP32 activations, prompt token IDs and 62 GiB expert cache. Each case runs one initial request and three warm repeats. Reported rates divide total warm work by total warm time; the initial request, including cold weight uploads, is excluded. No CUPTI tracing.', '',
             '| Case | Prefill tok/s | Decode tok/s | Decode ratio | All 65 logit vectors match control |',
             '|---|---:|---:|---:|---|']
    for name, c in cases.items():
        d = c.get('comparison', {})
        lines.append(f"| {name} | {c['prefill']['tokens_per_second']:.4f} | {c['decode']['tokens_per_second']:.4f} | {d.get('decode_speedup', float('nan')):.4f} | {d.get('all_logits_bitwise_equal', 'unpaired')} |")
    lines += ['', '| Case | HC norm ms/token | HC projection ms/token | HC mix ms/token | GPU / pool / serial calls per token | Sampled peak GPU GiB |',
              '|---|---:|---:|---:|---|---:|']
    for name, c in cases.items():
        d = c['decode']['per_token']
        lines.append(f"| {name} | {1000*d['hc_norm']:.4f} | {1000*d['hc_projection']:.4f} | {1000*d['hc_mix']:.4f} | {d['hc_gpu_calls']:.0f} / {d['hc_cpu_pool_calls']:.0f} / {d['hc_cpu_serial_calls']:.0f} | {c['sampled_peak_gpu_gib']} |")
    lines += ['', '## Placement and controls', '',
              '`DS41_HC_PROJECTION_GPU=1` keeps the 80 HC projection matrices resident on GPU and computes their matrix-vector products there. The actual matrices are F32 [20,480 inputs, 24 outputs], totaling 157,286,400 bytes (150 MiB), in addition to the preceding router implementation. No new activation or weight quantization is introduced. GPU allocation/shape/type failures retain the CPU fallback.', '',
              '`DS41_HC_PROJECTION_GPU=0` selects the previous CPU implementation. `DS41_HC_PROJECTION_SERIAL=1` also bypasses its worker pool and directly calls the same row-dot routine; this is a diagnostic control. The GPU switch takes precedence when both are enabled. Each command JSON records its complete DS41 environment.', '',
              'The first GPU implementation (`gpu-p32`) reuses the precise BF16/F32 primitive with two FP32 accumulators per AVX2 lane and the original reduction order, one eight-thread block per row. The cooperative implementation instead uses 256 threads to stage weight and input tiles of 2,048 values into 16 KiB of shared memory per block. The same eight lanes then extend the same FMA chains across tiles. Barriers protect each tile before reuse. This changes memory scheduling, without reassociating the sums; the independent scalar-order path remains covered.', '',
              'Final defaults are `DS41_HC_PROJECTION_GPU=1` and `DS41_HC_COOPERATIVE=1`. `DS41_HC_PROJECTION_SERIAL` remains off. Setting the GPU switch to zero restores CPU projection; setting the cooperative switch to zero restores the first GPU kernel. Cooperative staging applies only to projections with at most 32 rows and at least 2,048 columns; larger routers keep their previous kernel.', '',
              'Moving this operation alone adds 80 input uploads, 80 tiny output readbacks and 80 host completion waits per token. Logical input/output payload is 6,561,280 bytes/token (6.257 MiB), excluding the initial weight upload. Normalization, sigmoid/Sinkhorn coefficients, collapse and residual mixing remain on CPU. This is a measured placement step toward GPU-resident state, not elimination of the CPU boundary.', '',
              '## Validation and interpretation', '',
              'The initial router/HC test checks 60 F32/BF16 projections against AVX2 and independent scalar-order references. The final test expands this to 72, adding a 4,115-column partial-tile case, and exercises both cooperative and direct kernels. Shapes include the actual HC matrix, tails, changing inputs, scratch-buffer growth/reuse, disabled CUDA behavior and bounds. CTest also covers shared experts, backbone projections, MoE overlap, HC collapse/mix and causal prefill/reset using an external tiny fixture. Inspect ctest logs for the actual pass count.', '',
              'The initial three cases use `initial-source/` and `initial-environment.json`; `cooperative-p32` uses `cooperative-source/` and `cooperative-environment.json`. The `defaults-*` cases and `control-p128` use the final `source/` and `environment.json`. The last change selects the validated defaults, with unchanged arithmetic. Exact build commands, environments and binary/library hashes are retained for each stage.', '',
              'The benchmark checks finite logits and within-process repeat identity. Full-output hashes compare the prefill prediction plus all 64 decode predictions (8,403,200 FP32 values per case). `summary.json` also compares them with the previous resident-router baseline. This is a regression check for these inputs, not a general quality evaluation; identical logits are useful evidence, not a prerequisite for future optimizations.', '',
              'HC norm/projection/mix are nested host wall intervals across both sublayers. They overlap the existing top-level HC/other totals and must not be added to them. They include CPU submission and waits, not isolated device kernel time. Memory peaks are one-second telemetry samples, not allocator high-water marks. Three warm requests in one process do not establish independent process-level variance.', '',
              'Raw logs, outputs, telemetry, source/build hashes and the standalone analysis accompany this report. Model weights are external and are not bundled.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({n: {k: c[k] for k in ['prefill', 'decode', 'comparison'] if k in c} for n, c in cases.items()}, indent=2))

if __name__ == '__main__':
    main()
