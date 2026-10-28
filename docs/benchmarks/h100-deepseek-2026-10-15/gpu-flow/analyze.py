import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    cases={}
    for p in sorted(ROOT.glob('*/measurements.jsonl')):
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        setup=next(r for r in rows if r['kind']=='setup')
        completions=[r for r in rows if r['kind']=='completion']
        if len(completions)<2:continue
        name=p.parent.name
        cases[name]=dict(prompt_tokens=setup['prompt_tokens'],decode_steps=setup['decode_steps'],
            all_logits_sha256=sha(p.parent/'all-logits.f32'),final_logits_sha256=sha(p.parent/'final-logits.f32'),
            logit_values=(p.parent/'all-logits.f32').stat().st_size//4,
            predicted_ids=completions[0]['predicted_ids'],
            all_repeats_bitwise_equal=all(r['repeat_bitwise_equal'] for r in completions))
        for phase in ['prefill','decode']:
            selected=[r for r in rows if r.get('phase')==phase and r['round']>0]
            total_tokens=sum(r['tokens'] for r in selected)
            secs=sum(r['seconds'] for r in selected)
            skip={'kind','round','phase','tokens','seconds','token_seconds'}
            detail={k:sum(r[k] for r in selected)/total_tokens for k in selected[0] if k not in skip}
            cases[name][phase]=dict(tokens_per_second=total_tokens/secs,ms_per_token=secs/total_tokens*1000,
                warm_requests=len(selected),request_tokens_per_second=[r['tokens']/r['seconds'] for r in selected],
                per_token=detail)
        command=ROOT/(name+'-command.json')
        if command.exists():cases[name]['environment']=json.loads(command.read_text())['environment']
        resources=ROOT/(name+'-resources.txt')
        if resources.exists():
            for line in resources.read_text().splitlines():
                if 'Maximum resident set size (kbytes)' in line:cases[name]['peak_rss_gib']=int(line.split(':')[-1])/1024**2
    for name,c in cases.items():
        baseline=cases.get('control-p'+str(c['prompt_tokens']))
        if baseline:
            c['comparison']=dict(all_logits_bitwise_equal=c['all_logits_sha256']==baseline['all_logits_sha256'],
                predicted_ids_equal=c['predicted_ids']==baseline['predicted_ids'],
                decode_speedup=c['decode']['tokens_per_second']/baseline['decode']['tokens_per_second'],
                prefill_speedup=c['prefill']['tokens_per_second']/baseline['prefill']['tokens_per_second'])
    summary=dict(cases=cases,tests=json.loads((ROOT/'test-comparison.json').read_text()) if (ROOT/'test-comparison.json').exists() else None)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# DeepSeek scheduling and MXFP4 lookup experiment','',
        'One physical H100 (GPU 1), the same 10 GGUF shards, corpus, context and 62 GiB expert-cache budget. One initial request and three warm repeats per process unless stated by the per-case JSON. Rates divide total warm work by total warm time. No CUPTI tracing in timed runs.','',
        '| Case | Prefill tok/s | Decode tok/s | Decode ratio | All 65 logit vectors match control |',
        '|---|---:|---:|---:|---|']
    for name,c in cases.items():
        comp=c.get('comparison',{})
        lines.append(f"| {name} | {c['prefill']['tokens_per_second']:.4f} | {c['decode']['tokens_per_second']:.4f} | {comp.get('decode_speedup',float('nan')):.4f} | {comp.get('all_logits_bitwise_equal','no paired control')} |")
    lines += ['','The four independently selectable flags are `DS41_STAGING_POOL`, `DS41_SHARED_EARLY`, `DS41_QKV_READBACK_BATCH`, and `DS41_MXFP4_REGISTER_LUT`. Exact per-process settings and durations are recorded in command JSON files.',
        '', '| Case | QKV readback waits/token | Transient copy threads/token | Staged slices/token | Weight upload GiB/token |',
        '|---|---:|---:|---:|---:|']
    for name,c in cases.items():
        d=c['decode']['per_token']
        lines.append(f"| {name} | {d['prologue_readback_waits']:.0f} | {d['transient_copy_threads']:.3f} | {d['staged_slices']:.3f} | {d['routed_upload_bytes']/1024**3:.6f} |")
    lines += ['','Finite logits and within-process repeat identity are checked by the harness. Full output hashes compare the initial prefill prediction and all 64 subsequent predictions; identical hashes also require identical consumed continuations. This is a regression control for the tested inputs, not a general model-quality evaluation.',
        '', 'Synthetic tests cover small, model-sized and maximum routed shapes, all MXFP4 codes, cold/warm caches, duplicate experts, negative weights, callback exceptions and recovery. QKV and shared-expert tests include independent CPU references. `test-comparison.json` records cross-configuration output equality.',
        '', 'These changes retain FP32 activations, weight formats, reduction order and model architecture. They remove host scheduling overhead and a divergent constant-table lookup. They do not implement device-resident attention, peer experts or batched prefill.',
        '', 'Kernel lookup motivation: distinct constant-memory addresses within a warp are serviced separately ([NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#constant-memory)). The magnitude table fits into a 32-bit immediate; nibble extraction and sign application reproduce its exact float values. Measured ablations determine the effect in this engine.',
        '', 'Per-token phase values in `summary.json` are host wall intervals, including waits. They are not device kernel times. Small differences between separate processes should be treated cautiously; repeated warm requests do not establish independent process-level variance. Raw command records, telemetry, logs, output tensors, source snapshots and hashes accompany this report. No model weights are bundled.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:{'decode':v['decode']['tokens_per_second'],'prefill':v['prefill']['tokens_per_second'],'comparison':v.get('comparison')} for k,v in cases.items()},indent=2))
if __name__=='__main__':main()
