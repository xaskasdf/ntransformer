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
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    router = any(c['decode']['per_token'].get('router_gpu_calls',0) for c in cases.values())
    lines=['# DeepSeek '+('resident router and scheduling' if router else 'scheduling and MXFP4 lookup')+' experiment','',
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
    if router:
        lines += ['','## Resident router','',
            'The router matrix projection moves from CPU to GPU with resident BF16/F32 weights, FP32 inputs and FP32 accumulation. Eight-lane CUDA reductions reproduce the CPU AVX2 two-accumulator order; a scalar-order path is also available. Host softplus, expert selection and normalization remain unchanged. No new activation quantization is introduced.',
            '', '| Case | Router ms/token | GPU router calls/token |', '|---|---:|---:|']
        for name,c in cases.items():
            d=c['decode']['per_token'];lines.append(f"| {name} | {d['router']*1000:.4f} | {d.get('router_gpu_calls',0):.0f} |")
        lines += ['','Final defaults: `DS41_ROUTER_GPU=1`, `DS41_SHARED_EARLY=1`, `DS41_QKV_READBACK_BATCH=1`. `DS41_STAGING_POOL` and `DS41_MXFP4_REGISTER_LUT` default to 0 after their ablations showed no throughput gain. Setting `DS41_ROUTER_GPU=0` restores the CPU projection. Unsupported types or unavailable device allocations fall back to CPU.',
            '', 'The 40 router matrices are BF16, totaling 157,286,400 bytes (150 MiB). This is additional resident VRAM while the routed-expert cache stays at 62 GiB. The new projection adds one upload of 5,120 FP32 values and one readback of 384 FP32 values per layer, plus one host completion wait; it removes the CPU matrix work and its worker-pool dispatch. Eliminating those intermediate host boundaries requires moving expert selection and adjacent state onto GPU too.',
            '', 'The initial source/binary hashes are preserved in `initial-environment.json` and `initial-source/`. `defaults-*` and `control-p128` use the final source/defaults recorded by `environment.json`. Other timed cases use the initial binary with explicit switches; its computation is unchanged by final default selection.',
            '', 'Router tests compare 48 F32/BF16 projections bitwise with the CPU AVX2 and independent scalar-order references, including non-aligned dimensions, changing inputs, growth and reuse of buffers, bounds and disabled GPU behavior. Final CTest also exercises prefill state using the external tiny F32 fixture. Earlier scheduling tests at real/max shapes are archived in the sibling `gpu-flow` artifact.']
    lines += ['','Finite logits and within-process repeat identity are checked by the harness. Full output hashes compare the initial prefill prediction and all 64 subsequent predictions; identical hashes also require identical consumed continuations. This is a regression control for the tested inputs, not a general model-quality evaluation.',
        '', 'Scheduling tests cover all MXFP4 codes, cold/warm caches, duplicate experts, negative weights, callback exceptions and recovery. QKV and shared-expert tests include independent CPU references. The initial `gpu-flow/test-comparison.json` also records small, model-sized and maximum-shape cross-configuration output equality.',
        '', 'These changes retain FP32 activations, weight formats, reduction order and model architecture. Experimental controls examine host scheduling and a divergent constant-table lookup. They do not implement device-resident attention, peer experts or batched prefill.',
        '', 'Kernel lookup motivation: distinct constant-memory addresses within a warp are serviced separately ([NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#constant-memory)). The magnitude table fits into a 32-bit immediate; nibble extraction and sign application reproduce its exact float values. Measured ablations determine the effect in this engine.',
        '', 'Per-token phase values in `summary.json` are host wall intervals, including waits. They are not device kernel times. Small differences between separate processes should be treated cautiously; repeated warm requests do not establish independent process-level variance. Raw command records, telemetry, logs, output tensors, source snapshots and hashes accompany this report. No model weights are bundled.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({k:{'decode':v['decode']['tokens_per_second'],'prefill':v['prefill']['tokens_per_second'],'comparison':v.get('comparison')} for k,v in cases.items()},indent=2))
if __name__=='__main__':main()
