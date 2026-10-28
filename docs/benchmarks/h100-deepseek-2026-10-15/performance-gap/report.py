from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'summary.json').read_text())
lines=['# DeepSeek performance-gap probes','',
'Same model and consumed IDs as the completed llama.cpp comparison. One physical H100 (GPU 1), 52 CPU threads, 31 CPU-resident routed-expert layers in llama.cpp. All GPU probes pass the same training-lock and idle-device admission checks.',
'','## Uninstrumented controls','',
'One fresh process per configuration; one initial request and three warm requests. These controls isolate configuration effects, not independent process-to-process variance. Rates aggregate warm work divided by warm time. CUDA graphs and automatic operation offload are independent interventions.',
'','| Case | Prefill tok/s | Decode tok/s | Final logit relative L2 vs control |','|---|---:|---:|---:|']
for name in ['llama-control-p32','llama-no-graphs-p32','llama-control-p128','llama-no-op-offload-p128']:
    c=s['cases'][name]
    rates=[]
    for phase in ['prefill','decode']:
        rows=[r for r in c['phases'] if r['round']>0 and r['phase']==phase]
        rates.append(sum(r['tokens'] for r in rows)/sum(r['seconds'] for r in rows))
    delta=c.get('final_vs_control',{}).get('relative_l2',0)
    lines.append(f'| {name} | {rates[0]:.4f} | {rates[1]:.4f} | {delta:.6g} |')
lines+=['','## Instrumented timeline','',
'CUPTI concurrent kernel, memcpy and CUDA runtime activities. Only the final warm prefill and first eight decode steps are captured. Tracing changes execution cost, so these are diagnostic intervals, not replacement throughput benchmarks. Kernel/DMA union avoids double-counting overlap. Time outside observed GPU activity is not automatically CPU computation. Host synchronization intervals overlap GPU execution and must not be added to it. See [NVIDIA CUPTI activity tracing and overhead](https://docs.nvidia.com/cupti/13.0.0/main/main.html#cupti-overhead).',
'','| Capture | Wall ms | Kernel union ms | DMA union ms | GPU union ms | Host sync union ms |','|---|---:|---:|---:|---:|---:|']
for name,t in sorted(s['traces'].items()):
    lines.append('| '+name+' | '+' | '.join(f'{t[k]:.3f}' for k in ['wall_ms','kernel_union_ms','memcpy_union_ms','gpu_union_ms','host_sync_union_ms'])+' |')
trace_status=json.loads((ROOT/'performance-complete.json').read_text())['trace_returncodes']
lines+=['','Trace process return codes: `'+json.dumps(trace_status)+'`. Nonzero-return traces are excluded from timing analysis; their logs remain evidence. The first attempt using CUPTI 13.0.1 failed with SIGSEGV before a complete trace. The queued retry uses isolated CUPTI 13.1.115, without replacing any system or training libraries.','',
'## First-token numerical audit','',
'Both engines consume token ID 0 (BOS), with empty state. Dumps compare matching named FP32 intermediates. Flattening follows GGML strides. NTransformer grouped projection intermediate `attn_wo_a` is excluded because its host placeholder is not populated on the GPU path. Duplicate HC mix scratch names are not used to locate divergence.',
'','The llama evaluation callback can change graph boundaries/fusion. The following controls compare first-token final logits with the same engine without dump instrumentation; interpret intermediate differences in light of these controls.',
'','| Engine | Dump vs ordinary execution, relative L2 | Bitwise equal |','|---|---:|---|']
for engine,m in s['dump_controls'].items():lines.append(f"| {engine} | {m['relative_l2']:.6g} | {m['bitwise_equal']} |")
lines+=['','| Stage | Max absolute difference | Relative L2 (llama denominator) |','|---|---:|---:|']
for name in ['hc_init','hc_attn_pre-0','attn_norm-0','qr-0','q-0','kv-0','attn_out-0','hc_attn_post-0','ffn_norm-0','ffn_moe_out-0','ffn_shexp-0','l_last-0','l_last-10','l_last-20','l_last-39','result_output']:
    m=s['stages'].get(name)
    if m and 'relative_l2' in m:lines.append(f"| {name} | {m['max_abs']:.6g} | {m['relative_l2']:.6g} |")
lines+=['','### First projection arithmetic replay','',
'For each engine, replay its captured layer-0 attention-normalized input through the same dequantized Q8_0 query-A weights and RMS normalization. Compare FP32 input with Q8_1 input quantization (32-value blocks, max/127, signed nearest rounding, FP16 scale). Reference accumulation and RMS normalization use NumPy FP64, so this is an arithmetic diagnostic, not a bitwise CUDA emulator. Model-weight extracts remain on the remote host and are excluded from this artifact.',
'','| Replay against observed engine output | Relative L2 |','|---|---:|']
for name,m in s.get('first_projection_replay',{}).items():lines.append(f"| {name} | {m['relative_l2']:.6g} |")
lines+=['','These probes do not establish full-context numerical equivalence or downstream quality. The architectural analysis and priority order are in `docs/DEEPSEEK_PERFORMANCE_GAP.md`.','',
'## Reproducibility','',
'`environment.json`, command JSON files, admission evidence, source harnesses, logs, raw tensors, activity TSVs, and `summary.json` accompany this report. The pinned llama engine and original NTransformer library remain the ones documented in the preceding benchmark. `run.py` runs controls and traces; `run_dump.py` runs numerical probes; `analyze.py` and `report.py` regenerate this report. `sha256.json` covers the payload. No model weights are bundled.','']
(ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
