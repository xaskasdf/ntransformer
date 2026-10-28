import json
from pathlib import Path
import sys

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
env=json.loads((root/'environment.json').read_text())
choice=s['selection']
serial=s['serial_prefill_control']
lines=[
'# NTransformer versus llama.cpp: DeepSeek-V4.1-Flash on one H100',
'',
'2026-10-15. The exact same ten GGUF shards, prompt IDs and 64 consumed continuation IDs were evaluated on the same physical H100. Both engines run one sequence. The tables describe throughput on this fixed token workload. **Numerical equivalence between the engines is not established: their logits and some greedy predictions differ substantially. These results are not a claim of acceleration at equal output quality.**',
'',
'## Throughput',
'',
'Three repeated requests per context, after one initial pass in a fresh process. Each repeat clears attention/history state but reuses loaded weights. Rates are pooled tokens divided by pooled time. The final llama.cpp measurements filter DEBUG logs, matching normal CLI logging; the initial verbose run is retained under `initial-debug/` and excluded from this table.',
'',
'| Prompt tokens | NTransformer prefill tok/s | llama.cpp prefill tok/s | NTransformer decode tok/s | llama.cpp decode tok/s | llama.cpp / NT decode |',
'|---:|---:|---:|---:|---:|---:|'
]
for c in s['cases']:
    nt=c['ntransformer_reused'];ll=c['reused']
    lines.append(f"| {c['setup']['prompt_tokens']} | {nt['prefill']['tokens_per_second']:.3f} | {ll['prefill']['tokens_per_second']:.3f} | {nt['decode']['tokens_per_second']:.3f} | {ll['decode']['tokens_per_second']:.3f} | {c['llama_over_ntransformer_throughput']['decode']:.3f}x |")
lines += [
'',
'| Prompt | llama.cpp reused TTFT (s) | Reused prefill range (s) | Reused 64-step decode range (s) | Decode p95 (ms) |',
'|---:|---:|---:|---:|---:|'
]
for c in s['cases']:
    p=c['reused']['prefill'];d=c['reused']['decode']
    lines.append(f"| {c['setup']['prompt_tokens']} | {p['ttft_mean_seconds']:.3f} | {p['min_seconds']:.3f}-{p['max_seconds']:.3f} | {d['min_seconds']:.3f}-{d['max_seconds']:.3f} | {d['p95_token_ms']:.2f} |")
lines += [
'',
'TTFT starts after model/context initialization with already tokenized input. Each decode step includes execution, synchronization/readback and greedy selection; validation-buffer copies and file I/O are excluded, with outer wall durations recorded separately. Decode percentiles use 192 individual steps per context. Setup and first-pass times are recorded separately in the raw JSONL and `summary.json`; model-ready timing is not process-start latency.',
'',
'The NTransformer numbers are the controlled benchmark from earlier the same day, preserved under `baseline/`. This is not an interleaved A/B experiment or an exhaustive tuning search. Three repeats in one process per context do not establish a confidence interval across independent launches.',
'',
'## Same work and numerical differences',
'',
'NTransformer originally generated greedily. llama.cpp consumes that exact saved sequence (teacher forcing), while its own argmax at each position is recorded but not fed back. Thus both engines evaluate the same prompt and subsequent contexts even when their preferred token differs. Prefill supplies one prediction, followed by 64 decode forwards: 65 full output vectors per request. EOS is ignored for fixed-work timing; the original 512-token workload contains EOS after 32 non-EOS predictions. These are raw prose prefixes without chat templates, not instruction-following evaluations.',
'',
'| Prompt | Matching greedy predictions / 65 | Final-vector max absolute difference | Final-vector relative L2 | Final-vector cosine |',
'|---:|---:|---:|---:|---:|'
]
for c in s['cases']:
    e=c['final_logits_vs_ntransformer']
    lines.append(f"| {c['setup']['prompt_tokens']} | {c['greedy_agreement_by_round'][0]} | {e['max_abs']:.6f} | {e['relative_l2']:.6f} | {e['cosine']:.6f} |")
lines += [
'',
f"A control with the 128-token prompt, the same 64 continuation IDs and `n_ubatch=1` still matches only {serial['greedy_agreement_by_round'][0]}/65 greedy predictions. Its final-vector max absolute difference from NTransformer is {serial['versus_ntransformer']['max_abs']:.6f}, relative L2 {serial['versus_ntransformer']['relative_l2']:.6f}. Versus llama.cpp with native batched prefill, the control has max absolute difference {serial['versus_native_prefill']['max_abs']:.6f} and relative L2 {serial['versus_native_prefill']['relative_l2']:.6f}. The discrepancy therefore is not explained solely by the batched prefill schedule.",
'',
'Within each implementation, repeated requests reproduce their own logits bit for bit in these measurements. Disabling DEBUG logs also leaves all saved llama.cpp final vectors bitwise unchanged. The cross-engine differences are reproducible and much larger than the small intra-NTransformer arithmetic differences previously validated. Their cause remains unresolved; this benchmark does not attribute them exclusively to quantization, floating-point association, or an implementation bug. A per-layer comparison against the pinned reference is needed before claiming equivalent inference quality.',
'',
'All primary llama.cpp and NTransformer passes checked 65 x 129,280 logits each for finiteness. The saved final vectors are independently re-read and compared by `analyze.py`. Greedy agreement is checked at all 65 positions; the error norms in the table cover only the saved final vector, not all intermediate predictions.',
'',
'## Hardware, placement and execution policy',
'',
'- One H100 80GB HBM3, physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`), same CPU/host as NTransformer. GPU 0 remained unused during measurements. No P2P or second-GPU compute in either benchmark.',
'- Same model: `DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf`, about 473.069 GiB across 10 files; routed experts MXFP4, Q8_0 shared/backbone projections, BF16 head. The NTransformer baseline preserves shard sizes/mtimes; full weight payloads were not rehashed.',
f"- llama.cpp: pinned local fork `vcruz305/llama.cpp`, `runtime/deepseek41`, commit `{env['source_commit']}`. The existing reference build was CPU-only; this experiment built an isolated CUDA version from its clean Git archive. Results apply to this fork/revision and settings, not every llama.cpp version.",
'- GCC 14.4.0, CUDA toolkit 13.1.115, native CPU instructions, CUDA architecture 90. CUDA graphs, MMQ/cuBLAS dispatch and flash-attention auto-selection retain their defaults. The model requests F32 K/V cache, matching the NTransformer state dtype; this does not make every intermediate arithmetic operation identical.',
f"- llama.cpp uses {choice['threads']} CPU threads, selected from 26 and 52 on the same short 32+8-token pilot. Warm 8-step decode: 26 threads {float(choice['pilot_decode_seconds']['26']):.6f} s; 52 threads {float(choice['pilot_decode_seconds']['52']):.6f} s. Pilots used the original DEBUG logger; all main reported measurements use INFO filtering. This limited pilot does not prove a globally optimal configuration.",
'- All layers are GPU-eligible, with explicit CPU buffer overrides for routed experts in layers 0-30. The remaining nine layers keep their routed weights on GPU. Shared experts and backbone projections are GPU-resident; large Engram tables use the fork\'s lazy row-loading path. mmap is enabled.',
'- These are weight-placement settings, not a claim that every CPU-resident operation executes on CPU. Standard operation offload is enabled; the CUDA backend uses a batch-size heuristic. Native prompt processing can therefore schedule CPU-resident operations differently from single-token decode. Actual placement and buffer sizes are preserved in each log.',
'- llama.cpp: `n_ctx=1024`, `n_batch=512`, `n_ubatch=512`, one sequence. NTransformer processes the prompt sequentially through token GEMVs. Allowing each engine its own prefill implementation preserves the same requested work without forcing identical scheduling.',
'- NTransformer: the previously validated shared-expert build, 62 GiB dynamic expert cache, approximately 70.48 GiB maximum observed VRAM. Its weights move on cache misses; llama.cpp uses the placement described above. GPU memory usage is measured, not artificially forced to the same allocation size.',
'',
'| GPU | Maximum sampled memory (MiB) | Mean sampled utilization | Mean sampled power (W) |',
'|---:|---:|---:|---:|'
]
for index,g in s['telemetry'].items():
    lines.append(f"| {index} | {g['max_memory_mib']:.0f} | {g['mean_utilization_pct']:.1f}% | {g['mean_power_w']:.1f} |")
lines += ['', '| Prompt | llama.cpp peak process RSS (GiB) | Model load (s) | Context init (s) |',
          '|---:|---:|---:|---:|']
for c in s['cases']:
    lines.append(f"| {c['setup']['prompt_tokens']} | {c['max_rss_kib']/1024**2:.2f} | {c['setup']['model_load_seconds']:.3f} | {c['setup']['context_init_seconds']:.3f} |")
lines += [
'',
f"Telemetry covers the final suite including setup, gaps and the serial-prefill control. Samples are observed values, not guaranteed instantaneous peaks. {s['compute_process_observations']} compute-process observations found no foreign GPU process. Shared admission locks were held, active Brandon stages had finished, and no training process was stopped. Host page cache was not flushed. RSS includes mapped file-backed pages and is not the whole machine's RAM use.",
'',
'## Reproduction and artifacts',
'',
'`tools/benchmark_llama_ds41.cpp` calls the pinned library API directly, just as the NTransformer helper calls its Session API. It preserves prompt/continuation IDs, performs greedy selection for comparison, and synchronizes outputs before recording timings. The standalone `llama-bench` was also built, but its usual synthetic tokens do not reproduce this workload; the reported measurements use the API helper.',
'',
'`prepare.py` records the CUDA build. cuBLAS comes from the installed NVIDIA CUDA 13 Python package; only its cuBLAS headers are exposed to the build so its CUDA 13.0 runtime headers do not shadow the 13.1 compiler toolkit. The runtime library and source archive are fingerprinted. The CUDA driver stub is used only at link time; runtime resolves the real driver.',
'',
'`run.py` records admission, pilots and the initial series; `final_validation.py` preserves that initial series, filters DEBUG logging, adds the serial-prefill control and reruns the primary cases. `compile-final-command.json` records the final link command. Representative invocation with the recorded `LD_LIBRARY_PATH`:',
'',
'```bash',
'CUDA_VISIBLE_DEVICES=1 ./benchmark_llama_ds41 \\',
'  /path/to/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf \\',
'  workload-p128.json p128-d64 3 52 31 512',
'```',
'',
f"Final helper SHA-256: `{env['binary_sha256']}`.",
f"Final helper source SHA-256: `{env['source_sha256']}`.",
'',
'Local results: `docs/benchmarks/h100-deepseek-2026-10-15/llama-comparison/`.',
'Remote work: `/home/ubuntu/ntransformer-h100/llama-bench-20261015`.',
'Persistent copy: `/lambda/nfs/btiny/ntransformer-training/deepseek-llama-comparison-20261015`.',
'',
'Raw per-step timing, workload IDs, final logits for each round, initial DEBUG measurements, serial control, telemetry, resource logs, build settings, source snapshot and SHA-256 manifest accompany the report. No full model weights are bundled.',
]
assert all(c['repeat_bitwise_equal'] and c['repeat_top1_equal'] for c in s['cases'])
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('report written')
