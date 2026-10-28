# DeepSeek performance-gap probes

Same model and consumed IDs as the completed llama.cpp comparison. One physical H100 (GPU 1), 52 CPU threads, 31 CPU-resident routed-expert layers in llama.cpp. All GPU probes pass the same training-lock and idle-device admission checks.

## Uninstrumented controls

One fresh process per configuration; one initial request and three warm requests. These controls isolate configuration effects, not independent process-to-process variance. Rates aggregate warm work divided by warm time. CUDA graphs and automatic operation offload are independent interventions.

| Case | Prefill tok/s | Decode tok/s | Final logit relative L2 vs control |
|---|---:|---:|---:|
| llama-control-p32 | 8.2341 | 26.3028 | 0 |
| llama-no-graphs-p32 | 8.4942 | 26.8797 | 0 |
| llama-control-p128 | 17.8727 | 29.1130 | 0 |
| llama-no-op-offload-p128 | 67.1224 | 28.8503 | 0.108825 |

## Instrumented timeline

CUPTI concurrent kernel, memcpy and CUDA runtime activities. Only the final warm prefill and first eight decode steps are captured. Tracing changes execution cost, so these are diagnostic intervals, not replacement throughput benchmarks. Kernel/DMA union avoids double-counting overlap. Time outside observed GPU activity is not automatically CPU computation. Host synchronization intervals overlap GPU execution and must not be added to it. See [NVIDIA CUPTI activity tracing and overhead](https://docs.nvidia.com/cupti/13.0.0/main/main.html#cupti-overhead).

| Capture | Wall ms | Kernel union ms | DMA union ms | GPU union ms | Host sync union ms |
|---|---:|---:|---:|---:|---:|

Trace process return codes: `{"nt-trace-p32": 139, "llama-trace-p32": 139}`. Nonzero-return traces are excluded from timing analysis; their logs remain evidence. The first attempt using CUPTI 13.0.1 failed with SIGSEGV before a complete trace. The queued retry uses isolated CUPTI 13.1.115, without replacing any system or training libraries.

## First-token numerical audit

Both engines consume token ID 0 (BOS), with empty state. Dumps compare matching named FP32 intermediates. Flattening follows GGML strides. NTransformer grouped projection intermediate `attn_wo_a` is excluded because its host placeholder is not populated on the GPU path. Duplicate HC mix scratch names are not used to locate divergence.

The llama evaluation callback can change graph boundaries/fusion. The following controls compare first-token final logits with the same engine without dump instrumentation; interpret intermediate differences in light of these controls.

| Engine | Dump vs ordinary execution, relative L2 | Bitwise equal |
|---|---:|---|
| nt | 0 | True |
| llama | 0 | True |

| Stage | Max absolute difference | Relative L2 (llama denominator) |
|---|---:|---:|
| hc_init | 0 | 0 |
| hc_attn_pre-0 | 0 | 0 |
| attn_norm-0 | 2.5332e-07 | 2.45604e-06 |
| qr-0 | 0.0149604 | 0.00336534 |
| q-0 | 0.0511516 | 0.00420576 |
| kv-0 | 0.00156248 | 0.0019976 |
| attn_out-0 | 0.020546 | 0.0086541 |
| hc_attn_post-0 | 0.0501792 | 0.055563 |
| ffn_norm-0 | 0.0185302 | 0.0309172 |
| ffn_moe_out-0 | 0.115157 | 0.0109618 |
| ffn_shexp-0 | 0.0219002 | 0.0213148 |
| l_last-0 | 2.04907 | 0.090626 |
| l_last-10 | 752.039 | 0.594081 |
| l_last-20 | 7275.76 | 0.109729 |
| l_last-39 | 61234.5 | 0.191128 |
| result_output | 2.40453 | 0.139525 |

### First projection arithmetic replay

For each engine, replay its captured layer-0 attention-normalized input through the same dequantized Q8_0 query-A weights and RMS normalization. Compare FP32 input with Q8_1 input quantization (32-value blocks, max/127, signed nearest rounding, FP16 scale). Reference accumulation and RMS normalization use NumPy FP64, so this is an arithmetic diagnostic, not a bitwise CUDA emulator. Model-weight extracts remain on the remote host and are excluded from this artifact.

| Replay against observed engine output | Relative L2 |
|---|---:|
| nt_fp32_input | 1.35831e-07 |
| nt_q8_1_input | 0.00336582 |
| llama_fp32_input | 0.00336534 |
| llama_q8_1_input | 7.00573e-08 |

These probes do not establish full-context numerical equivalence or downstream quality. The architectural analysis and priority order are in `docs/DEEPSEEK_PERFORMANCE_GAP.md`.

## Reproducibility

`environment.json`, command JSON files, admission evidence, source harnesses, logs, raw tensors, activity TSVs, and `summary.json` accompany this report. The pinned llama engine and original NTransformer library remain the ones documented in the preceding benchmark. `run.py` runs controls and traces; `run_dump.py` runs numerical probes; `analyze.py` and `report.py` regenerate this report. `sha256.json` covers the payload. No model weights are bundled.
