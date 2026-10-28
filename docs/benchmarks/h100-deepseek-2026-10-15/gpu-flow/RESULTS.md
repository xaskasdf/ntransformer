# DeepSeek scheduling and MXFP4 lookup experiment

One physical H100 (GPU 1), the same 10 GGUF shards, corpus, context and 62 GiB expert-cache budget. One initial request and three warm repeats per process unless stated by the per-case JSON. Rates divide total warm work by total warm time. No CUPTI tracing in timed runs.

| Case | Prefill tok/s | Decode tok/s | Decode ratio | All 65 logit vectors match control |
|---|---:|---:|---:|---|
| candidate-p32 | 6.0313 | 6.2047 | 0.9954 | True |
| control-p32 | 6.0712 | 6.2334 | 1.0000 | True |
| lut-only-p32 | 6.0700 | 6.2302 | 0.9995 | True |

The four independently selectable flags are `DS41_STAGING_POOL`, `DS41_SHARED_EARLY`, `DS41_QKV_READBACK_BATCH`, and `DS41_MXFP4_REGISTER_LUT`. Exact per-process settings and durations are recorded in command JSON files.

| Case | QKV readback waits/token | Transient copy threads/token | Staged slices/token | Weight upload GiB/token |
|---|---:|---:|---:|---:|
| candidate-p32 | 40 | 0.000 | 86.906 | 0.507227 |
| control-p32 | 120 | 86.906 | 86.906 | 0.507227 |
| lut-only-p32 | 120 | 86.906 | 86.906 | 0.507227 |

Finite logits and within-process repeat identity are checked by the harness. Full output hashes compare the initial prefill prediction and all 64 subsequent predictions; identical hashes also require identical consumed continuations. This is a regression control for the tested inputs, not a general model-quality evaluation.

Synthetic tests cover small, model-sized and maximum routed shapes, all MXFP4 codes, cold/warm caches, duplicate experts, negative weights, callback exceptions and recovery. QKV and shared-expert tests include independent CPU references. `test-comparison.json` records cross-configuration output equality.

These changes retain FP32 activations, weight formats, reduction order and model architecture. They remove host scheduling overhead and a divergent constant-table lookup. They do not implement device-resident attention, peer experts or batched prefill.

Kernel lookup motivation: distinct constant-memory addresses within a warp are serviced separately ([NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#constant-memory)). The magnitude table fits into a 32-bit immediate; nibble extraction and sign application reproduce its exact float values. Measured ablations determine the effect in this engine.

Per-token phase values in `summary.json` are host wall intervals, including waits. They are not device kernel times. Small differences between separate processes should be treated cautiously; repeated warm requests do not establish independent process-level variance. Raw command records, telemetry, logs, output tensors, source snapshots and hashes accompany this report. No model weights are bundled.
