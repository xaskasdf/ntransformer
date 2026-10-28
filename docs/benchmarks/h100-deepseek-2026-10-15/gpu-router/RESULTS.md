# DeepSeek resident router and scheduling experiment

One physical H100 (GPU 1), the same 10 GGUF shards, corpus, context and 62 GiB expert-cache budget. One initial request and three warm repeats per process unless stated by the per-case JSON. Rates divide total warm work by total warm time. No CUPTI tracing in timed runs.

| Case | Prefill tok/s | Decode tok/s | Decode ratio | All 65 logit vectors match control |
|---|---:|---:|---:|---|
| control-p128 | 6.1451 | 6.1817 | 1.0000 | True |
| control-p32 | 6.0824 | 6.2385 | 1.0000 | True |
| defaults-p128 | 6.6619 | 6.6836 | 1.0812 | True |
| defaults-p32 | 6.5517 | 6.7163 | 1.0766 | True |
| router-p32 | 6.4537 | 6.6501 | 1.0660 | True |
| router-schedule-p32 | 6.5356 | 6.7175 | 1.0768 | True |

The four independently selectable flags are `DS41_STAGING_POOL`, `DS41_SHARED_EARLY`, `DS41_QKV_READBACK_BATCH`, and `DS41_MXFP4_REGISTER_LUT`. Exact per-process settings and durations are recorded in command JSON files.

| Case | QKV readback waits/token | Transient copy threads/token | Staged slices/token | Weight upload GiB/token |
|---|---:|---:|---:|---:|
| control-p128 | 120 | 71.906 | 71.906 | 0.419680 |
| control-p32 | 120 | 86.906 | 86.906 | 0.507227 |
| defaults-p128 | 40 | 71.906 | 71.906 | 0.419680 |
| defaults-p32 | 40 | 86.906 | 86.906 | 0.507227 |
| router-p32 | 120 | 86.906 | 86.906 | 0.507227 |
| router-schedule-p32 | 40 | 86.906 | 86.906 | 0.507227 |

## Resident router

The router matrix projection moves from CPU to GPU with resident BF16/F32 weights, FP32 inputs and FP32 accumulation. Eight-lane CUDA reductions reproduce the CPU AVX2 two-accumulator order; a scalar-order path is also available. Host softplus, expert selection and normalization remain unchanged. No new activation quantization is introduced.

| Case | Router ms/token | GPU router calls/token |
|---|---:|---:|
| control-p128 | 13.4854 | 0 |
| control-p32 | 13.2853 | 0 |
| defaults-p128 | 3.7104 | 40 |
| defaults-p32 | 3.6690 | 40 |
| router-p32 | 3.6892 | 40 |
| router-schedule-p32 | 3.6770 | 40 |

Final defaults: `DS41_ROUTER_GPU=1`, `DS41_SHARED_EARLY=1`, `DS41_QKV_READBACK_BATCH=1`. `DS41_STAGING_POOL` and `DS41_MXFP4_REGISTER_LUT` default to 0 after their ablations showed no throughput gain. Setting `DS41_ROUTER_GPU=0` restores the CPU projection. Unsupported types or unavailable device allocations fall back to CPU.

The 40 router matrices are BF16, totaling 157,286,400 bytes (150 MiB). This is additional resident VRAM while the routed-expert cache stays at 62 GiB. The new projection adds one upload of 5,120 FP32 values and one readback of 384 FP32 values per layer, plus one host completion wait; it removes the CPU matrix work and its worker-pool dispatch. Eliminating those intermediate host boundaries requires moving expert selection and adjacent state onto GPU too.

The initial source/binary hashes are preserved in `initial-environment.json` and `initial-source/`. `defaults-*` and `control-p128` use the final source/defaults recorded by `environment.json`. Other timed cases use the initial binary with explicit switches; its computation is unchanged by final default selection.

Router tests compare 48 F32/BF16 projections bitwise with the CPU AVX2 and independent scalar-order references, including non-aligned dimensions, changing inputs, growth and reuse of buffers, bounds and disabled GPU behavior. Final CTest also exercises prefill state using the external tiny F32 fixture. Earlier scheduling tests at real/max shapes are archived in the sibling `gpu-flow` artifact.

Finite logits and within-process repeat identity are checked by the harness. Full output hashes compare the initial prefill prediction and all 64 subsequent predictions; identical hashes also require identical consumed continuations. This is a regression control for the tested inputs, not a general model-quality evaluation.

Scheduling tests cover all MXFP4 codes, cold/warm caches, duplicate experts, negative weights, callback exceptions and recovery. QKV and shared-expert tests include independent CPU references. The initial `gpu-flow/test-comparison.json` also records small, model-sized and maximum-shape cross-configuration output equality.

These changes retain FP32 activations, weight formats, reduction order and model architecture. Experimental controls examine host scheduling and a divergent constant-table lookup. They do not implement device-resident attention, peer experts or batched prefill.

Kernel lookup motivation: distinct constant-memory addresses within a warp are serviced separately ([NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#constant-memory)). The magnitude table fits into a 32-bit immediate; nibble extraction and sign application reproduce its exact float values. Measured ablations determine the effect in this engine.

Per-token phase values in `summary.json` are host wall intervals, including waits. They are not device kernel times. Small differences between separate processes should be treated cautiously; repeated warm requests do not establish independent process-level variance. Raw command records, telemetry, logs, output tensors, source snapshots and hashes accompany this report. No model weights are bundled.
