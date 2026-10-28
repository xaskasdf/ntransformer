# DeepSeek resident attention experiment

One physical H100 (GPU 1), the same ten GGUF shards, FP32 activations and a 62 GiB expert cache. Each process runs one initial request and three warm repeats, with 64 decode steps. EOS is recorded but does not stop this fixed-work benchmark. Rates divide total warm work by total warm time. Initial uploads, validation and output files are outside token timings. No CUPTI tracing.

| Case | Prefill tok/s | Decode tok/s | Decode ratio | Matched argmax / aligned vectors | Relative L2 |
|---|---:|---:|---:|---:|---:|
| control-p128 | 7.6314 | 7.6443 | 1.0000 | 65 / 65 | 0 |
| control-p32 | 7.4929 | 7.7385 | 1.0000 | 65 / 65 | 0 |
| control-p512 | 7.4491 | 6.4298 | 1.0000 | 65 / 65 | 0 |
| defaults-p32 | 8.2718 | 8.6048 | 1.1120 | 65 / 65 | 5.10097613e-07 |
| resident-p128 | 8.4863 | 8.6118 | 1.1266 | 65 / 65 | 5.58609431e-07 |
| resident-p32 | 8.2559 | 8.5951 | 1.1107 | 65 / 65 | 5.10097613e-07 |
| resident-p512 | 8.4130 | 7.3845 | 1.1485 | 65 / 65 | 6.98598934e-07 |

| Case | Attention total ms/token | Prologue readback + tail input MiB/token | New compressed + metadata KiB/token | Resident calls/token | Sampled peak GPU GiB |
|---|---:|---:|---:|---:|---:|
| control-p128 | 50.0811 | 10.273438 | 0.0000 | 0 | 70.7919921875 |
| control-p32 | 45.1359 | 10.273438 | 0.0000 | 0 | 70.7919921875 |
| control-p512 | 61.5474 | 10.273438 | 0.0000 | 0 | 70.7919921875 |
| defaults-p32 | 33.0972 | 0.273438 | 22.2891 | 40 | 70.8037109375 |
| resident-p128 | 36.7638 | 0.273438 | 33.1641 | 40 | 70.8037109375 |
| resident-p32 | 33.1207 | 0.273438 | 22.2891 | 40 | 70.8037109375 |
| resident-p512 | 43.2526 | 0.273438 | 76.0293 | 40 | 70.8095703125 |

## Dataflow

The session owns a device raw KV ring (10 MiB for 40 layers, window 128, dimension 512), independent query/output scratch, RoPE tables and growing compressed-row buffers. Prologue writes Q directly into that session and appends its new KV row with a device copy. It still reads back the 1,280-value query latent for CPU indexing and a 512-value KV mirror for CPU fallback. It removes the 32,768-value Q readback and the equally sized attention-output upload at every layer: 10 MiB/token of logical transfer payload. Host prologue/tail waits remain; removing these also requires moving compression/indexing and residual state.

Compressed rows retain their CPU computation and selection. Each owner uploads its newly visible immutable prefix; capacity growth recopies that prefix and reset invalidates it. Selection order, repeated keys and layer sharing are preserved. Copies use separate offsets in a session-owned pinned staging allocation, which remains alive through the tail completion wait.

The score kernel cooperatively stages sixteen keys into shared memory and computes ordered FP32 products/sums. The second kernel includes attention sinks, computes exponentials in double then rounds them to FP32, accumulates the denominator and values in key order, and applies inverse RoPE. The output projection consumes its device result directly. Tensor precision and model architecture are unchanged; library/FMA rounding can differ from CPU, so output differences are measured rather than assumed harmless.

## Validation and interpretation

Tests cover the actual 64-head, 512-dimensional shape, raw ring wrap, changing queries, sinks, compressed-prefix growth and sharing, selected order/duplicates, two independent owners, reset, query recovery and invalid bounds. Prologue tests compare resident Q and host mirrors against ordinary prologue bitwise, and count avoided readback bytes. Twenty registered tests include these cases and prior HC/router/shared/backbone/MoE/prefill checks.

The benchmark checks finite values and full-logit repeat identity. Numerical comparison uses only vectors whose input contexts are aligned; it reports the aligned count explicitly if greedy paths diverge. KL, argmax agreement and logit norms are regression diagnostics, not an independent language-quality benchmark. Identical logits are useful but not mandatory.

The existing CPU attention_loop/tail intervals describe the control. The resident path reports a combined resident_attention host interval that includes GPU attention, inverse RoPE, output projection and final wait; it is nested inside attention total. Do not compare that interval with the CPU loop alone or add nested phases together. Transfer counters are successful logical payloads, not PCIe transactions. Memory peaks are one-second samples, not allocator peaks. Warm repeats do not establish independent process-level variance.

Resident attention is enabled by default; `DS41_ATTENTION_GPU=0` selects the CPU attention control. Detailed stage dumping retains the CPU attention path. Allocation/shape limits retain ordinary prologue or recover Q for CPU attention; failure to recover an in-flight query fails loudly. Reset clears visibility metadata without exposing stale rows. The existing single-stream engine remains non-reentrant.

The six control/resident cases use initial-source and initial-environment.json. The final source additionally enables the default, invalidates stale query metadata before replacement, hardens probe failure recovery, and extends tests to zero RoPE and fallback. It preserves the successful-path arithmetic. defaults-p32 and ctest-final validate that final build with the attention flag unset. Both source snapshots and their hashes are retained.

Raw logs, outputs, telemetry, source/build hashes and this analysis accompany the report. Model weights remain external.
