# Selective QKV readback — two H100s

DeepSeek-V4.1-Flash on the corrected hyper-connection graph. The same executable runs both modes: `DS41_QKV_SELECTIVE_READBACK=0` restores the former host mirrors; the default (`1`) returns only the query latent needed by indexer-source layers. Q and the sliding-window KV ring remain on device. No weight format or precision changes.

## Controlled performance

Two independent launches per mode and paired workload, with reversed mode order. Each process runs two warmups and three measured requests. The same corpus tokens, 64 teacher-forced continuation tokens, 62 GiB expert cache per GPU, SLRU 80%, FP32 routed compute, cooperative Q8 backbone and P2P owner execution are preserved. Builds and initial correctness tests finished before the performance cases. A strengthened isolation test was compiled and rerun only after all measurements completed.

| Workload | Phase | Baseline tok/s | Selective tok/s | Throughput gain | Gain by launch |
|---|---|---:|---:|---:|---|
| p32 | prefill | 25.2516 | 25.5638 | +1.24% | +0.87%, +1.57% |
| p32 | decode | 23.5669 | 23.8196 | +1.07% | +0.69%, +1.39% |
| p512 | prefill | 20.1740 | 20.3772 | +1.01% | +0.83%, +1.24% |
| p512 | decode | 18.1769 | 18.3122 | +0.74% | +0.68%, +0.86% |

Throughput uses tokens divided by median warm phase time; each paired cell has six measurements. The repeated requests within a process are not independent hardware runs. Whole-work time sums prefill and decode compute intervals; startup, logging and validation are outside these intervals. Initial requests remain diagnostic because the OS file cache was not reset.

| Workload | Baseline warm work s | Selective warm work s | Whole-work throughput gain |
|---|---:|---:|---:|
| p32 | 3.983490 | 3.938351 | +1.15% |
| p512 | 28.899612 | 28.621050 | +0.97% |

## Removed dependencies

| p512 prefill counter | Baseline | Selective |
|---|---:|---:|
| prologue_calls | 20,480 | 20,480 |
| prologue_readback_waits | 20,480 | 4,093 |
| prologue_readback_bytes | 146,800,640 | 20,956,160 |
| prologue_deferred_calls | 0 | 16,387 |
| prologue_staging_waits | 0 | 0 |
| raw_recovery_calls | 0 | 0 |

An asynchronous prologue owns a separate pinned input buffer and a CUDA event protecting its reuse. The buffer holds input plus RoPE tables (20,736 bytes for this model). No new device tensor is allocated by this change. The normal tail completes the stream before the next layer. If GPU attention fails, the host recovers Q and all valid raw ring slots, including history omitted by prior calls. Sequence checks prevent stale device history from overwriting newer CPU state.

QKV host intervals now stop earlier, so some pending GPU work is charged to later compressor or attention waits. Do not interpret the QKV timer reduction as faster projection kernels or add nested timers to parent totals. The global throughput change, rather than the wait-count reduction, measures the speed benefit.

| p512 prefill host interval | Baseline s | Selective s |
|---|---:|---:|
| attention | 9.973801 | 9.682382 |
| qkv | 2.371965 | 1.055607 |
| compression | 1.772397 | 1.757020 |
| resident_attention | 5.801452 | 6.841791 |

`qkv`, `compression` and `resident_attention` are nested inside `attention`; their intervals include CPU work, stream submissions and waits, and are not isolated GPU kernel times.

## Correctness and scope

All 33,612,800 unique logits across 260 prediction positions match the saved fixed-HC reference bitwise. Every process also matches across its five requests. The two additional quality cases (p128 and heldout-p512) have one candidate launch each; their timings are diagnostics, not paired speedups.

Four prologue test configurations cover legacy/cooperative Q8 kernels and batched/serial readback. They check exact selective Q/QR/KV, caller-input mutation, global scratch reuse, deferred submissions, window wrap, isolated states/layers, failed-tail recovery, reset/destruction and stale-read rejection. Attention, tail, compressor, indexer and HC tests also pass. The final isolation test uses distinct inputs for each layer and session; GPU-disabled behavior is checked separately. Test assertions are enabled. Compute Sanitizer was unavailable in the configured toolchain, so this run does not claim a CUDA memory-sanitizer pass.

GPU monitoring recorded 311 observations without foreign compute processes. Peak sampled VRAM by physical GPU (MiB): {'0': 64017.0, '1': 72511.0}. Monitoring samples are not allocation peaks and do not detect unrelated CPU-only work. Warm page faults and I/O are recorded in summary.json.

The arithmetic and validated outputs are unchanged; these corpus checks do not extend task-quality coverage. Prefill remains token-serial, and HC, router, compression/indexer and attention-output boundaries still involve the CPU. The next substantial prefill target is a causal layer-wise token-tile path with batched projections and expert grouping. The smaller follow-up is to avoid allocating/zeroing host Q/KV buffers that the resident path never consumes.

## Reproduction

The bundle retains per-round measurements, full logits, validation hashes, command environments, source snapshots, test logs, telemetry and this analyzer. The library differs from the fixed-HC library only in ds41_gpu.cu.o and ds41_forward.cpp.o; CUDA compiler flags and architecture targets are preserved. The failed first harness compile (missing benchmark_resources.h) and successful retry are both recorded. It occurred before testing and measurement.
