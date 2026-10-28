# DS41: remove pool dispatch from small hyper-connection operations

2026-10-15. DeepSeek-V4.1-Flash Q8_0/MXFP4, physical H100 GPU1,
GCC 14.4.0, CUDA 13.1.115, Xeon Platinum 8480+ with 52 logical CPUs.
Adaptive CPU chunks, the final-position prefill head and shared/routed MoE
overlap are retained in both comparison arms.

## Dependency analysis and implementation

Each layer collapses four hyper-connection streams before attention and
before the FFN. Each branch also combines its result with the residual
streams. For hidden coordinate `d`, the operations are:

```text
collapse[d] = sum_h stream[h,d] * pre[h]
post_mix[dst,d] = branch[d] * post[dst]
                 + sum_src residual[src,d] * comb[dst + src*hc]
```

At H=5,120 and hc=4, a collapse performs four products per coordinate; a
post-mix performs twenty. Previously each operation dispatched a job to the
52-participant pool, claimed row chunks through a shared atomic counter and
waited for all 51 workers. The preceding chunk optimization reduced atomic
traffic but retained that wake-up and completion round for each small job.

The two helpers in `ds41_hc.h` now execute directly on the calling thread by
default. `DS41_HC_DIRECT=0` restores pool execution for controlled comparisons.
Both modes use the same scalar body and preserve the accumulation order for
each output. Attention, HC projection matvecs, expert matvecs, router, GPU
kernels and the general thread pool retain their existing execution policies.
Output buffers must be distinct from inputs, as they already are in Session.

There are four removed pool dispatches per layer and an additional terminal
collapse whenever logits are requested. With 40 layers this removes **2,561
dispatches per 16-token prompt** with only its final head, or **644 dispatches
per four-step decode window**. These are dispatch counts from the code, not
measured OS context switches. The arithmetic and model state updates remain.

## Validation

`test_ds41_hc` compares direct execution, pool execution and an independent
scalar reference bitwise, checking guard elements before and after outputs.
It covers hidden sizes 0, 1, 31, 65, 256, 5,120 and 8,193; stream counts 1, 2,
4 and 8; ordinary values, mixed magnitudes and signed zeros. All 84 cases
pass in Release, using explicit failures rather than disabled assertions.

Five existing thread-pool tests also pass. The tiny-model prefill/state test
passes with CPU fallback, including sliding-window rollover, compression,
mixed head schedules, four-token continuation, reset and a single-token case.
Against the preceding binary, the tiny model's final prompt logits match
bitwise and all 147 printed stage sums agree. The final-logit SHA-256 is
`f4b7c281576b067260339fc9f37c8a1dc202a633d84967dba92e912af4b15f73`.

The complete-model benchmark acquired all existing pipeline locks and
required idle GPUs. Brandon's `mapping-mixture-v1` began during compilation;
its training and evaluation completed before exclusive admission. The first
bounded admission wait expired; `run_gpu.py` resumed the prepared experiment.
No Brandon process was stopped.

Full-model final prompt logits and all four continuation vectors are bitwise
identical across the four timing processes and the preceding, unmodified
MoE-overlap binary. Each comparison covers **646,400 finite FP32 values**.
Repeated windows within each process also match bitwise. Their concatenated
SHA-256 remains
`8c8ea79cda4da03fe1d1bbd8774b17eaf71b1a7602bc0a8952d061a9a37cd470`.
The separate full-model state test passes final-only and mixed head schedules,
continuation, reset and the single-token case.

## Full-model results

Six warm windows per arm from two processes, with the same candidate binary:

| Workload | Pool | Direct | Throughput increase |
| --- | ---: | ---: | ---: |
| 16-token prefill | 2.78437 s | 2.23845 s | **24.39%** |
| Prefill tokens/s | 5.74636 | 7.14781 | |
| Four autoregressive decode steps | 0.70263 s | 0.56803 s | **23.70%** |
| Decode tokens/s | 5.69290 | 7.04194 | |

Latency falls 19.61% for prefill and 19.16% for decode. Prefill saves 545.93 ms
per prompt. Prefill ranges are 2.77240–2.83096 s with the pool and
2.23110–2.24124 s direct; decode ranges are 0.69921–0.70521 s and
0.56625–0.57050 s respectively. These are short-context, one-GPU warm-cache
results, not training, cold streaming, long-context scaling or two-GPU scaling.

Every measured prefill has 640 routed-GPU calls and every decode window has
160, all with shared-CPU overlap and zero CPU routed fallbacks. Expert-cache
deltas show **100% hits and zero evictions** in every measured window: 11,520
slice lookups per prompt and 2,880 per decode window. Both processes in each
arm first warm the same 6,393 distinct slices. Aggregate process hit rates
include these initial misses and are lower than the measured-window rates.

Mean warm prefill phase times:

| Phase | Pool, s | Direct, s |
| --- | ---: | ---: |
| Initial attention HC/norm | 0.33198 | 0.20967 |
| Other, including residuals, FFN HC/norm and terminal head | 0.75106 | 0.30236 |
| Complete MoE | 0.95907 | 0.97972 |
| Attention | 0.69867 | 0.70176 |
| Engram | 0.04359 | 0.04494 |

The savings occur in the expected HC/residual categories. Other phases do
not all improve: router time within MoE rises from 0.17942 to 0.20629 s.
The measurements do not establish the cause of that small regression. The
net end-to-end saving includes it. After this change MoE plus attention
account for about 75% of prompt wall time, directing further profiling there.

No GPU allocation code changes: the same 10,622 expert-cache slots (62 GiB
budget), resident weights and scratch are used. Peak VRAM was not newly
sampled. This change adds no temporary arrays and reuses the existing Session
inputs/outputs. Host profiling counters are cumulative; the harness uses
window deltas. The `hc` category covers only the initial attention-side HC
and norm, with other HC operations in `other`.

## Measurement protocol

The fixed prompt is BOS followed by `3 + (i * 37) % (vocab_size - 3)` for
positions 1 through 15, followed by four greedy decode steps. Each process
warms this path once and measures three repeats, resetting model state before
each timed prefill. Process order is pool/direct/direct/pool, with the same
candidate binary and `DS41_HC_DIRECT` set to 0 or 1. This gives six warm
windows per arm from two processes, not six independent workloads.

Both arms use a 62 GiB expert cache, GPU attention, GPU LM head and the
previous MoE overlap. The unmodified preceding binary supplies a separate
correctness comparison. Per-window cache and GPU-call deltas are checked;
failed GPU execution or unequal logits invalidate a run.

The CPU microbenchmark uses H=5,120, hc=4, preallocated arrays and 400
repetitions per operation/window, in pool/direct/direct/pool order. It does
not represent full inference throughput.

| Operation | Pool, mean microseconds/call | Direct, mean microseconds/call |
| --- | ---: | ---: |
| Collapse four streams | 224.31 | 10.04 |
| Residual post-mix | 234.46 | 52.92 |

These local-operation reductions are about 22.34x and 4.43x respectively.
Each mean combines two windows of 400 calls. A compiler barrier prevents
elimination of repeated stores; it emits no hardware memory fence. Arrays
are warm and this isolated cache state differs from the full-model pipeline.

```bash
ctest --test-dir build -V -R 'test_ds41_parallel|test_ds41_hc'
DS41_GPU=0 DS41_TEST_MODEL=/home/ubuntu/tiny-deepseek41.gguf \
  ctest --test-dir build -V -R '^test_ds41_prefill$'
./build/test_ds41_hc --benchmark
CUDA_VISIBLE_DEVICES=1 DS41_GPU=1 DS41_GPU_ATTN=1 DS41_VRAM_GB=62 \
  DS41_PROFILE=1 DS41_HC_DIRECT=1 ./profile-candidate MODEL.gguf 16 3 output.bin
```

## Artifacts

Sources and experiment helpers are isolated at
`/home/ubuntu/ntransformer-h100/ds41-hc-20261015`.
The benchmark bundle records raw logs, outputs, command environments, source
snapshot, original forward sources and hashes. Model assets remain external;
the model layout is the prior ten-shard GGUF header inventory, not a fresh
full-payload hash.

Local copy: this directory. Persistent Lambda copy:
`/lambda/nfs/btiny/ntransformer-training/deepseek-hc-20261015`.
`archive.json` records the compressed bundle's SHA-256; `SHA256SUMS` records
each payload file. Local verification recomputes the means, checks cache/GPU
coverage, compares binary logits and matches current source hashes.
