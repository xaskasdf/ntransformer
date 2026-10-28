# H100 DeepSeek: batched routed experts and MoE tracing

Date: 2026-10-15. This experiment continues the grouped attention-output
baseline using the same DeepSeek-V4.1-Flash Q8_0/MXFP4 shards and token IDs.

## Change

Submit independent routed experts in four kernels: all gate/up GEMVs, all
clamped activations, all down GEMVs, and one ordered weighted sum. Each row's
MXFP4 dot-product operation order is unchanged. The final sum visits selected
experts in the original order, starting at FP32 +0. Intermediate values remain
FP32. There is no extra quantization or parallel tree reduction. The batched
sum overwrites every output, so it also avoids the old output memset.

The six-expert case goes from **30 to 4 kernels per layer**, an 86.67% reduction
in these launches, plus removal of the separate memset. Across 40 layers that
removes 16,640 kernel launches per 16-token prompt and 4,160 per four-token
decode window. `DS41_MOE_BATCH=0` restores the serial schedule; default is 1.
The flag is read once per process. `gpu_moe_stats()` reports successful calls,
batched calls, kernel launches and uploaded weight bytes.

Eight-expert scratch adds **573,440 bytes (560 KiB)** of GPU allocation. The
62 GiB expert-cache budget and 10,622 slots are unchanged. The descriptor
travels in launch arguments, without an explicit device metadata buffer or
copy. CPU routing, cache reservation/staging, the shared-expert CPU callback
and the final routed-plus-shared addition retain their ordering.

## Controlled comparison

| Metric | Serial experts | Batched experts | Change |
|---|---:|---:|---:|
| 16-token prefill, mean | 2.227457 s | 2.129959 s | -4.38% latency |
| Prefill throughput | 7.18308 tokens/s | 7.51188 tokens/s | **+4.58%** |
| Four-step decode, mean | 0.565150 s | 0.542721 s | -3.97% latency |
| Decode throughput | 7.07776 tokens/s | 7.37027 tokens/s | **+4.13%** |
| MoE phase during prefill | 0.980084 s | 0.875334 s | -10.69% |
| MoE phase during decode | 0.245861 s | 0.220258 s | -10.41% |

Prefill windows span 2.224063-2.232517 s for serial and 2.117298-2.141242 s
for batched. Decode spans 0.563182-0.569184 s and 0.540332-0.546151 s,
respectively. The observed ranges do not overlap. This is evidence for this
controlled workload, not a statistical generalization across prompts.

Each arm uses two independent processes, with three measured warm repetitions
per process after one full warmup: six windows per arm. Process order is
serial/batched/batched/serial. The repetitions use the same workload, not six
independent prompts. Each prefill consumes 16 fixed token IDs with the head
only on the last token, then four greedy decode steps. Sessions reset between
repetitions; resident weights and expert cache remain warm. All preceding
optimizations remain enabled in both arms.

Only physical GPU1 is used for inference, an H100 80 GB SXM5 on the existing
two-H100 host. Both GPUs are required idle for admission, together with the
shared pipeline/experiment locks and complete Brandon stage statuses. Brandon's
concurrent `mapping-sampling-v1` run was left untouched; full-model timing
waited for it to finish. Small synthetic correctness tests ran concurrently
and their durations are not performance evidence.

Every measured window must show 100% expert-cache hits, no evictions, no
weight uploads, no routed CPU fallback, and the expected kernel counts. The
performance comparison disables `DS41_MOE_TRACE`; existing phase profiling
is enabled identically in both arms. This short warm-cache workload does not
establish throughput for cold cache, cache churn, long context, or two-GPU
expert execution. The extra scratch figure is an allocation delta, not a
measurement of peak process VRAM or total device memory.

## Diagnostic trace

Separate diagnostic processes use two warm repetitions per arm, following
one first pass with an initially empty expert cache. Warm prefill means:

| Diagnostic interval, per 16-token prompt | Serial | Batched |
|---|---:|---:|
| Host input enqueue | 10.18 ms | 5.95 ms |
| Host cache lookup | 4.35 ms | 4.41 ms |
| Host kernel/readback enqueue | 56.36 ms | 12.16 ms |
| Host shared-expert callback | 687.99 ms | 638.74 ms |
| Host final completion wait | 20.40 ms | 2.58 ms |
| Stream compute/launch span | 752.41 ms | 576.96 ms |
| Stream readback span | 4.73 ms | 4.58 ms |

Kernel/readback submission falls 78.43%, the final wait 87.35%, and the
compute/launch stream span 23.32%. The shared CPU callback is unchanged in
code but also runs faster under the new schedule. The untraced comparison
shows the same direction (685.24 -> 642.14 ms); the cause of that secondary
CPU timing change is not isolated. The total speedup must not be attributed
exclusively to fewer launches.

With batching, the shared CPU callback exceeds the routed GPU span. That
identifies the shared-expert boundary as the next warm-path target; moving
that work would need its own arithmetic-equivalence and overlap evaluation.

The first diagnostic prompt uploads **35,570,810,880 bytes** in both arms:
5,676 expert slices. Serial/batched host staging is 0.81165/0.80693 s and
the upload stream span is 0.67405/0.67418 s. The following four-token first
decode uploads another 4,493,352,960 bytes (717 slices). Warm repetitions
upload zero weight bytes. Thus batching improves dispatch/compute but leaves
the miss pipeline in place. Those single first-pass diagnostics do not
establish cold-cache speedup or storage bandwidth; host page-cache state was
not flushed. Persistent staging, upload/compute overlap and peer-owned
experts remain distinct opportunities.

`DS41_MOE_TRACE=1` records eight host sections and five elapsed CUDA stream
intervals using six reusable events. There is no new intermediate GPU wait.
Durations are collected only after the existing final stream synchronization;
aborted/failed traces increment a separate counter and contribute no partial
phase totals. Tracing is disabled by default. Initialization, cache allocation
and pinned readback reservation happen before its first marker.

Host sections: input enqueue, cache lookup, miss staging (including thread
creation/join), weight-copy enqueue, kernel/readback enqueue, CPU callback,
final completion wait, and output publication. Stream intervals: input and
serial-path zeroing, preparation gap, weight uploads, compute/launch span,
and readback. Both timelines overlap and must not be added together.

These events measure elapsed stream spans, **not pure GPU kernel busy time**.
They include possible launch starvation and scheduling gaps. Recording events
also perturbs the trace, especially on small intervals. See NVIDIA's
[CUDA 13.1 event timing documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__EVENT.html).
Cold first passes are retained in the raw logs but excluded from warm means.

## Correctness and reproducibility

All seven full-model processes (four speed runs, two diagnostic runs and the
preceding binary) produce the same five finite logit vectors, **646,400 FP32
values**, bitwise. Every warm repetition within each process also agrees
bitwise. SHA-256 of that output is
`8c8ea79cda4da03fe1d1bbd8774b17eaf71b1a7602bc0a8952d061a9a37cd470`.
Real-model last-only and mixed head schedules pass state, four-step
continuation and reset checks. Traced model windows have full coverage and
zero trace failures. This does not establish equivalence on every possible
context or GPU architecture.

The synthetic test compares serial and batched dumps from fresh processes at
H/FF = 256/128, 5120/2304 and 8192/4096. Each shape exercises 1/6/8 selected
experts and changing inputs, duplicate IDs, negative/signed-zero weights,
cache misses/hits, callback ownership, unpublished output during the callback,
exception propagation, cache recovery and invalid counts. All **122,112
compared floats** agree bitwise. Counters verify 20 successful calls per
process, 510 serial or 80 batched launches, and, when traced, 20 complete
traces plus the deliberately aborted callback trace.

Nine registered GPU/backbone/disabled-path tests and the tiny prefill test
pass without skips. CPU-only old/new logits agree bitwise with 147 identical
printed stage summaries. The maximum-shape test initially used a 1 GiB cache,
which cannot hold the required minimum of 64 slices at that size; rerunning
that fixture with 2 GiB resolved the setup error. The failed attempt is
preserved, not treated as a kernel failure or a passing test. No CUDA hardware
fault was injected; exception recovery tests the host callback failure path.

Artifacts include per-command environments, raw logs, output arrays, source
snapshots/hashes, old source files, model metadata, toolchain/device identity,
admission records, analysis code and a SHA-256 manifest. The model payloads
are external; the manifest does not hash the entire 474 GiB shard set.

Remote experiment: `/home/ubuntu/ntransformer-h100/ds41-moe-batch-20261015`.
Persistent copy: `/lambda/nfs/btiny/ntransformer-training/deepseek-moe-batch-20261015`.
Local copy: this directory, with verification recorded in `local-verification.json`.

Reproduce from `source.tar` using the compiler and CUDA paths in
`environment.json`. `common.py` contains configuration, build/link commands,
environment and admission checks. The separate `changes.tar` captures the
trace-only baseline overlay on the preceding experiment; `candidate.tar`
captures the final batched implementation. `experiment.py` records the exact
order and arguments. Use `analyze.py` after all runs; `seal.py` packages them.
