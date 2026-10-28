# DS41: amortize CPU row scheduling

2026-10-15. Complete DeepSeek-V4.1-Flash Q8_0/MXFP4 model, physical H100 GPU1,
62 GiB expert cache, GPU attention and LM head. Host: Xeon Platinum 8480+,
52 logical CPUs / 26 cores; GCC 14.4.0, CUDA 13.1.115. Both comparison arms
include the preceding GPU upload fixes and final-position-only prefill head.

## Result

| Warm workload | Before | Adaptive chunks | Throughput increase |
| --- | ---: | ---: | ---: |
| Prefill, 16 input tokens | 4.38649 s | 3.48030 s | **26.04%** |
| Prefill tokens/s | 3.64756 | 4.59730 | |
| Four autoregressive decode steps | 1.11753 s | 0.87817 s | **27.26%** |
| Decode tokens/s | 3.57931 | 4.55494 | |

Latency falls 20.66% for prefill and 21.42% for the four-step decode window.
These are one-GPU, short-context, warm-cache results. They do not measure
training, two-GPU scaling, long-context generation or cold-storage throughput.
Decode is four greedy feedback steps after the fixed prompt; it includes
greedy selection and copying the logits for comparison, but no tokenization,
text rendering or network serving.

Final prompt logits and all four continuation logit vectors are **bitwise
identical** across baseline, candidate and the candidate's grain-1 control:
646,400 finite FP32 values per run. Repetitions within each process are also
bitwise equal. The concatenated output SHA-256 is
`8c8ea79cda4da03fe1d1bbd8774b17eaf71b1a7602bc0a8952d061a9a37cd470`.

## Change and rationale

The persistent pool previously used one shared atomic `fetch_add(1)` for every
row, on the main thread and 51 workers. Cheap hyper-connection/residual rows
paid this scheduling cost thousands of times per layer.

The pool now claims adjacent rows in chunks:

```text
grain = clamp(rows / (4 * participants), 1, 64)
```

The workers, wake-up protocol and per-row arithmetic stay the same. Each
claimed chunk runs its rows in order; chunks remain dynamically assigned.
There are normally at least four chunks per participant for substantial work.
For 5,120 rows and 52 participants, the grain is 24: 214 successful atomic
claims instead of 5,120. This calculated count excludes each participant's
final unsuccessful claim; it is not a hardware-counter measurement.

This changes the scheduling of independent output rows, not the order of a
dot product or residual reduction. It changes no CUDA kernel, weight format,
precision, model state or GPU allocation. Peak VRAM was not newly sampled;
both arms allocate the same 10,622 expert-cache slots and resident GPU buffers.
No peer-GPU path is introduced by this change.

`DS41_CPU_GRAIN=1` restores per-row dispatch for diagnosis. A positive value
up to 1,048,576 forces that grain; an invalid value warns and uses the adaptive
default. The variable is read once at pool creation. The existing global pool
still wakes all its workers and still ignores the `n_threads` argument; these
are separate scheduling opportunities, not changes included here.

## Measurement design

`profile.cpp` links against the previous experiment's static library for the
baseline and the new isolated library for the candidate. Compiler options
are the same. The runner uses process order **A1, B1, B2, A2**. Each process
performs one full prompt plus four-step continuation for warm-up, followed by
three measured repetitions. State reset happens before timing. The prompt is
BOS followed by `3 + (i * 37) % (vocab_size - 3)` for positions 1 through 15.

There are six warm samples per arm, drawn from two processes. These repeats
are descriptive measurements, not six independent workload samples or a
confidence interval. Individual warm prefill ranges are 4.36428–4.42513 s
before and 3.45316–3.54189 s after. Four-step decode ranges are
1.10181–1.17037 s before and 0.86939–0.89181 s after.

The cache-counter deltas confirm **100% expert slice hits and zero evictions
in every measured window**, for both arms: 11,520 lookups per prefill and
2,880 per decode window. Each process first loads the same 6,393 distinct
slices. Aggregate cache hit percentages printed at process end include
warm-up misses and must not be mistaken for the measured-window hit rate.

An additional one-sample control using the new binary with
`DS41_CPU_GRAIN=1` gives 4.28029 s prefill and 1.08246 s decode, with identical
outputs. It supports attribution to the scheduling change; it is excluded
from the headline ABBA means.

Both GPUs were free at admission, checked twice under the existing pipeline
locks. The suite completed at **08:35:03 UTC**. The next Brandon evaluation
process started at **08:35:06 UTC**, after lock release; it was left running.

## Warm phase trace

Mean seconds for the 16-token prefill, from `DS41_PROFILE` deltas:

| Existing profile category | Before | After |
| --- | ---: | ---: |
| MoE, including CPU router/shared expert | 1.86062 | 1.65553 |
| Attention | 0.70528 | 0.70479 |
| Engram | 0.06383 | 0.04397 |
| Initial attention hyper-connection/norm | 0.50187 | 0.32348 |
| Other, including remaining mixes/residuals and terminal head | 1.25489 | 0.75253 |

The `hc` timer does not cover all hyper-connection operations; `other` contains
the remaining FFN mixes and residual work. These are host wall-time intervals,
not CUDA-event kernel timings or an Nsight trace. The MoE category does not
yet distinguish routed GPU experts from the CPU shared expert. Attention
subphases and every raw window are retained in `analysis.json`.

The largest savings occur in the CPU-heavy categories while attention remains
almost unchanged. Next, split router/routed/shared expert timing and measure
small-job wake-up overhead. This evidence also supports continuing toward
GPU-owned intermediate state and shared-expert execution; it does not yet
establish the benefit of either implementation.

## Validation and reproduction

Five Release CTest cases pass: adaptive chunks, grain 1, grain 32, a grain
larger than the work, and invalid environment input. Each checks exactly-once
row visitation across sizes including empty/single/uneven ranges, repeated
dispatch, nested calls, and bitwise F32/Q8_0 matvecs and slices against serial
row evaluation. Failures return nonzero independently of `assert`/`NDEBUG`.

The tiny external-model prefill test also passes with GPU disabled: 18-token
prompt, raw-window rollover at 16, compression/index paths, mixed head
schedules, four-token continuation, reset and single-token prompt. Full-model
cross-binary validation covers the 16-token prompt and four-step continuation;
it is not a new long-context or model-quality evaluation.

```bash
ctest --test-dir build -V -R test_ds41_parallel
DS41_GPU=0 DS41_TEST_MODEL=/home/ubuntu/tiny-deepseek41.gguf \
  ctest --test-dir build -V -R '^test_ds41_prefill$'
CUDA_VISIBLE_DEVICES=1 DS41_GPU=1 DS41_GPU_ATTN=1 DS41_VRAM_GB=62 \
  DS41_PROFILE=1 ./profile-candidate MODEL.gguf 16 3 output.bin
```

`experiment.py` contains the exact builds, environment, locking and ABBA
sequence; `analyze.py` checks warm cache deltas and calculates the results.
The baseline archive is the preceding `deepseek-prefill-20261015` experiment.
This bundle contains the new full source snapshot, original scheduling source,
harness, source/binary hashes, raw logs, numerical outputs and result analysis.
Model files remain external. `model-layout.json` records the prior GGUF-header
inventory, not a new hash of the approximately 473 GiB weight payloads.

Locations:

- Local: `docs/benchmarks/h100-deepseek-2026-10-15/cpu-chunks/`.
- Isolated executable: `/home/ubuntu/ntransformer-h100/ds41-schedule-20261015/build/ntransformer`.
- Persistent bundle: `/lambda/nfs/btiny/ntransformer-training/deepseek-cpu-chunks-20261015/`.

The shared remote `source` and `build-release` directories were not replaced.
