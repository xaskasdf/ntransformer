# H100 DeepSeek: resident shared expert with CPU-compatible Q8 reductions

Date: 2026-10-15. The control retains the preceding batched routed-MoE,
grouped attention tail, direct HC loops and adaptive CPU work scheduling.
Only the shared expert moves from CPU to the same GPU as routed experts.

## Implementation

The three shared Q8 matrices per layer are uploaded once and remain resident.
Gate/up share a two-dimensional grid; activation and down projection add two
more kernels. Shared and routed experts use the same device input and one
completion wait. Their output buffers remain separate until the original CPU
addition. This removes three CPU matrix-vector jobs per layer, or 1,920 jobs
per 16-token prompt, without changing routing, quantized weights or the final
addition order. Routed experts still use four kernels; the GPU shared expert
adds three. They execute in order on one CUDA stream.

For each Q8 block, eight lanes reproduce the CPU AVX2 arithmetic: one rounded
multiply, three FMAs, the low/high add, and the same two horizontal adds.
Block sums are produced in parallel, then one lane accumulates scaled blocks
in the original order using FP32 FMAs. Thus the independent work is moved to
GPU while preserving the reference's reduction structure. Each eight-lane
group uses an explicit membership mask; it does not infer membership from
the lanes currently executing. See NVIDIA's
[warp primitive guidance](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).

The nonlinear activation uses a **FP64 exponential rounded to FP32**, followed
by the original FP32 add/divide/multiply. Tensor storage and GEMVs remain
FP32/Q8. This CUDA object is compiled separately without `--use_fast_math`.
It reduces approximation differences against the CPU `expf` library, but does
not promise bitwise equality across CPU and GPU implementations. See NVIDIA's
[floating-point guidance](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html).

`DS41_SHARED_GPU=0` restores the CPU shared expert and its overlap with routed
GPU work. The default is enabled for compatible Q8 tensors when the CPU build
uses AVX2; unsupported formats/shapes and unavailable offload fall back to the
previous path. `DS41_GPU_TEST` retains its existing CPU comparison behavior.
The mode is selected when constructing a Session. The global resident registry
requires immutable weights that stay alive, as do the existing DS41 registries.

Both shared and routed results are published after successful completion.
Shared setup failure retries routed GPU work with the CPU shared callback.
The benchmark checks actual shared/routed GPU coverage and rejects silent
fallback. New counters distinguish shared CPU calls, shared GPU calls,
successful shared kernel launches and resident shared-weight bytes.

## Performance

| Metric | Shared expert on CPU | Shared expert on GPU | Change |
|---|---:|---:|---:|
| 16-token prefill mean | 2.159595 s | 2.089979 s | -3.22% latency |
| Prefill throughput | 7.40880 tokens/s | 7.65558 tokens/s | **+3.33%** |
| Four-step decode mean | 0.548142 s | 0.530617 s | -3.20% latency |
| Decode throughput | 7.29738 tokens/s | 7.53839 tokens/s | **+3.30%** |
| Prefill MoE phase | 0.891415 s | 0.850863 s | -4.55% |
| Decode MoE phase | 0.223229 s | 0.213113 s | -4.53% |

Final prefill ranges are 2.150930-2.171902 s for CPU shared and
2.085207-2.098678 s for GPU shared; decode ranges are 0.543973-0.550797 s
and 0.528569-0.534420 s. These observed ranges do not overlap. The initial
measurement showed +3.03% prefill and +3.06% decode. The final comparison
reverses process order and confirms a modest benefit. These are repeated
windows of one workload, not independent prompts or a general quality study.

The final measurement uses GPU/CPU/CPU/GPU process order, three warm
repetitions per process after a full warmup: six windows per arm. The initial
CPU/GPU/GPU/CPU measurement, before tightening the subgroup membership mask,
is preserved separately in `initial-*` files. Final prompt and context outputs
must match that initial GPU implementation bitwise; the final performance
table uses only the final binary's measurements.

The workload is the same 16-token prompt with last-only head evaluation and
four greedy decode steps. Both arms use physical GPU1, one H100 80 GB, with
62 GiB reserved for the expert cache. Both GPUs and the shared pipeline locks
must be free for admission. The concurrent Brandon `mapping-kl-v1` pipeline
finished before full-model runs; no training process was stopped. Small
synthetic checks ran concurrently and their elapsed times are not benchmarks.

Every measured warm window has 100% routed expert-cache hits, zero evictions,
zero weight uploads and zero routed CPU fallback. Shared-GPU windows also
have zero shared CPU fallback and exactly three shared kernels per layer.
CPU-control windows retain their shared callback overlap. Extra event tracing
is disabled for performance measurements. These repeated short prompts do
not establish cold-cache, long-context or two-GPU throughput. The longer
context runs below test numerics; their ordered wall times are not a speedup
comparison.

## Memory and transfers

Shared resident weights add **1,504,051,200 bytes (1.40076 GiB)** across 40
layers, plus **48,128 bytes (47 KiB)** of reusable device scratch at H=5120,
FF=2304. Expert cache slots and budget remain unchanged. These are allocation
deltas, not measured process-memory peaks. Shared weights upload once per
process; no extra activation H2D is needed because the routed input is reused.

Keeping the original CPU addition returns a second 5,120-float result per
layer: **819,200 additional D2H bytes/token (0.78125 MiB)**, with 20 KiB more
pinned readback capacity. There is still one completion wait for the combined
call. This implementation uses one GPU; peer transfers and execution on the
other H100 are separate future experiments.

## Numerical validation and discrepancy mitigation

| Comparison against CPU shared | Logit values | Maximum absolute error | Maximum relative L2 | Top-1 / ordered top-8 |
|---|---:|---:|---:|---|
| Short benchmark, final FP64-exp activation | 646,400 | 2.86102e-5 | 1.35367e-6 | 5/5 agree |
| Short benchmark, initial FP32-exp activation | 646,400 | 3.14713e-5 | 1.41677e-6 | 5/5 agree |
| Two contexts through length 129, final implementation | 2,585,600 | 1.71661e-5 | 9.58112e-7 | 20/20 agree |

The final implementation checks **3,232,000 logit values** over 25 vectors.
All values are finite. All sampled top-1 predictions, ordered top-8 lists and
greedy continuations agree. FP64 reduces maximum absolute logit error by
9.09% and maximum relative L2 by 4.45% on the short comparison; its much
larger isolated-activation improvement does not eliminate propagation in
the complete model. This is not a guarantee on every context or architecture.

CPU-control output SHA-256:
`8c8ea79cda4da03fe1d1bbd8774b17eaf71b1a7602bc0a8952d061a9a37cd470`.
GPU-shared output SHA-256:
`5d46d14c9ac7bc5b4089a2bd7fe2424aef514ec09e6c604896f8e2125d1b39c0`.
The subgroup-mask correction preserves both short and full context-probe
outputs bitwise relative to the first FP64 implementation.

Acceptance limits were fixed before the full-model comparison: maximum
absolute logit error <=1e-3, relative L2 <=1e-4 and unchanged sampled greedy
IDs. Ordered top-8 agreement is reported as an additional observation.
CPU-control outputs remain bitwise equal to the preceding binary, and GPU
repetitions are internally bitwise repeatable. The CPU/GPU difference is a
reproducible arithmetic difference, not nondeterministic run-to-run variation.

Synthetic tests cover H/FF = 96/64, 5120/2304 and 8192/8192, four changing
inputs, unclamped and clamped activations including the 1e-6 boundary,
negative zero, resident reuse, unsupported dimensions, GPU-disabled behavior,
and integrated routed/shared output ownership. Gate/up projections and down
projections with identical activation inputs match the actual CPU Model
implementation bitwise. Final full shared-expert outputs also match CPU
bitwise in all these fixtures, although tiny activation differences remain
in some large-shape elements.

The original FP32 exponential failed a pointwise tolerance in the maximum,
unclamped synthetic fixture: maximum FFN error 0.00390625 with reference
magnitude about 44,207 and relative L2 1.63e-8. A normwise diagnostic isolated
amplification around cancellation. After using the FP64 exponential, all
fixtures pass **stricter pointwise and relative-L2 checks**, with zero final
FFN error. Failed and diagnostic attempts are preserved. No full-model
acceptance threshold was relaxed. The final tests use atol=3e-4 and rtol=1e-6
pointwise, plus relative L2 <=1e-6 for nonzero FFN references.

The full-model context probe uses two fixed token sequences with snapshots
at lengths 1, 16, 32, 64, 128 and 129, then four greedy continuation steps.
It crosses the 128-token raw-window boundary and exercises compression state.
It does not exceed the model's sparse top-k capacity of 512. The last-only
and mixed head schedules also pass continuation/reset checks on the real
model. The CPU-only tiny fixture retains identical logits and 147 stage sums.
Eleven registered GPU/backbone/disabled-path tests pass without skips, with
additional real/max-shape checks. No CUDA hardware fault was injected.

## Diagnostic trace

Separate final diagnostic runs use two warm repetitions per arm.
Mean intervals per 16-token prompt:

| Interval | Shared on CPU | Shared on GPU |
|---|---:|---:|
| Host kernel/readback enqueue | 12.15 ms | 19.43 ms |
| Host shared callback | 655.61 ms | 0.04 ms (empty callback section) |
| Host completion wait | 2.46 ms | 612.18 ms |
| GPU compute/launch stream span | 577.07 ms (routed) | 622.77 ms (routed + shared) |
| GPU readback stream span | 4.68 ms | 5.98 ms |

The former CPU callback largely overlapped the routed GPU span. Removing
655.61 ms of CPU work therefore does not remove 655.61 ms of latency. The
combined GPU span grows by about 45.69 ms, but finishes sooner than the old
CPU shared work. Other host phases also change; the total gain cannot be
attributed only to the shared kernel durations. Routed GPU execution now
accounts for most of this MoE segment, motivating a separate kernel-level
investigation rather than assuming more shared-expert offload will yield
another large gain.

CUDA events measure elapsed stream spans, including possible idle gaps,
not pure kernel busy time. CPU and GPU intervals overlap and must not be
added. Initialization and resident uploads precede the per-call trace.
The old `shared_seconds` category measures CPU work only; with GPU shared
execution its work is included in the GPU submission/wait category and the
combined stream span. A larger CPU completion wait is expected when the CPU
callback has been removed and is not, by itself, a regression.

## Artifacts

Remote: `/home/ubuntu/ntransformer-h100/ds41-shared-20261015`.
Persistent: `/lambda/nfs/btiny/ntransformer-training/deepseek-shared-20261015`.
Local: this directory, verified against the SHA-256 manifest and current
sources in `local-verification.json`.

The archive contains raw command/environment records, finite logit arrays,
source snapshots, compiler flags, admission records and analysis code.
`changes.tar` preserves the initial FP32-exp implementation; `final-changes.tar`
contains the final implementation. `shared-exp32.cu` and
`shared-before-mask-review.cu` preserve the intermediate arithmetic variants.
`source.tar` is the final complete build snapshot. Model shards remain external;
the entire 474 GiB payload was not rehashed. `common.py`, `experiment.py`,
`final_validation.py`, `run_probes.py`, `compare.py` and `analyze.py` record
the build, measurement, comparison and analysis procedure.
