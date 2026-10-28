# DeepSeek inference: dataflow audit and next experiments

2026-10-15. Model on the H100 host: `DeepSeek-V4.1-Flash-Q8_0`,
`general.architecture=deepseek41`. This audit concerns the new DS41 path, not
the Llama streaming path or the Brandon training adapter.

Evidence and validation: [local results](benchmarks/h100-deepseek-2026-10-15/RESULTS.md).
The model inventory reads the ten GGUF headers, not their weight payloads.
The [end-to-end benchmark](benchmarks/h100-deepseek-2026-10-15/end-to-end/RESULTS.md)
measures real-text prefixes of 32, 128 and 512 tokens with 64 fixed decode
steps, including initial and reused expert-cache conditions on one H100.

## What is actually running

The reference checkout is `vcruz305/llama.cpp`, branch `runtime/deepseek41`,
commit `5210c7c5ed61dddaee6ed476623abf4b63093d16`. The
[converter's model card](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF)
describes a runtime still under development. The
[original model configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)
identifies the V4.1 architecture; numerical acceptance must use this model and
the pinned reference, rather than assuming V3/V4 equivalence.

`src/main.cpp` dispatches DS41 before constructing the normal inference engine.
The DS41 options currently do not forward the Llama peer-cache, speculative
decoding or streaming options. `ExpertCache::ensure_device()` selects logical
CUDA device 0; the existing `PeerCache` is not connected to this path.

For each input token, `Session::forward()` traverses 40 layers. The current
schedule is approximately:

```text
CPU embedding / hyper-connections / norms
  -> GPU Q/KV projections + norms + RoPE
  -> CPU raw KV, compression bookkeeping, sparse selection, attention, inverse RoPE
  -> GPU grouped output projection
  -> CPU residual / hyper-connections / norm / expert routing
  -> CPU mmap -> pinned staging -> queue GPU expert GEMVs and readback
       GPU routed experts/readback || CPU shared expert
  -> completion wait -> ordered CPU sum / residual
  -> next layer
GPU LM head -> CPU full logits -> CPU sampling
```

The accelerator executes islands of work inside a CPU-owned forward pass.
Removing those boundaries requires moving the dependent operations and state
together. Merely making every copy asynchronous does not remove a dependency.

## Correctness fixes made before measuring speed

The saved `gputime.log` repeatedly reports a failed attention prologue and CPU
fallback. The original Q8 upload code used this pattern:

```cpp
size_t bytes;
get(q8_row_bytes_ok(weight, rows, cols, &bytes), bytes);
```

Argument evaluation order can read `bytes` before the helper writes it. The
first read can be uninitialized; subsequent calls can use the preceding
matrix's size. Recompiling the saved source with GCC 14.4 confirms
`-Wmaybe-uninitialized` in both prologue and tail. Oversized copies can also
consume VRAM needed by later layers. This is not numerical nondeterminism.

The fix calculates each matrix's size with a pure function. Attention,
compressor and indexer now reserve their input buffer independently of the
expert cache. Previously the buffer was allocated only by expert initialization,
although attention executes first. Failed input staging is propagated.

Four numerical tests now run in fresh processes, without an expert cache:
Q/KV prologue including RoPE, grouped output, gated/ungated compression and
Q8/BF16 indexer with input-buffer growth. All pass on H100; largest absolute
error is `1.43051147e-6` in these synthetic cases. Release CMake/CTest also passes
all four. These tests establish the corrected primitives, not full-model parity
or a measured generation speedup. Existing fast-math kernels were preserved.

Follow-up: the [prefill experiment](benchmarks/h100-deepseek-2026-10-15/prefill-head/RESULTS.md)
also ran the complete model. For the fixed single-token CPU-attention versus
GPU-attention comparison, maximum logit error was `3.33786011e-6`, relative L2
`2.60817e-7`, with identical top eight. This is a short numerical check of the
previous correction, not validation of long-context sparse attention.

## Capacity and bandwidth: different objects need different policies

The 1,046 tensor payloads total **473.064 GiB**:

| Tensor category | GiB | Proposed treatment |
| --- | ---: | --- |
| Routed experts, MXFP4 | 268.945 | Local/peer expert cache, bounded in-flight slots |
| Two Engram tables, Q8_0 | 194.555 | Gather selected rows; prefetch by token history |
| Attention and indexer | 5.091 | Reserve resident capacity first |
| Shared experts | 1.401 | CPU overlapped with routed GPU work; evaluate GPU residency separately |
| LM head | 1.233 | Resident; execute only when logits are consumed |
| Token embeddings | 1.233 | Gather rows; full residency optional |
| Other, including Engram projections and hyper-connections | 0.605 | Mostly resident; inspect individual tensors |

Everything except routed experts and Engram tables is **9.563 GiB**; excluding
the input embedding table leaves **8.330 GiB**. KV state, scratch, cache slots,
CUDA context and concurrent workloads require additional capacity. A global
budget must reserve these before choosing the expert cache size. The current
expert budget is independent of the lazily growing resident-weight registry.

Each expert's gate/up/down triplet is **18,800,640 bytes (17.930 MiB)**.
With six experts in every one of 40 layers, all misses imply **4,512,153,600
weight bytes per token**. At the earlier measured 55.041 GB/s pinned H2D rate,
the transfer-only floor is about **82 ms/token**. If every byte were already in
peer VRAM, the measured 304.365 GB/s GPU1-to-GPU0 rate implies about **14.8 ms**.
These are ideal payload bounds, not measured DS41 latencies; they omit CPU
staging, faults, launch overhead, computation and bandwidth contention.
See [the bandwidth measurements](H100_BENCHMARKS.md#transfer-bandwidth).

GPU P2P replaces transfers between GPU memories. It does not by itself remove
the CPU/staging path for an expert still in mmap or storage.

## Prioritized experiments

### 1. Make placement explicit and measure a functioning GPU path

Add a model-level memory plan, resident-byte accounting and per-layer GPU/CPU
fallback counters. Keep the current CPU reference available for parity.
Compare identical token IDs, weights and build flags; count failures as
failures, rather than treating a completed fallback run as GPU coverage.

The saved six-token prefill profile reports 3.21 s attention, 2.11 s MoE,
0.11 s Engram, 0.19 s hyper-connections and 0.46 s other. Attention's QKV and
tail account for 2.11 s and 1.01 s. These are historical phase totals including
initial uploads/fallbacks, not a steady-state decode profile. Re-profile after
the fixes and split first-use upload, warm prefill, decode, and host wait time.

### 2. Eliminate operations whose results are unused

Before this change, prefill called the entire `forward()` including the
1.233 GiB LM head for every prompt token, although the CLI consumes only the
final prompt logits.
Implemented `Session::forward(token, compute_logits=true)`: maintain every
layer's KV/history for every token, and compute the terminal collapse, norm and
head only for requested positions. The default preserves per-position logits
for API callers. The CLI requests only the last prompt position;
`DS41_PREFILL_ALL_LOGITS=1` restores the old schedule. `DS41_DUMP_STAGES` also
retains the full schedule so stage-sum diagnostics remain comparable.

The [measured result](benchmarks/h100-deepseek-2026-10-15/prefill-head/RESULTS.md)
is **+0.425% warm prefill throughput** on the complete model, across four
alternating pairs of 16-token prompts. Head evaluations fall from 16 to 1,
while final logits and four-token continuations remain bitwise identical.
This removes unused work, but its small wall-time effect directs the next
investigation toward the remaining layer pipeline. Decode speed was not
measured by this experiment.

CPU fallback scratch (`partials`, `hbuf`) now allocates only after GPU MoE
fails, as described below. Further candidates are preallocating other
per-session scratch and binding tensor handles once. The CPU thread pool is
already persistent, but expert-miss staging still creates one thread per
missing slice (up to 18 per layer for six experts) and joins all before H2D.
Use persistent workers only where a host copy is still required.

The next [warm-cache experiment](benchmarks/h100-deepseek-2026-10-15/cpu-chunks/RESULTS.md)
found substantial CPU scheduling cost. The persistent pool now claims adjacent
rows in adaptive chunks instead of performing a shared atomic operation for
every row. With 52 logical CPUs, 5,120 rows use 214 successful claims instead
of 5,120; each row keeps its original arithmetic and reduction order.
`DS41_CPU_GRAIN=1` restores the previous per-row schedule for diagnosis.

On one H100, the complete model's warm 16-token prefill improves from
**4.38649 to 3.48030 s (+26.04% throughput)**; four-step autoregressive decode
improves from **1.11753 to 0.87817 s (+27.26%)**. ABBA process order, six warm
windows per arm, identical prompt and continuation, 100% expert-cache hits
within every measured window. Final prompt logits and all four continuation
vectors remain bitwise identical. This measures short warm workloads, not
training, cold streaming or two-GPU scaling. Attention stays near 0.705 s per
prompt; most savings occur in CPU-heavy mixes/residuals and the MoE category.

### 2b. Overlap independent shared and routed expert work

The [next completed experiment](benchmarks/h100-deepseek-2026-10-15/moe-overlap/RESULTS.md)
splits MoE profiling into router, routed execution/wait and shared CPU work.
Both expert branches consume the same layer input; the shared expert does not
depend on the routed output. The GPU helper now queues its original kernels
and pinned readback, runs scoped CPU work, then waits and publishes the result.
The final ordered sum is preserved. `DS41_MOE_OVERLAP=0` selects the serial
control; stage-sum and GPU comparison diagnostics also retain serial execution.

On one H100 with warm expert cache, prefill improves from **3.44908 to
2.79074 s (+23.59% throughput)** and four-step decode from **0.87954 to
0.70774 s (+24.27%)**. Both arms already include adaptive CPU chunks. Six
windows per arm in ABBA process order show 100% cache hits, zero CPU routed
fallbacks, and bitwise-identical prompt/continuation logits, including comparison
with the preceding binary. MoE wall time falls from 1.634 to 0.960 s per prompt.

CPU fallback buffers now allocate only on fallback, removing 174 KiB of unused
array payload per layer on GPU success. The two timing arms share this cleanup;
its isolated speed contribution is not measured. The callback runs on the
calling thread, uses separate CPU output, and must not reenter DS41 GPU helpers.
Queued work is drained on callback exceptions; CUDA failures invalidate cache
entries. GPU memory allocation and numerical kernels are unchanged.

### 2c. Remove dispatch overhead from small hyper-connection operations

The [direct HC experiment](benchmarks/h100-deepseek-2026-10-15/hc-direct/RESULTS.md)
targets the remaining pool dispatches for stream collapse and residual
post-mixing. With four streams, these operations perform only four or twenty
products per hidden coordinate. They now execute on the calling thread,
preserving each output's original accumulation order. `DS41_HC_DIRECT=0`
restores pool execution; projection matvecs and attention retain their
existing policies.

For 40 layers, this removes 2,561 pool dispatches per 16-token final-head-only
prefill and 644 per four-step decode window. This removes coordination, not
model arithmetic. Direct/pool/scalar comparisons cover 84 shape/value cases;
tiny-model logits match the preceding binary bitwise, with identical stage
sums. The complete-model comparison uses the same warm-cache ABBA protocol.

On one H100, six warm windows per arm improve prefill from **2.78437 to
2.23845 s (+24.39% throughput)** and four-step decode from **0.70263 to
0.56803 s (+23.70%)**. Prompt and continuation logits remain bitwise identical
to the preceding binary. Every measured window has 100% expert-cache hits,
zero routed CPU fallbacks and the same MoE overlap. The HC/other categories
shrink substantially; MoE and attention now account for about 75% of prefill
wall time. GPU allocation code and cache capacity are unchanged.

### 2d. Submit independent attention output groups in one grid

The [grouped-tail experiment](benchmarks/h100-deepseek-2026-10-15/grouped-tail/RESULTS.md)
targets the eight independent Q8 GEMVs in attention's first output projection.
The original path launched each group separately, then projected their
concatenation. A two-dimensional grid now makes all eight groups available in
one launch. Each row retains its original warp, Q8 block traversal and FP32
reduction; the second projection remains ordered on the same stream.

This changes output-projection kernel counts from nine to two per layer,
eliminating 4,480 launches per 16-token prompt. `DS41_GROUPED_TAIL=0` restores
separate launches. Cumulative `gpu_backbone_stats()` counters expose successful
tail calls, grouped calls and projection launch counts, so measured window
deltas can verify both GPU coverage and the selected schedule. There are no
new GPU allocations, transfers, precision changes or intermediate tensors.

The full-model result is modest: **+0.36% prefill throughput and +0.55% decode**
across 12 warm windows per arm from four processes, using ABBA then BAAB order.
The output-projection phase itself takes **4.36% less time** in prefill, while
its launch count falls 77.78%. Both sequences show a small positive total
effect, with overlapping end-to-end timing ranges. All prompt/continuation
logits remain bitwise identical to the previous binary. This confirms a useful
local reduction, but identifies computation, movement and the remaining MoE
critical path as more consequential targets than this set of host launches.

### 2e. Batch independent routed experts and trace the exposed waits

The routed MoE path now submits four kernels per layer: a two-dimensional
gate/up grid across the selected experts, elementwise activation, a down grid
across experts, and an ordered weighted sum. Each dot product retains its
MXFP4 decode, per-row warp and FP32 reduction order. Intermediate gate, up,
activation and down values remain FP32. The final sum follows the router's
original expert order; it is not a parallel tree reduction.

For six experts this removes 26 of 30 launches per layer, or 16,640 launches
per 16-token prompt across 40 layers. The separate output memset is also
unnecessary because the final kernel writes every output from a local zero
accumulator. Eight-expert scratch adds 560 KiB of VRAM; weight-cache slots,
uploaded weight formats and CPU shared-expert overlap are unchanged.
`DS41_MOE_BATCH=0` restores the serial expert schedule. The switch is read
once per process, and successful call/launch/upload counters are available
through `gpu_moe_stats()`.

`DS41_MOE_TRACE=1` enables eight host phases and five CUDA-event intervals.
The six events are reused after the existing completion wait; tracing adds
no intermediate synchronization. Host phases separate input enqueue, cache
lookup, miss staging, weight-copy enqueue, kernel/readback enqueue, the CPU
callback, final wait and result publication. Stream intervals separate input
(and serial-path zeroing), the preparation gap, weight uploads, compute/launch
span and readback. Initialization and cache allocation precede these phases.
Only complete, successful traces contribute timings; failed or aborted traces
have a separate counter. Host and GPU intervals overlap and are not additive.

CUDA-event intervals are **elapsed stream spans, not pure GPU busy time**.
They can include host launch starvation, scheduling gaps and interference;
see the [CUDA 13.1 event timing documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__EVENT.html).
Performance comparisons disable this extra tracing and use separate diagnostic
runs to locate waits. Tests cover 1/6/8 selected experts, repeated IDs, negative
and signed-zero weights, cache misses/hits, callback ownership and exception
recovery, including real and maximum supported dimensions.

The [batched-MoE experiment](benchmarks/h100-deepseek-2026-10-15/moe-batch/RESULTS.md)
improves full-model warm prefill from **2.22746 to 2.12996 s (+4.58% throughput)**
and four-step decode from **0.56515 to 0.54272 s (+4.13%)**, across six windows
per arm in ABBA order. All 646,400 prompt/continuation logits remain bitwise
identical to the preceding binary, with zero routed CPU fallbacks and 100%
warm expert-cache hits. Synthetic comparisons also agree bitwise for all
three tested shapes, up to eight experts at H=8192/FF=4096.

Separate warm traces show kernel/readback submission falling from 56.36 to
12.16 ms per prompt, final completion waits from 20.40 to 2.58 ms, and the
GPU compute/launch span from 752.41 to 576.96 ms. These overlapping intervals
are diagnostic, not an additive explanation of the total speedup. Shared CPU
work also became faster under the new schedule despite unchanged arithmetic;
the experiment does not isolate why. Its approximately 639 ms callback now
exceeds the GPU span, making that CPU/GPU boundary the next warm-cache target.

The first diagnostic prompt uploads 35.57 GB of expert slices. Both schedules
still spend approximately 0.81 s staging those misses and 0.67 s in the upload
stream span. This is an initially empty expert cache, not a cold-storage test.
Batching does not remove the miss pipeline: persistent staging resources,
copy/compute overlap and peer-owned expert execution remain separate targets.

### 2f. Keep the shared expert resident and reproduce CPU Q8 reductions on GPU

The [shared-expert experiment](benchmarks/h100-deepseek-2026-10-15/shared-expert/RESULTS.md)
moves the shared gate/up/down projections and activation onto the same GPU as
the routed experts. Three new kernels replace three CPU matrix-vector jobs
per layer. They reuse the routed input, keep outputs separate until the original
final addition, and use the existing single completion wait. Shared weights
add **1.40076 GiB** of residency and **47 KiB** of reusable scratch; the routed
expert cache remains 62 GiB. A second result adds 0.78125 MiB of D2H per token.

The Q8 kernel reproduces the CPU AVX2 block operations and ordered FP32
accumulation, verified bitwise for identical inputs. An exponential evaluated
in FP64 and rounded to FP32 reduces the CPU/CUDA library discrepancy; the rest
of the activation remains FP32. Explicit eight-lane masks preserve collective
membership even with partially populated warps. `DS41_SHARED_GPU=0` restores
the CPU shared expert. The GPU path is enabled by default for compatible Q8
tensors in AVX2 CPU builds, with the old path retained as fallback.

The final BAAB comparison measures **+3.33% warm prefill throughput**
(2.15960 -> 2.08998 s) and **+3.30% decode** (0.54814 -> 0.53062 s), six windows
per arm. An earlier ABBA run before tightening subgroup membership showed
+3.03%/+3.06%; final outputs match that run bitwise. No measured window uses
routed CPU fallback or shared CPU fallback in the GPU arm. The gain is much
smaller than the removed CPU work because that work previously overlapped
routed GPU execution. The GPU now performs both experts on the same stream.

Across **3,232,000 compared logit values** from the short benchmark and two
contexts through length 129 plus continuation, maximum absolute error is
**2.86102e-5**, maximum relative L2 **1.35367e-6**, and all 25 sampled top-1 and
ordered top-8 results agree. This is a reproducible arithmetic difference,
not run-to-run nondeterminism. CPU controls remain bitwise equal to the old
binary. The isolated shared FFN is bitwise equal in the synthetic fixtures,
but small activation differences still propagate in the complete model.

The first GPU version used FP32 `expf`. FP64 reduces short-benchmark maximum
logit error from 3.14713e-5 to 2.86102e-5; the strong local improvement does
not make the full model bitwise equal. Separate final traces show a routed-only
GPU span of 577.07 ms per prompt versus 622.77 ms with the shared expert, while
the 655.61 ms CPU callback disappears. The final host wait increases because
it no longer contains useful CPU work. These overlapping intervals are not
additive; they point to routed GPU execution as the next MoE target.

### 2g. End-to-end requests expose expert-cache pressure

The [real-text benchmark](benchmarks/h100-deepseek-2026-10-15/end-to-end/RESULTS.md)
uses the final shared-expert engine, one H100, a 62 GiB expert cache and batch
size one. Each prompt length starts in a separate process, then repeats the
same request three times with reset attention/history state and reused weights.
Prefill is still sequential token execution. Throughput over the three repeats:

| Prompt tokens | Prefill tokens/s | Decode tokens/s, 64 fixed steps | Ready-model TTFT |
|---:|---:|---:|---:|
| 32 | 6.067 | 6.229 | 5.275 s |
| 128 | 6.162 | 6.178 | 20.771 s |
| 512 | 6.052 | 5.364 | 84.605 s |

EOS is ignored for fixed-work timing. The 512-token case emits EOS after 32
non-EOS predictions; its pre-EOS prefix runs at 5.627 tokens/s. The other two
cases do not emit EOS within the 65 predictions (one from prefill plus 64
decode forwards). These are raw text completions, without a chat template.

Unlike the earlier 16+4 synthetic workload, even the repeated requests evict
experts. Decode hit rates are 87.93%, 90.01% and 82.45%, with 32.46, 26.86 and
47.19 GiB of routed H2D uploads per 64 steps. These are cache-slice hits and
host-to-device bytes, not disk-read measurements. Initial prefill rates are
2.455, 4.157 and 4.749 tokens/s; host page cache was not flushed. The changed
workload prevents interpreting the difference from 7.54 decode tokens/s as
a code regression.

All 12 passes complete with finite logits and full routed/shared/tail GPU
coverage; each repeated request reproduces its initial pass bit for bit.
Observed maximum GPU memory is 72,173 MiB (70.48 GiB); sampled GPU utilization
averages 41.1% over setup, requests and gaps. The largest process reaches
155.85 GiB host RSS. The second GPU remains unused. The measurements make
expert residency/transfer overlap and the growing attention cost relevant
targets alongside kernel compute; they do not measure a two-GPU speedup.

### 3. Move computation to resident experts as an alternative to copying them

Test two distinct two-GPU paths:

* **Peer cache:** on a local miss, fetch a ready triplet from GPU1 through
  `cudaMemcpyPeerAsync`, then compute on GPU0. Reuse the existing peer-cache
  concepts at expert granularity, rather than assuming its Llama layer layout.
* **Expert ownership:** keep a subset of experts on GPU1 and execute them there.
  Send the activation and route metadata, then return expert outputs. For
  three remote experts, transferring weights costs 56,401,920 bytes; one
  5,120-float input plus three outputs costs 81,920 bytes, excluding metadata.
  That is about **688.5 times fewer communication bytes**, not a speedup claim.

The second path makes both H100s compute and avoids moving a weight matrix
every time a cached remote expert is used. Preserve expert output order when
reducing: returning one pre-summed remote result changes FP32 association.
Measure launch/transfer latency, load balance and cache hit rates before
choosing either strategy; a single token has only six routed experts per layer.

### 4. Reserve blocks early; overlap only work with known dependencies

Current MoE stages every miss, enqueues every H2D, then runs all expert kernels
on the same stream. Introduce separate copy/compute streams and completion
events per triplet. Compute a ready expert while another triplet is loading.
Reserve fixed slots before dispatch and prevent eviction until consumers finish.
Keep the host staging source alive until its DMA completion event.

Block reservation can precede routing; the identity of a required MoE expert
cannot generally be known before the router sees that layer's activation.
Speculative weight prefetch therefore needs a bounded budget and must be
measured by useful bytes, pollution and stalls. It must never change routing.
The ideal overlapped interval approaches `max(copy, compute)` rather than their
sum, but shared hardware and dependencies can prevent that bound.

Pinned memory and explicit completion ownership matter: CUDA's `Async` suffix
alone does not guarantee host-asynchronous behavior for pageable memory.
See [CUDA 13.1 synchronization semantics](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/api-sync-behavior.html).

### 5. Prefetch Engram exactly, without predicting hidden states

`Session::engram_add()` derives table row IDs from the current token, previous
tokens and fixed hash parameters. All Engram row addresses for the token can
therefore be computed before layer 0. For prefill, the prompt tokens are
already known. Schedule gathers ahead of the layer that consumes them.

The current configuration has two layers, 24 buckets each, 256 Q8_0 values per
row: only **13,056 logical quantized bytes** are selected per token from the
194.555 GiB tables, before reuse. Actual storage traffic can be larger because
of page/read granularity. This argues for row-level gathering and a hot-row
cache, not reserving entire tables in VRAM. Keep the small Engram projections
resident and overlap row fetching with unrelated layer work.

### 6. Keep the attention state and intermediate tensors on GPU

The QKV prologue currently performs three output copies, each followed by a
stream synchronization; the tail returns to the GPU after CPU attention.
Prologue+tail boundaries alone move **310,272 bytes/layer**, about
**11.836 MiB/token** across 40 layers, excluding compressor/indexer/MoE and
initial weights. More significant than those bytes may be their 160 explicit
host waits per token when both paths succeed.

Implement the raw KV ring, compressed state, sparse selection, attention,
inverse RoPE and output projection as one GPU-owned chain. Migrate one
subgraph at a time and preserve sinks, compression visibility, shared layer
roles and rotation semantics. Aggregate readback while migrating, but make
the eventual interface return the next consumed result rather than every
intermediate. Use stage-specific comparisons, not just final generated text.

### 7. Group expert dispatch and use a distinct prefill path

The four-launch expert batching described in section 2e preserves arithmetic
and ordered reduction. Further fusions should be evaluated separately, with
the same numerical checks and measurements of the exposed critical path.

For longer prompts or concurrent requests, group tokens by expert and evaluate
matrix-matrix kernels to reuse weights. The current prefill is a loop of
single-token GEMVs. Keep single-request decode and batched prefill metrics
separate; a higher prefill token rate does not establish faster decode.

## Acceptance for the next full run

Use the pinned reference and fixed input IDs at single-token, compression
boundaries, sliding-window rollover and contexts exceeding sparse top-k.
Measure stage errors, logits, top-k agreement and greedy continuations;
attribute any discrepancy before accepting it as numerical variation.
Record actual GPU coverage and reject silent fallback from the GPU experiment.

Collect warm/cold expert hits by layer, bytes from host/peer/storage, copy wait
time, kernel time, allocation events, reserved/used VRAM, prefill tokens/s,
time to first token and per-token decode latency. Run controlled before/after
benchmarks with the GPUs available. The initial synthetic GPU tests ran
alongside Brandon and do not provide throughput measurements. The subsequent
prefill-head and CPU-chunk experiments above use separate idle-GPU windows;
their short warm-cache measurements do not establish cold or long-context
performance.
