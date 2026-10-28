# DeepSeek inference: dataflow audit and next experiments

2026-10-15. Model on the H100 host: `DeepSeek-V4.1-Flash-Q8_0`,
`general.architecture=deepseek41`. This audit concerns the new DS41 path, not
the Llama streaming path or the Brandon training adapter.

Evidence and validation: [local results](benchmarks/h100-deepseek-2026-10-15/RESULTS.md).
The model inventory reads the ten GGUF headers, not their weight payloads.

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
  -> CPU mmap -> pinned staging -> GPU expert GEMVs -> CPU result
  -> CPU shared expert / residual
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

## Capacity and bandwidth: different objects need different policies

The 1,046 tensor payloads total **473.064 GiB**:

| Tensor category | GiB | Proposed treatment |
| --- | ---: | --- |
| Routed experts, MXFP4 | 268.945 | Local/peer expert cache, bounded in-flight slots |
| Two Engram tables, Q8_0 | 194.555 | Gather selected rows; prefetch by token history |
| Attention and indexer | 5.091 | Reserve resident capacity first |
| Shared experts | 1.401 | Resident; used on every layer/token |
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

Prefill currently calls the entire `forward()` including the 1.233 GiB LM head
for every prompt token, while the CLI consumes only the final prompt logits.
Add an explicit `compute_logits` option: maintain every layer's KV/history for
every token, and compute the head only for requested positions. Default it on
for callers that need all logits. This preserves the causal computation and
avoids work before changing any matrix kernel.

CPU fallback scratch (`partials`, `hbuf`) is allocated before knowing whether
GPU MoE succeeds. Move it into the fallback branch; preallocate other
per-session scratch and bind tensor handles once. The CPU thread pool is
already persistent, but expert-miss staging still creates one thread per
missing slice (up to 18 per layer for six experts) and joins all before H2D.
Use persistent workers only where a host copy is still required.

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

Six routed experts currently require 30 kernel launches per layer: gate,
up, activation, down and weighted add for each expert. A grid over expert IDs
with per-expert scratch can batch those launches while preserving each expert's
arithmetic and ordered final reduction. Then evaluate fusions separately.

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
benchmarks with the GPUs available; the current small tests ran alongside
Brandon and do not provide throughput measurements.
