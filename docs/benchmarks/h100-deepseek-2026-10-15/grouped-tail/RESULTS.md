# DS41: launch independent attention output groups together

2026-10-15. DeepSeek-V4.1-Flash Q8_0/MXFP4, physical H100 GPU1,
GCC 14.4.0, CUDA 13.1.115. Adaptive CPU chunks, direct HC operations,
final-position-only prefill head and shared/routed MoE overlap are retained.

## Dependency analysis and implementation

The preceding warm profile spent approximately 0.702 s of a 2.238 s prompt
in attention. Its output projection executed eight independent GEMVs in
sequence on the same CUDA stream, then one GEMV over their concatenation:

```text
z[g] = A[g] @ attention_output[g], for g = 0,...,7
output = B @ concat(z[0],...,z[7])
```

Each group has 1,024 output rows and 4,096 input columns. The existing kernel
uses one warp per output row and eight warps per block: 128 blocks per
individual group launch. The candidate submits the same eight groups in a
single 128-by-8 grid (1,024 blocks), followed by the unchanged B projection.
This removes seven host launches and the intervening kernel boundaries per
layer, while making all independent group work available to the scheduler.
Grid dimensionality only supplies indexing; it is not itself an acceleration
mechanism. CUDA describes multidimensional grid and block indexing in its
[C++ programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html).

`k_gemv_q8_0<Grouped>` offsets weights, input and output by `blockIdx.y`.
The default specialization retains the original one-matrix implementation.
Every row uses the same Q8 blocks, FP32 accumulators, traversal order and warp
reduction. Groups write disjoint output slices. The subsequent B kernel on
the same stream consumes all completed slices, in their original order.
There is no new stream, allocation, format conversion or reduction across
groups. Existing transfers and completion waits remain.

Grouped execution is enabled by default. `DS41_GROUPED_TAIL=0` restores the
separate group launches; the setting is read on the first tail invocation.
Group counts beyond the CUDA grid-y limit retain the separate-launch path.
`gpu_backbone_stats()` exposes cumulative successful tail calls, grouped
calls and associated GEMV launch counts. Failed calls do not count as GPU
coverage. Window deltas verify that the intended path actually ran.

For the 40-layer model, successful output projections use **two kernels per
layer instead of nine**: 1,280 rather than 5,760 launches per 16-token prompt,
or 320 rather than 1,440 per four-token decode window. Arithmetic and tensor
payloads are unchanged. These counts cover the attention output projections,
not all model kernels.

## Full-model results

The launch reduction is substantial, but the end-to-end gain is small.
Combined results from the initial ABBA sequence and a BAAB confirmation:

| Workload | Separate groups | One grouped launch | Throughput increase |
| --- | ---: | ---: | ---: |
| 16-token prefill | 2.23353 s | 2.22552 s | **0.36%** |
| Prefill tokens/s | 7.16355 | 7.18933 | |
| Four autoregressive decode steps | 0.56827 s | 0.56514 s | **0.55%** |
| Decode tokens/s | 7.03887 | 7.07785 | |

There are **12 warm windows per arm from four processes**. The prefill saves
8.01 ms per prompt; the decode window saves 3.13 ms. Initial and confirmation
throughput changes are +0.549% and +0.171% for prefill, +0.447% and +0.660%
for decode. Both sequences favor grouped execution, but end-to-end ranges
overlap: prefill 2.22050–2.24852 versus 2.21930–2.23855 s; decode
0.56323–0.58065 versus 0.56230–0.56910 s. This is a modest observed improvement,
not evidence of a large or workload-independent speedup. The experiment uses
one GPU, short context and warm expert cache; it does not measure training,
long-context scaling, cold streaming or two-GPU scaling.

The targeted output-projection phase shows a clearer reduction:

| Host wall-time phase per 16-token prompt | Separate, s | Grouped, s |
| --- | ---: | ---: |
| Attention output projection | 0.24864 | 0.23780 |
| Complete attention | 0.70050 | 0.69130 |
| Q/KV prologue | 0.21509 | 0.21602 |
| Compression/indexing | 0.04444 | 0.04522 |
| CPU attention loop and inverse RoPE | 0.19089 | 0.19086 |
| Complete MoE | 0.97630 | 0.97812 |

Output-projection time falls **4.36%** during prefill and **4.22%** during
decode. Its prefill ranges do not overlap: 247.82–249.52 ms separate versus
237.17–238.42 ms grouped. Other phase variation reduces the net saving.
These are host wall intervals, including calls, copies and completion waits;
they are not CUDA-event kernel times or measurements of SM occupancy.

Counters confirm 640 successful GPU tail calls per prefill and 160 per decode
window. The exact launch counts are 5,760→1,280 and 1,440→320 respectively:
**77.78% fewer output-projection launches**. Every measured window also has
100% expert-cache hits, zero evictions, no routed CPU fallback and shared-CPU
overlap on all 640/160 MoE calls. Each process warms the same 6,393 distinct
expert slices; aggregate process hit rates include the initial misses.

The same 10,622 expert-cache slots (62 GiB budget), resident weights and
scratch are used. GPU allocation code and requested capacities are unchanged;
peak VRAM was not newly sampled. Removing launches did not remove projection
arithmetic or the CPU/GPU data transfers. The limited overall gain directs
the next investigation toward computation, data movement and the shared MoE
branch, with device timing needed to separate those costs.

## Validation and protocol

The tail tests cover one, two, three and eight groups, three distinct inputs,
and an eight-group/rank-36 shape whose last block has unused warps. All weight
allocations remain alive because the resident registry keys them by address.
Results are checked against an independent CPU matrix-vector reference with
the existing tolerance `2e-4 + 5e-5 * abs(reference)`, and checked for finiteness.
The tests also require exact call and launch counters in both execution modes.

The same test executable dumps all 1,152 output floats in fresh serial and
grouped processes: they match bitwise, SHA-256
`2c0ff44c3f3530dde8a7c790ec5c0623210c5619541d60fbf8a13ecd461b99d9`.
The four existing backbone tests,
an added serial-tail test and two MoE overlap tests run under CTest. Tiny-model
CPU fallback, prefill/state schedules and cross-binary stage/logit checks
remain part of the suite.

All seven GPU CTest cases pass with no skips, both in the initial small check
and under exclusive admission. Tiny-model final prompt logits match the
preceding binary bitwise, and all 147 printed stage sums agree. The full-model
state test passes final-only/mixed head schedules, four-token continuation,
reset and a single-token case.

Prompt-final logits and all four continuation vectors match bitwise across
all eight timing processes and the preceding unmodified HC-direct binary:
646,400 finite FP32 values per comparison. Repeated windows within each
process also match. The concatenated SHA-256 remains
`8c8ea79cda4da03fe1d1bbd8774b17eaf71b1a7602bc0a8952d061a9a37cd470`.

The full-model prompt is BOS followed by `3 + (i * 37) % (vocab_size - 3)` for
positions 1 through 15, then four greedy decode steps. Each process warms
this path once and measures three repeats, resetting model state before
each timed prefill. Order is serial/grouped/grouped/serial with the same
candidate binary, giving six warm windows per arm from two processes in the
initial experiment. Because the initial gain was small, a second sequence in
reverse order (grouped/serial/serial/grouped) added another six windows per
arm and two processes. No results were discarded. Repeated windows are not
independent workload samples. `initial-analysis.json` preserves the first
analysis; `analysis.json` includes both sequences and their separate means.

Both arms use a 62 GiB expert cache, GPU attention, GPU LM head and the
previous optimizations. The preceding HC-direct binary is a separate
correctness reference. Cache-hit and GPU-coverage counters are checked per
window, and prompt-final plus continuation logits must match bitwise.

Small GPU correctness checks can run alongside authorized Brandon work;
their durations are not throughput measurements. Full-model timings require
all existing pipeline locks and idle GPUs. Brandon's `mapping-dose-v1`
acquired the training lock during preparation; admission waited for its
training and evaluation to finish without stopping it. The confirmation
sequence separately reacquired the same locks and checked idle GPUs.

```bash
ctest --test-dir build -V -R 'test_ds41_backbone|test_ds41_moe_overlap'
DS41_GPU=0 DS41_TEST_MODEL=/home/ubuntu/tiny-deepseek41.gguf \
  ctest --test-dir build -V -R '^test_ds41_prefill$'
CUDA_VISIBLE_DEVICES=1 DS41_GPU=1 DS41_GPU_ATTN=1 DS41_VRAM_GB=62 \
  DS41_PROFILE=1 DS41_GROUPED_TAIL=1 ./profile-candidate MODEL.gguf 16 3 output.bin
```

## Artifacts

Isolated build, logs and exact source:
`/home/ubuntu/ntransformer-h100/ds41-tail-20261015`.
Local copy: this directory. Persistent Lambda copy:
`/lambda/nfs/btiny/ntransformer-training/deepseek-tail-20261015`.
The bundle includes commands, environments, raw measurements, output vectors,
source snapshots and hashes. Model assets remain external; the layout file
is the prior ten-shard GGUF header inventory, not a new weight-payload hash.
`archive.json` records the compressed bundle SHA-256; `SHA256SUMS` records
each payload file. Local verification recomputes means, checks cache/GPU/launch
coverage, compares output bytes and matches current source hashes.
