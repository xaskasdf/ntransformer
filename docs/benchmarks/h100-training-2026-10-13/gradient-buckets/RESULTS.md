# Explicit gradient accumulation and communication buckets

Measured and archived on 2026-10-15 on 2× H100 SXM.

The single-bucket variant reaches **403,675 global tokens/s in normal mode**,
**1.27% above the current graph/DDP control**. In strict mode it reaches
**365,529 tokens/s**, 1.14% above stable batched DDP controls. All ten strict
100-update continuations preserve weights, AdamW, RNG and numerical metrics
exactly. The first strict control was anomalously slow; it remains in the
record and its inflated speed ratio is not the headline.

## Same work, three scheduling choices

The experiment retains the completed pretraining checkpoint, optimizer state,
tokenizer, corpus, batch hashes, model and evaluation from the previous suite:
107,604,288 parameters, 30 layers, hidden 576, FFN 1472, GQA 9/3, vocabulary
8192, context 1024, FP32 parameters and BF16 autocast. Batch is 16 sequences
per rank × 2 ranks × 2 microbatches = 65,536 tokens/update. AdamW and clipping
are unchanged. The native controls already contain all preceding optimizations,
including fused labels and captured model forward/backward.

* **Control:** the existing graph adapter with ordinary DDP bucket views.
* **Batched DDP:** `--batched-gradient-reduction` enables the installed runtime's
  `batched_grad_copy` implementation. It groups deferred copies and divides
  whole buckets. This behavior was checked in the
  [matching PyTorch reducer source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/distributed/c10d/reducer.cpp).
* **Explicit buckets:** `--bucketed-graph` obtains the captured gradients with
  `autograd.grad`, then accumulates them with multi-tensor operations into
  independent persistent buffers. It replaces DDP gradient hooks with explicit
  NCCL reductions. A 25 MiB nominal capacity produces 18 buckets; capacity
  512 MiB holds all gradients in one **430,417,152-byte (410.478 MiB)** buffer.

Parameters and tied embedding remain the same objects. `parameter.grad` views
point into the owned buffers. Each update copies the first microbatch and adds
later microbatches in the original order. Division by rank count precedes the
NCCL sum. All communication completes on the owner stream before the unchanged
finite checks, clipping and AdamW. Input and gradient ownership are checked.

This removes zeroing of accumulators that the first microbatch fully overwrites.
For gradient size G, replacing zero-plus-add with a copy removes 2G of logical
memory traffic per rank/update, approximately 0.802 GiB here. This is an
operation-level count, not measured physical HBM traffic.

## Throughput and the slow control

Each native cell contains two 100-update continuations, with reversed order on
the repeat. Global tokens/s are rounded. Warmup offsets 0–9 are excluded;
offsets 10–99 include sampling, hashes, communication, clipping, optimizer and
logging. Validation, capture, persistence and profiler runs are separately timed.

| Policy | Graph/DDP control | Batched DDP | 18 explicit buckets | One explicit bucket |
|---|---:|---:|---:|---:|
| Strict | 313,839 / 361,258 | 361,925 / 360,863 | 364,243 / 364,344 | 365,574 / 365,483 |
| Normal | 398,402 / 398,829 | 399,398 / 399,415 | 402,287 / 402,328 | 403,527 / 403,823 |

Normal means are **398,615 / 399,406 / 402,308 / 403,675**. The single bucket
improves 1.27% against ordinary graph/DDP and 1.07% against batched DDP. Its
two ordinary-control paired gains are 1.287% and 1.252%.

Strict means for batched DDP, 18 buckets and one bucket are **361,394 /
364,293 / 365,529**. The explicit variants improve 0.80% and 1.14% against
batched DDP. The reverse-order ordinary-control pair improves **1.17%** with
one bucket. The raw strict mean ratio against both ordinary controls is
8.29%, and the first pair is 16.48%; neither is a reliable isolated estimate
because the first control ran at 313,839 rather than about 361,000 tokens/s.

The first control's median step was 208.15 ms (range 199.69–221.89), versus
180.72 ms for the next batched run. Median forward stream times were both
47.22 ms, with backward 128.76/127.54 ms. These observations do not identify
the cause, and stream durations cannot be subtracted to assign CPU overhead.
All samples are preserved. Two repeats establish a small measured gain here,
not a confidence interval or a hardware-independent guarantee.

The single strict P2P pair is **271,642 → 272,438 (+0.293%)** for batched DDP.
It preserves the activation pool and transfer schedule. We do not claim a
substantial general P2P speedup from that one pair.

An additional strict confirmation pair was prepared after noticing the slow
control. Brandon's `robustness-replication-v1` acquired the shared pipeline
lock after our main suite completed. The confirmation launched no GPU work;
only its waiting helper and the waiting archive helper were stopped. The
18 completed runs were then verified and sealed. No Brandon process was
stopped. This deferred pair is recorded in [status.json](status.json).

## Correctness

Small two-layer and complete 30-layer tests compare graph/DDP-style local
accumulation with both bucket capacities. They require exact first-microbatch
and accumulated gradients, deferred scalar losses, clipping norm, final weights
and AdamW over two updates. The full check covers all **272 unique parameters**.
Missing accumulator ownership and premature reduction are rejected.

Four two-update distributed preflights cover ordinary DDP, batched DDP and
both explicit layouts; all three candidates match the preflight control
exactly. Eight strict native 100-update runs and two strict P2P runs match the
frozen deterministic reference in weights, optimizer/RNG, recorded loss and
gradient norms, and final validation. No acceptance limit was widened.

Normal relative update differences against the first normal control are:

| Variant | First run | Repeat |
|---|---:|---:|
| Batched DDP | 2.321% | 2.641% |
| 18 explicit buckets | 2.640% | 2.673% |
| One explicit bucket | 2.575% | 1.763% |
| Ordinary control repeat | — | 2.659% |

The one-bucket maximum training-loss deltas are 0.0001668/0.0002058; validation
deltas are 0.00001587/0.00003267. These pass the unchanged 0.03 training-loss,
0.02 validation-loss and 10% relative-update gates. Reference repetition is
descriptive variation, not a statistical bound. This is not a new instruction-
following or long-run convergence evaluation.

The initial test attempt failed because the test compared a CUDA optimizer
state with a CPU reference. The comparison was corrected to copy both snapshots
to CPU; no training arithmetic changed for that fix. Its log and exact source
snapshot are retained under `rejected-probes/gradient-buckets-20261015`.

## What the traces establish

Eight traces cover two warmed updates per rank in four separate 13-update
strict runs. Both ranks have matching counts. Profiles are diagnostics and
are excluded from throughput claims.

| Events per two updates/rank | Control | Batched DDP | 18 buckets | One bucket |
|---|---:|---:|---:|---:|
| CPU operator events | 9,078 | 8,058 | 5,314 | 5,212 |
| Direct kernel launches | 1,728 | 1,218 | 102 | 68 |
| Graph launches | 8 | 8 | 8 | 8 |
| GPU kernel events | 12,208 | 11,698 | 10,582 | 10,548 |
| CPU `aten::add_` events | 1,088 | 1,088 | 0 | 0 |
| CPU `aten::div_` events | 1,088 | 68 | 0 | 0 |

Including graph launches, the one-bucket path reduces host launch calls
**1,736 → 76 (−95.62%)**, CPU operator events 42.59%, and GPU kernel events
13.60%. Matrix kernel-name counts, including cuBLAS split-K helpers, remain identical across all four
variants. Mathematical model compute is preserved; fewer host calls do not
translate into proportionally faster training.

The explicit paths add two grouped copies, two grouped accumulations and two
grouped divisions per profiled window. Other optimizer multi-tensor operations
remain. One gradient bucket needs **100 gradient all-reduces per rank over
100 updates**, versus 1,800 with 18 buckets. Metric/validation collectives are
additional. Profiling records six total NCCL kernels for the one-bucket window,
versus 40 for 18 buckets, including the per-step metric collectives.

All four traces transfer 1,048,624 bytes host→device and 74 bytes device→host
per rank/window, plus 32 bytes of device copies. The gradient payload remains
on GPU. This change removes per-parameter dispatch and accumulation work; it
does not claim to have removed a previous CPU copy of the gradients.

## Memory, scope and remaining limits

| Native mode | Peak allocated GiB/rank | Peak reserved GiB/rank |
|---|---:|---:|
| Graph/DDP and batched DDP | 2.534 | 12.281 |
| 18 explicit buckets | 2.134 | 12.203 |
| One explicit bucket | 2.133 | 12.193 |

Reserved allocator memory saves **80 MiB** with 18 buckets and **90 MiB** with
one bucket. The allocated counter excludes retained graph temporary lifetimes;
2.133 GiB is not the VRAM required for training. Runtime/communication allocations
outside the allocator are also excluded. P2P keeps its additional 1.055 GiB
peer reservation and approximately 3.14 GiB compute allocation in this pair.

The explicit adapter is first order, with fixed graph inputs, contiguous FP32
parameters, a fixed microbatch count and owner stream. It assumes the same
checkpoint is restored on all ranks, as this trainer verifies. It does not
implement general DDP initialization broadcasts, join, unused parameters,
arbitrary hooks or higher-order differentiation. AdamW and finite checks remain
eager. All changes are opt-in.

The whole backward graph releases all gradients together; this experiment
does not overlap reduction with layer computation. The single bucket is best
among these tested layouts, not a universal choice. Capturing the existing
P2P host schedule and segmenting backward for earlier communication are separate
tasks; P2P hardware support remains intact.

## Reproduction and stored evidence

Add `--bucketed-graph --gradient-bucket-mb 512` to the existing native command
with `--training-graph`, fused labels and prior optimizations. The batched DDP
alternative uses `--batched-gradient-reduction` and is mutually exclusive with
explicit buckets. Exact commands are in [run-gradient-buckets.py](run-gradient-buckets.py).
No new CUDA library build was needed.

Library SHA256: `ffaecbb69a0d27b33aecb7b25b0dc22860f9b3cb8e20604c20449221521b89e6`.
Checkpoint SHA256: `c2d96fa49ba170246c4be6c017ac84321460379d2295472e5a46708094af7764`.
Archive SHA256: `cde7a77a3f0d177780b09ab72e1e23082dad194823cea67d341cd5fb424acf6e`.

The archive contains **370 verified payload files and its manifest**, including
eight compressed traces reanalyzed locally with exact agreement. Current
training sources match the executed snapshot. Four unrelated DS41 inference
source additions in current CMake are recorded separately; the executed CMake
file and library build record are preserved. Checkpoint files remain remote,
with their hashes included.

Evidence: [results](results.json), [aggregates](aggregates.json),
[gradient tests](gradient-bucket-tests.json), [verification](local-verification.json),
[provenance](provenance.json). Local directory:
`docs/benchmarks/h100-training-2026-10-13/gradient-buckets`.
The verified archive/library and report also have a persistent Lambda copy at
`/lambda/nfs/btiny/ntransformer-training/gradient-buckets-20261015`.
Full run directories remain at
`/home/ubuntu/ntransformer-h100/training/gradient-buckets-20261015-r2`.
