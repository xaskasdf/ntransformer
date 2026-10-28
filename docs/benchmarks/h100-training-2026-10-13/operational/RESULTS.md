# Removing operational work from Brandon training

The complete [dataflow review](../../../H100_TRAINING_DATAFLOW.md) traces storage,
RAM, VRAM, dtype/layout conversions, activation lifetime, backward, DDP, clipping,
AdamW and persistence. It distinguishes required dependencies from repeated
materialization and CPU coordination. The first measured removals improve the
native deterministic trainer by **19.46%**, with exact numerical equivalence in
two independent 100-step runs. The normal attention policy improves by **2.45%**.

## What changed

All three changes are opt-in in `scripts/train_brandon_experiment.py`:

* `--skip-uninitialized-fill`, together with `--deterministic`, disables the
  debug filling of uninitialized output buffers. Strict deterministic algorithms
  remain enabled; required zero initialization is retained.
* `--gradient-bucket-views` keeps DDP gradients as views of communication buckets,
  avoiding the copy of reduced gradients back into separate tensors.
* `--defer-loss-readbacks` reads both losses together after accumulation.
  Finiteness is still checked before clipping and AdamW; backward no longer
  waits for the CPU to inspect each individual forward loss.

Model, native kernels, master-weight precision, optimizer, global batch, sampling
and learning rate are unchanged. This remains a hybrid native/PyTorch trainer.
CUDA Graphs, shared activation casts, direct grouped-KV attention and fused RoPE
are research directions identified by the trace, not implemented results here.

## Throughput and memory

All numbers are **global tokens/s across two H100 SXM5 GPUs**. Each case restores
the same completed pretraining checkpoint, including AdamW and RNG state, then
takes 100 updates of 65,536 tokens. Steps 11–100 form a continuous measured
window including data preparation, hashing, communication, logging and timing.
Initialization, initial/final validation and checkpoint writing are separate.

| Policy and variant | Global tokens/s | Compute-device peak allocated GiB/rank | Exact deterministic audit |
|---|---:|---:|---|
| Deterministic native, original filling | 266,319 | 14.542 | Pass |
| Deterministic native, no debug filling | 308,753; 310,202 | 14.943 | Pass, both runs |
| Above + bucket views only | 310,875 | 14.943 | Pass |
| Above + deferred readbacks only | 314,281 | 14.542 | Pass |
| All three removals | 318,488; 317,799 | 14.943 | Pass, both runs |
| Normal policy, original trainer | 334,314 | 14.943 | Different attention policy |
| Normal policy + bucket views/readbacks | 342,491 | 14.943 | Default-policy numerical gate passed |
| Deterministic local recomputation, no debug filling | 236,025 | 4.878 | Pass |
| Deterministic P2P, no debug filling | 236,899 | 3.514 + 1.055 peer | Pass |
| Deterministic P2P, all three removals | 239,923 | 3.512 + 1.055 peer | Pass |

The no-fill pair averages 309,478 tokens/s: **+16.21%** versus the contemporaneous
filled deterministic run. The combined pair averages 318,144: **+19.46%** versus
that run, and **+2.80%** beyond no-fill alone. These are descriptive averages of
two runs, not confidence intervals. The other conditions each ran once.

The deterministic combined mode remains **7.11% slower than the newly optimized
normal mode**, or 4.84% slower than the previous normal mode. Speed has been
substantially recovered; the deterministic cost has not disappeared. Strict mode
selects deterministic Flash SDPA in this environment, while normal mode uses
cuDNN SDPA. Different attention policies must remain explicit in comparisons.

There is **no new peak-memory reduction for the native normal mode**. Removing
debug fills increases the measured deterministic allocator peak by approximately
0.400 GiB relative to the filled run; removing copies does not guarantee a lower
peak across asynchronous tensor lifetimes. The table reports observations
without assigning that difference to an unmeasured cause. In bidirectional P2P,
each physical GPU also stores the opposite rank's peer blocks. Adding separate
allocator peaks is an upper bound, not a simultaneous whole-device reading.

## Numerical acceptance

Ten deterministic 100-step cases all pass `--require-exact` against the archived
PyTorch deterministic reference: 273 state-dict entries (272 unique parameters),
AdamW state, CPU/CUDA RNG state, every recorded loss/gradient norm/LR/token count,
and final per-source validation. Batch hashes match. File hashes can differ
because the checkpoint containers also record the selected backend options.
The comparison JSON files' throughput ratios use that older PyTorch reference;
the ablation ratios above instead use this review's fresh native controls.

The normal-policy pair passes the unchanged numerical gate:

* Maximum training-loss difference: **0.000173390**.
* Final mean validation difference: **0.0000308193**.
* Parameter discrepancy divided by reference-update L2: **1.71715%**.
* Maximum absolute weight difference: **0.000325159**; RNG states remain equal.

This is comparable to the previously observed 1.729% and 1.774% variation between
independent PyTorch reference repetitions. It is accepted within that observed
scale for this experiment, supported by exact comparisons under a common strict
policy. It is not 1.717% of the full weight norm and not proof of long-run
convergence. The fully fused RMSNorm variant's earlier failed SFT audit remains
separate. This review does not produce a new instruction-following score.

## Work actually removed

The following counts are from **two warmed updates of rank 0**, captured in
separate profiling runs at offsets 11 and 12. Profiling rates are excluded from
the throughput table. Both ranks' full compressed traces are preserved locally.

| Observed operation | Deterministic filled | No-fill | Combined |
|---|---:|---:|---:|
| NaN-fill calls | 7,960 | 0 | 0 |
| Logical bytes written by those NaN fills | 146,976,392,496 | 0 | 0 |
| Required zero-fill calls | 130 | 130 | 130 |
| Logical bytes written by zero fills | 6,752,986,112 | 6,752,986,112 | 6,752,986,112 |
| `aten::mm` calls | 2,532 | 2,532 | 2,532 |
| CUDA kernel events | 22,076 | 13,958 | 13,944 |
| D2D memcpy calls | 544 | 544 | 0 |
| D2D memcpy bytes | 860,834,304 | 860,834,304 | 0 |
| GPU→CPU memcpy calls | 18 | 18 | 12 |
| GPU→CPU memcpy bytes | 78 | 78 | 74 |
| `cudaStreamSynchronize` calls | 30 | 30 | 24 |

The debug fills account for **68.441 GiB of logical writes per step and rank**.
That is a buffer-write inventory, not measured HBM traffic; caching matters.
The eliminated D2D copies correspond to **410.478 MiB per step and rank**.
GPU→CPU traffic was already tiny in bytes: the benefit of grouping it concerns
coordination and dependency placement rather than link bandwidth.

The zero fills needed by the computation remain. The no-fill experiment removes
prewrites whose outputs are replaced by the subsequent producer; it does not
remove gradient initialization or relax deterministic algorithm selection.
[PyTorch documents this distinction](https://docs.pytorch.org/docs/2.14/notes/randomness.html#filling-uninitialized-memory).

The warmed traces also show no `cudaMalloc` events, whereas the initial cold trace
contained 370. Consequently, the initial allocator costs are not attributed to
every steady-state update. Batch preparation totals approximately 7.8–8.3 ms and
hashing approximately 1.0 ms across the four profiled microbatches per rank.
These instrumented timings motivate a prepared input ring, but do not establish
an end-to-end speedup for one. The larger unimplemented candidates remain shared
activation conversion, RoPE temporaries, expanded KV heads and materialized logits.

## Design references

The project author identifies Karpathy's work with TinyStories and nanochat as
design inspirations. The [dataflow review](../../../H100_TRAINING_DATAFLOW.md#referentes-de-diseño)
records a source-level comparison with nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`: explicit activation precision,
persistent input buffers, host dispatch, distributed optimizer scheduling and
grouped attention. These are design references, not additional measured speedups;
this experiment ran Brandon only. Its frozen arithmetic and AdamW contract remain
the basis for all comparisons above.

## Reproduction and artifacts

Use the guarded, frozen-source runners in
[`fill-review/run-operational-review.py`](fill-review/run-operational-review.py)
and [`ablations/run-operational-review.py`](ablations/run-operational-review.py).
They waited for the existing Brandon pipeline locks and idle GPUs; the separate
`retention-v1` experiment completed before this review started. The ablation
suite required the first suite and its exact audit to complete successfully.

The deterministic combined flags, appended to the common launch command in the
[training guide](../../../H100_TRAINING_EXPERIMENT.md), are:

```bash
--mode native --norm-mode compatible --steps 100 --warmup-steps 10 \
  --deterministic --skip-uninitialized-fill \
  --gradient-bucket-views --defer-loss-readbacks
```

For normal mode omit `--deterministic --skip-uninitialized-fill`; for P2P select
`--mode peer`. The existing defaults are preserved so baseline comparisons remain
reproducible. `--profile --profile-start 11 --steps 13` produces the separate
warmed trace, with the same library environment and GPU admission gate.

Local records include summaries, per-step/rank metrics, batch hashes, exact and
normal-policy comparisons, Python source snapshots, execution scripts,
checkpoint/library hashes, and the compressed traces. All **203 payload files**
were checked against their manifests, as were the downloaded archives:

* Fill review archive SHA-256: `d3d5f2fb5315e97ceee017162b8757563cd33f30dc2128a183238f1b9a6187de`.
* Ablation archive SHA-256: `5916e7c5c53488ef994f86a13f8aee54db5a20c189af922d516e33d406f0a13e`.

The CUDA training library is unchanged. CPU validation passed the four measurement
tests, five queue tests and six exact-state audit tests. The audit tests used the
remote PyTorch environment because the local Windows Python lacks PyTorch.
The trace analyzer was reproduced locally against both compressed initial traces.

Full model/optimizer checkpoints remain external under
`/home/ubuntu/ntransformer-h100/training/operational-20261013` and
`/home/ubuntu/ntransformer-h100/training/operational-ablation-20261013`.
