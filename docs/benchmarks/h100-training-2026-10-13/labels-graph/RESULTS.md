# Label fusion and reserved forward/backward replay

Measured on 2026-10-14; archived and reviewed locally on 2026-10-15.

CUDA Graph replay improves mean throughput **3.57% in strict mode** and
**4.45% with the normal attention policy**, against new controls containing
all preceding optimizations. All eight strict 100-update continuations,
including the P2P pair, match the frozen deterministic reference exactly.
Label fusion alone removes 20 kernels per update/rank but does not establish
a consistent throughput improvement. Both changes remain opt-in.

## Work and controls

Same final pretraining checkpoint, 107,604,288 parameters, 30 layers, hidden
576, FFN 1472, GQA 9/3, vocabulary 8192, context 1024. Two H100 SXM GPUs,
FP32 master parameters, BF16 autocast, fused AdamW, restored optimizer/RNG,
16 sequences/rank and two microbatches: 65,536 global tokens per update.
Every comparison checks batch hashes. No architecture, corpus or tokenizer
change is involved.

Controls already enable compatible RMSNorm, native SwiGLU, shared projection
casts, direct GQA, fused RoPE, target-only fused loss, DDP bucket views and
deferred loss readback. Strict controls also enable deterministic algorithms
and omit the audited uninitialized-memory debug fills.

Admission acquired the existing pipeline locks and checked all Brandon stage
statuses plus two idle GPU inventories. The suite began at **05:11:45 UTC**,
after robustness-v1 completed at 05:11:34. It did not interrupt that pipeline.

Fourteen runs contain 100 updates each. Throughput uses offsets 10–99 and
includes data preparation, batch hashing, communication, clipping, AdamW and
logging. Capture, validation and checkpoint persistence are separately timed.
Two 2-update DDP preflights and three 13-update profiled runs are additional
diagnostics, excluded from throughput claims. Repeats reverse variant order.

## Throughput

Global tokens/s, rounded; each cell lists its two independent continuations.

| Policy | Fused-loss control | + fused labels | + labels and graph |
|---|---:|---:|---:|
| Strict | 347,275 / 350,289 | 348,675 / 348,493 | 361,179 / 361,312 |
| Normal | 385,017 / 378,432 | 382,311 / 382,175 | 399,002 / 398,405 |

| Comparison of arithmetic means | Strict | Normal |
|---|---:|---:|
| Control | 348,782 | 381,725 |
| Labels | 348,584 | 382,243 |
| Graph | 361,246 | 398,704 |
| Labels / control | −0.057% | +0.136% |
| Graph / control | **+3.573%** | **+4.448%** |
| Graph / labels | +3.632% | +4.306% |

The graph/control pairs improve 4.00% and 3.15% strict, 3.63% and 5.28%
normal. These are two repeats, not a confidence interval or a convergence
study. Labels/control pairs change sign: +0.40%/−0.51% strict and
−0.70%/+0.99% normal. The single P2P pair is **271,955 → 272,136 (+0.066%)**.
No substantial standalone speedup is attributed to label fusion.

Graph capture takes **0.658–0.739 s** in the four 100-update graph runs;
initialization is 6.30–6.36 s versus 5.60–5.70 s in the eager controls.
Capture is a fixed setup cost, not included in the warmed throughput above.

## What changed

`--fused-loss-labels` prepares the NLL proxy, applies the shifted-label layout
and validates each target inside the existing target-log-probability CUDA
kernel. It removes Python-dispatched predicates, their reduction and proxy
construction. The kernel supports strided labels and preserves ignore-index,
mask and shift semantics. Device validation remains enabled in Release;
invalid negative and out-of-vocabulary targets fail in isolated subprocesses.
Unsupported compute modes use ordinary cross entropy, including direct BF16
without autocast and FP64, which preserve their original loss dtype/reduction.

`--training-graph` captures the model forward and backward before DDP wrapping.
Inputs have persistent addresses; captured temporaries retain their graph
pool; parameters are updated in place. Each microbatch replays one forward
and one backward graph. DDP collectives, clipping, AdamW, finite checks,
data sampling and persistence remain outside capture. The captured callable
uses BF16 autocast with its cast cache disabled, following the
[PyTorch API contract](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.make_graphed_callables.html).

The adapter requires one live forward at a time, a fixed shape/dtype/device,
the original stream and persistent independent parameter-gradient storage.
`zero_grad(set_to_none=False)` is mandatory. It clones the scalar output so
deferred logging cannot read the next replay's loss. Evaluation bypasses the
graph. Parameter names and tied embedding identity are preserved.

This initial graph path covers native pretraining loss only. It rejects the
checkpoint/P2P host schedule rather than capturing Python counters and prefetch
decisions that would not run on replay. Fused labels work with P2P: its 30 peer
blocks (1.055 GiB), two staging blocks, 6,000 writes, 6,000 reads and 5,800
lookahead prefetches per rank are retained. Capturing that schedule is separate
future work; no H100 capability was removed to target a 3090.

## Storage lifetime bug found and corrected

The first small graph test failed gradient accumulation. PyTorch can let an
initially absent `.grad` adopt the tensor returned by backward. Graph replay
returns reusable gradient storage: the next replay overwrites the first
microbatch before accumulation. The CPU reproducer shows contributions 1 and
3 incorrectly becoming **6** when storage aliases, and **4** with an independent
accumulator. This is a correctness defect, not tolerated BF16 variation.

The fixed adapter preallocates separate gradients and rejects a later reset
to `None`. Small and full 30-layer tests now compare the first microbatch,
both accumulated microbatches, deferred losses and weights after each of two
updates exactly, across all **272 unique parameters**. They also exercise
shape rejection, overlapping forward rejection, missing accumulators and the
evaluation fallback. Rejected probe sources/logs are retained in the archive.

The two-update DDP preflight is exact. All six strict native 100-update cases
and both strict P2P cases match weights, optimizer state, RNG, recorded loss/
gradient norm and validation against the original deterministic reference.
Graph runs report 200 forward and 200 backward replays/rank, unchanged
parameter addresses and no live forward at completion.

Normal graph update differences are **1.740% and 1.729%** relative to the first
normal control; label-only differences are 1.714% and 1.715%. Repeating the
normal control itself produces **2.945%**. Graph maximum training-loss deltas
are 0.0002552/0.0002341, and validation deltas 0.00000629/0.00001553.
These pass the unchanged 0.03/0.02/10% continuation gates. The reference
variation is descriptive evidence, not a statistical bound. Instruction
following and long-run convergence were not reevaluated here.

## Trace evidence

Each row covers **two warmed updates on one rank**. Both ranks have the same
counts. Traces are instrumented diagnostics, not timing substitutes.

| Event | Control | Labels | Graph |
|---|---:|---:|---:|
| CPU operator events | 76,008 | 75,944 | 9,078 |
| Direct host kernel launches | 11,700 | 11,660 | 1,728 |
| Graph launches | 0 | 0 | 8 |
| GPU kernel events | 11,700 | 11,660 | 12,208 |

Labels eliminate **40 GPU kernels per two updates**, including predicates,
the target-validity reduction, standalone assertion and proxy creation.
The validation itself remains inside the fused kernel.

Graph replay cuts counted host launch calls, including graph launches, by
**85.1%** relative to labels. CPU operator events fall 88.0%. Captured matrix
operations still execute on GPU; their kernel-name counts are unchanged.
The graph has **548 additional GPU kernels**: 544 FP32 gradient additions,
two multi-tensor zero operations and two additional NCCL all-reduce kernels
in this DDP schedule. An additional 544 scalar-multiply kernels change operand
variant but not count. This is reduced host dispatch with modest extra device
work, not removal of the mathematical training work.

Rank 0's profiled uncovered device envelope is 103.35 ms for labels and
22.50 ms for graph, over two updates. This includes profiler distortion and
host/device scheduling effects; it is not an estimate of achievable speedup.
Synchronization calls remain, and their observed wait durations move as
work is submitted earlier. Only the unprofiled continuations establish speed.

## Memory accounting

GiB per rank on its compute device, during the 100-update training window.

| Mode | Peak allocated | Peak reserved |
|---|---:|---:|
| Strict control, two runs | 11.514 / 11.914 | 11.924 / 12.174 |
| Strict labels, both runs | 11.914 | 12.174 |
| Normal control, both runs | 11.914 | 12.174 |
| Normal labels, both runs | 11.514 | 11.924 |
| Graph, all four runs | **2.534** | **12.281** |

The graph's 2.534 GiB allocated counter **does not mean the model trains in
2.534 GiB of VRAM**. The captured addresses remain in a private pool across
replays; its storage is still reserved even after temporary tensor lifetimes
end. This follows [PyTorch graph memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management).
Reservation rises 0.107–0.357 GiB over the observed eager controls. We make no
graph memory-saving claim. These allocator counters also exclude some runtime
and communication allocations; they are not total board memory readings.

P2P remains approximately 2.743 GiB compute allocation plus 1.055 GiB peer
reservation per process. The tiny label proxy adds 128 KiB at the measured
peak. The larger 0.400 GiB eager allocation variation remains unisolated;
label fusion is not credited with causing or fixing it.

## Reproduction and evidence

Add `--loss-mode fused --fused-loss-labels --training-graph` to the native
experiment command with the preceding flags. Keep the normal/strict policy
explicit. The library is built separately in `build-training-labels`.
Exact commands, sources, build flags, logs and hashes are included here.

Library SHA256: `ffaecbb69a0d27b33aecb7b25b0dc22860f9b3cb8e20604c20449221521b89e6`.
Checkpoint SHA256: `c2d96fa49ba170246c4be6c017ac84321460379d2295472e5a46708094af7764`.
Archive SHA256: `a4e669967903e8ee34f70d3bb98e39cf937cc89f733d455a23db1274f9a4a6bd`.

The archive contains **274 verified payload files plus its manifest**, including
six compressed traces reanalyzed locally with exact agreement. Model weights
remain remote; their hashes are recorded. Current training code matches the
executed snapshot. Current CMake additionally lists four unrelated `ds41`
inference sources; that difference is recorded, and the executed build file
is preserved. No inference edits were reverted.

Machine-readable evidence: [results](results.json), [aggregates](aggregates.json),
[correctness tests](label-graph-tests.json), [storage reproduction](storage-tests.log),
[local verification](local-verification.json), [provenance](provenance.json).

Local files are under `docs/benchmarks/h100-training-2026-10-13/labels-graph`.
The verified archive and shared library also have a persistent copy at
`/lambda/nfs/btiny/ntransformer-training/labels-graph-20261014`, with this report
and the local verification records. Full run directories and checkpoints remain
at `/home/ubuntu/ntransformer-h100/training/labels-graph-20261014` on the instance.

The next measured target is gradient accumulation/DDP dispatch outside the
graph: 1,088 `add_` and 1,088 `div_` CPU calls remain per two updates. A block
schedule could combine that work and recover communication overlap, while
preserving independent lifetimes, exact accumulation order and tied weights.
That is a hypothesis for the next experiment, not a measured improvement here.
