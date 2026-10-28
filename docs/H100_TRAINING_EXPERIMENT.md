# H100 training accelerator experiment

This experiment starts a training backend for NTransformer on the two H100 SXM5
GPUs at `ubuntu@<h100-host>`. Its purpose is to measure full-model training
correctness, fused CUDA compute, activation memory, and a speculative P2P DMA
pipeline. Hardware capabilities determine the implementation; adapting it to a
3090 is a later portability exercise.

The user authorized a 100-step continuation of the current Brandon Mini
experiment, on the existing branch. The original pretraining, SFT and diagnostic
pipeline must finish successfully before any new GPU training or validation runs.

The current compatible RMSNorm/SwiGLU backend achieved **334,814 versus 301,551
global tokens/s (+11.0%)**, with peak allocated memory falling from **18.44 to
14.94 GiB per GPU** in the paired 100-step pretraining continuation. A separate
100-step instruction adjustment reproduced every final weight, AdamW state,
RNG state and recorded loss exactly. That checkpoint matches the previously
scored reference: **56/120** transformations and **0/60** original instructions.
The earlier fully fused RMSNorm reached +25.8% in pretraining but failed the
fresh-optimizer SFT check; it remains an explicit diagnostic option.

## Current implementation

The experimental backend is hybrid. NTransformer supplies native RMSNorm forward
and backward, native FP32/BF16 SwiGLU forward and backward, whole-block
recomputation, a pool of peer GPU activation blocks, and an event-ordered DMA
pipeline. PyTorch supplies GEMM, attention, loss, autograd scheduling, AdamW and
DDP. Every model parameter is trained, including the tied embedding/output matrix.
This is distinct from the existing inference executable and its read-only GGUF
weight cache.

`src/training/ops.cu` has a C ABI with caller-owned allocations and CUDA streams.
RMSNorm defaults to native pointwise kernels with PyTorch reductions. Its two
input-gradient paths stay separate until autograd accumulates them alongside
residual gradients, preserving the rounding order validated for SFT. The
optional `--norm-mode fused` reuses the inference forward kernel and adds
input/weight derivatives with a deterministic two-stage reduction.
SwiGLU preserves BF16 rounding between SiLU and multiplication, including the
corresponding intermediate in backward. Training excludes inference's
`--use_fast_math`: overriding fmad/ftz/div/sqrt individually did not prevent its
approximate `expf` mapping from changing small BF16 gradients. A regression test
now uses real SFT values at the affected rounding boundary. Inference retains
its existing fast-math flag. Norm inputs and master parameters are
FP32; GEMM and attention run under BF16 autocast.

`python/ntransformer_training/pipeline.py` reserves all checkpoint input blocks
on the peer GPU before training. On the compute GPU it reserves two staging
blocks, plus separate write and read streams. No block allocation occurs inside
the peer save/acquire/release path. The autograd engine and recomputation still
allocate their ordinary working tensors.

For each microbatch:

1. Forward saves each block input to peer VRAM with `cudaMemcpyPeerAsync`, after
   an event marking production of that input. The original storage remains live
   until the asynchronous copy finishes.
2. Backward fetches the last block's input, then speculatively fetches the next
   block in reverse order into the other staging buffer.
3. Recomputation and backward consume one buffer while DMA prepares the other.
   A consumed event protects reuse; a ready event protects reading.
4. All block generations must be released before the next microbatch. A stale
   generation or an attempt to overwrite a live buffer raises an error.

This first scheduler uses a known sequential layer order and one-layer lookahead.
It reserves enough blocks for the requested shape plus a 2 GiB free-memory margin
on the peer. It does not yet implement a dynamic admission controller or stream
optimizer states/gradients. Parameters and optimizer state stay on the compute
device, and only checkpoint inputs move over P2P.

The backward implementation returns every parameter gradient through the outer
autograd graph so DDP accumulates/reduces each gradient once. Dropout, shared
blocks, recurrent loops, MTP, register tokens and the other Brandon variants are
rejected in this experiment. Higher-order gradients and `torch.compile` are
outside this first backend.

## Controlled continuation

The source is the completed `pretrain/last.pt` from
`/lambda/nfs/btiny/brandon-mini/stages/stage1-v1`. The SFT stage and its diagnostics
also finish before the experiment starts; the continuation itself uses the
pretraining objective and corpus. It does not overwrite or resume inside the
original run directory.

| Setting | Value |
|---|---|
| Model | Brandon Mini, 107,604,288 parameters, 30 layers |
| Hidden and FFN dimensions | 576 and 1472 |
| Attention | 9 query heads, 3 KV heads, head dimension 64 |
| Vocabulary and context | 8192 and 1024 |
| Master weights | FP32 |
| Matrix compute | BF16 autocast |
| Optimizer | Restored AdamW state, original betas and weight decay |
| Learning rate | Final pretraining schedule floor, 0.00003 |
| DDP batch | 2 ranks × 16 sequences × 2 microbatches |
| Global tokens per optimizer step | 65,536 |
| Continuation | 100 steps per case, 6,553,600 tokens per case |
| Throughput sample | Steps 11–100; the first 10 updates are warmup |

All cases start independently from the same final checkpoint. The original
settings, tokenizer/corpus identities and rank RNG states are restored. Sampling
continues at the next original step, using Brandon's deterministic
seed/step/rank/microbatch sampler. The completed cosine schedule remains clamped
to its final learning rate; this is an explicitly new continuation experiment,
not an exact resume of a run whose planned length has been changed.

## Comparisons

The immutable historical reference at step 25,400 is preserved in
[baseline-stage1-20261013-a](benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a/REPORT.md),
including its JSON, original metrics, CSV curves, source snapshot and checksum
manifest. Its recent global throughput is **307,985.16 tokens/s**; aggregated
training-window throughput is 306,373.61 tokens/s and effective throughput since
stage launch is 298,102.34 tokens/s. Median/P95 of its 100-step window means are
212.83/214.67 ms. Those percentiles are not individual-step tail latencies.
The slow step-8,700 window also appears in the
[training snapshot captured during the earlier 8B/P2P experiments](benchmarks/h100-2026-10-13/training_after.jsonl).
That overlap is relevant workload contention; the recent stable 1,000-step
segment is the historical speed reference. The immutable series retains all
windows, including the slower ones.

The runner verifies the architecture, original settings, corpus identity, source
hashes and PyTorch/CUDA-build/NCCL versions against that reference before
training. Both implementations still start the controlled continuation from the
same final weights and AdamW state at LR 3e-5. Historical ratios are descriptive;
the speedup attributed to an implementation uses the paired continuation under
the same measurement code. A 1.25x historical rate would be 384,981 tokens/s;
1.5x would be 461,978 and 2x would be 615,970. These are comparison thresholds,
not performance forecasts.

The historical validation mean at step 25,000 is 2.309738, with FineWeb-Edu
2.788890 and Cosmopedia 1.830586. The continuation evaluates those same fixed
source samples and reports each source separately. Equal-work correctness is
the scope of the 100-step comparison. Quality at an equal time budget and time
to a quality threshold require a subsequent longer controlled run; the previous
curve cannot establish those properties for the new implementation.

| Case | Compute placement | Kernels | Saved block inputs |
|---|---|---|---|
| `ddp-reference` | Both GPUs, DDP | Original PyTorch | Original activation graph |
| `ddp-native` | Both GPUs, DDP | Native RMSNorm/SwiGLU | Original activation graph |
| `ddp-checkpoint` | Both GPUs, DDP | Native RMSNorm/SwiGLU | Local VRAM, recomputed blocks |
| `ddp-peer` | Both GPUs, DDP | Native RMSNorm/SwiGLU | Opposite GPU, recomputed blocks |
| `single-checkpoint` | GPU 0 | Native RMSNorm/SwiGLU | Local VRAM, recomputed blocks |
| `single-peer` | GPU 0; GPU 1 stores activations | Native RMSNorm/SwiGLU | GPU 1, recomputed blocks |

The single-compute-GPU cases consume the same sampled global batch by iterating
four microbatches with the original two logical rank identities. Their measured
throughput is compared against each other. DDP peer storage exchanges activation
capacity in both directions and does not reduce aggregate storage by itself;
the single-compute-GPU pair measures the capacity tradeoff more directly.

One input block occupies 36 MiB at batch 16, context 1024 and dimension 576.
Thirty inputs reserve 1080 MiB on the peer, and two staging buffers reserve
72 MiB on the compute GPU. Each microbatch transfers 2160 MiB in total across
the save/read directions. These are calculated sizes; the recorded DMA counters
and allocator peaks provide the actual run accounting.

The primary throughput is global tokens divided by one continuous measured
window, including data sampling, batch hashing, transfer, compute,
synchronization, gradient clipping, optimizer update, instrumentation and
logging. It is not a mean of step rates or a sum of timings with the gaps removed.
Validation and checkpoint serialization are excluded. Initialization, initial
and final validation, total training and checkpoint/batch-record writing are
timed separately. The wall-clock scope begins at Python entry before importing
PyTorch and ends after checkpoint serialization; it excludes torchrun launch.
Forward, backward and optimizer CUDA stream intervals are recorded
separately; those intervals can include waits and host launch gaps and are not
isolated kernel execution time. Peak allocated/reserved memory is recorded per
process and device, with `nvidia-smi` sampling as a separate process-level record.
Historical memory/power observations remain samples, not historical peaks or
energy estimates. GPU utilization is not a measure of FLOP efficiency.
Two short profiler runs follow the comparisons; their rates are excluded from
the throughput table.

## Validation and launch gate

CPU validation already passed:

- Whole-block recomputation matches reference loss, every gradient and one
  AdamW update exactly on a small Brandon model.
- Two-process Gloo DDP with gradient accumulation matches all weights and
  gradients exactly after three updates.
- Five CPU gate tests reject unfinished/failed/inconsistent pipeline states.
- Four measurement tests verify the complete historical snapshot's hashes,
  reject configuration/software/data drift and check continuous-window timing.
- The native training library compiles with the existing isolated GCC 14 and
  CUDA 13.1 toolchain.

GPU validation passed after the existing pipeline completed at 18:28 Santiago.
The first queued attempt failed while loading the shared library, before any
training update, because the CUDA link driver omitted the C++ runtime
(`__gxx_personality_v0`). The target now links with the C++ driver and rejects
undefined symbols at build time. The failed attempt is preserved separately;
the successful retry uses `experiment-20261013-r2`.

Validation checks RMSNorm against
independent FP64 derivatives and finite differences, FP32/BF16 SwiGLU against
PyTorch, stream behavior, full 30-layer model loss/gradients/AdamW, tied weights,
and two complete peer-pool generations. Peer gradients must also agree directly
with the local recomputation variant. A two-step GPU DDP continuation compares
the reference and peer backend before any 100-step run begins.

Measured full-model gradient relative L2 differences were approximately
1.66e-6 in FP32 and 0.00748 in BF16. Local recomputation and P2P satisfied the
direct gradient comparison. The BF16 results are numerical agreement within the
declared tolerance, not bitwise identity.

The supervisor requires `status.json` to report the entire original pipeline
complete, consistent pretraining completion metadata, the original exclusive
pipeline lock, and two idle-GPU observations. If the original pipeline fails,
it records a blocked experiment. A failed validation stops the experiment.
Only a process group created by this supervisor can be terminated on timeout.

Full-model gradient relative L2 limits are 0.002 for FP32 and 0.03 for BF16.
For continuation comparisons, maximum training loss difference must be at most
0.03, final validation mean difference at most 0.02, and the final parameter
discrepancy at most 10% of the reference update norm. These are fixed acceptance
limits, not measured precision claims. Every checkpoint tensor must have changed
and remain finite. Batch hashes must match across all six cases.

## Build and reproduce

Build with the normal Linux compiler/toolkit arguments and the additional option:

```bash
cmake -S . -B build-training -DNTRANSFORMER_BUILD_TRAINING=ON \
  -DNTRANSFORMER_BUILD_NTRPU_SIM=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=gcc-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build-training --target ntransformer_training -j 2
ctest --test-dir build-training -R test_training_queue -V
export NTRANSFORMER_TRAINING_LIB="$PWD/build-training/libntransformer_training.so"
```

Use the existing Brandon Python environment; the C ABI avoids rebuilding or
installing PyTorch. The native inference executable does not gain a training
flag. Its existing build and models remain independent.

```bash
python scripts/queue_brandon_training.py \
  --pipeline /lambda/nfs/btiny/brandon-mini/stages/stage1-v1 \
  --brandon-root /lambda/nfs/btiny/brandon-mini/sources/stage1-v1 \
  --output /home/ubuntu/ntransformer-h100/training/experiment-20261013
```

The supervisor holds an experiment lock, refuses an output directory that already
has a status, and writes status updates atomically. Input/checkpoint/source hashes
are captured before training. Local retrieval is read-only over SSH:

```powershell
$env:NT_REMOTE_HOST = "ubuntu@<h100-host>"   # or pass --host
python scripts/fetch_training_results.py --watch `
  --remote /home/ubuntu/ntransformer-h100/training/experiment-20261013 `
  --output docs/benchmarks/h100-training-2026-10-13
```

The local directory receives live `remote_status.json` and, at completion or
failure, the portable records archive and a generated `RESULTS.md`. Checkpoints
and large Chrome trace files remain under the remote experiment directory.

## Next optimization decisions

### First completed H100 measurements

The current precise-math binary has its own
[paired measurement](benchmarks/h100-training-2026-10-13/precise/RESULTS.md):
301,703.8 tokens/s reference versus 379,539.6 native, with mean step times
217.220 and 172.672 ms. The maximum allocated peak across compute GPUs was
18.036 versus 14.539 GiB. Final validation means were 2.1700738 and 2.1700669;
parameter discrepancy was 1.735% of the reference update norm. The corrected
library passed primitive, full-model and P2P validation, including the new
small-gradient rounding regression.

The initial six 100-step continuations completed on 2026-10-13. Portable records,
per-step timings, comparison results and profiler tables are local in
[`retry-r2`](benchmarks/h100-training-2026-10-13/retry-r2/RESULTS.md).
All variants used the identical sampled global batches and all five candidates
passed the fixed loss/validation/update checks against the paired reference.
These are one short run per configuration in a fixed order, not replicated
performance estimates or long-run convergence evidence.
This first table uses the initial library, before the subsequently discovered
SwiGLU rounding fix. The corrected library is measured separately under
`benchmarks/h100-training-2026-10-13/precise`; do not silently attribute these
initial measurements to a different binary.

| Mode | Global tokens/s | Ratio to same-topology reference |
|---|---:|---:|
| DDP reference | 302,255 | 1.000 |
| DDP native norms/SwiGLU | 381,597 | 1.263 |
| DDP native + local recomputation | 292,320 | 0.967 |
| DDP native + peer recomputation | 286,946 | 0.949 |
| One compute GPU + local recomputation | 149,873 | 1.000 |
| One compute GPU + peer recomputation | 149,317 | 0.996 |

Native DDP reduced peak PyTorch allocated memory from 18.04 to 14.94 GiB on
each compute GPU (17.2%); these are allocator peaks, not whole-process
`nvidia-smi` measurements. Its final validation mean was 2.170049 versus
2.170089 for the reference. Maximum training-loss difference was 0.0001621;
parameter discrepancy was 1.74% of the reference update L2 norm. All 273
state-dict tensors changed and remained finite. That tensor count includes tied
state-dict entries and is not the unique parameter count.

On one compute GPU, peer storage reduced its allocated peak from 4.079 to
3.107 GiB, using 1.055 GiB on the peer. It reserved 30 input blocks of 36 MiB
and two local 36 MiB staging buffers. Across 400 microbatches, the counters
recorded 12,000 writes, 12,000 reads, 11,600 lookahead prefetches and 843.75 GiB
transferred in both directions combined. Throughput was 0.37% below local
recomputation in this short measurement. Peer memory is additional capacity;
this does not reduce the sum of physical memory in use.

The separate two-step profile records 480 P2P copies with 47.7 ms total copy
duration. Profiler sums do not establish critical-path cost or an overlap
percentage; profiled throughput is excluded from the table. Full Chrome traces
remain on the instance, while profiler tables are in the local records.

For this fitting model the current fastest route is native DDP without
recomputation. P2P and recomputation provide a measured capacity tradeoff for
larger models/contexts. Adaptive selection should use memory requirements and
timing rather than enabling every mechanism unconditionally.

The first measurements determine whether fused elementwise kernels reduce step
time, how much memory recomputation recovers, and whether peer DMA is hidden by
backward compute. Native GEMM/attention training paths, tensor/pipeline
parallelism, sharded optimizer state, adaptive block admission and CPU/NVMe
fallbacks are subsequent work. Their scheduling must account for mutable weights
and optimizer updates, unlike inference caching. The H100 implementation is the
starting point; hardware-specific workarounds follow measured requirements.

## Verifiable instruction-following pilot

The original completed run reached pretraining validation loss 2.1707469 and
SFT validation loss 1.3737701, but strict instruction following remained 0/60.
That baseline is preserved under
`docs/benchmarks/h100-training-2026-10-13/baseline-stage1-final`.
Lower cross-entropy alone is not the success criterion of this next experiment.

`run_verified_instructions.py` runs after the numerical/performance experiment,
with the same original pipeline lock and idle-GPU check. It starts both reference
and native variants from the original `sft/best.pt`, using a new, identical AdamW
state for the new synthetic curriculum. `train_verified_instructions.py` wraps
the frozen Brandon trainer, retaining its supervised-token normalization across
variable sequence lengths, accumulation steps and DDP ranks. The initial run
changed both RMSNorm and SwiGLU. The current runner defaults to
`--native-mode native --norm-mode compatible`; `--native-mode swiglu` retains
reference RMSNorm. The older combined variant remains available explicitly
with `--norm-mode fused`. Batch hashes include every
train/validation fetch on each rank. The numerical audit is now part of the
supervisor's completion gate, and its failure makes the experiment fail.

The curriculum is the existing NTransformer teacher's verified output under
`/lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered`: 391 training
examples, 78 accepted validation examples. The 391 examples span uppercase (76),
join (78), replacement (79), marker removal (80) and brackets (78). All JSON
examples failed teacher filtering, so that family has no training examples.
No rejected answer is repaired or used as a training target. IDs, prompts,
expected answers and immutable file hashes are checked before launch.

Both variants train for 100 updates: batch 8 per GPU, two accumulation steps,
two GPUs (32 examples per update), BF16 autocast/FP32 parameters, fused AdamW
at 5e-5, 10 warmup steps, cosine decay over the final 80%, floor 10%, weight
decay 0.01 and clipping 1.0. Seed is 20261013. Validation loss is sampled at
steps 0, 50 and 100. The final step-100 checkpoint is evaluated without choosing
it using generation scores. These short, variable-length SFT tokens/s are not
comparable to the context-1024 pretraining rates.

The same frozen evaluator scores the initial checkpoint and both final ones:

- Every one of the 120 held-out transformation requests, including teacher
  failures and the 20 JSON cases. Generation limit is 64 tokens for all three
  checkpoints. Report each family as well as the total; these are held-out inputs
  and paraphrases, not an unseen-skill benchmark.
- The original, unchanged 60-case instruction diagnostic, with its original
  32-token limit. Its templates never enter this training set.

Both use greedy decoding, a required stop token and case-sensitive exact match
with only outer whitespace ignored. Saved records include responses, stop flags
and expected answers. Exact text without a stop token is recorded separately but
does not count as strict success. Performance and quality experiments use
separate output directories and never overlap GPU execution.

### Results and numerical audit

The initial paired experiment completed its 100 updates and all six generation
evaluations. The original checkpoint scored 0/120 transformations and 0/60 on
the original diagnostic. Reference SFT reached 56/120 and combined native SFT
55/120; both remained 0/60 on the original diagnostic. Full response records
and subsequent audits are local in
[`instructions`](benchmarks/h100-training-2026-10-13/instructions/RESULTS.md).

For the initial native run, marker removal and brackets reached 20/20 each,
joining reached 14/20, uppercase 1/20, and replacement/JSON 0/20. JSON had no
accepted training examples. These results show learning on several trained
transformation families, without evidence of general instruction following.
For example, an uppercase request returned the unchanged lowercase text; an
original context-lookup prompt produced unrelated curriculum words. More
training loss reduction alone would not resolve this evidence gap. A subsequent
curriculum should diversify instruction wording, include context-grounded tasks
and abstention, and retain independent evaluation templates and a retention set.

The initial full-native update discrepancy was 18.707% of the reference update
norm, exceeding the unchanged 10% limit even though loss differences passed.
A repeated reference run reproduced every final tensor exactly. Ablations with
the initial library gave 12.178% for SwiGLU alone and 19.258% for RMSNorm alone.
The native SFT was therefore not accepted as an equivalent implementation.

Capturing actual backward inputs found rare SwiGLU BF16 rounding differences
missed by unit-scale random tests. With precise `expf` and the ATen SiLU
arithmetic, the corrected SwiGLU-only variant reproduced all final weights,
training losses and validation losses **bit for bit across the full 100 steps**.
The corrected combined variant still differs by 19.258%, isolating the remaining
SFT trajectory discrepancy to the native RMSNorm path. Its primitive derivatives
can satisfy their tolerances while a fresh optimizer's 100-step trajectory fails
the stronger check. RMSNorm therefore remained on the reference path until the
compatible implementation below passed. The matured-optimizer pretraining comparison has its own
recorded acceptance result and does not establish SFT equivalence.

The original 60-case baseline was also reproduced with 60/60 identical response
texts and stop flags. No test was weakened and no threshold was increased.
The corrected SwiGLU-only checkpoint was explicitly decoded on both suites:
56/120 transformations and 0/60 original instructions, matching the reference
scores. Source snapshots, every generated answer, small-gradient probes, ablation
comparisons and logs are in the local portable records. Model/optimizer
checkpoints and the large profiler traces remain on the instance. Both GPUs
were free of compute processes when the experiments completed.

The standalone SFT reproduction command is:

```bash
python scripts/run_verified_instructions.py \
  --brandon-root /lambda/nfs/btiny/brandon-mini/sources/stage1-v1 \
  --pipeline /lambda/nfs/btiny/brandon-mini/stages/stage1-v1 \
  --distillation /lambda/nfs/btiny/brandon-mini/distillation/text-v1 \
  --performance /home/ubuntu/ntransformer-h100/training/precise-20261013 \
  --output /path/to/new/instruction-experiment
```

Use the training-library environment described above. The output must be a new
directory. The default compatible backend now requires exact final weights,
optimizer/RNG state and recorded losses through `--require-exact` in the audit.
The fully fused diagnostic retains the original numerical thresholds.

## Compatible RMSNorm results

The follow-up preserves PyTorch's reduction operations and order, while fusing
pointwise forward/backward work in CUDA. Matching a standalone RMSNorm derivative
was insufficient: the input also receives a residual gradient. Combining the
two norm paths inside one custom backward changed their association with that
residual. Passing the same input through two autograd edges allows separate
direct and variance gradients, restoring the tested accumulation order.

Six primitive shapes pass exact forward/input-gradient/weight-gradient checks,
including dimension 576 and small upstream gradients. The full 30-layer model's
real first SFT batch reproduces all 272 unique parameter gradients exactly.
The 100-step DDP SFT then reproduces all 273 state-dict tensors, AdamW state,
RNG state and recorded losses exactly. The duplicate state-dict entry is the
tied embedding/output weight. This is validated on the recorded PyTorch build
and Brandon graph; another graph or software version requires revalidation.

The pretraining experiment uses the same completed checkpoint and restored
optimizer, unchanged global batch and 100 updates per variant:

| Variant | Global tokens/s | Ratio to reference | Compute allocation per rank | Peer allocation per rank |
|---|---:|---:|---:|---:|
| PyTorch reference | 301,551 | 1.000 | 18.437 GiB | 0 |
| Compatible RMSNorm and native SwiGLU | 334,814 | 1.110 | 14.943 GiB | 0 |
| Plus local block recomputation | 260,471 | 0.864 | 4.878 GiB | 0 |
| Plus peer block storage | 254,509 | 0.844 | 3.908 GiB | 1.055 GiB |

These are allocator peaks. Bidirectional DDP also places the opposite rank's
1.055 GiB peer pool on each physical device, so the last row is not 3.908 GiB
of total device usage. Peer transfer costs 2.3% throughput relative to local
recomputation in this run. Both ranks together issued 12,000 saves and 12,000
reads, with 11,600 lookahead prefetches and 843.75 GiB transferred across both
directions. Reserved blocks and event ordering remain active; no NVMe adapter
is involved.

All three pretraining variants pass the unchanged numerical gate. They are
not bit-identical: final parameter discrepancy is 1.71–1.78% of the reference
update L2 norm, against the 10% limit. Exact SFT equivalence does not establish
exact pretraining equivalence. Each variant ran once in the listed order, with
steps 11–100 measured; this is a short comparison without a confidence interval
or evidence of long-run convergence. The earlier +25.8% used fully fused
reductions and remains a separate historical result.

The new `behavior-v1` pipeline was allowed to finish before these GPU runs.
It trained a separate 600-step curriculum and scored 52/120 transformations
and 0/60 original instructions. Different data and decoding limits prevent a
direct score comparison to the 100-step pilot. Its metadata is preserved as
context, without changing its checkpoint or training process.

All metrics, source snapshots, execution scripts, SFT reference metadata,
checkpoint hashes and audits are local in
[`improve/RESULTS.md`](benchmarks/h100-training-2026-10-13/improve/RESULTS.md).
Full model/optimizer checkpoints remain under
`/home/ubuntu/ntransformer-h100/training/improve-20261013` on the instance.
The compatible implementation is the default in the adapter and launch scripts.
The exact SFT audit is an enforced completion condition, not a relaxed tolerance.

## Pretraining repeatability review

The subsequent review found that the remaining 1.776% pretraining update
discrepancy is comparable to PyTorch's own variation: a new reference repetition
differs by 1.729%, and an earlier repetition by 1.774%. Profiling identified
cuDNN SDPA; replaying real attention inputs and upstream gradients confirmed
variation in its default backward. The short SFT batch reproduced exactly even
with that backend, so the backend name alone does not imply observable variation
for every shape.

`train_brandon_experiment.py --deterministic` now provides a separate diagnostic
mode, enabling strict deterministic algorithms before CUDA initialization and
setting a reproducible cuBLAS workspace configuration. In the measured workload
SDPA then selects deterministic Flash Attention. Four new 100-step DDP runs
confirmed exact model weights, AdamW/RNG state, recorded losses/gradient norms
and validation for native kernels, local recomputation and P2P versus their
common deterministic reference. `compare_training_experiment.py --require-exact`
enforces that complete check.

Normal performance runs retain their previous default. The deterministic pair
measured 265,642 native versus 257,272 reference tokens/s and has a different
attention policy from the earlier performance comparison. Details, tensor-level
statistics, controls, sources and reproduction commands are in
[`differences/DIFFERENCES.md`](benchmarks/h100-training-2026-10-13/differences/DIFFERENCES.md).
The result applies to the compatible implementation; the older fully fused
RMSNorm's failed SFT audit remains separate.

### Accepted numerical behavior

The compatible backend's default-mode discrepancy is **accepted for this
100-step experiment**: its 1.776% update discrepancy is at the observed scale
of independent reference repetitions (1.729% and 1.774%), and the same native,
recomputation and P2P implementations pass exact comparisons under a common
deterministic policy. This is evidence of reference nondeterminism, not a reason
to expand the original thresholds or accept arbitrary new kernel discrepancies.
It is not a statistical confidence interval or a long-convergence guarantee.

Subsequent optimizations must retain both controls: paired performance runs
with identical model/data/batch/precision and exact deterministic audits of
weights, optimizer/RNG state, recorded numerical metrics and validation. Changes
to attention policy are reported separately. The established improvements are
native compatible RMSNorm/SwiGLU (+11.0% in the paired default run), 19.0% lower
peak allocated memory, exact SFT updates, and validated local/P2P recomputation
with pre-reserved activation blocks and asynchronous prefetch.

## Operational review

The next optimization review traces the complete training state and data path,
including startup, corpus sampling, dtype/layout conversions, activation lifetime,
backward, DDP buckets, clipping, AdamW, telemetry and persistence. It distinguishes
mathematical dependencies from host coordination and redundant materialization.
See [the full dataflow review](H100_TRAINING_DATAFLOW.md) for measured transfer
counts, per-tensor volumes and the ordered removal experiments. CUDA Graphs is
one scheduling tool in that review, not a substitute for eliminating work inside
the schedule.

The first completed removals omit deterministic debug fills, keep DDP gradients
as bucket views and group loss readbacks after accumulation. They improve the
native deterministic trainer from 266,319 to 317,799–318,488 global tokens/s
(+19.46% using the two-run mean), with exact weights, AdamW/RNG state, numerical
metrics and validation against the deterministic reference. P2P also passes.
The normal-policy pair improves from 334,314 to 342,491 tokens/s (+2.45%) and
retains a 1.717% update discrepancy, comparable to the observed reference variation.
Optimized strict mode remains 7.11% slower than optimized normal mode.
These runtime options are opt-in; the measured flags, memory peaks, ablations,
profiles and local artifact hashes are in
[`operational/RESULTS.md`](benchmarks/h100-training-2026-10-13/operational/RESULTS.md).

## Shared projection activation casts

`--shared-projection-casts` shares the BF16 forward buffer across Q/K/V and
gate/up while retaining separate backward casts to FP32 before accumulation.
The fresh strict controls average 316,110 tokens/s versus 323,738 with sharing
(+2.41%, two repetitions each). The normal pair improves 343,239 → 349,241
(+1.75%); normal allocated peak falls 14.943 → 12.959 GiB on each GPU. Strict
memory falls 14.943 → 13.360 GiB. P2P improves 241,182 → 247,411 (+2.58%) with
its existing direct transfers, reserved blocks and prefetch protocol intact.

Six strict 100-step cases pass complete exact-state audits; the normal pair's
1.739% update discrepancy remains at the observed reference variation scale.
Separate warmed traces confirm removal of 180 forward conversions per update
and rank, with unchanged backward conversion and GEMM counts. The feature is
opt-in, defaults are preserved, and all results/source snapshots are local in
[`shared-casts/RESULTS.md`](benchmarks/h100-training-2026-10-13/shared-casts/RESULTS.md).

## Direct grouped-query attention and architecture research

`--direct-gqa` passes compact KV heads to SDPA instead of repeating them to the
query-head count. It composes with `--shared-projection-casts`, bucket views
and deferred loss readbacks. The fresh normal pair improves **349,067 → 360,951
global tokens/s (+3.40%)**; the strict two-run means improve **322,133 → 330,299
(+2.54%)**. Strict P2P improves **249,018 → 255,764 (+2.71%)**, preserving its
reserved blocks and direct DMA protocol.

Six strict 100-step cases pass the full exact-state audit. The normal pair's
1.700% update discrepancy is of the previously observed reference-variation
scale, with unchanged acceptance limits. Traces verify removal of explicit KV
expansion and preserve the attention backend within each pair. Normal allocated
peak falls 12.959 → 12.665 GiB per GPU. The option remains off by default.
Full results, numerical differences, memory variation, source snapshots and
compressed traces are local in
[`direct-gqa/RESULTS.md`](benchmarks/h100-training-2026-10-13/direct-gqa/RESULTS.md).

[`TRAINING_RESEARCH.md`](TRAINING_RESEARCH.md) derives the GQA adjoint, streaming
cross-entropy, parallel residual branches and reversible coupling. A reproducible
CPU tool checks the algebra and illustrates reconstruction rounding. The
architecture candidates are untrained; speed, convergence and instruction
following remain experimental questions rather than results of these checks.

## Fused rotary embeddings

`--fused-rope` requires `--direct-gqa` and a rebuilt `ntransformer_training`
library. One kernel handles Q/K forward and another handles backward, retaining
the original frequencies and verified FP32 FMA/BF16 rounding. Native support is
first order; unsupported inputs and trainable frequencies use the original
expression. The old shared library remains intact; this experiment builds in
`/home/ubuntu/ntransformer-h100/build-training-rope` and freezes a library copy
alongside each experiment's source snapshot.

Fresh controls already contain the preceding optimizations. Normal mode improves
**361,078 → 380,757 global tokens/s (+5.45%)**; strict means improve **329,996 →
346,301 (+4.94%)**; strict P2P improves **255,983 → 270,165 (+5.54%)**. All six
strict 100-update cases pass the complete exact-state audit. Normal update
discrepancy is 1.714%, comparable in scale to reference repetition, with no
change to the numerical acceptance limits.

The tradeoff is approximately 0.40 GiB additional allocated peak per GPU in
strict/P2P continuations; normal memory is effectively unchanged. Separate
instrumented runs do not reproduce that increase, so its cause remains
unisolated. Traces establish 720 fewer CUDA kernels per update and rank, with
unchanged GEMM counts and attention backends. Full measurements, primitive and
model-gradient tests, build records and verified local traces are in
[`fused-rope/RESULTS.md`](benchmarks/h100-training-2026-10-13/fused-rope/RESULTS.md).

## Target-only loss and vocabulary streaming

`--loss-mode fused` keeps the classifier BF16 GEMMs and removes dense
log-softmax/FP32 NLL buffers. It reproduces the installed runtime's intermediate
BF16 rounding; all six strict control/candidate/P2P continuations remain exact.
The tied embedding, ignored labels, masks and shift semantics are covered by
full-model gradient tests. The default forward API and default CLI stay unchanged.

Across two pairs, strict means are **342,321 → 348,482 tokens/s (+1.80%)** and
normal means **376,596 → 379,021 (+0.64%)**. Normal pairs range from -0.42% to
+1.70%, so this is not evidence of a reliable general speedup. P2P's single
pair improves 2.16%. The repeatable benefit is memory: strict allocated peak
falls **12.667 → 11.514 GiB/rank**, normal saves 0.351–0.753 GiB, and P2P
compute allocation falls about **3.51 → 2.74 GiB**, with peer reservations intact.

`--loss-mode vocab-stream` is an unpromoted experimental alternative. At block
2048 it removes full logits, but changes accumulation order: the worst initial
parameter-gradient discrepancy is 6.33%, failing the retained 2% diagnostic.
Its 100-step audit passes the broader continuation limits with a 1.91% update
difference. It reaches 344,981 tokens/s and saves only another 0.260 GiB over
the fused path. This does not justify selecting it for routine training here.

Eleven continuations, ten compressed traces, exact sources, rejected probes
and the head-gradient diagnosis are verified locally in
[`target-loss/RESULTS.md`](benchmarks/h100-training-2026-10-13/target-loss/RESULTS.md).
The isolated library is in `build-training-loss`; earlier libraries are retained.
This is a runtime/memory experiment, not a new instruction-following evaluation.

## Fused labels and reserved graph replay

`--fused-loss-labels` folds target validation, shifting and NLL proxy preparation
into the loss kernel. `--training-graph` additionally captures native pretraining
forward/backward with persistent storage; DDP, clipping, AdamW and finite checks
stay eager. Capture precedes DDP wrapping. Graph accumulation must use separate
persistent gradients and `zero_grad(set_to_none=False)`; resetting to `None`
is rejected because a subsequent replay could overwrite a previous microbatch.

Fourteen fresh 100-update continuations give **348,782 → 361,246 tokens/s
(+3.57%) strict** and **381,725 → 398,704 (+4.45%) normal**, averaging two runs
per native variant. Eight strict cases, including P2P, retain exact weights,
AdamW/RNG and numerical metrics. The full 272-parameter gradient and two-update
DDP checks are exact. Normal graph differences are 1.73–1.74% of the reference
update; the normal reference repeat itself differs by 2.94%.

Labels alone remove 20 kernels/update/rank but show no consistent speedup.
Graph replay reduces host launch calls 85.1% in separate profiles, while adding
274 GPU kernels/update for accumulation, zeroing and this DDP schedule. Capture
takes 0.66–0.74 s. Reserved memory is **12.281 GiB/rank**; the low 2.534 GiB
allocated counter excludes retained graph temporary lifetimes and must not be
reported as a VRAM reduction. P2P supports fused labels; its host checkpoint/
prefetch schedule is not yet part of this graph implementation.

Sources, failed aliasing reproduction, full tests, fourteen numerical audits,
six locally verified traces and limitations are in
[`labels-graph/RESULTS.md`](benchmarks/h100-training-2026-10-13/labels-graph/RESULTS.md).

## Explicit gradient buckets

Two independent opt-in paths address the dispatch left outside graph replay:

* `--batched-gradient-reduction` enables the installed PyTorch DDP
  `batched_grad_copy` path, grouping copies/scaling by communication bucket.
  It also works with the existing P2P activation schedule.
* `--bucketed-graph --gradient-bucket-mb 512` requires `--training-graph` and
  replaces DDP gradient hooks with explicit `autograd.grad`, grouped accumulation
  and NCCL reductions. The default 25 MiB capacity produces 18 buckets for
  Brandon Mini; 512 MiB holds its 410.5 MiB gradient surface in one bucket.

The explicit path borrows each graph gradient only until the next replay.
Independent persistent buckets back `parameter.grad`; the first microbatch
copies into them and later microbatches add in order. It divides before NCCL
sum, waits on the owner stream, and retains the existing clipping, finite checks
and AdamW. The trainer restores the same verified checkpoint on every rank;
this adapter does not provide general DDP initialization, unused-parameter,
join, higher-order or arbitrary-hook semantics.

The 18-run experiment retains all ten strict audits exactly. Normal single-
bucket means improve **398,615 → 403,675 tokens/s (+1.27%)** against the graph
control, or +1.07% against batched DDP. Strict single-bucket means improve 1.14%
against stable batched DDP; the original strict control has one unexplained
slow run, so its inflated mean ratio is not used as the headline. A planned
additional confirmation pair was deferred when Brandon's replication pipeline
acquired the GPUs; the main suite and eight traces were already complete.

The single-bucket reserved allocator peak is **12.193 GiB/rank**, versus 12.281
for graph/DDP. This is a modest 90 MiB reduction, not the much larger difference
suggested by allocated tensor counters. Grouping all gradients is specific to
the current whole-backward capture: it does not overlap layer communication.
Evidence and all samples are in
[`gradient-buckets/RESULTS.md`](benchmarks/h100-training-2026-10-13/gradient-buckets/RESULTS.md).

## API references

The autograd bridge follows PyTorch's
[custom Function contract](https://docs.pytorch.org/docs/stable/notes/extending.html)
and [CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).
Peer transfers use NVIDIA's
[CUDA peer access](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-runtime-api/group__CUDART__PEER.html)
and [asynchronous memory copy API](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-runtime-api/group__CUDART__MEMORY.html).
SwiGLU's arithmetic was checked against the installed PyTorch build's
[ATen SiLU implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/ActivationSiluKernel.cu).
Compatible RMSNorm follows the same build's
[autograd derivatives](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml).
