# H100 pretraining numerical differences

The residual discrepancy of the compatible backend is explained by variation in
the reference attention backward. A new 100-step PyTorch repetition differs from
the earlier PyTorch run by 1.729% of its update norm, compared with 1.776% for
NTransformer. With deterministic algorithms enabled, native kernels, local
recomputation and P2P all reproduce the reference weights, AdamW state, RNG state,
recorded losses/gradient norms and validation exactly across 100 DDP updates.

This review covers the compatible RMSNorm and precise SwiGLU implementation on
the recorded PyTorch 2.14.1 build and two H100 GPUs. It does not revalidate the
older fully fused RMSNorm that failed the separate fresh-optimizer SFT audit.

The residual difference is accepted for this bounded experiment because it is
at the observed scale of reference variation and disappears in the strict
deterministic comparison. Future changes retain the same tolerances and exact
control; these observations do not define a general nondeterminism allowance.

## Default execution

Every run starts from the same completed pretraining checkpoint, restored AdamW
state and sampled batches. Each processes 100 updates of 65,536 global tokens.
The reference is `improve-20261013/reference`. The new reference repetition and
the earlier precise-build reference provide two controls without native kernels.

| Compared with the reference | Update discrepancy | Maximum loss difference | Maximum absolute weight difference |
|---|---:|---:|---:|
| New PyTorch repetition | 1.729% | 0.00021648 | 0.00037213 |
| Earlier PyTorch repetition | 1.774% | 0.00018722 | 0.00046312 |
| Compatible native kernels | 1.776% | 0.00021440 | 0.00032687 |
| Native with local recomputation | 1.707% | 0.00023252 | 0.00039684 |
| Native with P2P recomputation | 1.730% | 0.00026107 | 0.00036172 |

Update discrepancy means `||candidate - reference|| / ||reference - initial||`.
It is not the fraction of weights that are wrong or the relative size of the
complete model. For native kernels the latter is approximately **0.012%**.
The table retains the original audit's 273 state-dict entries, including the
tied embedding/output entry twice. `checkpoint-analysis.json` also counts only
the 272 unique tensors: the new repetition and native update discrepancies are
1.746% and 1.794%, respectively. This accounting difference does not change the
finding, and historical thresholds were not altered.

All five comparisons have identical first losses at step 32,769, but different
gradient norms at that step; their losses first differ at step 32,770. After
100 updates, differences occur throughout all 272 unique parameter tensors.
For the native run, the largest aggregate contributions are layers 1, 2 and 3
(5.77%, 5.35% and 5.18% of squared discrepancy). The largest absolute individual
weight difference is in the tied embedding/output matrix. This is a distributed
training-trajectory difference rather than a concentration in normalization
weights or a particular peer block.

These controls establish the scale of reference variation in the recorded
runs. Two repetitions are not an uncertainty interval or evidence about
long-run convergence.

## Attention backward isolation

Profiling the real first pretraining batch, shape `[16, 1024]`, shows that
automatic SDPA dispatch selects **cuDNN attention**. Repeating the unchanged
model yields different gradients despite identical weights, inputs and forward
loss. The same test separately enables RMSNorm, SwiGLU, both, and recomputation.
Enforcing deterministic algorithms switches dispatch to Flash Attention; all
272 parameter gradients then match exactly for every variant.

A second probe captures the actual query/key/value inputs and upstream gradient
of layer 29. It replays only attention backward ten times for each configuration:

| Attention implementation | Differing gradient elements in each of nine repetitions |
|---|---:|
| cuDNN, default backward | 2–5 |
| Flash, default backward | 93–127 |
| Flash, deterministic backward | 0 |

Inputs and upstream gradient stay fixed within each comparison. The replay uses
the same BF16 autocast policy as the model. Switching to Flash alone therefore
does not guarantee repeatability; deterministic backward is required.
`real-attention.json` contains every repetition and gradient statistic.

This matches the documented behavior of
[PyTorch 2.14 SDPA determinism](https://docs.pytorch.org/docs/2.14/notes/randomness.html#cuda-scaled-dot-product-attention):
strict deterministic mode disables the cuDNN backend and enables deterministic
Flash backward. Different attention backends can also produce different forward
rounding, so exact comparisons are made within the same execution policy.

## Full deterministic continuation

All four variants use `--deterministic`, compatible RMSNorm, the same final
checkpoint, restored optimizer, batch, precision and sampling. No training
data, learning rate, tolerance or loss was adjusted to obtain agreement.

| Candidate against deterministic reference | Model weights | AdamW and RNG | Losses and gradient norms | Validation |
|---|---|---|---|---|
| Native | Exact | Exact | Exact | Exact |
| Local recomputation | Exact | Exact | Exact | Exact |
| P2P recomputation | Exact | Exact | Exact | Exact |

The comparison's `--require-exact` flag fails on any difference in these fields.
All 273 final state-dict tensors changed from initialization and matched the
deterministic reference. All variants reached validation mean 2.17004219815135.
The checks cover two DDP ranks, two accumulation microbatches and 100 updates,
including the speculative peer prefetch pipeline.

These diagnostic runs measured 257,272 global tokens/s for reference, 265,642
for native, 208,900 for local recomputation and 202,783 for P2P. Strict determinism
changes attention dispatch and other PyTorch behavior, so these rates are a
separate comparison from the normal 334,814 versus 301,551 tokens/s experiment.
The regular launch default remains unchanged; determinism is an explicit option.

## Why the SFT check already matched

The real first SFT batch has shape `[8, 93]`, compared with `[16, 1024]` in
pretraining. Its default dispatch also selects cuDNN, but the repeated complete
backward and the operator variants match exactly on this shorter batch.
The previous 100-step SFT audit likewise matched exactly. Repeatability depends
on the selected kernel and workload; the backend name alone does not establish
that a particular run will differ. Deterministic Flash also matches all variants
on the SFT probe, with its own common forward rounding.

## Reproduction and records

`scripts/probe_training_differences.py` profiles and compares the real first
batch for either objective. `scripts/analyze_training_differences.py` compares
saved tensors by layer, operator family, unique storage and trajectory metrics.
The execution scripts preserve the original pipeline locks and reject active
GPU processes before launching. Source snapshots and input/checkpoint hashes
are included; model/optimizer checkpoints remain on the instance.

Add these options to the established pretraining commands and comparator:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_brandon_experiment.py \
  --brandon-root /lambda/nfs/btiny/brandon-mini/sources/stage1-v1 \
  --checkpoint /lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt \
  --data-dir /lambda/nfs/btiny/brandon-mini/stages/stage1-v1/data \
  --output /path/to/new/deterministic-native \
  --mode native --norm-mode compatible --steps 100 --warmup-steps 10 --deterministic

python scripts/compare_training_experiment.py \
  --reference /path/to/deterministic-reference \
  --candidate /path/to/new/deterministic-native \
  --source-checkpoint /lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt \
  --output /path/to/exact-comparison.json --require-exact
```

Use the documented training-library environment and the existing launch gate.
The complete executed commands are in `execution/`. The authoritative probes
are `probe-pretrain.json`, `probe-sft.json` and `real-attention.json`; the earlier
`first-batch.json` retains the exploratory replay with random upstream gradients.
