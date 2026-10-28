# Target-only loss on two H100 SXM5 GPUs

The opt-in `--loss-mode fused` removes the dense log-softmax and FP32 NLL
intermediates while preserving the full BF16 classifier GEMMs. Six strict
100-update controls/candidates pass the complete exact-state audit. Allocated
memory decreases consistently; speed changes are small and variable. The
vocabulary-streamed candidate also runs, but is **not promoted**: it loses
exact gradients and offers only another 0.26 GiB over the fused path here.

## Paired measurements

Every run restores the final pretraining checkpoint, AdamW and RNG. The model,
data hashes, tokenizer, precision, schedule and global batch are unchanged:
107,604,288 parameters, 30 layers, dimension 576, FFN 1472, GQA 9/3,
8192 vocabulary, context 1024, FP32 parameters and BF16 autocast. Two GPUs,
16 sequences/GPU and two microsteps produce 65,536 global tokens/update.

Controls already enable shared projection casts, direct GQA, fused RoPE,
DDP bucket views and deferred loss readbacks. Strict cases additionally
use deterministic algorithms and omit uninitialized-memory debug fills.
Each throughput run has 100 updates, ten warmup updates and a continuous
90-update measurement window including batch preparation, hashing,
communication, clipping, AdamW and logging. Validation and persistence are
outside that window. Profiles are separate 13-update runs, active at offsets
11 and 12; their timing is not used for speed claims.

| Case | Global tokens/s | Compute allocated peak GiB, ranks 0 / 1 |
|---|---:|---:|
| Strict control | 345,721 | 12.667 / 12.667 |
| Strict fused loss | 348,353 | 11.514 / 11.514 |
| Normal control | 375,186 | 12.265 / 12.265 |
| Normal fused loss | 373,616 | 11.914 / 11.914 |
| Strict fused loss, repeat | 348,612 | 11.514 / 11.514 |
| Strict control, repeat | 338,921 | 12.667 / 12.667 |
| Strict P2P control | 266,395 | 3.507 / 3.510 |
| Strict P2P fused loss | 272,136 | 2.742 / 2.743 |
| Strict vocabulary stream, block 2048 | 344,981 | 11.253 / 11.253 |
| Normal fused loss, repeat | 384,427 | 11.914 / 11.914 |
| Normal control, repeat | 378,006 | 12.667 / 12.667 |

Strict means are **342,321 → 348,482 tokens/s (+1.80%)**. Individual pairs
are +0.76% and +2.86%, with reversed order in the repeat. Normal means are
**376,596 → 379,021 (+0.64%)**, with individual pairs **-0.42% and +1.70%**.
The normal repeat was added after the first ambiguous pair, with the source
and library frozen. These small samples do not establish a reliable general
speedup. P2P's single pair improves **2.16%**.

Memory peaks cover all 100 training updates, including warmup, before final
validation. Fused loss saves **1.153 GiB/rank in both strict repetitions**,
**0.351–0.753 GiB in normal mode**, and **0.765–0.766 GiB of compute-device
allocation in P2P**. The normal control peak varies between repetitions;
do not attribute that variation to a fixed tensor lifetime without further
evidence. P2P additionally retains 1.055 GiB of peer storage per process,
30 reserved blocks, two staging slots, 6000 writes/reads and 5800 lookahead
prefetches. These per-process peaks are not simultaneous aggregate GPU peaks.

## Numerical contract and implementation

The observed runtime uses **BF16 log-softmax followed by FP32 NLL**. An initial
prototype incorrectly assumed FP32 log-softmax; it failed the head test and
was corrected before the continuation suite. Rejected probes are retained
under [rejected-probes](rejected-probes/). The accepted reduction follows the
installed [ATen softmax implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/SoftMax.cu):
eight strided values per thread and a two-stage warp reduction at width 8192.

Let `Qb` round to BF16, `z` be the BF16 GEMM result, `m=max(z)` and
`s=sum(exp(float(z)-m))`. The kernel emits only

```
logp_target = float(Qb((float(z_target) - m) - log(s)))
g = float(Qb(incoming_target_gradient))
logp_j = float(Qb((float(z_j) - m) - log(s)))
dz_j = Qb(fma(-exp(logp_j), g, j == target ? g : 0))
```

PyTorch NLL on one proxy class preserves the row reduction, ignored labels
and mean denominator. Backward reconstructs rounded log probabilities from
the saved BF16 logits and row statistics. Classifier GEMMs remain unchanged.
This is a first-order adapter for the tested CUDA/BF16, vocabulary-8192
configuration, not a universal bitwise guarantee across runtimes or hardware.

The [CUDA kernels](../../../../src/training/cross_entropy.cu) and
[autograd adapter](../../../../python/ntransformer_training/loss.py) preserve
the tied embedding Parameter and state-dict inventory. `loss_only=True`
explicitly requests `(None, loss)`; ordinary forward still returns logits.
CPU/unsupported compute shapes use ordinary CE. Smoothing, auxiliary losses
and nondefault prediction-head requests use the original model forward.
Shifted SFT keeps the original full classifier row shape, including zero
gradients for unused final rows. Weighted masks retain their original
denominator, including masked positions whose label is ignored.

The [tests](loss-tests.json) cover three logit scales, masks, ignored labels,
all-zero masks, all-ignored means, label shifts, CPU fallback, default output
and smoothing fallback. Fused loss is exact for the loss and all **272 unique
parameter gradients** on a real full pretraining batch and a shifted/masked
version of that batch. The latter is a semantics test, not an SFT quality eval.

All six strict control/fused/P2P cases match the frozen deterministic reference
in weights, AdamW, RNG, recorded loss/norm/LR/tokens and validation. Batch hashes
also match. Normal fused comparisons pass the unchanged gates:

| Comparison | Max training loss difference | Validation mean difference | Relative update difference |
|---|---:|---:|---:|
| Normal fused / control | 0.000192583 | 0.0000145733 | 2.0654% |
| Normal fused / control, repeat | 0.000284880 | 0.0000444502 | 1.7204% |
| Normal control repeat / control | 0.000192404 | 0.0000416040 | 2.0460% |
| Strict stream / strict reference | 0.000198364 | 0.0000313520 | 1.9080% |

Normal differences are comparable in scale to the newly measured control
variation, not bounded by a statistical confidence interval. Gates remain
0.03 training loss, 0.02 validation loss and 0.10 relative update difference.

## Why vocabulary streaming remains experimental

`--loss-mode vocab-stream --loss-block-size 2048` computes online row maxima
and exponential sums over vocabulary blocks, recomputes block logits during
backward and accumulates partial activation-gradient GEMMs in FP32 before
rounding once to BF16. It never allocates the full logits matrix. This algebra
matches dense CE over the reals; it does not preserve every floating-point
reduction order. FP32 output for BF16 inputs uses the documented
[`torch.mm` output dtype](https://docs.pytorch.org/docs/2.14/generated/torch.mm.html).

The original **2% per-tensor gradient diagnostic fails**: the worst relative
L2 difference is 6.3343% (`layers.3.ffn_norm.weight`) on pretraining, versus
0.8807% on the shifted/masked case. That failed gate remains in the record;
passing the broader continuation gate does not promote this candidate.

The [isolated head probe](head-probe/results.json) holds the actual final
activations/weights fixed. A single 8192-entry block is exact. Blocks of
4096/2048/1024 yield activation-gradient relative differences of
0.02373% / 0.02835% / 0.03027%. Holding `dz` fixed gives almost the same
differences, isolating vocabulary partitioning of the gradient GEMM as the
dominant cause in that probe. A full FP32-output GEMM rounded to BF16 is exact;
disabling reduced-precision GEMM reduction also leaves that full result exact.
The probe therefore does not implicate that flag as the cause. These results
agree with PyTorch's warning that
[slice and full computations can use different reduction orders](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations).

The stream continuation reaches 344,981 tokens/s, between the two strict
controls and about 1.00% below the fused mean. Its extra memory saving over
fused loss is only 0.260 GiB/rank for this model. It is useful evidence about
the tradeoff, not a speed improvement to enable routinely.

## What the traces establish

Counts below cover **two warmed updates per rank** and agree on both ranks.

| Event | Strict control | Strict fused | Strict stream |
|---|---:|---:|---:|
| CUDA kernels | 11,664 | 11,700 | 11,948 |
| `aten::mm` calls | 2,532 | 2,532 | 2,584 |
| `aten::copy_` calls | 2,798 | 2,790 | 2,802 |
| Dense `[16384,8192]` loss conversions | 8 | 0 | 0 |

Fused loss removes **3 GiB of logical conversion reads/writes per update and
rank**, plus approximately **1 GiB of dense zero writes**, replacing them
with small row buffers. These are tensor-level counts, not a hardware HBM
traffic measurement. The full **256 MiB BF16 logits** still exist in fused
mode. Stream mode uses 64 MiB logits blocks instead and adds **26 GEMMs per
update** through partitioning/recomputation.

Fused loss adds 18 CUDA kernels/update overall: target validation and proxy
label construction are still separate small operations. This is primarily a
buffer-lifetime improvement, not a reduction in launch count. Strict attention
remains Flash and normal attention remains cuDNN within each pair.

In the active profile window, strict and normal control peaks are 12.265 GiB,
fused peaks 11.514 GiB and stream peaks 11.251–11.253 GiB. These are
distinct from peaks over the entire 100-update runs; instrumentation does not
explain their lifetime differences by itself.

## Reproduction, artifacts and next experiment

The isolated build is `/home/ubuntu/ntransformer-h100/build-training-loss`;
previous training and RoPE libraries remain intact. Add `--loss-mode fused`
to the existing optimized training command and point
`NTRANSFORMER_TRAINING_LIB` to this rebuilt library. Defaults stay unchanged.
[run-target-loss.py](run-target-loss.py) and
[repeat-target-loss.py](repeat-target-loss.py) record exact commands, source
snapshots, both pipeline locks and idle-GPU admission. No user process was
stopped. All benchmark jobs finished before another Brandon evaluation began.

Local artifacts include [results.json](results.json), build commands, test
results, eleven checkpoint hashes, ten compressed traces, failed probes,
source snapshots, [provenance.json](provenance.json) and
[local-verification.json](local-verification.json). The immutable payload has
334 SHA256-verified files; the archive includes its manifest as file 335.
Checkpoint and shared-library binaries remain on Lambda; hashes are local.
Archive SHA256: `a57e3cca1df39a0e813ba581203a12c4bf08446da214e139af0bda13a0881502`.

The next concrete target is to merge label checks/proxy construction into the
loss plan and measure repeated CPU dispatch with reserved buffers. A fully
tiled classifier/loss needs its own Tensor Core scheduling and rounding
analysis; a Python loop over smaller GEMMs did not establish a speed benefit.
P2P allocation/prefetch remains available. No new Transformer architecture
or instruction-following improvement is claimed by this experiment.
