# Fused rotary embeddings on two H100 SXM5 GPUs

Fusing Q/K dtype conversion and rotary arithmetic improves the already
optimized direct-GQA trainer by **5.45% in normal mode**, **4.94% under strict
determinism** and **5.54% with strict P2P activation storage**. All six strict
100-update cases pass the complete exact-state audit. Normal mode passes the
unchanged numerical gate. The measured tradeoff is approximately **0.40 GiB
more allocated peak per GPU in the strict and P2P continuations**; normal
allocated peak is effectively unchanged.

## Implementation and arithmetic

The opt-in `--fused-rope` requires `--direct-gqa` and a rebuilt training library.
[`rotary.cu`](../../../../src/training/rotary.cu) processes Q and K in one kernel
for forward and another for backward. It reads BF16 or FP32 inputs, performs
FP32 arithmetic in registers and writes the original dtype. The original
complex64 frequency table, positions, model parameters and attention layout
are preserved; frequencies are not recomputed or approximated.

For a pair `(a,b)` and frequency `(c,d)`, the implementation computes

```
real = fma(a, c, -round_fp32(b*d))
imag = fma(a, d,  round_fp32(b*c))
```

Backward uses the same operations with `d` negated, then rounds to the input
dtype. The contraction order matters: replacing either expression with a
different FMA arrangement or two separately rounded products can change its
result. This follows the complex multiplication expression in the installed
[PyTorch source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/headeronly/util/complex.h),
with contraction verified empirically for this build.

The [rounding probe](rounding-probe/results.json) tests five arrangements over
12 cases of 1,000,003 complex pairs, including FP32/BF16 input values, three
scales and conjugate frequencies. The selected arrangement has zero FP32
output mismatches across those cases. This is evidence for the tested runtime,
not a guarantee for every compiler, input or future GPU.

The [autograd bridge](../../../../python/ntransformer_training/rotary.py) saves
only the frequency tensor. Constant-coefficient RoPE does not mathematically
need its input in backward; this is not a claim that the old PyTorch graph
necessarily retained those inputs. Strided and broadcast gradients are read
directly. Unused outputs preserve `None` gradients. CPU, unsupported dtypes,
mixed Q/K dtypes, trainable frequencies and lazy conjugate/negative views use
the original expression. The native path supports first-order autograd.

The CUDA training target retains precise math flags and uses explicit `fmaf`
only where needed. RMSNorm, SwiGLU, GEMMs, SDPA, DDP, clipping and AdamW retain
their prior implementations. Strict controls use Flash SDPA; normal controls
use cuDNN SDPA. Traces verify that the backend is unchanged within each pair.

## Paired measurements

All cases restore the final pretraining checkpoint at step 32,768, AdamW and
RNG. Model: 107,604,288 parameters, 30 layers, dimension 576, FFN 1472, 9 Q/3 KV
heads, vocabulary 8192 and context 1024. BF16 autocast, FP32 parameters, 16
sequences per GPU and two accumulation microsteps give 65,536 global tokens
per update. Bucket views, deferred loss readbacks, shared projection casts
and direct GQA are active in every case; only fused RoPE differs within pairs.

Each run has 100 updates and ten warmup updates. Throughput covers the
continuous remaining window, including batch preparation, hashes,
communication, clipping, AdamW and logging. Initialization, validation and
checkpoint persistence are separate. Tokens/s sum both GPUs.

| Case | Global tokens/s | Compute allocated peak GiB, ranks 0 / 1 |
|---|---:|---:|
| Strict control | 329,557 | 12.266 / 12.266 |
| Strict fused RoPE | 346,047 | 12.667 / 12.667 |
| Normal control | 361,078 | 12.266 / 12.266 |
| Normal fused RoPE | 380,757 | 12.265 / 12.265 |
| Strict fused RoPE, repeat | 346,556 | 12.667 / 12.667 |
| Strict control, repeat | 330,435 | 12.266 / 12.266 |
| Strict P2P control | 255,983 | 3.511 / 3.510 |
| Strict P2P fused RoPE | 270,165 | 3.909 / 3.904 |

Strict means are **329,996 → 346,301 tokens/s (+4.94%)**; the second pair
reverses execution order. Normal mode improves **361,078 → 380,757 (+5.45%)**;
P2P improves **255,983 → 270,165 (+5.54%)**. Two strict repetitions and one
normal/P2P pair are descriptive measurements, not confidence intervals. The
historical pipeline rate is not the causal implementation control.

Memory peaks cover all 100 training updates, including warmup, before final
validation. Strict peak increases by **0.401 GiB** in both repetitions. P2P
increases by **0.398 / 0.394 GiB**, respectively. Normal mode decreases by only
**1.055 MiB**, which is not a material whole-model memory improvement.

The separate instrumented 13-update runs do not reproduce the higher strict
peak: their allocator peaks, including the active profile window at updates
12–13, are 12.266 GiB for controls and 12.265 GiB for candidates. That does not
locate or explain the peak in the uninstrumented 100-update runs. The measured
peak tradeoff remains part of the result; removed temporaries alone do not
determine the maximum live allocation.

P2P additionally reserves **1.055 GiB on the peer GPU per rank**, with 30
blocks and two local staging buffers. Both cases retain 6,000 writes, 6,000
reads and 5,800 lookahead prefetches per rank across 200 microbatches. Each
physical GPU also holds the other rank's peer pool. Direct DMA and reservation
semantics remain intact; resident activations remain faster for this model.

## Correctness

All six strict cases pass `--require-exact` against the frozen deterministic
PyTorch reference: all 273 state-dict entries (272 unique parameters), AdamW,
CPU/CUDA RNG, recorded loss/gradient norm/LR/token metrics, batch identities
and final per-source validation. Neither the acceptance limits nor the
training objective changed.

The normal pair reports:

* Maximum training-loss difference: **0.000176430**.
* Final mean validation difference: **0.0000273436**.
* Weight discrepancy relative to the control update L2: **1.71389%**.
* Maximum absolute weight difference: **0.000322547**; RNG is exact.

The update discrepancy is of the previously observed 1.729–1.774% reference
repetition scale, slightly below that observed range. This is not a percentage
of total weight norm, a statistical determinism bound or evidence of long-run
quality. The exact strict controls provide the additional implementation check.

[`rotary-tests.json`](rotary-tests.json) contains eight primitive cases with
exact forward/backward results across FP32/BF16, dimensions 2/6/64, position
offsets, strided inputs and gradients, broadcast gradients and unused outputs.
Five fallback cases pass. Attention tests cover causal, explicit-mask, prefill
and decode semantics. Full-model loss and all 272 parameter gradients match on
a real pretraining batch and a partially masked variant. The latter is not a
new SFT or instruction-following evaluation.

## Removed work

Separate profiles capture two warmed updates per rank. Both ranks have the
same counts in their respective variants:

| Events in two updates per rank | Control | Fused RoPE |
|---|---:|---:|
| `aten::copy_` | 3,998 | 2,798 |
| `aten::mul` | 1,026 | 546 |
| `aten::clone` | 240 | 0 |
| `aten::view_as_complex` | 480 | 0 |
| Native rotary pair kernels | 0 | 240 |
| All CUDA kernels, strict | 13,104 | 11,664 |
| All CUDA kernels, normal | 12,984 | 11,544 |
| `aten::mm` | 2,532 | 2,532 |

That is **720 fewer CUDA kernel launches per update and rank**, with unchanged
GEMM counts. The eliminated 1,200 copies comprise four groups of 240 Q/K
FP32↔BF16 conversions and 240 copies of complex frequency tensors. Their
standalone logical read/write inventory is **16.934 GiB per update and rank**.
This is not net measured HBM traffic saved: the fused kernels still read and
write Q/K, frequency reuse affects physical traffic, and complex products also
had their own accesses. Views themselves did not copy data. H2D and DtoH
transfer counts and bytes remain unchanged.

## Reproduction and local evidence

Build `ntransformer_training` from the current source. This experiment uses
CUDA Toolkit 13.1.115, GCC 14.4.0 and the existing PyTorch 2.14.1+cu130 / NCCL
2.30.7 environment. [Build commands and logs](build-record/) are preserved.
The experimental library is isolated at
`/home/ubuntu/ntransformer-h100/build-training-rope/libntransformer_training.so`;
the previous library was not replaced.

Set `NTRANSFORMER_TRAINING_LIB` to the rebuilt library and append:

```bash
--mode native --steps 100 --warmup-steps 10 --norm-mode compatible \
  --gradient-bucket-views --defer-loss-readbacks --shared-projection-casts \
  --direct-gqa --fused-rope
```

Add `--deterministic --skip-uninitialized-fill` for strict mode; use
`--mode peer` for P2P. The guarded commands are in
[`run-fused-rope.py`](run-fused-rope.py). They acquired both Brandon pipeline
locks, checked idle GPUs and used a frozen copy of the library and sources.
All GPU work completed and released the locks before archiving.

Library SHA-256:
`6159c1268668e526f48f083f30d15e23c0e51d3e6b4b13dcf6297d23f666327e`.
Archive SHA-256:
`ab9525f625129f05bc234233df54b83052dbdce92230214d784a59d84bcf2ebe`.

All **163 payload files** are downloaded and verified by
[`SHA256SUMS.json`](SHA256SUMS.json); local CUDA/Python/CMake sources match the
executed snapshot. All eight compressed traces were independently reanalyzed
locally and reproduce their summaries exactly. Python compilation, four
measurement tests and five queue tests pass. See
[`local-verification.json`](local-verification.json). `LOCAL_SHA256SUMS.json`
also covers this report and the local archive.

Full checkpoints and the frozen shared library remain remote under
`/home/ubuntu/ntransformer-h100/training/fused-rope-20261014`.
Next candidates remain vocabulary-block loss and execution scheduling; the
parallel/reversible architecture proposals in
[`TRAINING_RESEARCH.md`](../../../TRAINING_RESEARCH.md) remain untrained.
