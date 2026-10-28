# Direct grouped-query attention on two H100 SXM5 GPUs

Passing three KV heads directly to SDPA removes Brandon's explicit expansion
to nine heads. It improves the previously optimized trainer by **3.40% under
the normal attention policy**, **2.54% under strict determinism** and **2.71%
with strict P2P activation storage**. All six strict 100-step continuations pass
the complete exact-state audit. The normal pair passes the unchanged numerical
gate. This changes the implementation of the existing GQA architecture, not
the model, its parameter count or its training objective.

## Implementation and controls

`--direct-gqa` installs [`attention.py`](../../../../python/ntransformer_training/attention.py).
Q/K/V projections, the frozen Brandon RoPE function, residual values, masks,
compact KV caches, output projection and dropout setting are preserved. SDPA
receives compact K/V with `enable_gqa=True` instead of materialized
`repeat_interleave(3)` tensors. The option is disabled by default.

Traces establish the actual backend: strict runs use PyTorch Flash SDPA;
normal runs use cuDNN SDPA, for both controls and candidates. This is not an
installation or benchmark of FlashAttention-3. PyTorch GEMMs, compatible native
RMSNorm, SwiGLU, AdamW and the CUDA training library are unchanged. The library
SHA-256 remains `b5039135d5c50d4c019ea57046d2ae9595269ea8584cd6fee10f9455f9f450ee`.

Every case restores the final pretraining checkpoint at step 32,768, AdamW and
RNG. The 107,604,288 parameters, corpus/tokenizer, BF16 autocast with FP32
parameters, 16 sequences per GPU and two accumulation microsteps are fixed:
65,536 global tokens per update. All cases use bucket views, deferred loss
readbacks and shared projection casts. Strict cases additionally select
deterministic algorithms and omit the previously audited debug memory fills.
Only direct GQA changes within each pair.

## Paired measurements

Each continuation runs 100 updates, with ten warmup updates. Throughput covers
the continuous remaining window, including data preparation, hashes,
communication, clipping, AdamW, logging and timing. Initialization, validation
and checkpoint persistence are measured separately. Tokens/s sum both GPUs.

| Case | Global tokens/s | Compute allocated peak GiB, ranks 0 / 1 |
|---|---:|---:|
| Strict control | 322,103 | 13.360 / 13.360 |
| Strict direct GQA | 330,025 | 12.665 / 12.665 |
| Normal control | 349,067 | 12.959 / 12.959 |
| Normal direct GQA | 360,951 | 12.665 / 12.665 |
| Strict direct GQA, repeat | 330,572 | 12.266 / 12.266 |
| Strict control, repeat | 322,162 | 13.360 / 13.360 |
| Strict P2P control | 249,018 | 3.511 / 3.513 |
| Strict P2P direct GQA | 255,764 | 3.509 / 3.509 |

The strict two-run means are **322,133 → 330,299 tokens/s (+2.54%)**; execution
order is reversed in the second pair. The normal pair is **349,067 → 360,951
(+3.40%)**, and P2P is **249,018 → 255,764 (+2.71%)**. These are short paired
measurements, not confidence intervals or comparisons with a newly trained
architecture. The historical pipeline rate is not the causal control.

Normal allocated peak decreases **0.294 GiB (2.27%) per GPU**. Strict candidates
have different peaks despite exact numerical states: 12.665 and 12.266 GiB.
Both are below 13.360 GiB, but this experiment does not isolate the cause of
their variation. Reported values are allocator peaks, not whole-device VRAM.

P2P additionally reserves **1.055 GiB in the peer GPU per rank**, with 30 blocks
and two local staging buffers. Each GPU also hosts the other rank's pool.
Both P2P cases preserve 6,000 writes, 6,000 reads and 5,800 lookahead prefetches
per rank across 200 microbatches. Direct DMA, reservation and lifetime ordering
remain active. Keeping this small model's activations local is still faster.

## Numerical evidence

All six strict cases pass `--require-exact` against the archived deterministic
PyTorch reference: 273 state-dict entries representing 272 unique parameters,
AdamW state, CPU/CUDA RNG state, recorded losses/gradient norms/LR/token counts
and final per-source validation. Batch hashes match.

[`attention-tests.json`](attention-tests.json) also records exact attention
outputs, input/parameter gradients and caches for causal, explicit-mask,
prefill and decode cases. Full-model loss and all 272 parameter gradients are
exact on a real pretraining batch and a variant with half its targets masked.
The masked test is not a new SFT or instruction-following evaluation.

The normal-policy pair has:

* Maximum training-loss difference: **0.000163615**.
* Final mean validation difference: **0.0000195093**.
* Weight discrepancy relative to the control update L2: **1.69980%**.
* Maximum absolute weight difference: **0.000433203**; RNG remains exact.

The update discrepancy is slightly below the previously observed 1.729–1.774%
reference-repetition range and of the same scale. That range is descriptive,
not a statistical acceptance bound; this is not a percentage of total weight
norm. No tolerance was widened. The deterministic controls provide an exact
implementation check, without claiming long-run quality from 100 updates.

## Work removed

Separate 13-update runs profile updates 12 and 13. Both ranks reproduce the
following event counts; profiler timing is not used as benchmark throughput.

| Events in two updates per rank | Control | Direct GQA |
|---|---:|---:|
| `aten::repeat_interleave` | 240 | 0 |
| `aten::clone` | 720 | 240 |
| `aten::copy_` | 4,478 | 3,998 |
| `aten::mm` | 2,532 | 2,532 |
| CUDA kernels, strict | 13,584 | 13,104 |
| CUDA kernels, normal | 13,464 | 12,984 |

The missing copies comprise 240 expanded BF16 tensors of shape
16×3×3×1024×64, plus 120 BF16 copies of shape 16×1024×3×64 and 120 FP32
copies of shape 16×1024×3×32×2. The explicit expansion alone wrote
**2.109 GiB per update and rank**. All three removed groups account for
**6.328 GiB of logical reads plus writes per update and rank**, using the
trace analyzer's shape inventory. This is not a measurement of HBM traffic;
broadcast inputs, caches and kernel-internal buffers affect physical traffic.

The KV gradient still requires summing contributions from the query heads.
Strict traces retain 728 `aten::sum` calls in both variants. Normal traces
change from 728 to 488, while direct cuDNN attention performs 240
`fmha_reduce_head` kernel calls inside its backend. Removing a Python operation does not remove that
mathematical dependency. H2D and DtoH transfer counts and byte totals are
unchanged: this removal is within GPU execution.

Before the full experiment, an isolated probe captured actual Q/K/V and
upstream gradients from layers 0 and 29. Direct Flash attention reproduced
forward and all Q/K/V gradients exactly. cuDNN showed tiny Q-gradient
differences and its own repeat variability. Its isolated forward/backward
median improved from roughly 0.518 to 0.428 ms; these are diagnostic kernel
measurements, not the full-step gain. See
[`attention-probe/results.json`](attention-probe/results.json).

## Reproduction and local archive

Append the option to the previous optimized launch:

```bash
--mode native --steps 100 --warmup-steps 10 --norm-mode compatible \
  --gradient-bucket-views --defer-loss-readbacks --shared-projection-casts \
  --direct-gqa
```

Add `--deterministic --skip-uninitialized-fill` for strict runs, and select
`--mode peer` for P2P. The complete commands, environment and pipeline-lock
checks are in [`run-direct-gqa.py`](run-direct-gqa.py). The suite acquired both
Brandon pipeline locks, checked idle GPUs and released the locks on completion.
No environment upgrade or NVMe driver change was required.

All **150 payload files** are local and verified against
[`SHA256SUMS.json`](SHA256SUMS.json). The executed source snapshot matches the
local implementation byte for byte. Four compressed rank-0 traces were
independently reanalyzed locally and reproduce the archived summaries exactly.
Python compilation, four measurement tests and five queue tests pass.
[`local-verification.json`](local-verification.json) records those checks.

Archive SHA-256:
`1e2984391f1a907687e8c68ddd2199b480a69a0d58ab85238cb01f054ae8387d`.
`LOCAL_SHA256SUMS.json` additionally covers this report and the local archive.
Full continuation checkpoints remain under
`/home/ubuntu/ntransformer-h100/training/direct-gqa-20261014`;
the isolated probe's tensor captures remain remote, with hashes in its JSON.

The next implementation candidates and mathematical architecture analysis are
in [`TRAINING_RESEARCH.md`](../../../TRAINING_RESEARCH.md). Parallel and
reversible blocks there are proposals with small CPU algebra checks; no new
architecture has been trained or demonstrated better instruction following.
