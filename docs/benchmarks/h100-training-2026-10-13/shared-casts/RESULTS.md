# Shared BF16 projection inputs on two H100 SXM5 GPUs

Sharing the forward activation conversion across Q/K/V and gate/up improves
the already optimized native trainer by **2.41% under strict determinism** and
**1.75% under the normal attention policy**. The normal-mode allocated peak falls
from **14.943 to 12.959 GiB per GPU**. All six deterministic 100-step cases pass
the complete exact-state audit; the normal pair passes the unchanged numerical
gate. The optional P2P mode also improves, by **2.58%**, with exact state.

## Implementation and numerical contract

`--shared-projection-casts` enables scoped sharing in
[`projections.py`](../../../../python/ntransformer_training/projections.py).
Each attention/FFN group creates one detached BF16 activation and presents its
storage to separate autograd cast nodes. Each backward branch converts its own
gradient to FP32 **before** accumulation at the original activation. A single
differentiable shared cast would instead accumulate in BF16 and change rounding.
The scope is cleared after forward, including exceptional exits; GEMM's saved
tensors retain the shared storage for weight gradients. Recomputed blocks create
fresh scopes. No tensor cache survives between optimizer steps.

PyTorch GEMMs, weight autocast, attention, AdamW, compatible native RMSNorm and
SwiGLU are unchanged. Parameters and state-dict keys are preserved. The CUDA
training library remains `b5039135d5c50d4c019ea57046d2ae9595269ea8584cd6fee10f9455f9f450ee`.

The supported path is FP32 nonleaf activations under CUDA BF16 autocast, with
first-order autograd and sequential host calls per module. Other dtypes/devices,
inference tensors and trainable leaves use the original linear path. An initial
isolated test caught the leaf case: PyTorch's existing autocast cache can already
share a trainable leaf's differentiable cast. Preserving that path avoids changing
its accumulation. That prototype stopped before any 100-step benchmarks. The
final tests exercise both ordinary activations and this fallback.

## Paired measurements

Every run restores the same final pretraining checkpoint, AdamW and RNG states.
Architecture, corpus, tokenizer, BF16 autocast/FP32 parameters, 16 sequences per
GPU, two accumulation microsteps and 65,536 global tokens per update are fixed.
All cases include the prior bucket-view and deferred-loss-readback options;
strict cases also omit audited uninitialized-memory debug fills. Only the shared
projection option differs within each pair.

Each continuation has 100 updates; the first ten are warmup. The continuous
measurement window includes sampling, hashing, communication, clipping, AdamW,
logging and timing. Initialization, validation and checkpoint persistence are
separate. Tokens/s below sum both GPUs. Profiles are separate runs.

| Case | Global tokens/s | Compute allocated peak GiB, ranks 0 / 1 |
|---|---:|---:|
| Strict control | 315,716 | 14.943 / 14.943 |
| Strict shared | 323,408 | 13.360 / 13.360 |
| Normal control | 343,239 | 14.943 / 14.943 |
| Normal shared | 349,241 | 12.959 / 12.959 |
| Strict shared, repeat | 324,067 | 13.360 / 13.360 |
| Strict control, repeat | 316,505 | 14.943 / 14.943 |
| Strict P2P control | 241,182 | 3.905 / 3.909 |
| Strict P2P shared | 247,411 | 3.908 / 3.912 |

The strict means are **316,110 → 323,738 tokens/s (+2.41%)**. The second pair
reverses execution order. Two repetitions are descriptive, not a confidence
interval; normal and P2P pairs each ran once. Strict shared mode remains **7.30%
slower** than normal shared mode, which uses a different attention policy.

Strict allocated peak drops **1.583 GiB (10.59%)** on both GPUs; normal mode drops
**1.984 GiB (13.28%)**. These are allocator peaks, not whole-device VRAM readings.
The normal-mode reduction exceeds the direct saved-activation inventory; the
additional difference is not assigned to an unmeasured cause.

P2P has effectively unchanged compute peak and an additional **1.055 GiB peer
pool per rank**, with 30 reserved blocks and two staging buffers. Each physical
GPU also holds the opposite rank's peer pool. Both P2P cases report exactly
6,000 writes, 6,000 reads and 5,800 lookahead prefetches per rank across 200
microbatches. This experiment preserves P2P DMA and its reservation/lifetime
protocol; it does not make offloading faster than keeping this small model's
activations local.

## Correctness

All six strict cases pass `--require-exact` against the archived deterministic
PyTorch reference: 273 state-dict entries representing 272 unique parameters,
AdamW state, CPU/CUDA RNG state, every recorded loss/gradient norm/LR/token count,
and final per-source validation. Batch hashes match. Tests also establish exact
full-model loss and all 272 parameter gradients on one real pretraining batch
and a variant with half its targets masked. The latter is a rounding test, not
a new SFT or instruction-following evaluation.

The normal-policy pair passes the original acceptance limits:

* Maximum training-loss difference: **0.000248075**.
* Final mean validation difference: **0.0000242554**.
* Weight discrepancy relative to the reference update L2: **1.73949%**.
* Maximum absolute weight difference: **0.000442415**; RNG remains exact.

That update discrepancy lies at the previously measured 1.729–1.774% scale of
reference repetition. It is not 1.739% of total weight norm, a formal statistical
bound or evidence of long-run convergence. Strict exact-state controls provide
the additional implementation check; no tolerances were widened.

## Work removed, measured independently of speed

Two warmed updates per rank were profiled after eleven preceding updates.
Both ranks show the same counts below:

| Events in two updates | Control | Shared |
|---|---:|---:|
| FP32 → BF16 copies, shape 16×1024×576 | 844 | 484 |
| BF16 → FP32 copies, same shape | 604 | 604 |
| All `aten::copy_` calls | 4,838 | 4,478 |
| CUDA kernel events | 13,944 | 13,584 |
| `aten::mm` calls | 2,532 | 2,532 |

Thus **180 forward conversions per update and rank disappear**, while the
backward conversion count stays intact. The removed tensors account for
**9.492 GiB of logical reads plus writes per update and rank**. This is computed
from tensor shapes and dtypes, not measured HBM traffic. H2D/DtoH transfer counts
and bytes are unchanged. The change removes device-side representation work,
without claiming that weights previously traveled through CPU RAM each step.

## Reproduction and local evidence

Append `--shared-projection-casts` to the prior optimized launch:

```bash
--mode native --steps 100 --warmup-steps 10 --norm-mode compatible \
  --gradient-bucket-views --defer-loss-readbacks --shared-projection-casts
```

For strict mode add `--deterministic --skip-uninitialized-fill`; select
`--mode peer` for the capacity experiment. Defaults remain unchanged. The full
guarded commands and immutable source snapshot are in
[`run-shared-casts.py`](run-shared-casts.py) and [`source-snapshot`](source-snapshot/).
The runner acquired both existing Brandon pipeline locks and checked idle GPUs.
All GPU runs finished and released the locks before archive processing.

[`projection-tests.json`](projection-tests.json) records branch-rounding,
storage-alias, fallback and full-model checks. Group tests also cover multiple
outstanding forwards, exception cleanup, deepcopy and CPU fallback. Local
Python compilation and the four measurement/five queue tests pass. The CUDA
checks use the existing remote PyTorch environment; no environment upgrade was
needed. Both compressed rank-0 traces were independently reanalyzed locally and
reproduced their archived summaries exactly.

All **121 payload files** were downloaded and verified against
[`SHA256SUMS.json`](SHA256SUMS.json), and the final local implementation matches
the executed source snapshot byte for byte. Compressed traces, trace summaries,
per-step/rank metrics, comparison JSON, checkpoint hashes and scripts are local.
Archive SHA-256: `1bdf43628a2f6eb9ff1c08b6dbf03d9ea1806086afc4ecda75498682d9a31a99`.
`LOCAL_SHA256SUMS.json` additionally covers this report and the local archive.
Full checkpoints remain under
`/home/ubuntu/ntransformer-h100/training/shared-casts-20261014-r2`.

See the [complete dataflow and design references](../../../H100_TRAINING_DATAFLOW.md)
for the connection to Karpathy's TinyStories/llama2.c and nanochat work and the
remaining RoPE, grouped-KV and logits materialization candidates.
