# Verified instruction-following pilot

Status: **complete**.

**The initial native SFT failed the numerical acceptance check.** Completed execution and similar generation scores do not establish update equivalence.

Both implementations start from the completed original SFT checkpoint and train 100 updates on the same 391 teacher-verified examples with a fresh AdamW. Native mode supplies CUDA RMSNorm/SwiGLU; PyTorch supplies the rest of training. All train and validation batch hashes match.

| Checkpoint | All held-out transformations | Original instruction diagnostic |
|---|---:|---:|
| before | 0/120 | 0/60 |
| reference | 56/120 | 0/60 |
| native | 55/120 | 0/60 |

Success requires greedy generation to emit a stop token and exactly the expected text, ignoring only outer whitespace. Transformation decoding allows 64 new tokens; the original diagnostic retains its 32-token limit. The final step-100 checkpoints are scored without selection using generation results.

| Transformation family | Before | Reference | Native |
|---|---:|---:|---:|
| uppercase | 0/20 | 3/20 | 1/20 |
| join | 0/20 | 14/20 | 14/20 |
| replace | 0/20 | 0/20 | 0/20 |
| remove_marker | 0/20 | 20/20 | 20/20 |
| brackets | 0/20 | 19/20 | 20/20 |
| json_record | 0/20 | 0/20 | 0/20 |

All 120 validation requests are included, even when the teacher failed. JSON has no accepted training examples. Inputs and paraphrases are held out within these task families; the original 60 cases test different skills and never enter training.

| Validation loss on the accepted synthetic subset | Initial | Final |
|---|---:|---:|
| reference | 2.781040 | 0.301906 |
| native | 2.780424 | 0.308337 |

These losses use the 78 accepted validation examples, not all 120 generation cases, and are not comparable to the original SFT corpus loss. A gain on transformations alone does not establish general instruction following.

Configuration: 2 H100, batch 8/GPU, two accumulation steps (32 examples/update), BF16 autocast and FP32 parameters; AdamW 5e-5, 10 warmup steps, decay over the final 80%, floor 10%, weight decay 0.01, clipping 1.0 and seed 20261013. Checkpoints remain on the remote instance; local records contain all generated responses, expected answers, stop flags, metrics, batch hashes and provenance.

Paired numerical check: **failed**. Maximum training-loss delta 0.00607754; final validation delta 0.00643078; parameter discrepancy 18.707% of the reference update L2 norm. See `analysis.json` for limits and response agreement.

| Numerical diagnostic | Result | Update discrepancy |
|---|---|---:|
| native-fixed | failed | 19.2576% |
| norms-only | failed | 19.2576% |
| reference-repeat | passed | 0.0000% |
| swiglu-fixed | passed | 0.0000% |
| swiglu-only | failed | 12.1780% |

The limit remains 10% of the reference update norm. The reference repetition tests reproducibility. `*-only` uses the original library; `*-fixed` uses the precise-math build after removing inherited inference fast math.

Corrected SwiGLU-only SFT: **56/120 transformations, 0/60 original instructions**. Its weights and quality records are separate from the failed initial full-native run. RMSNorm stays on the reference path for this SFT configuration.
