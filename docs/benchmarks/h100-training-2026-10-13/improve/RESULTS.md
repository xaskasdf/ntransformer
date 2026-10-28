# H100 training experiment results

Status: **complete**.

Experimental hybrid backend: native CUDA RMSNorm/SwiGLU and peer activation staging; PyTorch supplies GEMM, attention, AdamW and DDP.

| Case | Compute GPUs | Tokens/s | Same-topology ratio | Seconds/step | Validation loss |
|---|---:|---:|---:|---:|---:|
| reference | 2 | 301550.9 | 1.000x | 0.2173 | 2.170061 |
| native | 2 | 334813.9 | 1.110x | 0.1957 | 2.170073 |
| checkpoint | 2 | 260471.2 | 0.864x | 0.2516 | 2.170076 |
| peer | 2 | 254508.6 | 0.844x | 0.2575 | 2.170069 |

Each measured case continues the same checkpoint for 100 steps; the first 10 steps are excluded from throughput. Single-GPU cases process the same global batch with four microbatches. Ratios compare DDP cases to the DDP reference, and single-compute-GPU cases to single-checkpoint. Throughput includes the continuous measured window, including logging and instrumentation gaps.

| Case | Compute-device peak allocated GiB per rank | Peer-device allocation GiB per rank |
|---|---:|---:|
| reference | 18.437 | 0.000 |
| native | 14.943 | 0.000 |
| checkpoint | 4.878 | 0.000 |
| peer | 3.908 | 1.055 |

These are PyTorch allocator peaks, not whole-device occupancy. In bidirectional DDP each physical GPU also holds the other rank's peer blocks. Adding independent peaks is an upper bound, not a simultaneous device measurement. Peer staging trades locality for capacity; it does not reduce total storage by exchanging equal blocks.

Historical stage1 snapshot at step 25400: 307985.2 global tokens/s. This is contextual evidence; paired ratios above isolate the implementation change using common weights, AdamW state, batches, final LR and timing.

## Compatible RMSNorm audit

This run uses native CUDA pointwise RMSNorm with PyTorch reductions, plus precise native SwiGLU. The 100-step SFT comparison uses the unchanged 391-example curriculum, same batches and initialization, and a fresh AdamW for both implementations.

SFT audit: **passed**. All 273 final model tensors match exactly; optimizer state exact: True; RNG state exact: True. Recorded training-loss and final validation-loss differences: 0.0 and 0.0. `--require-exact` enforces this stronger gate.

The separate primitive/full-model test reproduces all 272 unique parameter gradients on the real first SFT batch. There are 273 state-dict entries because embedding/output weights are tied. Exact SFT equivalence does not extend to the pretraining comparison: its three variants pass the original numerical tolerances with nonzero differences.

The fully fused RMSNorm variant remains available with `--norm-mode fused`. Its earlier +25.8% pretraining result is a different implementation; it failed the fresh-optimizer SFT check. The compatible mode is now the default.

One 100-step run per pretraining variant, in reference/native/checkpoint/peer order; no repeated-run uncertainty interval or long-convergence claim. The SFT checkpoint is identical to the previously decoded reference (56/120 transformations, 0/60 original instructions); no new generation score or quality gain is claimed here.

Portable records include source snapshots, execution scripts, reference SFT metadata and checkpoint hashes in `provenance.json`. Full model/optimizer checkpoints remain on the instance. `behavior-reference` contains a separately completed curriculum experiment: 52/120 transformations and 0/60 original instructions, with different data and decoding limits; it is not this acceleration comparison.

See `validation.json`, `comparison-*.json`, per-case `summary.json` and the saved logs for diagnostics.
