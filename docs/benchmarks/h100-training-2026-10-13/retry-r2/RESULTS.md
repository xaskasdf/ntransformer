# H100 training experiment results

Status: **complete**.

Experimental hybrid backend: native CUDA RMSNorm/SwiGLU and peer activation staging; PyTorch supplies GEMM, attention, AdamW and DDP.

| Case | Compute GPUs | Tokens/s | Same-topology ratio | Seconds/step | Validation loss |
|---|---:|---:|---:|---:|---:|
| ddp-reference | 2 | 302255.0 | 1.000x | 0.2168 | 2.170089 |
| ddp-native | 2 | 381597.1 | 1.263x | 0.1717 | 2.170049 |
| ddp-checkpoint | 2 | 292319.7 | 0.967x | 0.2242 | 2.170067 |
| ddp-peer | 2 | 286946.4 | 0.949x | 0.2284 | 2.170081 |
| single-checkpoint | 1 | 149873.0 | 1.000x | 0.4373 | 2.170048 |
| single-peer | 1 | 149316.8 | 0.996x | 0.4389 | 2.170088 |

Each measured case continues the same checkpoint for 100 steps; the first 10 steps are excluded from throughput. Single-GPU cases process the same global batch with four microbatches. Ratios compare DDP cases to ddp-reference, and single-compute-GPU cases to single-checkpoint. Throughput includes the continuous measured window, including logging and instrumentation gaps.

Historical stage1 snapshot at step 25400: 307985.2 global tokens/s. This is contextual evidence; paired ratios above isolate the implementation change using common weights, AdamW state, batches, final LR and timing.

Last supervisor message: Validation and all bounded comparisons completed

See `validation.json`, `comparison-*.json`, per-case `summary.json` and `logs/` for diagnostics.
