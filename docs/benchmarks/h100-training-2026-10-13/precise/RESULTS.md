# Corrected precise-math H100 results

The training library excludes inference fast math; the SwiGLU small-gradient regression test passes. These measurements use the corrected binary, whose hash is in `inputs.json`.

| Mode | Global tokens/s | Mean step ms | Peak allocated GiB/compute GPU | Final validation |
|---|---:|---:|---:|---:|
| Reference | 301703.8 | 217.220 | 18.036 | 2.1700738 |
| Native RMSNorm + SwiGLU | 379539.6 | 172.672 | 14.539 | 2.1700669 |

Paired throughput ratio: **1.2580x**. One 100-step continuation per backend from the identical completed pretraining checkpoint and AdamW state, using identical batches; first 10 updates excluded from throughput. The denominator is one continuous 90-step window including sampling, transfers, synchronization, logging and instrumentation, excluding evaluation/checkpoint saving. This is a short fixed-order measurement, not a full-pretraining convergence or energy comparison.

Numerical comparison: **passed**; max training loss delta 0.0002102852, validation delta 6.902963e-06, parameter discrepancy 1.7348% of the reference update norm. All 273 state-dict tensors changed and remained finite.

The full native path passes this matured-optimizer pretraining experiment but fails the fresh-optimizer SFT update check. The corrected SwiGLU-only SFT path matches reference weights exactly after 100 steps; see `../instructions/RESULTS.md`. P2P capacity/throughput measurements under `../retry-r2/` use the initial binary; the corrected library passes the GPU primitive/model/P2P tests in `validation.json`.
