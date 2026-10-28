# H100 DeepSeek: cooperative Q8 backbone loads

Controlled comparison on two H100 80 GB SXM5 GPUs, NVLink, DeepSeek-V4.1-Flash-Q8_0 (10 shards). Both variants use owner-GPU expert execution, 62 GiB expert cache per GPU and FP32 routed arithmetic over MXFP4 weights.

The only runtime difference is `DS41_Q8_COOPERATIVE=0|1`. Q8_0 weights, FP32 activations, block dot-product order and warp reduction are unchanged. This is a memory-access optimization, independent of the routed FP32/F16/Q8 selector.

## End-to-end results

64 teacher-forced continuation steps, identical corpus token IDs, one cold request and three warm repeats. Decode throughput is 64 divided by median warm decode time. Context order is 32, 128, 512; candidate/control order reverses at 128.

| Prompt tokens | Legacy decode tok/s | Cooperative decode tok/s | Gain | Legacy prefill tok/s | Cooperative prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 32 | 16.3772 | 23.5314 | 43.68% | 17.1573 | 25.2541 |
| 128 | 15.4544 | 21.6056 | 39.80% | 16.5632 | 23.8683 |
| 512 | 13.3465 | 17.6649 | 32.36% | 14.1706 | 19.1550 |

All 25,209,600 logits match bitwise across variants; every cold/warm repetition also matches within its variant. This proves equivalence on these 195 prediction positions, not general task quality. Exact logits remain diagnostic, rather than a requirement for future optimizations.

## Stage measurements

| Context | Variant | Attention ms/token | QKV ms/token | Resident attention + tail ms/token | Compression ms/token | MoE ms/token |
|---:|---|---:|---:|---:|---:|---:|
| 32 | legacy | 33.210 | 12.267 | 17.629 | 3.263 | 11.701 |
| 32 | cooperative | 14.681 | 4.598 | 6.894 | 3.137 | 11.621 |
| 128 | legacy | 36.852 | 12.252 | 21.115 | 3.431 | 11.695 |
| 128 | cooperative | 18.465 | 4.650 | 10.391 | 3.371 | 11.685 |
| 512 | legacy | 43.332 | 12.264 | 26.999 | 4.005 | 15.399 |
| 512 | cooperative | 24.901 | 4.662 | 16.304 | 3.870 | 15.439 |

Host-side Session timers include submission and waits; QKV, resident attention/tail and compression are attention substages and must not be added again to total attention. These are not isolated GPU busy times.

## Kernel microbenchmarks

CUDA-event spans, median of seven batches of 20 launches, alternating variant order, five warmups. Operands are reused and may hit device cache: these are warm isolated projections, not a full-model bandwidth measurement.

| Rows | Columns | Groups | Legacy ms | Cooperative ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1280 | 5120 | 1 | 0.047867 | 0.012182 | 3.929x |
| 32768 | 1280 | 1 | 0.188480 | 0.037666 | 5.004x |
| 512 | 5120 | 1 | 0.025224 | 0.011349 | 2.223x |
| 1024 | 4096 | 8 | 0.147874 | 0.024984 | 5.919x |
| 5120 | 8192 | 1 | 0.185018 | 0.036006 | 5.138x |

## Memory and remaining host work

| Context | Variant | GPU 0 peak MiB | GPU 1 peak MiB | RSS peak MiB | Warm weight H2D MiB/token |
|---:|---|---:|---:|---:|---:|
| 32 | legacy | 64017 | 72505 | 91044 | 0.000 |
| 32 | cooperative | 64017 | 72505 | 91020 | 0.000 |
| 128 | legacy | 64017 | 72505 | 112663 | 0.000 |
| 128 | cooperative | 64017 | 72505 | 112646 | 0.000 |
| 512 | legacy | 64017 | 72511 | 151865 | 68.917 |
| 512 | cooperative | 64017 | 72511 | 151879 | 68.917 |

VRAM peaks are one-second telemetry samples. RSS is time(1) high-water memory. The cooperative kernel uses 12,928 bytes of shared memory per CTA, with no additional global weight/activation allocation or repacking.

Each warp loads contiguous words for 32 Q8 blocks; a CTA loads one activation tile shared by eight rows. The arithmetic follows the same lane/block sequence. Columns not divisible by 64 retain the legacy path, avoiding unaligned row reads. Partial rows still participate in CTA barriers.

The change does not remove CPU scheduling, QKV mirror readback, compressor/indexer host work or tail readback. Expert cache misses still stage through pinned CPU memory. At context 512, even warm requests upload about 68.9 MiB/token because of cache churn. P2P expert ownership and cache budgets are unchanged. Further progress should target those remaining boundaries and attention reduction costs with separate evidence.

## Validation and reproduction

48 CTests passed without skips/failures. The new direct CUDA test covers zero/random/large activations, varied scales, partial CTAs, grouped matrices, aligned tiles and odd-block fallback, comparing both kernels bitwise and against an independent double-precision CPU dot. Eight cross-process prologue/tail fixture outputs also match within their paired controls.

`prepare.py`, `run.py`, `analyze.py`, source snapshots, command/environment hashes, corpus/token IDs, raw logits, timing JSONL, telemetry and resource logs are included. `DS41_Q8_COOPERATIVE=1` is opt-in; `0` is the control. Recalibrate precision policies after changing the binary; old policies are intentionally bound to their recorded binary hash.

The previous llama.cpp numbers used another execution/token protocol. This experiment does not establish an updated cross-engine performance gap.
