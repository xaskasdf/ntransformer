# DeepSeek MXFP4 memory-access experiment

One physical H100 (GPU 1), the same ten GGUF shards, FP32 activations and the unchanged 62 GiB expert cache. Each process runs one initial request and three warm repeats with 64 decode steps. EOS is recorded without stopping fixed work. Throughput divides total warm work by total warm time; loading, validation and output files are excluded. Warm repeats do not establish independent process-level variance.

| Case | Prefill tok/s | Decode tok/s | Decode ratio | Matched argmax | Relative L2 |
|---|---:|---:|---:|---:|---:|
| combined-p128 | 11.3676 | 11.5965 | 1.3444 | 65 / 65 | 0 |
| combined-p32 | 10.9232 | 11.5367 | 1.3436 | 65 / 65 | 0 |
| combined-p512 | 11.2750 | 9.4616 | 1.2799 | 65 / 65 | 0 |
| control-p128 | 8.4986 | 8.6260 | 1.0000 | 65 / 65 | 0 |
| control-p32 | 8.2587 | 8.5866 | 1.0000 | 65 / 65 | 0 |
| control-p512 | 8.4168 | 7.3923 | 1.0000 | 65 / 65 | 0 |
| cooperative-p32 | 8.6890 | 9.0613 | 1.0553 | 65 / 65 | 0 |
| cooperative-register-p32 | 10.4114 | 11.5429 | 1.3443 | 65 / 65 | 0 |
| cooperative-register-trace | 10.8952 | 11.5143 | 1.3410 | 65 / 65 | 0 |
| cooperative-trace | 8.6577 | 9.0227 | 1.0508 | 65 / 65 | 0 |
| lut-control | 8.2674 | 8.6067 | 1.0023 | 65 / 65 | 0 |
| lut-trace | 8.1692 | 8.5723 | 0.9983 | 65 / 65 | 0 |
| pilot-control | 8.2737 | 8.6070 | 1.0024 | 65 / 65 | 0 |
| pilot-trace | 8.2302 | 8.5733 | 0.9985 | 65 / 65 | 0 |

## MoE phase measurements

Trace-enabled cases record host intervals and adjacent events on the existing CUDA stream. Host staging overlaps device work; these columns must not be added. Preparation includes the early shared expert and possible host submission gaps. Compute spans include launches and any stream idle gaps, not isolated GPU busy time. Trace overhead is measured separately; throughput claims use untraced cases.

| Traced case | Host staging ms/token | Upload stream ms/token | Routed compute stream ms/token | Host wait ms/token |
|---|---:|---:|---:|---:|
| cooperative-register-trace | 14.5248 | 10.3748 | 6.4048 | 17.5111 |
| cooperative-trace | 14.5801 | 10.3603 | 30.1771 | 41.3029 |
| lut-trace | 14.5663 | 10.3758 | 35.9812 | 47.1463 |
| pilot-trace | 14.5492 | 10.3607 | 36.0378 | 47.1953 |

| Case | MoE wall ms/token | Routed upload MiB/token | Cooperative launches/token | Sampled peak GPU GiB | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| combined-p128 | 33.3511 | 429.7522 | 80 | 70.8037109375 | 112.1396 |
| combined-p32 | 37.3063 | 519.4006 | 80 | 70.8037109375 | 89.5648 |
| combined-p512 | 46.0907 | 755.0079 | 80 | 70.8095703125 | 155.8192 |
| control-p128 | 62.9888 | 429.7522 | 0 | 70.8037109375 | 112.1690 |
| control-p32 | 66.9799 | 519.4006 | 0 | 70.8037109375 | 89.5589 |
| control-p512 | 75.7515 | 755.0079 | 0 | 70.8095703125 | 155.8394 |
| cooperative-p32 | 61.0311 | 519.4006 | 80 | 70.8037109375 | 89.5536 |
| cooperative-register-p32 | 37.3101 | 519.4006 | 80 | 70.8037109375 | 89.5658 |
| cooperative-register-trace | 37.6663 | 519.4006 | 80 | 70.8037109375 | 89.5917 |
| cooperative-trace | 61.5480 | 519.4006 | 80 | 70.8037109375 | 89.6059 |
| lut-control | 66.8652 | 519.4006 | 0 | 70.8037109375 | 89.5903 |
| lut-trace | 67.3199 | 519.4006 | 0 | 70.8037109375 | 89.5847 |
| pilot-control | 66.8965 | 519.4006 | 0 | 70.8037109375 | 89.5646 |
| pilot-trace | 67.3475 | 519.4006 | 0 | 70.8037109375 | 89.5810 |

## Implementation and validation

The cooperative kernel keeps one warp per output row and the existing ordered FP32 accumulation. Warps load contiguous words for each 32-block MXFP4 tile, then decode from shared memory. The CTA loads the activation tile once, with a 33-float pitch to avoid shared-memory bank conflicts. Shared storage is 8,576 bytes per CTA; the weight cache and model representation do not change. Rows require columns divisible by 128 for aligned word loads; other shapes use the original kernel. Partial rows participate in all barriers and cannot write past output bounds.

Tests include all 16 quantized codes, changing activations, cache hits/misses, partial CTAs/tiles, alignment fallback, actual and maximum expert dimensions, routing duplicates, signed weights, callback ownership and exception recovery. Independent FP64 GEMV references and cross-process binary comparisons complement full-model finite/repeat checks. Identical logits are a regression result for these inputs, not an independent language-quality evaluation.

The register-only, cooperative-only and combined probes separate the interaction between the two changes. NVIDIA documents serialization when warp lanes read distinct constant-memory addresses and the benefit of coalesced global accesses in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#constant-memory). The observed interaction is consistent with one memory-access cost hiding another; no hardware-counter trace was obtained to prove individual stall causes.

The pilot and isolated register-LUT probes use benchmark_pilot.cpp with the prior attention library. Candidate cases use the new harness with its additional cooperative-launch counter. Compiler flags, binary/library hashes, environments and source snapshots are archived. Model assets stay external.

Final source enables both flags by default. Set DS41_MXFP4_COOPERATIVE=0 and DS41_MXFP4_REGISTER_LUT=0 for the original control; set just one flag to zero to isolate its contribution. The final build passes 28 selected DS41 CTest cases, 18 cross-process numerical cases with both flags unset in the default fixtures. These final correctness checks use a 2 GiB expert-cache budget alongside Brandon training; no throughput is measured concurrently. The isolated full-model timings come from the candidate build with both flags explicitly enabled. A source check confirms that final production changes are only the two flag defaults and a size comment. candidate-source and candidate-environment.json preserve the opt-in build used for the factorial and paired-context cases; initial-source preserves the prior attention source used by the pilot library. The compiled harness is the root benchmark_ds41.cpp (benchmark_pilot.cpp for the pilot), whose hash is recorded separately. The final source/tools copy is synchronized with that harness.

Memory peaks are one-second samples. Byte counters are successful logical payloads, not physical bus transactions. Only one GPU is used; peer ownership/cache and copy/compute overlap remain separate experiments.
