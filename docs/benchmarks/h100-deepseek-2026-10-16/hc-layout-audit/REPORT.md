# DeepSeek HC layout correction and prefill audit

The large cross-engine distribution discrepancy was mainly caused by a destination/source transpose in NTransformer hyper-connection mixing. The correction changes the directed residual connections to match the pinned llama.cpp graph. It retains the model weights, FP32 routed arithmetic, GPU kernels, P2P expert execution and cache configuration.

## Cause and correction

Sinkhorn internally used row-major `matrix[dst * hc + src]`. Its final flat copy populated `comb`, but the consumer reads the GGUF/ggml layout `comb[dst + src * hc]`. This applied the transposed mixing matrix. Both orientations retain the same row/column sums, so those checks and direct-versus-thread-pool equality did not detect the error. The corrected producer explicitly exports destination-fast storage.

The intended operation is `out[dst, d] = post[dst] * branch[d] + sum_src C[dst, src] * residual[src, d]`. The producer and consumer now agree on how `C` is stored.

Reference: `reference/deepseek4.cpp`, `build_hc_sinkhorn` and `build_hc_post`, from pinned fork commit `5210c7c5ed61dddaee6ed476623abf4b63093d16`. Only `ds41_forward.cpp.o` differs between the baseline and corrected static libraries; `object-validation.json` verifies all archive members.

## Same-token numerical results

Each workload uses the exact fair-benchmark prompt and 64 continuation tokens, with 65 full-vocabulary prediction vectors. Corrected runs use one independent process and five requests; every request is finite and repeats bitwise. The rows cover 260 prediction positions from two corpora, not a downstream task-quality certification. KL is NTransformer to native batched llama.cpp.

| Workload | Old NT NLL | Corrected NT NLL | llama NLL | Old KL | Corrected KL | Corrected argmax agreement |
|---|---:|---:|---:|---:|---:|---:|
| p32 | 2.169373 | 1.750829 | 1.757762 | 0.975198 | 0.007380 | 95.38% |
| p128 | 2.244304 | 2.148210 | 2.137048 | 0.513642 | 0.008898 | 96.92% |
| p512 | 2.533017 | 2.131490 | 2.128646 | 0.620707 | 0.007260 | 96.92% |
| heldout-p512 | 1.463863 | 1.380446 | 1.370942 | 0.423722 | 0.011263 | 93.85% |

The residual differences remain measurable. The serial-versus-batched llama control itself changes distributions; that places the remaining discrepancy in context but does not prove all residual error is rounding or establish equal task quality.

## Prefill control and remaining work

The same llama binary, two-GPU placement, 52 CPU threads and disabled operation-offload setting were rerun with ubatch 1 instead of 512. The control uses one warmup and one measured request per workload; historical native numbers use six warm requests. This isolates an execution-path change but is a diagnostic rather than a counterbalanced speedup estimate. Warm control phases recorded zero major page faults and input blocks.

| Workload | NT original prefill s | llama native prefill s | llama ubatch-1 prefill s | llama native-to-serial KL | Corrected NT-to-serial KL |
|---|---:|---:|---:|---:|---:|
| p32 | 1.267524 | 0.380551 | 0.943554 | 0.012550 | 0.017305 |
| p512 | 25.313871 | 4.896709 | 14.931220 | 0.005321 | 0.006347 |

NTransformer prefill still calls `Session::forward` once per prompt token. Each token traverses all layers. The native llama graph performs layer operations over multiple prompt tokens. The control confirms that batching contributes substantially to the prefill gap, while llama also retains an advantage on the single-token path.

At p512 in the original fair run, median warm host intervals for launch 0 were 9.952 s attention, 7.428 s MoE, 1.404 s Engram, 2.152 s HC and 4.366 s other, out of 25.307 s total. These are host intervals containing submissions/waits; nested stage times must not be summed with their parents or treated as GPU busy time. The HC top-level interval covers only the attention-side HC work; FFN-side HC work is accounted under other.

A p512 warm prefill executes 20,480 QKV readback waits, 40,960 HC projections and 20,480 router projections. QKV alone returns 140 MiB of intermediate data to the CPU. The code also synchronizes when returning dense projection outputs and the attention tail. Small transfer volume does not remove the cost of these repeated dependencies.

The next implementation should provide a layer-wise prompt path with a bounded token tile, dense GEMMs and experts grouped by owner GPU. Keep HC/norm/router/attention intermediates on the device; precompute token-history-dependent Engram row IDs once per prompt and stage those rows asynchronously. Preserve P2P execution and explicit expert-block reservation. Causal compressed-state publication and the delayed HC pre-mix must be checked against the corrected serial path before benchmarking. This audit does not implement that prefill path.

## Validation and timing scope

The asymmetric stochastic-matrix regression fails with the former export and passes with the correction. Release and ASan/UBSan runs pass four directed layout cases, 36 independent double-precision Sinkhorn cases and 84 existing direct/pool cases. The guarded outputs and asymmetric basis residuals test producer-to-consumer behavior, including production epsilon and 20 iterations.

Corrected phase times are retained in audit-summary.json as diagnostics. They are not a paired speedup claim: the correction uses one process per workload, and CPU-only validation ran concurrently with part of the p512 numerical case. timing-scope.json records overlapping commands. GPU monitoring found no foreign GPU processes; that monitor cannot exclude CPU-only interference. No kernel or precision change was made.

Historical cache traces, predicted tokens and precision-selector calibration used the pre-correction graph. They remain historical measurements, not correctness references for the corrected model; recompute them before accepting new cache or arithmetic policies.

Artifacts include inputs by reference to the locally preserved fair benchmark, full logits, raw per-round measurements, command environments, before/after sources, test results and GPU telemetry. The local collector verifies every payload hash and reproduces audit-summary.json from the full logits.
