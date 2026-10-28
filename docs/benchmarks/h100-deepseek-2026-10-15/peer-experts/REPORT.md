# DeepSeek: peer expert execution and precision calibration

Two H100 80 GB SXM5, NV18. Logical GPU 0 is physical GPU 1. Owner = expert ID modulo 2.
Single GPU: 62 GiB expert cache. Copy/execute: 62 GiB per GPU (124 GiB total); copy adds bounded transient slots.
Same MXFP4 weight bytes in every arithmetic mode. FP16/Q8 change routed dot products; scales, nonlinearities, accumulation, shared expert, attention and final reduction remain FP32.
Fixed corpus tokens, contexts 32/128, 64 decode steps, three warm repeats after one cold request. These are teacher-forced throughput measurements, not the earlier autoregressive workload.

| Context | Mode | Decode tok/s | Prefill tok/s | vs single | vs peer-copy | Host uploads MiB/token | P2P weights MiB/token | P2P activations+outputs MiB/token |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | single-fp32 | 11.5246 | 10.9763 | 1.0000x | 0.8427x | 503.15 | 0.00 | 0.000 |
| 32 | copy-fp32 | 13.6763 | 14.2576 | 1.1867x | 1.0000x | 0.00 | 2268.39 | 0.000 |
| 32 | execute-fp32 | 16.3525 | 17.1241 | 1.4189x | 1.1957x | 0.00 | 0.00 | 3.243 |
| 32 | execute-f16 | 16.0306 | 16.8153 | 1.3910x | 1.1721x | 0.00 | 0.00 | 3.242 |
| 32 | execute-q8 | 16.0730 | 16.8723 | 1.3947x | 1.1752x | 0.00 | 0.00 | 3.242 |
| 128 | single-fp32 | 11.4826 | 11.4582 | 1.0000x | 0.8796x | 426.11 | 0.00 | 0.000 |
| 128 | copy-fp32 | 13.0547 | 13.8776 | 1.1369x | 1.0000x | 0.00 | 2258.30 | 0.000 |
| 128 | execute-fp32 | 15.4324 | 16.5327 | 1.3440x | 1.1821x | 0.00 | 0.00 | 3.234 |
| 128 | execute-f16 | 15.2051 | 16.2865 | 1.3242x | 1.1647x | 0.00 | 0.00 | 3.234 |
| 128 | execute-q8 | 15.2190 | 16.2875 | 1.3254x | 1.1658x | 0.00 | 0.00 | 3.229 |

FP32 single/copy/execute logits are bitwise equal on both contexts. Each candidate also repeated bitwise within its own arithmetic mode.

## Automatic precision decision

Selected: **fp32**. Acceptance limits: `{"mean_kl": 0.005, "nll_increase": 0.01, "p99_kl": 0.05, "relative_l2": null, "speedup": 1.03}`.

| Mode | Context | Mean KL | P99 KL | NLL increase | Perplexity ratio | Max relative L2 | Argmax agreement | Speed vs same-placement FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f16 | 32 | 0.00042557 | 0.00530075 | 0.00387809 | 1.00389 | 0.0667733 | 98.4615% | 0.9803x |
| f16 | 128 | 0.00116979 | 0.00824613 | -0.00739148 | 0.992636 | 0.0578958 | 100.0000% | 0.9853x |
| q8 | 32 | 0.00477306 | 0.039696 | -0.021416 | 0.978812 | 0.319801 | 96.9231% | 0.9829x |
| q8 | 128 | 0.00500014 | 0.0211951 | 0.010573 | 1.01063 | 0.229418 | 100.0000% | 0.9862x |

f16: workload 0: speed improvement not established; workload 1: speed improvement not established


q8: workload 0: speed improvement not established; workload 1: mean_kl exceeds 0.005; workload 1: nll_increase exceeds 0.01; workload 1: speed improvement not established

Raw logit L2 is diagnostic by default; a common logit offset does not change softmax probabilities. The optional L2 gate is disabled for the final selection.

## Memory

| Context | Mode | Physical GPU 0 peak MiB | Physical GPU 1 peak MiB | Peak RSS MiB |
|---:|---|---:|---:|---:|
| 32 | single-fp32 | 4 | 72505 | 90808 |
| 32 | copy-fp32 | 64015 | 72649 | 90989 |
| 32 | execute-fp32 | 64017 | 72505 | 90983 |
| 32 | execute-f16 | 64017 | 72505 | 90934 |
| 32 | execute-q8 | 64017 | 72505 | 91230 |
| 128 | single-fp32 | 4 | 72505 | 112386 |
| 128 | copy-fp32 | 64015 | 72649 | 112609 |
| 128 | execute-fp32 | 64017 | 72505 | 112664 |
| 128 | execute-f16 | 64017 | 72505 | 112703 |
| 128 | execute-q8 | 64017 | 72505 | 112842 |

VRAM values are maxima of one-second telemetry samples, not allocation high-water counters. RSS is the process high-water value reported by time(1).

## Remaining pipeline time

| Context | FP32 placement | MoE ms/token | Attention ms/token | HC ms/token | Engram ms/token | Other ms/token |
|---:|---|---:|---:|---:|---:|---:|
| 32 | single-fp32 | 36.889 | 33.277 | 4.272 | 2.776 | 9.355 |
| 32 | copy-fp32 | 23.383 | 33.129 | 4.189 | 2.705 | 9.536 |
| 32 | execute-fp32 | 11.644 | 33.174 | 4.234 | 2.691 | 9.241 |
| 128 | single-fp32 | 33.723 | 36.917 | 4.258 | 2.762 | 9.264 |
| 128 | copy-fp32 | 23.325 | 36.789 | 4.220 | 2.710 | 9.348 |
| 128 | execute-fp32 | 11.649 | 36.837 | 4.220 | 2.694 | 9.216 |

These are existing host-side Session stage measurements, not isolated device busy time. Substage counters and overlapping work must not be added as independent costs.

Numerical calibration covers 130 prediction positions of one corpus. It does not establish general task accuracy or instruction following. Token equality is diagnostic, not an acceptance requirement. Limits are explicit experimental defaults and are configurable.

## Ownership and execution

Both GPU caches reserve all selected slices before submitting kernels. With at least 64 slots and at most 24 selected slices, this synchronous call keeps them alive until both streams finish. Cold misses still stage through CPU pinned memory; resident peer weights no longer traverse the CPU.
Execute mode uses cross-device input/ready events, P2P input/output copies and an ordered primary-GPU reduction. Callback exceptions drain both streams and invalidate both caches before reuse. Copy mode uses the same ownership/cache capacity, transferring the selected peer weight slices to temporary primary slots.
GPU placement and precision are independent opt-in controls. The default remains single-GPU FP32. Calibration happens outside decode, with no per-token CPU selector.

## Validation and artifacts

`ctest.log`, `numerical-tests.json`, per-case `*-command.json`, raw `all-logits.f32`, token IDs, timing JSONL, source snapshots, `environment.json`, `telemetry.csv`, and `precision-policy.json` preserve reproduction evidence.
Native CUDA arithmetic and peer ordering references: [half2 arithmetic](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html), [integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html), [multi-GPU programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html).
