# DeepSeek precision calibration and peer experts

The controls select **routed expert arithmetic**, not a new weight format.
DeepSeek's expert weights remain MXFP4 in the GGUF files and in both GPU
caches. This avoids expanding the cached weights to FP16 or FP32.

| Control | Behavior |
|---|---|
| `DS41_MXFP4_COMPUTE=fp32` | Existing cooperative MXFP4 kernel; default and calibration reference. |
| `DS41_MXFP4_COMPUTE=f16` | FP16 activation rounding and packed FP16 products; FP32 accumulation and exponent scaling. Blocks with activation magnitude above 4096 use the reference products to avoid overflow. |
| `DS41_MXFP4_COMPUTE=q8` | Symmetric activation quantization per 32 values; MXFP4 integer significands use signed DP4A with INT32 block accumulation, followed by FP32 scaling and accumulation. |
| `DS41_PEER_MODE=copy` | Two owner caches, expert ID modulo two; selected remote weights move by P2P to primary scratch. |
| `DS41_PEER_MODE=execute` | Same ownership/cache capacity; input moves to the owner GPU, which computes the expert and returns its output by P2P. |

Unset peer mode uses one GPU. `DS41_VRAM_GB` and `DS41_PEER_VRAM_GB`
control the respective cache budgets. Reduced arithmetic requires cooperative
loads and columns divisible by 128; other shapes use the existing FP32 path.
FP32 scales, activations after the GEMVs, nonlinearities and ordered final
expert reduction are retained. Shared experts, backbone, router and LM head
keep their existing arithmetic. This is not whole-model FP16 or Q8 conversion.

Peer execution uses CUDA cross-device events to order input and output copies.
Selected cache blocks remain reserved until both streams finish, including
failure and callback exception paths. Resident peer weights avoid CPU staging;
cold owner-cache misses still use pinned host buffers. The current interface
is synchronous and non-reentrant, with the calling thread managing dispatch.

## Reproduce and select

Build the engine and link `tools/benchmark_ds41.cpp` against the resulting
library. Use the same binary, model, corpus, visible GPU order, placement and
cache budgets for every precision candidate. Set:

```sh
export CUDA_VISIBLE_DEVICES=1,0
export DS41_GPU=1 DS41_GPU_ATTN=1 DS41_PROFILE=1
export DS41_VRAM_GB=62 DS41_PEER_VRAM_GB=62
export DS41_PEER_MODE=execute
export DS41_BENCH_ALL_LOGITS=1 DS41_BENCH_TEACHER_FORCING=1
```

Run the benchmark for `fp32`, `f16` and `q8`, with 32 and 128 prompt tokens,
64 continuation steps and three warm repeats. The benchmark consumes actual
corpus continuation tokens in all candidates, even when their predictions
diverge. It saves the identical inputs/targets and all 65 logit vectors.
Each process records its own repeatability independently of cross-mode error.

The experiment runner in the accompanying benchmark artifacts preserves the
required command metadata, execution identity and source hashes. Then run:

```sh
python tools/select_ds41_precision.py /path/to/experiment
```

The selector writes `precision-policy.json` with the selected mode, evidence,
acceptance thresholds, runtime environment and binary/GPU identities. It
selects the fastest eligible candidate across both contexts. Default limits:

- Mean KL divergence ≤ 0.005; P99 KL ≤ 0.05.
- Mean target NLL increase ≤ 0.01.
- Median decode speedup ≥ 3% on each workload, with the candidate's slowest
  warm repeat faster than the reference's fastest warm repeat.

Raw relative logit L2 is diagnostic by default: adding the same constant to
every logit leaves softmax probabilities unchanged. `--max-relative-l2` can
enable a separate numerical-distance gate when a task specifically needs it.

These are explicit experimental thresholds, configurable through the tool;
they are not a guarantee of general task accuracy. Argmax agreement is
reported, **not required to be 100%**. FP32 is selected if no reduced mode
passes both numerical and speed gates. Different tokens, cache budgets,
devices, binaries, runtime fallback or missing repeat validation invalidate
the comparison instead of producing a recommendation.

Optional `--run /path/to/bench MODEL CORPUS PROMPT 64 3 OUTPUT LABEL` executes
the measured benchmark with the selected environment after checking its
binary, GPU/driver, model path and corpus identity. The selector runs outside
decode, once per calibration/launch; it adds no per-token host decision.
It is not a general runtime policy loader, and `DS41_MXFP4_COMPUTE=auto` is not
a supported kernel setting. For a different executable or workload, calibrate
again before applying the selected mode.

## Evaluation scope

Two contexts from one corpus are a first numerical calibration. Acceptance
for broader use additionally needs held-out corpora, longer contexts and
task-specific checks, including instruction following. A faster mode can
have different logits without being worse on those tasks, and close logits
alone cannot certify task quality. Keep that evaluation separate from the
placement experiment and from autoregressive throughput comparisons.

Synthetic coverage includes independent CPU arithmetic references, zero/tiny
inputs, large FP16 overflow-control inputs, unaligned fallback, partial CTA
tiles, real/max expert shapes, duplicate/negative expert weights, cold/hot
caches and callback exception recovery. Run `ctest --test-dir build -V`
and `python tests/test_ds41_precision_selector.py`; inspect GPU skips.

CUDA semantics: [packed half multiplication](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html),
[signed DP4A](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html),
[multi-GPU event ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html).
