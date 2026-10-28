# DeepSeek DS41 audit — 2026-10-15

Implemented a correction to Q8 resident-upload byte counts and input-buffer
initialization in `src/model/ds41/ds41_gpu.cu`. Added four fresh-process CTest
cases in `tests/test_ds41_backbone.cpp`. Full investigation and proposed
experiments: [DeepSeek inference optimization](../../DEEPSEEK_INFERENCE_OPTIMIZATION.md).

## Validation

Hardware: NVIDIA H100 80 GB SXM, physical GPU1 selected with
`CUDA_VISIBLE_DEVICES=1`. CUDA 13.1.115, GCC/G++ 14.4.0.
Brandon training was active on both GPUs. Tests used small synthetic matrices,
without loading model weights or allocating an expert cache.

| Case | Coverage | Result |
| --- | --- | --- |
| Prologue | Q8 projections, weighted RMSNorm, RoPE; 3 different inputs | Pass |
| Tail | Two grouped Q8 projections; 3 inputs | Pass |
| Compressor | BF16 gated and ungated paths; 3 inputs | Pass |
| Indexer | Q8/BF16, input buffer 32→256 elements, repeated reuse | Pass |
| Disabled GPU | `DS41_GPU=0` | Exit 77, explicit skip |

Largest observed absolute error against the synthetic CPU calculation:
**1.43051147e-6**. The test gate is `2e-4 + 5e-5 * abs(reference)` per element,
with an explicit finite check. This is a primitive-level regression threshold,
not a full-model quality or determinism criterion.

Standalone release compilation and all four cases passed. The complete
`ntransformer_lib` and new test target also built through CMake, followed by:

```sh
ctest --test-dir cmake-build -V -R test_ds41_backbone
```

**4/4 passed, zero skips**, 1.01 s total CTest wall time. This duration includes
process/CUDA initialization and concurrent load; it is not a benchmark.
The CMake build retained the inference `--use_fast_math` configuration and
compiled SM80/86/89/90 targets; execution was on SM90 only.

The saved original source compiled with `-Wall -Wextra -Wuninitialized` emits
`bytes may be used uninitialized` in both attention prologue and tail.
The historical `gputime.log` documents GPU resident-upload failures and CPU
fallback. No new full-model run was performed and no speedup is claimed.
The main remote source/build used by ongoing DeepSeek work was not replaced;
the corrected code was built in an isolated audit directory.

## Inventory and historical evidence

`layout.json` records all 1,046 tensors, ten shard header hashes, file sizes,
types, dimensions and byte counts. It contains **header hashes only**, not
weight hashes. Tensor payload: **507,948,390,368 bytes / 473.0638 GiB**.
The model assets remain external.

`evidence/real_gpu_gen.log` contains a six-token prompt profile: attention
3.21 s, MoE 2.11 s, Engram 0.11 s, hyper-connections 0.19 s, other 0.46 s.
The final cache report spans prompt and generation: 10,080 slice lookups,
5,106 hits (50.7%), zero evictions. These two observations cover different
intervals. They do not establish a decode rate, an LRU eviction problem, or
performance of the corrected code.

## Reproduction and preservation

* Remote work: `/home/ubuntu/ntransformer-h100/ds41-audit-20261015`.
* Persistent copy: `/lambda/nfs/btiny/ntransformer-training/deepseek-audit-20261015`.
* Local evidence: this directory, including source snapshot, commands, test
  outputs, original compiler diagnostics, historical logs and `SHA256SUMS`.
* `checks.json` records standalone commands, GPU/process observations and hashes.
* `cmake-checks.json`, `configure.log`, `cmake-build.log`, `ctest.log` record
  repository integration checks.
* `cmake-source.tar` preserves the complete build input (including pre-existing
  unrelated source changes); `source/model/ds41/ds41_gpu.before.cu` preserves
  the original file from this audit.
* `layout.py` uses the preserved `gguf_scan.py` header reader. Adjust its
  explicit host paths to reproduce elsewhere. `run_checks.py` and
  `build_cmake.py` preserve the exact validation procedure.

Full-model parity and isolated performance measurement remain the next gate
before promoting the new offload behavior as a measured optimization.
