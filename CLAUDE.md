# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NTransformer is a C++20/CUDA LLM inference engine with no dependencies beyond the CUDA Toolkit (no cuBLAS, no PyTorch). It is built to run models far larger than VRAM on a single GPU; the original target was Llama 70B on one RTX 3090. The repo now holds four loosely coupled tracks:

| Track | Where | State |
|---|---|---|
| Llama inference + tiered layer streaming | `src/{core,cuda,memory,model,inference}` | Mature. 8B Q8_0 ≈ 49 tok/s resident; 70B tiered 0.2–0.5 tok/s |
| MoE models: DeepSeek-V4.1-Flash (`ds41`), GLM-5.3-Flash (`glm5`) | `src/model/ds41/`, `src/model/glm5/` | Active. Exact CPU reference + GPU offload |
| NTRPU: custom FPGA inference accelerator | `src/ntrpu/` (simulator), `fpga/` (HLS/RTL), `tools/` | Active research. Needs no CUDA |
| Experimental H100 training ops | `src/training/`, `python/ntransformer_training/` | Opt-in, Linux only |

Design records, benchmarks, and decisions for each track live in `docs/` (see the docs map below), not in commit messages.

## Build

CMake options: `NTRANSFORMER_ENABLE_CUDA` (ON; the GPU engine, CUDA or HIP), `USE_HIP` (OFF), `NTRANSFORMER_BUILD_NTRPU_SIM` (ON), `BUILD_TESTING` (ON, via `include(CTest)`), `USE_GPUNVME` (OFF; links prebuilt static libs from `../gpu-nvme-direct/build-hw/`), `NTRANSFORMER_BUILD_TRAINING` (OFF), `NTRANSFORMER_NATIVE_ARCH` (ON; OFF swaps `-march=native` for `x86-64-v3`), and `NTRANSFORMER_STATIC_CUDART` (OFF). The default CUDA architectures are `80;86;89;90`. Pass `-DCMAKE_CUDA_ARCHITECTURES=86` (RTX 3090) or `90` (H100) for faster builds.

**CI** (`.github/workflows/build.yml`) runs compile-only jobs, because the runners have no GPU: Linux CUDA in a `nvidia/cuda` container (it also builds the CUDA tests), Windows CUDA (network installer plus MSVC plus Ninja), and Linux HIP in a `rocm/dev-ubuntu-24.04` container. The CUDA and ROCm versions are pinned in the workflow. The jobs build with `NATIVE_ARCH=OFF` and `STATIC_CUDART=ON` and upload the binaries as artifacts. A `v*` tag also creates a GitHub Release. Pushes that touch only docs, `fpga/`, `tools/`, `scripts/`, `python/`, or `linux_port/` skip the workflow. The NTRPU simulator is not in CI.

**HIP/ROCm** (`-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`) covers the Llama path only.
- The `.cu` files compile as HIP unchanged.
- `src/core/hip_compat/` resolves `<cuda_runtime.h>`/`<cuda_fp16.h>` to `src/core/cuda_to_hip.h`, a 1:1 alias list. Add an alias there when Llama-path code starts using a new CUDA API.
- Warp shuffles use `NT_WARP_MASK` (64-bit on HIP).
- The DS41/GLM5 sources are excluded. `src/model/moe_unavailable.cpp` replaces their `run_cli` and reports the model as unsupported.

**Linux** (gcc-14 + CUDA 13.1). gcc-15 is incompatible with CUDA 13.1, and CMake automatically picks `gcc-14` as the CUDA host compiler:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build -j
```

**Windows** (MSVC 14.5x + CUDA 13.2 + Ninja, run from a VS x64 developer environment). This matches the existing `build/win` configuration:
```powershell
cmake -S . -B build/win -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DNTRANSFORMER_BUILD_NTRPU_SIM=OFF -DBUILD_TESTING=OFF
cmake --build build/win
```
OS-specific code is confined to `src/core/filemap_{posix,win}.inl` (file mapping and host-RAM queries). gpu-nvme-direct/VFIO is Linux-only.

**NTRPU simulator only** (any C++20 compiler, no CUDA toolkit):
```bash
cmake -S . -B build/ntrpu-sim -DNTRANSFORMER_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/ntrpu-sim --config Release --parallel
```

**Training library** (Linux; uses `-Wl,--no-undefined`): configure with `-DNTRANSFORMER_BUILD_TRAINING=ON -DNTRANSFORMER_BUILD_NTRPU_SIM=OFF`, build target `ntransformer_training`, then `export NTRANSFORMER_TRAINING_LIB=$PWD/build-training/libntransformer_training.so`. The Python package loads it through ctypes.

`build/` also contains per-experiment output directories, not just CMake trees. Don't clean it wholesale.

### Fast-math gotcha
Release CUDA builds add `--use_fast_math` to every target **unless** the target sets the `NTRANSFORMER_PRECISE_MATH` property. The training library and the `ntransformer_ds41_shared` object library (`ds41_shared.cu`, `ds41_attention.cu`) set it so their results match the CPU reference reduction order. Put new CUDA code that must agree numerically with a CPU reference in such a target. Overriding `--fmad`/`--ftz` alone does not undo fast-math's `expf` mapping.

## Run

```bash
./build/ntransformer -m model.gguf -p "Hello" -n 128                 # resident
./build/ntransformer -m model.gguf -p "Hello" -n 32 --streaming      # tiered (model > VRAM)
./build/ntransformer -m model.gguf --benchmark -n 64
./build/ntransformer -m model.gguf --streaming --skip-threshold 0.98 # layer skip
./build/ntransformer -m model.gguf --self-spec --draft-k 3           # self-speculative
CUDA_VISIBLE_DEVICES=0,1 ./build/ntransformer -m model.gguf --peer-gpu 1 --local-layers 16 -p "..."
# Parity hooks (all model families): bypass the tokenizer, dump prompt logits as F32
./build/ntransformer -m model.gguf --token-ids 785,6722,315 --dump-logits out.bin -n 12 -t 0 --repeat-penalty 1.0
```
`-c/--ctx-size` (default 4096) caps the model's context and sizes the KV cache. The KV cache competes with resident layers for VRAM: 70B Q6_K gets 29 VRAM layers at ctx 512 versus 24 at ctx 4096. For deterministic comparisons, use `-t 0 --repeat-penalty 1.0`.

NVMe direct (Linux, root, `-DUSE_GPUNVME=ON`): `sudo GPUNVME_PCI_BDF=0000:01:00.0 GPUNVME_GGUF_LBA=0 ./build/ntransformer ... --streaming`. The GGUF must first be `dd`'d to the raw device, and the device bound to VFIO via `scripts/setup_nvme.sh` (re-run after every reboot). Read `docs/GPU_NVME_DIRECT_INTEGRATION.md` first.

## Tests

There is no test framework. Each test is a standalone executable (`tests/test_<feature>.cpp|.cu`) registered in the root `CMakeLists.txt` or `src/ntrpu/CMakeLists.txt`, often several times with different argv modes or env vars.

```bash
ctest --test-dir build -V                                # everything
ctest --test-dir build -R '^test_ds41_router' -V         # one test plus its env variants
./build/test_ds41_backbone tail                          # run directly with a mode argument
DS41_GPU=0 ./build/test_ds41_router disabled             # reproduce a CTest env variant by hand
ctest --test-dir build/ntrpu-sim -C Release -R 'ntrpu_(hls|q8_0)' --output-on-failure
python -B -m unittest discover -s tests -p test_ntrpu_buffer_pool_model.py -v
```

- **Exit code 77 means SKIP** (no CUDA, fewer than two GPUs, no P2P). A green CTest run can hide skips, and some failures don't set the exit status. Read the verbose output for `FAIL`/`SKIP`.
- Release defines `NDEBUG`. Reconfigure with `-DCMAKE_BUILD_TYPE=Debug` when a test relies on `assert`.
- The CUDA tests use synthetic tensors and need no model files.
- `tests/test_training_*.py` need PyTorch, CUDA, and `NTRANSFORMER_TRAINING_LIB`. Many `tests/test_ntrpu_*_r<N>.cpp` and the HLS/RTL checks are not in CTest. `tools/run_ntrpu_*.py` and `scripts/run_ntrpu_hls.py` drive them, and synthesis/co-simulation needs Vivado/Vitis (`docs/fpga/HLS.md`, `docs/fpga/CLOUD_HLS.md`).
- `test_nvme_layer` needs real NVMe hardware and is intentionally not registered.

## Architecture

### Model dispatch (`src/main.cpp`)
The CLI peeks at the GGUF's `general.architecture` and tries `glm5::run_cli` (`"glm5-next"`), then `ds41::run_cli` (`"deepseek41"`). Each returns `-1` to decline. Anything else falls through to the Llama `Engine`. The Llama-only flags (`--streaming`, `--peer-gpu`, `--self-spec`, `--skip-threshold`, `--draft-model`, …) do **not** reach DS41/GLM5. Those paths are configured entirely through `DS41_*` / `GLM5_*` environment variables (grep `getenv` in `src/model/{ds41,glm5}`).

### Llama path
`inference/engine.cpp` (generate/chat/benchmark/speculative) → `model/transformer.cpp` (`forward`, `forward_streaming`, `forward_tiered`, `forward_draft*`) → `memory/streamer.cu` (`LayerStreamer`).

- At load time, `TierConfig::compute` assigns each layer a `LayerTier` of `VRAM`, `PEER`, `RAM`, or `NVME`. Sizing comes from `cudaMemGetInfo`, `host_avail_ram()`, and the KV/workspace reserve.
  - Non-resident layers flow through two GPU slots: the H2D copy of layer N+1 on `STREAM_TRANSFER0/1` overlaps compute of layer N on `STREAM_COMPUTE`, and CUDA events order them.
  - Tier-C sources, auto-selected: gpu-nvme-direct (including BAR1 NVMe→VRAM DMA), pinned mmap, or a worker thread doing memcpy into pinned staging.
- `memory/peer_cache.cu` implements the `PEER` tier: overflow layers live in a second GPU's VRAM and are staged by P2P DMA into the same double buffers.
- The tiered bottleneck is PCIe H2D bandwidth, not compute (≈100 ms per 70B layer at Gen3 x8 versus <1 ms of compute). Evaluate optimizations by bytes moved per token.
- Embedding lookup is CPU dequant plus an H2D copy. Prefill attention loops GEMV per token.
- Delta-encoded streaming (`--delta-model`, `.ntd`) and CUDA Graphs were tried and documented as negative results (`docs/OPTIMIZATIONS.md`, `docs/research/`).

### DS41 / GLM5 (MoE) path
- Each model has a CPU `Session` that is the exact reference and correctness oracle, plus GPU offload toggled by env (`DS41_GPU=0` disables it; `GLM5_GPU=1` enables the GPU engine and `GLM5_FAST=1` the device-resident fast path in `glm5_gpu2.cu`).
- Routed experts stream into a VRAM LRU expert cache (`ds41_cache.h`).
- GLM5 reuses DS41's `ExpertCacheIndex` and the mHC hyper-connections in `ds41_hc.h`. Keep both models working when touching these files.
- Both reuse `GGUFLoader` and `Tokenizer`.

Correctness bar: greedy parity with pinned llama.cpp references (DS41: `vcruz305/llama.cpp` branch `runtime/deepseek41`; GLM5: llama.cpp `glm5-next`, PR #27754). Bugs are found by bisecting stage dumps against the CPU reference.

The stated DS41 direction (`docs/DEEPSEEK_PERFORMANCE_GAP.md`) is to remove the CPU from the steady-state per-token critical path. Bit-identical logits are not required, but any arithmetic change needs likelihood/quality measurements, and a logit diff alone proves nothing either way.

Known GLM5 parity hazards:
- Large prefill chunks race in the union expert load (evict versus H2D in flight). The verified configuration is `GLM5_CHUNK=3`.
- `GLM5_PREFETCH=2` races.
- On Windows/WDDM, the VRAM cache plus resident weights must stay under ≈21.5 GB or the driver silently pages.
- On Linux gpnvme, NaN logits or `!!!!` output mean expert DMA is dead. Every VFIO rebind needs `setpci 84.W=0x0008` and `COMMAND=0x0006`, both read back.

The GLM5 gpu-nvme-direct path is merged into `src/model/glm5/`: `GLM5_GPNVME`, `GLM5_BACKBONE_PACK`, `GLM5_LBA_BIN`, all guarded by `USE_GPUNVME`. `linux_port/` holds what is specific to the Linux host: the runbook, runner scripts, LBA indexes, and the gpu-nvme-direct `layer_loader` patch. `tools/linux/` holds the Windows-side `D:\linux_kit` bridge tools. Linux runs from a live USB because the model's NVMe disk is bound to VFIO and the live USB needs a different kernel.

### NTRPU (FPGA)
The goal is an accelerator with up to 512 GB DDR4 that runs inference **without an NVIDIA GPU or CUDA**. Adding another data source to `LayerStreamer` would not meet that goal.

- `src/ntrpu/` (`nt::ntrpu`, C++20) is a deterministic, single-threaded functional simulator: sparse memory, tensor directory, DMA queues, distributed GEMV/FFN.
- The HLS kernels in `fpga/hls/*/` are C++14 and are compiled into the simulator as `ntrpu_hls_host`, so the same source is CPU-tested and Vitis-synthesized.
- SystemVerilog lives in `fpga/rtl/` and `fpga/experimental/`, testbenches in `fpga/tb/`. The current hardware target is Llama 3.2 1B (`fpga/profiles/`).
- Revisions are append-only: a changed block becomes a new `_r<N+1>` file (RTL, HLS, tools, and tests together). Previously qualified revisions are not edited in place because validation records reference them.

### Training
`src/training/*.cu` exposes a C ABI (`nt_train_*`) for RMSNorm/SwiGLU/rotary/cross-entropy kernels. `python/ntransformer_training/` wraps it for PyTorch, which handles GEMM, attention, AdamW, and DDP. Experiments target the "Brandon" model on H100 (`scripts/train_*.py`, `scripts/queue_brandon_training.py`).

## Conventions

- Namespaces: `nt`, `nt::cuda` (kernels), `nt::ds41`, `nt::glm5`, `nt::ntrpu`. Kernel launchers are named `launch_*()`. C-linkage device helpers for non-nvcc `.cpp` files are named `nt_cuda_*()`.
- Style: PascalCase types, snake_case functions and files, trailing `_` on private members, four-space indent, same-line braces. No formatter is configured; match surrounding code.
- `NT_CHECK(cond, msg)` / `NT_CUDA_CHECK(err)` call `abort()`. There is no error recovery.
- `DType` (internal, `src/core/types.h`) is **not** numbered like ggml types. `GGMLType` holds the on-disk ids, and `types.h` maps between them. Supported formats are F32, F16, BF16, Q8_0, Q4_0, Q4_K_M, Q5_K, Q6_K, and MXFP4.
- Any compile definition that changes a struct layout (e.g. `USE_GPUNVME` inside `LayerStreamer`) must be `PUBLIC` on `ntransformer_lib`. A mismatch between translation units silently corrupts adjacent objects.
- `include/ntransformer.h` declares a C API that is not implemented yet.
- Commits use descriptive subjects prefixed by subsystem (e.g. `Async NVMe prefetch: ...`). Performance changes should report before/after tok/s and VRAM, plus the hardware used.
- Docs language varies: `docs/fpga/`, `docs/GLM5_*`, and `linux_port/` are in Spanish, the rest in English. Match the file you are editing.

## Hardware Safety

`scripts/setup_system.sh`, `setup_nvme.sh`, and `restore_nvme.sh`, raw `dd` to `/dev/nvme*`, and the `linux_port/tools` disk-fill scripts rebind PCIe devices and overwrite whole drives. A wrong-disk fill has already destroyed data on this project (`docs/GLM5_LINUX_BENCH.md`). Never run these without explicit user confirmation and a verified target (BDF and serial). Never target the boot drive.

## Docs Map

- Llama engine: `docs/INFERENCE_PIPELINE.md`, `docs/TIERED_CACHING.md`, `docs/OPTIMIZATIONS.md`, `docs/GPU_NVME_DIRECT_INTEGRATION.md`, `docs/DEVELOPMENT.md` (history), `docs/research/`
- DS41: `docs/DEEPSEEK_INFERENCE_OPTIMIZATION.md`, `docs/DEEPSEEK_PERFORMANCE_GAP.md`, `docs/DEEPSEEK_PRECISION_SELECTION.md`
- GLM5: `docs/GLM5_PROJECT.md` (master: env flags, findings, Linux setup), `docs/GLM5_LINUX_BENCH.md` (bench, verification, incidents), `docs/GLM5_LINUX_HANDOFF.md`, `docs/GLM5_GPU_UNLOCK.md`, `linux_port/README_LINUX_PORT.md`
- H100 / peer cache / training: `docs/H100_BENCHMARKS.md`, `docs/H100_TRAINING_EXPERIMENT.md`, `docs/TRAINING_RESEARCH.md`
- NTRPU: `docs/fpga/README.md` (start here), `docs/fpga/SIMULATOR.md`, `docs/fpga/HLS.md`
- Benchmark records: `docs/benchmarks/<host>-<date>/`
