# NTransformer - Development Log

## Project Overview
High-efficiency LLM inference engine in C++/CUDA designed to run Llama 70B at Q8-equivalent quality on a single RTX 3090 (24GB VRAM).

### 6 Core Innovations
1. **SLEP** - Streaming Layer Execution with Predictive Prefetch
2. **Adaptive Mixed Precision** - Per-layer sensitivity-based quantization
3. **TransMLA** - GQA to Multi-head Latent Attention retrofit via SVD
4. **RotateKV** - Walsh-Hadamard KV-cache quantization to 2-4 bits
5. **Neuromorphic Sparsity** - PowerInfer-style hot/cold neuron splitting
6. **Self-Speculative Decoding** - Using early layers as draft model

---

## Phase 1: Foundation (Core + Basic Inference)

**Goal:** Minimal functional pipeline to run Llama 7B GGUF.

**Status:** WORKING - Compiled and tested on RTX 3090 (Windows/MSVC/CUDA 12.4)

### Architecture Decisions

#### Memory Layout
- All tensors track device (CPU/CUDA) and support zero-copy views
- Pool allocator manages VRAM and pinned RAM separately
- GPU device abstraction manages 2 transfer streams + 1 compute stream (for Phase 2 SLEP)

#### GGUF Compatibility
- Direct mmap of GGUF files for zero-copy weight loading
- Support for Q4_0, Q8_0, Q4_K_M, Q6_K quantization formats
- Metadata parsing for model config (Llama architecture)

#### CUDA Kernels
- Fused RMSNorm kernel (single-pass with Welford-style reduction)
- RoPE with interleaved/non-interleaved support
- Online softmax for numerical stability
- Quantized GEMV kernels for Q4_0, Q8_0, Q4_K_M decode
- Flash Attention style decode kernel with GQA support

#### Inference Pipeline
- BPE tokenizer parsed from GGUF vocabulary
- Top-k, top-p, temperature sampling
- Simple greedy + multinomial decode loop

### Files Implemented
- [x] `src/core/types.h` - Data types, quantization block layouts, GGUF constants
- [x] `src/core/tensor.h/cpp` - Multi-device tensor with views
- [x] `src/core/allocator.h/cpp` - Pool allocator for VRAM/pinned RAM
- [x] `src/core/device.h/cu` - GPU management, streams, events
- [x] `src/cuda/rmsnorm.cu` - Fused RMSNorm kernel
- [x] `src/cuda/rotary.cu` - RoPE kernel
- [x] `src/cuda/softmax.cu` - Online softmax kernel
- [x] `src/cuda/gemm.cu` - Quantized GEMV/GEMM kernels (Q4_0, Q8_0, Q4_K_M, F16, F32)
- [x] `src/cuda/attention.cu` - Flash Attention decode + prefill with GQA
- [x] `src/cuda/elementwise.cu` - Add, add_inplace, copy kernels
- [x] `src/cuda/kernels.h` - All kernel launcher declarations
- [x] `src/model/config.h/cpp` - ModelConfig from GGUF metadata
- [x] `src/model/loader.h/cpp` - GGUF parser with mmap zero-copy
- [x] `src/model/norm.h/cpp` - RMSNorm wrapper
- [x] `src/model/attention.h/cpp` - Attention with GQA
- [x] `src/model/ffn.h/cpp` - SwiGLU FFN
- [x] `src/model/transformer.h/cpp` - Complete transformer with residual connections
- [x] `src/inference/tokenizer.h/cpp` - BPE tokenizer (GPT-2 BPE + SentencePiece auto-detect)
- [x] `src/inference/sampler.h/cpp` - Top-k, top-p, temperature sampling
- [x] `src/inference/engine.h/cpp` - Main inference engine with streaming output
- [x] `src/main.cpp` - CLI entry point
- [x] `CMakeLists.txt` - Build system (CMake 3.24+, CUDA, C++20)
- [x] `src/utils/timer.h` - CPU timer utility
- [x] `src/utils/logger.h` - Logging utility
- [x] `src/utils/profiler.h/cpp` - CUDA event-based profiler
- [x] `include/ntransformer.h` - Public C API header
- [x] `tests/test_tensor.cpp` - Tensor unit tests
- [x] `tests/test_gemm.cpp` - GEMM/kernel unit tests

### Validation Results
- [x] Run Llama 3.1 8B Instruct Q8_0 GGUF successfully
- [x] Correct factual completions verified ("capital of France" → "Paris")
- [x] All unit tests passing (7/7 tensor, 4/4 kernel)
- VRAM: 10.9 GB (8B Q8_0, ctx=4096) — higher than target due to larger model + Q8_0
- Decode: **38.8 tok/s** — within 30-50 target ✅ (was 15.8 before kernel optimization)
- Prefill: **42.5 tok/s** — below 500+ target (GEMV-per-token, no batched GEMM)

### Kernel Optimization (Post Phase 1)

Applied to `src/cuda/gemm.cu` — all GEMV kernels (Q4_0, Q8_0, Q4_K_M, F16, F32):

1. **Shared memory x cache** — input vector loaded once into shared memory,
   shared by all warps. Eliminates redundant global memory reads (8x savings).
2. **8 warps per block** (was 4) — better latency hiding and SM occupancy.
3. **Dynamic shared memory** + `cudaFuncSetAttribute` for FFN layers
   (in_features=14336 needs 56KB, exceeds default 48KB limit).
4. **Vectorized half2 loads** for F16 GEMV (LM head: output.weight).
5. **Vectorized float4 loads** for x into shared memory.

**Results: 2.5x speedup on decode (15.8 → 38.8 tok/s)**

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Prefill | 16.6 tok/s | 42.5 tok/s | 2.56x |
| Decode  | 15.8 tok/s | 38.8 tok/s | 2.46x |

Bandwidth analysis: 38.8 tok/s × ~8 GB/token = 310 GB/s effective (33% of RTX 3090 theoretical 936 GB/s).
Remaining gap is from non-GEMV overhead (attention, norms, kernel launches) and memory access patterns.

---

## Q6_K Support

**Goal:** Enable 70B Llama Q6_K quantization for Q8-equivalent quality at smaller size.

**Status:** COMPLETE

### Implementation
- `src/cuda/gemm.cu` — New Q6_K GEMV kernel with GGML interleaved layout
  - 256 weights per block: 128 ql (lower 4 bits) + 64 qh (upper 2 bits) + 16 int8 scales + FP16 d
  - Two 128-weight halves with pointer advancement pattern matching GGML
  - Shared memory x cache + 8 warps per block (consistent with Q8_0/Q4_0 kernels)
- `src/model/transformer.cpp` — Q6_K embedding dequant on CPU (same interleaved layout)
- `src/core/types.h` — `BlockQ6_K` struct: `uint8_t ql[128], qh[64]; int8_t scales[16]; uint16_t d;`

### Validation
- 70B Llama 3.1 Instruct Q6_K: loads and runs correctly on RTX 3090
- VRAM: 4.2 GB (streaming), 55 GB model streamed from CPU via PCIe
- Coherent output verified on factual prompts

---

## Phase 2: Memory Optimization (SLEP) - COMPLETE

**Goal:** Run Llama 70B via layer streaming on RTX 3090.

**Status:** COMPLETE — 70B Q6_K running on single RTX 3090 with 4.2 GB VRAM.

### Implementation

#### Core Streaming Engine (`src/memory/streamer.h/cu`)
- Double-buffer GPU slots: two 669 MB buffers (for 70B Q6_K) hold alternating layers
- CUDA events for synchronization: `transfer_done_[2]` and `compute_done_[2]`
- Pinned memory strategy:
  - First tries `cudaHostRegister` on mmap'd region (true async DMA)
  - Falls back to double pinned staging buffers + worker thread (for large models)

#### Worker Thread Pipeline (streaming speed optimization)
- **Problem:** `cudaHostRegister` fails for 55 GB mmap (insufficient pinnable memory).
  Single staging buffer → synchronous 669 MB CPU memcpy blocks host thread. ~0.02 tok/s.
- **Solution:** Double staging buffers + dedicated worker thread for CPU memcpy.
  Three-stage pipeline overlapping on different hardware:
  ```
  Worker thread:  CPU memcpy from mmap to staging[slot]     (CPU cores)
  DMA engine:     async H2D from staging[slot] to gpu[slot]  (PCIe controller)
  GPU SMs:        compute layer using gpu[slot]               (GPU)
  ```
- `prefetch_staging(layer_idx, slot)` — Non-blocking: queues memcpy to worker thread
- `begin_h2d(layer_idx, slot)` — Waits for staging ready, issues async H2D transfer
- `begin_transfer()` — Legacy API, calls both methods sequentially
- Steady state: **~27ms per layer** (PCIe 4.0 limited). 80 layers = 2.2s = ~0.45 tok/s target.

#### Streaming Forward Pass (`src/model/transformer.cpp`)
- `forward_streaming()` — Pipelined loop with 2-layer lookahead:
  ```
  Pre-fill: prefetch_staging(0,0) → begin_h2d(0,0) → prefetch_staging(1,1)
  Loop i:   wait_transfer(slot) → begin_h2d(i+1) → prefetch_staging(i+2) → compute(i)
  ```
- Norm weights preloaded to contiguous GPU buffer (tiny: 1 MB for 80 layers)
- Layer weights set via `set_weights()` non-owning views into GPU buffer

#### Streaming Setup (`load_streaming()`)
- All norm weights preloaded to GPU in single contiguous buffer
- Attention/FFN initialized in streaming mode (no GPU weights until forward pass)
- `LayerStreamer::init()` builds per-layer tensor layout, allocates double buffers

### Files Modified/Added
- [x] `src/memory/streamer.h` — LayerStreamer class with worker thread + double staging
- [x] `src/memory/streamer.cu` — Full implementation: init, shutdown, pipeline, worker loop
- [x] `src/model/transformer.h` — Streaming mode flag, LayerStreamer member
- [x] `src/model/transformer.cpp` — `load_streaming()`, `forward_streaming()`, Q6_K embedding
- [x] `src/model/attention.h/cpp` — `init_streaming()`, `set_weights()` for non-owning views
- [x] `src/model/ffn.h/cpp` — `init_streaming()`, `set_weights()` for non-owning views
- [x] `src/model/norm.h/cpp` — `init_streaming()`, `set_weight()` for preloaded GPU norms
- [x] `src/main.cpp` — `--streaming` CLI flag
- [x] `src/cuda/gemm.cu` — Q6_K GEMV kernel

### Validation Results
- [x] 8B Q8_0 streaming: **bit-identical output** vs resident mode ✅
- [x] 8B streaming: 0.9 tok/s decode (PCIe limited, expected)
- [x] 8B resident: 42.2 tok/s decode, 10.9 GB VRAM
- [x] 70B Q6_K streaming: coherent output, 4.2 GB VRAM ✅
- [x] 70B Q6_K streaming: ~0.02 tok/s (staging fallback, pre-worker-pipeline)
- [ ] 70B Q6_K with worker pipeline: target 0.3-0.5 tok/s (needs re-test)

### Memory Budget (70B Q6_K Streaming)
| Component | Size |
|-----------|------|
| GPU layer buffers (×2) | 1,338 MB |
| KV cache (80 layers, ctx=512) | 640 MB |
| Workspace | ~500 MB |
| Norm weights | 1.3 MB |
| Output weight (LM head) | 571 MB |
| Pinned staging (×2) | 1,338 MB (host) |
| **Total VRAM** | **~4.2 GB** |

---

## Phase 3: Advanced Quantization - PLANNED

**Goal:** RotateKV + Adaptive precision for better quality and longer context.

### Key Design
- Walsh-Hadamard transform to redistribute outliers before KV quantization
- Per-layer sensitivity analysis for mixed precision assignment
- KV-cache: 1.3GB -> 170MB, context 4K -> 16K+

### Planned Files
- `src/cuda/hadamard.cu` — Walsh-Hadamard transform kernel
- `src/quant/kv_quant.h/cpp` — KV-cache quantization (INT2/INT4)
- `src/quant/adaptive.h/cpp` — Per-layer adaptive precision
- `scripts/convert_model.py` — Model conversion utility

---

## Phase 4: Novel Architectures - PLANNED

**Goal:** MLA retrofit, SSM support, Sparsity, Speculative decoding.

### Planned Files
- `src/model/ssm.h/cpp`, `src/cuda/ssm.cu` — Mamba/SSM support
- `src/cuda/sparsity.cu` — Neuromorphic hot/cold neuron splitting
- `src/inference/speculative.h/cpp` — Self-speculative decoding

---

## Phase 5: Polish - PLANNED

**Goal:** Optimization, benchmarks, public C API.

---

## Performance Targets

| Metric | Phase 1 (7B) | Phase 2 (70B) | Phase 3 (+Quant) | Phase 4 (Full) |
|--------|-------------|--------------|-----------------|----------------|
| VRAM | 4 GB | 6 GB | 5 GB | 8 GB |
| Decode tok/s | 30-50 | 0.7-1.0 | 0.8-1.2 | 1.5-3.0 |
| Prefill tok/s | 500+ | 20-40 | 25-50 | 40-80 |
| Quality (PPL) | Baseline | Baseline | +0.05-0.1 | +0.1-0.2 |
| Max Context | 4K | 4K | 16K+ | 16K+ |

---

## Blackwell Build Notes

### Overview

sm_120 (Blackwell) is now included in `CMAKE_CUDA_ARCHITECTURES`. This enables native codegen for RTX 5060 Ti, 5080, and 5090 without PTX JIT fallback at runtime.

### Why it matters

Without sm_120, CUDA compiles PTX (portable assembly) and the driver JIT-compiles it on first launch. This adds startup latency (~1–3 seconds for large binaries) and prevents Blackwell-specific instruction selection (wgmma, cp.async.bulk improvements in sm_120 vs sm_90a).

### Compiler constraint

gcc-14 is required. gcc-15 is incompatible with CUDA 13.1 due to C++ standard library ABI changes that break device code compilation.

### Build command

```bash
cmake .. -DCMAKE_C_COMPILER=gcc-14 \
         -DCMAKE_CXX_COMPILER=g++-14 \
         -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
         -DCMAKE_CUDA_HOST_COMPILER=g++-14
```

The `-DCMAKE_CUDA_HOST_COMPILER=g++-14` flag is needed if your system default (`/usr/bin/c++`) resolves to gcc-15.

### Architecture targets

| SM | GPU family | Notes |
|----|-----------|-------|
| sm_80 | A100, RTX 30xx | Ampere |
| sm_86 | RTX 3090 (primary dev target) | Ampere |
| sm_89 | RTX 40xx | Ada Lovelace |
| sm_90 | H100 | Hopper |
| sm_120 | RTX 5060 Ti, 5080, 5090 | Blackwell (new) |

---

## Port: Windows/CUDA 12.4 → Linux/CUDA 13.1 (2026-02-19)

Ported the entire codebase from Windows (MSVC 2022 / CUDA 12.4) to Linux (gcc-14 / CUDA 13.1).
This is a prerequisite for gpu-nvme-direct integration, which requires Linux (VFIO, /proc/self/pagemap).

### Changes Made

| File | Change |
|------|--------|
| `CMakeLists.txt` | Removed MSVC branch, CUDA C++17→C++20, auto-detect gcc-14 as CUDA host compiler, `-Xcompiler=-march=native` |
| `src/core/types.h` | Removed `_MSC_VER` ifdefs, `_aligned_malloc`/`_aligned_free` branches |
| `src/core/allocator.h` | Added `#include <memory>` (MSVC included it transitively, gcc does not) |
| `src/model/loader.h` | Removed Windows HANDLE members, added `tensor_file_offset()` and `file_data_offset()` for NVMe integration |
| `src/model/loader.cpp` | Removed ~65 lines of Windows API code (CreateFile, MapViewOfFile, etc.), kept only POSIX mmap path |

### Build Verified
- **Toolchain:** nvcc 13.1.115 + g++-14.3.0 + C++20 + SM 86
- **Tests:** 7/7 tensor, 6/6 kernel — all passing
- **Binary:** ntransformer runs, help output verified

### Notes
- gcc-15 is incompatible with CUDA 13.1; CMake auto-detects gcc-14
- CUDA 13.1 supports C++20 for device code (was C++17 with nvcc 12.4)
- `#include <memory>` is not transitive via MSVC headers but required explicitly by gcc
- `tensor_file_offset()` / `file_data_offset()` added to GGUFLoader for NVMe LBA calculation (Phase 2.5 prep)

---

## gpu-nvme-direct Integration into LayerStreamer (2026-02-19)

Integrated the gpu-nvme-direct Layer Loader API as an optional I/O backend for SLEP streaming.
When `USE_GPUNVME=ON` and NVMe env vars are set, layer data is read from NVMe via GPU-initiated
DMA instead of CPU memcpy from mmap. Graceful fallback to worker thread if NVMe unavailable.

### Changes Made

| File | Change |
|------|--------|
| `CMakeLists.txt` | Added `USE_GPUNVME` option, links pre-built gpu-nvme-direct libs, added `test_nvme_layer` target |
| `src/memory/streamer.h` | Added `#include <gpunvme/layer_loader.h>`, `gpunvme_layer_loader_t`, `NvmeLayerInfo`, NVMe state members |
| `src/memory/streamer.cu` | NVMe init in `init()` (env vars, layer_loader_init, pre-compute LBAs), NVMe path in `prefetch_staging()` (gpunvme_load_layer), cleanup in `shutdown()` |
| `tests/test_nvme_layer.cu` | Standalone test: reads layer 0 via NVMe, compares byte-for-byte with mmap'd data |
| `CLAUDE.md` | Updated with Layer Loader API docs, NVMe pipeline diagram, integration section |
| `GPU_NVME_DIRECT_INTEGRATION.md` | Updated with Layer Loader API, simplified integration spec, dev/test guide, troubleshooting |

### Build Verified
- `USE_GPUNVME=OFF`: all targets compile, 13/13 tests pass (no regression)
- `USE_GPUNVME=ON`: all targets compile including `test_nvme_layer`, 13/13 tests pass
- Hardware NVMe test requires VFIO setup (`sudo ./test_nvme_layer <model.gguf> <pci_bdf>`)

### Architecture
- Layer Loader API: `gpunvme_layer_loader_init()` → `gpunvme_load_layer()` (repeated) → `gpunvme_layer_loader_destroy()`
- Integration point: `LayerStreamer::prefetch_staging()` — replaces CPU worker thread memcpy with NVMe DMA
- Fallback: if NVMe init fails or read fails, falls back to original worker thread path
- Env vars: `GPUNVME_PCI_BDF`, `GPUNVME_GGUF_LBA`
