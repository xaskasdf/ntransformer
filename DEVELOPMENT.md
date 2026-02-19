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

## Phase 2: Memory Optimization (SLEP) - PLANNED

**Goal:** Run Llama 70B via layer streaming on RTX 3090.

### Key Design
- Double-buffer: while layer N computes, layer N+1 transfers via PCIe
- PCIe 4.0 ~25 GB/s -> layer INT4 (~428MB) in ~17ms
- Only 2 layer slots in VRAM at any time
- KV-cache in FP16 initially

### Target
- VRAM < 8GB for 70B model
- ~0.7-1 tok/s decode

---

## Phase 3: Advanced Quantization - PLANNED

**Goal:** RotateKV + Adaptive precision for better quality and longer context.

### Key Design
- Walsh-Hadamard transform to redistribute outliers before KV quantization
- Per-layer sensitivity analysis for mixed precision assignment
- KV-cache: 1.3GB -> 170MB, context 4K -> 16K+

---

## Phase 4: Novel Architectures - PLANNED

**Goal:** MLA retrofit, SSM support, Sparsity, Speculative decoding.

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
