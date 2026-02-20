# NTransformer - Project Guide

## What Is This
High-efficiency C++/CUDA LLM inference engine. Goal: run Llama 70B at Q8-equivalent quality on a single RTX 3090 (24GB VRAM) by combining 6 memory-optimization techniques.

## Current State
**Phase 2 (SLEP) - COMPLETE. Ported to Linux/CUDA 13.1. Ready for gpu-nvme-direct integration.**
- Phase 1 fully working: Llama 3.1 8B Q8_0 at 38.8 tok/s decode (resident), 0.9 tok/s (streaming)
- Phase 2 SLEP streaming: pipelined layer streaming via PCIe with worker thread
- Q6_K quantization support: 70B Llama running on single RTX 3090 (4.2 GB VRAM)
- `--streaming` CLI flag enables SLEP mode
- All unit tests passing (7/7 tensor, 6/6 kernel)
- 8B streaming verified: bit-identical output vs resident mode
- 70B Q6_K streaming: working, ~0.02 tok/s (pre-optimization, staging fallback)
- Worker thread pipeline implemented: overlaps CPU memcpy, H2D DMA, and GPU compute
- **Ported from Windows/MSVC/CUDA 12.4 to Linux/gcc-14/CUDA 13.1 (C++20 unified)**
- gpu-nvme-direct integration spec complete (`GPU_NVME_DIRECT_INTEGRATION.md`)

## Development Setup
- **Platform:** Linux (Ubuntu, kernel 6.17+)
- **Compiler:** gcc-14 / g++-14 (gcc-15 is incompatible with CUDA 13.1)
- **CUDA:** Toolkit 13.1, C++20 for both host and device code
- **GPU:** RTX 3090 24GB, Compute 8.6
- **Build requirements:** CMake 3.24+, CUDA Toolkit 13.1, C++20, gcc-14
- **No external dependencies** beyond CUDA Toolkit (no PyTorch, no cuBLAS)
- **Test models:** Configure paths via `-m` flag

### Build Commands
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build . --config Release -j
# Tests
./Release/test_tensor
./Release/test_gemm
# Run
./Release/ntransformer -m /path/to/model.gguf -p "Hello" -n 128
./Release/ntransformer -m /path/to/model.gguf --chat
./Release/ntransformer -m /path/to/model.gguf --benchmark -n 64
# Phase 2: Streaming mode (streams layers from CPU via PCIe)
./Release/ntransformer -m /path/to/model.gguf -p "Hello" -n 128 --streaming
```

## Architecture Overview

### Namespace & Conventions
- Everything in `namespace nt`, CUDA kernels in `namespace nt::cuda`
- CUDA launchers: `launch_*()` (e.g., `launch_rmsnorm`, `launch_gemv`)
- C-linkage device ops: `nt_cuda_*()` (for .cpp files compiled without nvcc)
- Error macros: `NT_CHECK(cond, msg)`, `NT_CUDA_CHECK(err)` — both call `abort()` on failure

### Directory Layout
```
src/
├── core/           # Fundamental: types, tensor, allocator, GPU device
├── cuda/           # CUDA kernels (.cu files) + kernels.h declarations
├── memory/         # Phase 2: SLEP layer streaming (streamer.h/cu)
├── model/          # Transformer components: config, loader, norm, attention, ffn, transformer
├── inference/      # High-level: tokenizer, sampler, engine
├── utils/          # Timer, logger, profiler
├── main.cpp        # CLI entry point
include/            # Public C API (ntransformer.h)
tests/              # Unit tests
scripts/            # Python utilities (Phase 3+)
```

### Data Flow (Forward Pass)
```
tokens[seq_len] (CPU)
  → embed_tokens(): CPU lookup + H2D copy → hidden_state[seq_len, hidden_size] (GPU)
  → for each layer:
      norm → attention(+RoPE, KV cache, GQA) → residual add
      norm → FFN(SwiGLU: gate↦SiLU, up, down) → residual add
  → output_norm (last token only)
  → LM head GEMV → logits[vocab_size] (GPU)
  → D2H copy → sampling (CPU) → next token
```

### Streaming Pipeline (Phase 2)
```
Worker thread:  [memcpy L0→stg0][memcpy L1→stg1][memcpy L2→stg0]...
H2D DMA:        [              ][stg0→gpu0     ][stg1→gpu1     ]...
GPU Compute:    [              ][              ][layer 0       ]...
```
Three-stage pipeline: worker memcpy (CPU cores), async H2D (PCIe DMA), GPU compute (SMs).

### Key Buffers & Memory Layout
| Buffer | Shape | Notes |
|--------|-------|-------|
| hidden_buf | [max_seq, hidden_size] | Main hidden state, F32 GPU |
| residual_buf | [max_seq, hidden_size] | Temp for norm→attn/ffn output |
| k_cache | [n_layers, max_seq, n_kv_heads, head_dim] | F32 GPU |
| v_cache | [n_layers, max_seq, n_kv_heads, head_dim] | F32 GPU |
| workspace | [max(attn_ws, ffn_ws)] | Shared, reused between attn/ffn |
| logits_buf | [vocab_size] | Final output, F32 GPU |
| gpu_buf_[2] | [layer_size] | Streaming: double-buffer for layer weights |
| staging_buf_[2] | [layer_size] | Streaming: pinned staging for CPU→GPU pipeline |

### GGUF Tensor Names (Llama)
```
token_embd.weight                    # Embedding table
output.weight                        # LM head (may share with embedding)
output_norm.weight                   # Final RMSNorm
blk.{i}.attn_norm.weight             # Pre-attention norm
blk.{i}.attn_q.weight               # Query projection
blk.{i}.attn_k.weight               # Key projection
blk.{i}.attn_v.weight               # Value projection
blk.{i}.attn_output.weight          # Output projection
blk.{i}.ffn_norm.weight             # Pre-FFN norm
blk.{i}.ffn_gate.weight             # SwiGLU gate
blk.{i}.ffn_up.weight               # SwiGLU up
blk.{i}.ffn_down.weight             # SwiGLU down
```

### Quantization Formats Supported
| Format | Block Size | Bytes/Block | Layout |
|--------|-----------|-------------|--------|
| Q4_0 | 32 weights | 18 bytes | FP16 scale + 16 nibble bytes |
| Q8_0 | 32 weights | 34 bytes | FP16 scale + 32 int8 |
| Q4_K_M | 256 weights | 144 bytes | FP16 d/dmin + 12 sub-scales + 128 nibbles |
| Q6_K | 256 weights | 210 bytes | 128 ql + 64 qh + 16 scales + FP16 d |
| F16 | 1 | 2 bytes | IEEE 754 half |
| F32 | 1 | 4 bytes | IEEE 754 float |

### CUDA Streams
- `STREAM_COMPUTE` (0): All kernel execution
- `STREAM_TRANSFER0` (1): For Phase 2 SLEP buffer A
- `STREAM_TRANSFER1` (2): For Phase 2 SLEP buffer B

## Known Limitations

1. **Embedding lookup on CPU** — CPU dequant (Q8_0/Q4_0/Q6_K/F16/F32) + H2D copy. Acceptable for Phase 1-2.
2. **GEMV per-token for prefill** — attention.cpp loops GEMV per token instead of batched GEMM. Slow prefill.
3. **No graceful error recovery** — `NT_CHECK` calls `abort()`. Missing tensor = crash.
4. **Chat mode is stateless** — each turn is independent, no conversation history management.
5. **Special token handling** — Chat template tokens (`<|start_header_id|>` etc.) not parsed; use raw text prompts.
6. **70B streaming throughput** — ~0.02 tok/s with staging fallback; worker thread pipeline targets 0.4-0.5 tok/s (needs 70B re-test).

## Phase Roadmap

### Phase 1: Foundation ✅ (complete)
Run Llama 8B Q8_0. Actual: 38.8 tok/s decode, 10.9 GB VRAM (after kernel optimization).

### Phase 2: SLEP (Streaming Layer Execution) ✅ (complete)
Double-buffer layer streaming via PCIe. Llama 70B Q6_K on single RTX 3090 (4.2 GB VRAM).
- `src/memory/streamer.h/cu` — Pipelined engine with worker thread + double staging
- Pinned memory: tries `cudaHostRegister` on mmap, falls back to double pinned staging
- Worker thread overlaps CPU memcpy (mmap→staging) with H2D DMA and GPU compute
- `transformer.cpp` — `forward_streaming()` with 3-stage pipeline
- Norm weights preloaded to GPU; layer weights streamed per-token
- CLI: `--streaming` flag
- Q6_K GEMV kernel + embedding dequant for 70B support

### Phase 3: Advanced Quantization
RotateKV (Walsh-Hadamard + INT2 KV-cache) + adaptive per-layer precision.
- `src/cuda/hadamard.cu`
- `src/quant/kv_quant.h/cpp`
- `src/quant/adaptive.h/cpp`
- `scripts/convert_model.py`

### Phase 4: Novel Architectures
MLA retrofit via SVD, Mamba/SSM support, neuromorphic sparsity, speculative decoding.
- `src/model/ssm.h/cpp`, `src/cuda/ssm.cu`
- `src/cuda/sparsity.cu`
- `src/inference/speculative.h/cpp`

### Phase 5: Polish
Optimization, benchmarks, public C API, documentation.

## Performance Results

### 8B Q8_0 (Llama 3.1 8B Instruct)
| Mode | Decode | Prefill | VRAM |
|------|--------|---------|------|
| Resident | 42.2 tok/s | 42.5 tok/s | 10.9 GB |
| Streaming | 0.9 tok/s | 0.4 tok/s | 4.2 GB |

### 70B Q6_K (Llama 3.1 70B Instruct)
| Mode | Decode | VRAM | Notes |
|------|--------|------|-------|
| Streaming (staging fallback) | ~0.02 tok/s | 4.2 GB | Pre-pipeline optimization |
| Streaming (worker pipeline) | target 0.4-0.5 tok/s | 4.2 GB | Needs re-test |

### Performance Targets
| Metric | Ph1 (7B) | Ph2 (70B) | Ph3 (+Quant) | Ph4 (Full) |
|--------|---------|----------|-------------|------------|
| VRAM | 4 GB | 6 GB | 5 GB | 8 GB |
| Decode | 30-50 t/s | 0.7-1 t/s | 0.8-1.2 t/s | 1.5-3 t/s |
| Prefill | 500+ t/s | 20-40 t/s | 25-50 t/s | 40-80 t/s |

## Key Technical Constants
```
GGUF_MAGIC       = 0x46554747  (little-endian "GGUF")
GGUF alignment   = 32 bytes (or metadata-specified)
RoPE theta       = 10000.0 (default Llama), 500000.0 (Llama 3.1)
RMSNorm eps      = 1e-5
CUDA target SMs  = 80, 86, 89, 90
```

## Bugs Fixed During Bringup
- **GGUF_MAGIC byte order** — was `0x46475547`, correct is `0x46554747` (LE "GGUF")
- **BlockQ8_0 size** — GGML uses FP16 scale (2 bytes, total 34), not float (4 bytes, total 36)
- **output_weight_ on CPU** — was passing mmap'd CPU pointer to CUDA kernel; now copied to GPU
- **Windows mmap** — ported POSIX mmap to `CreateFileMapping`/`MapViewOfFile`
- **aligned_alloc** — MSVC uses `_aligned_malloc`/`_aligned_free`
- **CUDA C++20** — nvcc 12.4 + MSVC 14.42 crashed with C++20; resolved with CUDA 13.1 + gcc-14
- **Context size OOM** — Llama 3.1 has 131K context; cap with `--ctx-size`
- **Tokenizer encoding** — Llama 3 uses GPT-2 byte BPE (Ġ), not SentencePiece (▁)

## Documentation
- `DEVELOPMENT.md` — Detailed progress log, per-file status, design decisions
- This file (`CLAUDE.md`) — Project context for AI-assisted development
