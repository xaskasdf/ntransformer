# NTransformer - Project Guide

## What Is This
High-efficiency C++/CUDA LLM inference engine. Goal: run Llama 70B at Q8-equivalent quality on a single RTX 3090 (24GB VRAM) by combining 6 memory-optimization techniques.

## Current State
**Phase 1 (Foundation) - CODE COMPLETE, pending compilation/testing on RTX 3090.**
- 43 files, ~5,600 lines of C++/CUDA
- Full inference pipeline: GGUF loading → tokenization → transformer forward → sampling → text output
- Not yet compiled. Next step: build and debug on remote machine with RTX 3090 via SSH.

## Development Setup
- **Write code:** This machine (local)
- **Compile and test:** Remote PC with RTX 3090 via SSH
- **Build requirements:** CMake 3.24+, CUDA Toolkit 12.x, C++20, Linux
- **No external dependencies** beyond CUDA Toolkit (no PyTorch, no cuBLAS)
- **Test models:** Llama 7B GGUF (Phase 1), Llama 70B GGUF (Phase 2+)

### Build Commands
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Tests
./test_tensor
./test_gemm
# Run
./ntransformer -m /path/to/model.gguf -p "Hello" -n 128
./ntransformer -m /path/to/model.gguf --chat
./ntransformer -m /path/to/model.gguf --benchmark -n 64
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

### Key Buffers & Memory Layout
| Buffer | Shape | Notes |
|--------|-------|-------|
| hidden_buf | [max_seq, hidden_size] | Main hidden state, F32 GPU |
| residual_buf | [max_seq, hidden_size] | Temp for norm→attn/ffn output |
| k_cache | [n_layers, max_seq, n_kv_heads, head_dim] | F32 GPU |
| v_cache | [n_layers, max_seq, n_kv_heads, head_dim] | F32 GPU |
| workspace | [max(attn_ws, ffn_ws)] | Shared, reused between attn/ffn |
| logits_buf | [vocab_size] | Final output, F32 GPU |

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
| Q8_0 | 32 weights | 36 bytes | F32 scale + 32 int8 |
| Q4_K_M | 256 weights | 144 bytes | FP16 d/dmin + 12 sub-scales + 128 nibbles |
| Q6_K | 256 weights | 210 bytes | 128 ql + 64 qh + 16 scales + FP16 d |
| F16 | 1 | 2 bytes | IEEE 754 half |
| F32 | 1 | 4 bytes | IEEE 754 float |

### CUDA Streams
- `STREAM_COMPUTE` (0): All kernel execution
- `STREAM_TRANSFER0` (1): For Phase 2 SLEP buffer A
- `STREAM_TRANSFER1` (2): For Phase 2 SLEP buffer B

## Known Limitations (Phase 1)

1. **Quantized embedding lookup not implemented** — falls back to zeros with a warning. F32 and F16 embeddings work. Most GGUF models use F16 embeddings so this is usually fine.
2. **Embedding lookup on CPU** — CPU dequant + H2D copy. Acceptable for Phase 1, will need GPU kernel for large batch prefill.
3. **GEMV per-token for prefill** — attention.cpp loops GEMV per token instead of batched GEMM. Slower prefill but correct. Optimize later.
4. **No graceful error recovery** — `NT_CHECK` calls `abort()`. Missing tensor = crash.
5. **Chat mode is stateless** — each turn is independent, no conversation history management.

## Phase Roadmap

### Phase 1: Foundation ✅ (code complete)
Run Llama 7B. Target: 30-50 tok/s decode, ~4GB VRAM.

### Phase 2: SLEP (Streaming Layer Execution)
Double-buffer layer streaming via PCIe. Run Llama 70B with ~6GB VRAM.
- `src/memory/streamer.h/cpp` — Double-buffer engine
- `src/memory/offloader.h/cpp` — CPU↔GPU policy
- `src/memory/kv_cache.h/cpp` — KV-cache manager
- `src/memory/prefetcher.h/cpp` — Layer prefetch
- Modify `transformer.cpp` to support streaming mode

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

## Performance Targets
| Metric | Ph1 (7B) | Ph2 (70B) | Ph3 (+Quant) | Ph4 (Full) |
|--------|---------|----------|-------------|------------|
| VRAM | 4 GB | 6 GB | 5 GB | 8 GB |
| Decode | 30-50 t/s | 0.7-1 t/s | 0.8-1.2 t/s | 1.5-3 t/s |
| Prefill | 500+ t/s | 20-40 t/s | 25-50 t/s | 40-80 t/s |

## Key Technical Constants
```
GGUF_MAGIC       = 0x46475547
GGUF alignment   = 32 bytes (or metadata-specified)
RoPE theta       = 10000.0 (default Llama)
RMSNorm eps      = 1e-5
CUDA target SMs  = 80, 86, 89, 90
```

## Documentation
- `DEVELOPMENT.md` — Detailed progress log, per-file status, design decisions
- This file (`CLAUDE.md`) — Project context for AI-assisted development
