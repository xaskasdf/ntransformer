# NTransformer - Project Guide

## What Is This
High-efficiency C++/CUDA LLM inference engine. Goal: run Llama 70B at Q8-equivalent quality on a single RTX 3090 (24GB VRAM) by combining 6 memory-optimization techniques.

## Current State
**Phase 2 (SLEP) - COMPLETE. 3-tier adaptive caching — 33x speedup on 70B.**
- Phase 1 fully working: Llama 3.1 8B Q8_0 at 48.9 tok/s decode (resident), 0.9 tok/s (streaming)
- Phase 2 SLEP streaming: pipelined layer streaming via PCIe with worker thread
- **3-tier adaptive caching**: VRAM resident + pinned RAM + NVMe/mmap (auto-sized from hardware)
- 70B Q6_K: 29 VRAM + 51 RAM → **0.2 tok/s** (33x over mmap baseline), 23 GB VRAM
- 8B Q8_0: auto-promotes all 32 layers to VRAM → 48.8 tok/s (equivalent to resident)
- Q6_K quantization support: 70B Llama running on single RTX 3090
- `--streaming` CLI flag enables tiered mode (auto-selects best tier per layer)
- All unit tests passing (7/7 tensor, 6/6 kernel)
- **gpu-nvme-direct integrated and verified** — NVMe reads at 3,315 MB/s sustained (95% PCIe 4.0 x4)
- Worker thread pipeline: fallback path when NVMe unavailable or mmap pinning succeeds
- **Ported from Windows/MSVC/CUDA 12.4 to Linux/gcc-14/CUDA 13.1 (C++20 unified)**
- Setup/restore scripts: `scripts/setup_nvme.sh`, `scripts/restore_nvme.sh`

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

### Build with gpu-nvme-direct (NVMe backend for streaming)
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPUNVME=ON \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build . --config Release -j

# Setup NVMe for VFIO (once after reboot)
sudo ./scripts/setup_nvme.sh 0000:01:00.0

# Write GGUF to NVMe raw device (once per model)
sudo ./scripts/restore_nvme.sh 0000:01:00.0   # bind to kernel driver
sudo dd if=/path/to/model.gguf of=/dev/nvme0n1 bs=1M oflag=direct status=progress
sudo ./scripts/setup_nvme.sh 0000:01:00.0      # rebind to VFIO

# Run with NVMe backend
sudo GPUNVME_PCI_BDF=0000:01:00.0 GPUNVME_GGUF_LBA=0 \
  ./build/ntransformer -m /path/to/model.gguf -p "Hello" -n 128 --streaming

# Restore NVMe to kernel driver when done
sudo ./scripts/restore_nvme.sh 0000:01:00.0
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

**With gpu-nvme-direct (eliminates CPU from data path):**
```
GPU MMIO:       [doorbell L0  ][doorbell L1  ][doorbell L2  ]...
NVMe DMA:       [DMA L0→stg0  ][DMA L1→stg1  ][DMA L2→stg0 ]...
H2D:            [              ][stg0→gpu0    ][stg1→gpu1    ]...
GPU Compute:    [              ][              ][layer 0      ]...
```
See `GPU_NVME_DIRECT_INTEGRATION.md` for full integration guide.

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
6. **NVMe requires root** — `/proc/self/pagemap` needs `CAP_SYS_ADMIN` for physical address translation. Run with `sudo`.
7. **NVMe setup required after reboot** — Must run `scripts/setup_nvme.sh` to force D0 power state + enable BusMaster before NVMe backend works.

## Phase Roadmap

### Phase 1: Foundation ✅ (complete)
Run Llama 8B Q8_0. Actual: 38.8 tok/s decode, 10.9 GB VRAM (after kernel optimization).

### Phase 2: SLEP (Streaming Layer Execution) ✅ (complete)
Double-buffer layer streaming via PCIe. Llama 70B Q6_K on single RTX 3090 (7.3 GB VRAM).
- `src/memory/streamer.h/cu` — Pipelined engine with worker thread + double staging
- **gpu-nvme-direct backend**: NVMe reads directly to pinned staging at 3,315 MB/s
- Three data paths (auto-selected): (1) mmap pinned, (2) NVMe direct, (3) CPU worker memcpy
- Worker thread overlaps CPU memcpy (mmap→staging) with H2D DMA and GPU compute
- `transformer.cpp` — `forward_streaming()` with 3-stage pipeline
- Norm weights preloaded to GPU; layer weights streamed per-token
- CLI: `--streaming` flag + env vars `GPUNVME_PCI_BDF` / `GPUNVME_GGUF_LBA`
- Q6_K GEMV kernel + embedding dequant for 70B support
- Setup/restore scripts for NVMe VFIO binding

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
| Mode | Decode | Prefill | VRAM | Tier Split |
|------|--------|---------|------|------------|
| Resident | 48.9 tok/s | 50.9 tok/s | 10.0 GB | N/A |
| Tiered (auto) | 48.8 tok/s | 50.9 tok/s | 10.3 GB | 32 VRAM + 0 RAM |
| Streaming (mmap pinned) | 0.9 tok/s | 1.8 tok/s | 3.4 GB | Pure streaming |

### 70B Q6_K (Llama 3.1 70B Instruct)
| Mode | Decode | VRAM | Tier Split | Speedup |
|------|--------|------|------------|---------|
| Streaming (mmap staging) | 0.006 tok/s | 7.3 GB | Pure streaming | 1x |
| Streaming (NVMe direct) | 0.03 tok/s | 7.3 GB | Pure streaming | 5x |
| **Tiered (ctx=4096)** | **0.2 tok/s** | **23.0 GB** | **24 VRAM + 54 RAM + 2 NVMe** | **33x** |
| **Tiered (ctx=512)** | **0.2 tok/s** | **23.1 GB** | **29 VRAM + 51 RAM + 0 NVMe** | **33x** |

### Tiered Caching Performance
- **Bottleneck**: PCIe H2D at Gen3 x8 (~6.5 GB/s) — B450 runs GPU at x8 due to M.2 lane sharing
- **Per tier B layer**: 669 MB / 6.5 GB/s ≈ 103 ms H2D (compute is only ~0.7 ms)
- **Auto-sizing**: VRAM reserve computed from KV cache + workspace + buffers. RAM from `/proc/meminfo` MemAvailable
- **With B550/X570 (Gen4 x16)**: expected ~0.5-0.7 tok/s (H2D at 25 GB/s, compute-bound)

### NVMe Direct Performance
- **Sustained throughput**: 3,300–3,330 MB/s (95% of PCIe 4.0 x4 theoretical max)
- **Per-layer read**: ~202 ms for 670 MB (80 layers, 670 NVMe commands each)
- **Hardware**: WD SN740 512GB (DRAM-less), MDTS=1024K, pipeline depth=32
- **Speedup vs mmap staging**: 4.7–5x (126s vs 633s total for 5 tokens)

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

## gpu-nvme-direct Integration

**Status**: Fully integrated and verified. 5x speedup on 70B Q6_K.

**Layer Loader API** (`../gpu-nvme-direct/include/gpunvme/layer_loader.h`):
```c
gpunvme_layer_loader_init(&loader, "0000:01:00.0", max_layer_bytes, 32);
gpunvme_load_layer(&loader, start_lba, size_bytes, dest_pinned);  // repeated
gpunvme_layer_loader_destroy(&loader);
```

**Integration point**: `LayerStreamer::prefetch_staging()` — NVMe reads replace worker thread
memcpy when available. Falls back to mmap if NVMe init fails or env vars not set.

**Build with NVMe**: `cmake .. -DUSE_GPUNVME=ON` (links against pre-built gpu-nvme-direct libs)

**Environment variables**: `GPUNVME_PCI_BDF=0000:01:00.0 GPUNVME_GGUF_LBA=0`

**Setup scripts**:
- `scripts/setup_nvme.sh` — Loads VFIO, binds device, forces D0, enables BusMaster
- `scripts/restore_nvme.sh` — Rebinds to kernel nvme driver for dd/mount

**Full guide**: `GPU_NVME_DIRECT_INTEGRATION.md`

## Bugs Fixed During NVMe Integration
- **Struct layout corruption** — `#ifdef USE_GPUNVME` in `LayerStreamer` caused sizeof mismatch between compilation units, corrupting adjacent `Tokenizer` object. Fixed by ensuring consistent compile definitions across all sources.
- **NVMe controller fatal state** — After dd + VFIO rebind, controller stuck in D3 power state. Fixed with `setup_nvme.sh` that forces D0 via `setpci` and enables BusMaster.
- **Pagemap requires root** — `/proc/self/pagemap` physical address translation needs `CAP_SYS_ADMIN`. NVMe backend must run with sudo.

## Documentation
- `GPU_NVME_DIRECT_INTEGRATION.md` — NVMe integration spec, dev/test guide, troubleshooting
- `DEVELOPMENT.md` — Detailed progress log, per-file status, design decisions
- This file (`CLAUDE.md`) — Project context for AI-assisted development
