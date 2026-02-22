# NTransformer

High-efficiency C++/CUDA LLM inference engine. Runs Llama 70B on a single RTX 3090 (24GB VRAM) by streaming model layers through GPU memory via PCIe, with optional NVMe direct I/O that bypasses the CPU entirely.

## Key Results

| Model | Mode | Decode | VRAM | Notes |
|-------|------|--------|------|-------|
| Llama 3.1 8B Q8_0 | Resident | 48.9 tok/s | 10.0 GB | All layers in VRAM |
| Llama 3.1 8B Q8_0 | Tiered (auto) | 48.8 tok/s | 10.3 GB | 32/32 layers auto-promoted to VRAM |
| Llama 3.1 70B Q6_K | Streaming (mmap) | 0.006 tok/s | 7.3 GB | Page cache thrashing (53 GB > 48 GB RAM) |
| Llama 3.1 70B Q6_K | Tiered (auto) | 0.2 tok/s | 23.1 GB | 26 VRAM + 54 RAM + 0 NVMe |
| Llama 3.1 70B Q4_K_M | Tiered (auto) | 0.3 tok/s | 22.9 GB | 36 VRAM + 44 RAM (50% faster) |
| Llama 3.1 70B Q4_K_M | **Tiered + layer skip** | **0.5 tok/s** | **22.9 GB** | **36 VRAM + 44 RAM, 20 layers skipped** |

**3-tier adaptive caching** auto-sizes from hardware: VRAM-resident layers (zero I/O) + pinned RAM (H2D only) + NVMe/mmap fallback. Achieves **83x speedup** over mmap baseline for 70B on consumer hardware (RTX 3090 + 48 GB RAM).

Bottleneck is PCIe H2D bandwidth at Gen3 x8 (~6.5 GB/s). Q4_K_M fits 10 more layers in VRAM (36 vs 26), reducing tier B transfers. Layer skip (cosine similarity calibration) eliminates 20/80 layers per token with minimal quality loss.

## Features

- **Zero external dependencies** beyond CUDA Toolkit (no PyTorch, no cuBLAS)
- **GGUF model format** with Q4_0, Q8_0, Q4_K_M, Q5_K, Q6_K, F16, F32 quantization
- **3-Tier Adaptive Caching**: auto-sized VRAM resident + pinned RAM + NVMe/mmap tiers
- **SLEP streaming**: double-buffered layer pipeline overlaps NVMe reads, PCIe DMA, and GPU compute
- **gpu-nvme-direct backend**: userspace NVMe driver reads model weights directly to pinned GPU-accessible memory
- **Layer skip**: cosine-similarity calibration skips redundant layers (20/80 skipped at threshold 0.98)
- **Self-speculative decoding**: VRAM-resident layers as draft model (no extra model needed)
- **Four data paths** (auto-selected): VRAM resident > pinned RAM H2D > mmap pinned > CPU worker memcpy
- Llama architecture: RoPE, GQA, SwiGLU, RMSNorm, KV cache

## Requirements

- Linux (tested on Ubuntu, kernel 6.17+)
- CUDA Toolkit 13.1
- gcc-14 / g++-14
- NVIDIA GPU with Compute Capability 8.0+ (RTX 3090 tested)
- CMake 3.24+
- (Optional) NVMe SSD on separate PCIe slot + [gpu-nvme-direct](../gpu-nvme-direct) library

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build . -j

# Run (resident mode — model fits in VRAM)
./ntransformer -m /path/to/llama-8b-q8_0.gguf -p "Hello" -n 128

# Run (streaming mode — model larger than VRAM)
./ntransformer -m /path/to/llama-70b-q6_k.gguf -p "Hello" -n 32 --streaming

# Run with layer skip (fastest for 70B)
./ntransformer -m /path/to/llama-70b-q4_k_m.gguf -p "Hello" -n 32 --streaming --skip-threshold 0.98

# Self-speculative decoding (VRAM layers as draft, no extra model)
./ntransformer -m /path/to/llama-70b-q6_k.gguf -p "Hello" -n 32 --self-spec --draft-k 3

# Chat mode
./ntransformer -m /path/to/model.gguf --chat

# Benchmark
./ntransformer -m /path/to/model.gguf --benchmark -n 64
```

## NVMe Direct Streaming

For models that don't fit in VRAM, the NVMe backend eliminates the CPU from the data path:

```
NVMe SSD → (DMA) → Pinned Staging → (PCIe H2D) → GPU Buffers → Compute
```

### Setup

```bash
# Build with NVMe support (requires gpu-nvme-direct library)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPUNVME=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build . -j

# Write GGUF model to NVMe raw device
sudo ./scripts/restore_nvme.sh           # ensure kernel driver is bound
sudo dd if=model.gguf of=/dev/nvme0n1 bs=1M oflag=direct status=progress

# Bind NVMe to VFIO for userspace access
sudo ./scripts/setup_nvme.sh             # loads VFIO, forces D0, enables BusMaster

# Run with NVMe backend
sudo GPUNVME_PCI_BDF=0000:01:00.0 GPUNVME_GGUF_LBA=0 \
  ./build/ntransformer -m /path/to/model.gguf -p "Hello" -n 32 --streaming

# Restore NVMe to kernel driver when done
sudo ./scripts/restore_nvme.sh
```

### How It Works

1. The GGUF model file is written to raw NVMe blocks via `dd`
2. `setup_nvme.sh` binds the NVMe to VFIO, forces PCIe D0 power state, enables BusMaster
3. gpu-nvme-direct initializes the NVMe controller from userspace (admin queues, I/O queues)
4. During inference, each layer (~670 MB for 70B Q6_K) is read via 670 NVMe commands in ~202 ms
5. Data lands in CUDA pinned staging memory, then async DMA to GPU compute buffers
6. Pipeline overlaps NVMe reads, H2D DMA, and GPU compute across double buffers

## Architecture

```
src/
├── core/           # Tensor, allocator, GPU device management
├── cuda/           # CUDA kernels: GEMV, RMSNorm, RoPE, SwiGLU, softmax
├── memory/         # SLEP layer streaming engine (NVMe + mmap backends)
├── model/          # Transformer: config, GGUF loader, attention, FFN, norms
├── inference/      # Tokenizer, sampler, engine
├── utils/          # Timer, logger
├── main.cpp        # CLI entry point
scripts/
├── setup_nvme.sh   # Bind NVMe to VFIO, configure for gpu-nvme-direct
├── restore_nvme.sh # Restore NVMe to kernel driver
tests/              # Unit tests (tensor, GEMM kernels, NVMe layer loader)
```

## 3-Tier Adaptive Caching

```
forward_tiered() — hybrid pipeline:

Tier A (VRAM resident, layers 0..28):
  GPU Compute:  [layer 0][layer 1]...[layer 28]     (zero I/O, weights permanent)

Tier B (pinned RAM, layers 29..79, double-buffered):
  H2D DMA:     [L29→gpu0][L30→gpu1][L31→gpu0]...   (async from pinned RAM)
  GPU Compute: [         ][layer 29][layer 30]...    (overlapped with H2D)

Tier C (NVMe/mmap fallback, if needed):
  NVMe/memcpy: [read L→stg0][read L→stg1]...
  H2D DMA:     [            ][stg0→gpu0  ]...
  GPU Compute: [            ][            ][layer]...
```

Tier sizes auto-computed from `cudaMemGetInfo()` + `/proc/meminfo` MemAvailable.

## Quantization Formats

| Format | Bits/Weight | Block Size | Supported |
|--------|------------|-----------|-----------|
| Q4_0 | 4.5 | 32 | Yes |
| Q8_0 | 8.5 | 32 | Yes |
| Q4_K_M | 4.5 | 256 | Yes (mixed: Q4_K + Q5_K + Q6_K) |
| Q5_K | 5.5 | 256 | Yes |
| Q6_K | 6.6 | 256 | Yes |
| F16 | 16 | 1 | Yes |
| F32 | 32 | 1 | Yes |

## Phase Roadmap

- **Phase 1** - Foundation (complete): Llama 8B Q8_0, custom CUDA kernels, 48.9 tok/s
- **Phase 2** - SLEP Streaming (complete): 70B on single GPU, 3-tier caching, 33x speedup
- **Phase 3** - Optimization (complete): Q4_K_M/Q5_K support, layer skip (0.5 tok/s), self-speculative decoding, F16 KV cache
- **Phase 4** - NVMe Direct: gpu-nvme-direct backend for tier C (GPU-initiated NVMe reads, 3.35 GB/s)
- **Phase 5** - Polish: speculative decoding with draft model, benchmarks, public C API

## License

BSD-2-Clause
