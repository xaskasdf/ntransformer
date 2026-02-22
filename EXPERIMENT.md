# ntransformer — Streaming Experiment (ren/streaming-experiment)

## Setup (2026-02-22)

### What's here
Worktree off main repo at `~/workspace/ntransformer` for our PCIe streaming experiment.
Branch: `ren/streaming-experiment`

### Build status
- ✅ Built with: gcc-14 + nvcc 13.1 (CUDA host compiler: g++-14)
- ✅ All 13 tests pass (7 tensor, 6 kernel) on RTX 5060 Ti
- PTX JIT path: binary compiled for sm_80/86/89/90, JIT-compiled to sm_120 (Blackwell) at runtime

### Rebuild with sm_120 target (recommended for perf)
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build . --config Release -j$(nproc)
```

### Hardware
- GPU: RTX 5060 Ti 16GB (sm_120, Blackwell), PCIe Gen 5 x16
- RAM: 32GB (23GB available for pinned tier B)
- Boot NVMe: nvme1n1 (Viper 1TB) — DO NOT TOUCH
- Experiment NVMe: nvme0n1 (ARSDS 2TB) — safe for VFIO/direct path
  - PCI BDF: **0000:02:00.0** (MAXIO MAP1602 controller)

## Phase 1: RAM Streaming (no kernel hacks)

Get a 30B model and benchmark tier B (RAM) streaming:

```bash
# Download Llama 3.1 30B Q4_K_M (~17GB) — pick a mirror
# e.g. huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF ...
# or smaller first: Llama-3.1-8B-Q8_0 (~8GB) as sanity check

# Benchmark resident (8B, should hit ~48 tok/s per upstream)
./build/ntransformer -m /path/to/8b-q8.gguf --benchmark -n 64

# Benchmark streaming (30B, tier B via RAM)
./build/ntransformer -m /path/to/30b-q4km.gguf --benchmark -n 64 --streaming
```

## Phase 2: NVMe Direct (requires VFIO setup)

Only attempt on nvme0n1 after Phase 1 benchmarks look promising.
See CLAUDE.md for setup steps. Key risk: binding nvme0n1 to VFIO removes it from /dev/.

## Notes
- Their RTX 3090 (24GB, PCIe Gen 4 x8): 0.2 tok/s on 70B Q6_K
- Our advantage: PCIe Gen 5 x16 = ~5x bandwidth → tier B transfers dramatically faster
- Expected 30B Q4_K_M: fits ~9GB in VRAM, ~8GB streamed from RAM
- Target: beat Ollama's mmap baseline (should be easy), hopefully 1-3 tok/s on 30B
