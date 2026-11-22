# NTransformer

High-efficiency C++/CUDA LLM inference engine. Runs Llama 70B on a single RTX 3090 (24GB VRAM) by streaming model layers through GPU memory via PCIe, with optional NVMe direct I/O that bypasses the CPU entirely. It also runs the 320B-parameter GLM-5.3-Flash MoE (Q8_0, 341 GB) on the same RTX 3090 by streaming experts straight from a raw NVMe drive, and DeepSeek-V4.1-Flash on H100s.

## Key Results

| Model | Mode | Decode | VRAM | Notes |
|-------|------|--------|------|-------|
| Llama 3.1 8B Q8_0 | Resident | 48.9 tok/s | 10.0 GB | All layers in VRAM |
| Llama 3.1 8B Q8_0 | Tiered (auto) | 48.8 tok/s | 10.3 GB | 32/32 layers auto-promoted to VRAM |
| Llama 3.1 70B Q6_K | Streaming (mmap) | 0.006 tok/s | 7.3 GB | Page cache thrashing (53 GB > 48 GB RAM) |
| Llama 3.1 70B Q6_K | Tiered (auto) | 0.2 tok/s | 23.1 GB | 26 VRAM + 54 RAM + 0 NVMe |
| Llama 3.1 70B Q4_K_M | Tiered (auto) | 0.3 tok/s | 22.9 GB | 36 VRAM + 44 RAM (50% faster) |
| Llama 3.1 70B Q4_K_M | **Tiered + layer skip** | **0.5 tok/s** | **22.9 GB** | **36 VRAM + 44 RAM, 20 layers skipped** |
| GLM-5.3-Flash Q8_0 (320B MoE) | Windows, expert LRU + pinned RAM rings | 0.24–0.26 tok/s | 6.6 GiB resident + 13.3 GiB expert cache | Expert fill via ReadFile at 1.33 GiB/s |
| GLM-5.3-Flash Q8_0 (320B MoE) | **Linux, experts via gpu-nvme-direct** | **0.45–0.5 tok/s** | 6.6 GiB resident + 13.7 GiB expert cache | Disk-bound (SN5000S ~3.4 GB/s); greedy output bit-exact vs llama.cpp |
| DeepSeek-V4.1-Flash Q8_0 | 2× H100 SXM5, owner-GPU experts | 18.2–23.5 tok/s | 64 + 72 GiB | llama.cpp on the same setup: 31.4–33.9 tok/s |

**3-tier adaptive caching** auto-sizes from hardware: VRAM-resident layers (zero I/O) + pinned RAM (H2D only) + NVMe/mmap fallback. Achieves **83x speedup** over mmap baseline for 70B on consumer hardware (RTX 3090 + 48 GB RAM).

Bottleneck is PCIe H2D bandwidth at Gen3 x8 (~6.5 GB/s). Q4_K_M fits 10 more layers in VRAM (36 vs 26), reducing tier B transfers. Layer skip (cosine similarity calibration) eliminates 20/80 layers per token with minimal quality loss.

## Features

- **Zero external dependencies** beyond CUDA Toolkit (no PyTorch, no cuBLAS)
- **GGUF model format** with Q4_0, Q8_0, Q4_K_M, Q5_K, Q6_K, F16, F32 quantization
- **3-Tier Adaptive Caching**: auto-sized VRAM resident + pinned RAM + NVMe/mmap tiers
- **SLEP streaming**: double-buffered layer pipeline overlaps NVMe reads, PCIe DMA, and GPU compute
- **Native CUDA peer VRAM cache**: overflow layer blocks on a second GPU, staged by async P2P DMA through the existing double buffers
- **gpu-nvme-direct backend**: userspace NVMe driver reads model weights directly to pinned GPU-accessible memory
- **Layer skip**: cosine-similarity calibration skips redundant layers (20/80 skipped at threshold 0.98)
- **Self-speculative decoding**: VRAM-resident layers as draft model (no extra model needed)
- **Four data paths** (auto-selected): VRAM resident > pinned RAM H2D > mmap pinned > CPU worker memcpy
- Llama architecture: RoPE, GQA, SwiGLU, RMSNorm, KV cache

## Requirements

- Linux (tested on Ubuntu, kernel 6.17+) with CUDA Toolkit 13.1 and gcc-14 / g++-14
- or Windows with MSVC (VS 2026 Build Tools) + CUDA 13.2 + Ninja (no NVMe direct I/O)
- NVIDIA GPU with Compute Capability 8.0+ (RTX 3090 tested)
- or AMD GPU with ROCm 7.x (`-DUSE_HIP=ON`, Linux or Windows): **Llama path only**. The DS41/GLM5 MoE engines are CUDA-only, and a HIP build reports them as unavailable.
- CMake 3.24+
- (Optional) NVMe SSD on separate PCIe slot + [gpu-nvme-direct](https://github.com/xaskasdf/gpu-nvme-direct) library

Prebuilt binaries: every push runs `.github/workflows/build.yml`, which compiles Linux CUDA, Windows CUDA and Linux ROCm builds and attaches them to the run as artifacts. Pushing a `v*` tag also publishes them as a GitHub Release. CUDA builds target sm_80/86/89/90 and link the CUDA runtime statically, so they need only the NVIDIA driver. The Windows build also needs the Visual C++ 2015–2022 Redistributable. Linux builds need glibc 2.39+ (Ubuntu 24.04). The runners have no GPU, so the pipeline compiles but does not run tests.

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

## H100 CUDA P2P

The native peer cache builds with `USE_GPUNVME=OFF`. On two H100 SXM5 GPUs under
shared training load, Llama 3.1 8B Q8_0 with 16 layers in peer VRAM decoded at
24.75 tok/s versus 12.5 tok/s with those layers in pinned RAM. The all-resident
baseline reached 32.25 tok/s. All cache modes generated identical text.

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/ntransformer -m /path/to/model.gguf \
  --peer-gpu 1 --local-layers 16 -p "The capital of France is" \
  -n 128 -c 256 -t 0 --repeat-penalty 1.0
```

`--peer-gpu` automatically enables tiered streaming. Omit `--local-layers` to
cache only overflow layers; use `--local-layers 0` to stage every layer from the
peer. Compute and KV caches stay on the first visible GPU. See the
[H100 benchmark report](docs/H100_BENCHMARKS.md) for measurements, validation and
reproduction commands.

## MoE Models: GLM-5.3-Flash and DeepSeek-V4.1

The CLI picks the engine from the GGUF's `general.architecture`: `glm5-next`
and `deepseek41` run their own MoE paths; everything else uses the Llama
engine. Both MoE paths have an exact CPU reference and GPU offload with a VRAM
LRU expert cache, and are configured through `GLM5_*` / `DS41_*` environment
variables rather than the Llama flags. `--token-ids` and `--dump-logits` feed
fixed token IDs and dump prompt logits for parity checks against llama.cpp.

**GLM-5.3-Flash on one RTX 3090** (320B total / 18B active, 288 experts, Q8_0
341 GB; RTX 3090 + Ryzen 5800X + 48 GB RAM). Each decode token touches ~9.2 GB
of experts, so throughput is set by how fast experts reach the GPU:

| | Windows (expert fill via ReadFile) | Linux (experts read from raw NVMe by gpu-nvme-direct) |
|---|---|---|
| Decode | 0.24–0.26 tok/s | **0.45–0.5 tok/s** (disk ceiling) |
| Prefill | 1.05 s/tok at 138 tokens | ~29 s per chunk (union expert load dominates)¹ |
| Expert fill | 1.33 GiB/s | 2.7–3.2 GB/s |
| VRAM | 6.57 GiB resident + 13.3 GiB expert cache | 6.57 GiB resident + 13.71 GiB (1652 slots) |
| Greedy parity vs llama.cpp | bit-exact | bit-exact |

¹ Measured on a single chunk. Large prefill chunks race in the union expert
load; the parity-verified configuration is `GLM5_CHUNK=3`.

```bash
# experts filled from the GGUF shards (ReadFile on Windows, mmap memcpy elsewhere)
GLM5_GPU=1 ./build/ntransformer -m GLM-5.3-Flash-Q8_0-00001-of-00008.gguf \
  --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512
# →  Paris. In French, Paris is spelled "Paris", but
```

The Linux path reads experts in place from the raw NVMe blocks of the original
GGUF (LBA index regenerated with FIEMAP; no repack) while non-expert weights
come from a 9.8 GB backbone pack. Projected next steps: a second fast NVMe in
stripe (~0.9–1.0 tok/s) or a PCIe 4.0 drive (~1.5–1.9 tok/s). Setup, the
runbook and every finding are in [GLM5_PROJECT.md](docs/GLM5_PROJECT.md),
[GLM5_LINUX_BENCH.md](docs/GLM5_LINUX_BENCH.md) and [linux_port/](linux_port/README_LINUX_PORT.md).

**DeepSeek-V4.1-Flash on 2× H100 SXM5**: 23.5 / 21.6 / 18.2 tok/s decode at
32 / 128 / 512 prompt tokens versus llama.cpp's 33.7 / 33.9 / 31.6 on the same
GPUs, with 62 GiB expert cache per GPU. The outputs are not bit-identical to
llama.cpp; NLL/KL and argmax agreement are reported alongside speed. See
[the fair comparison](docs/benchmarks/h100-deepseek-2026-10-16/fair-comparison/REPORT.md)
and [DEEPSEEK_PERFORMANCE_GAP.md](docs/DEEPSEEK_PERFORMANCE_GAP.md).

## NTRPU FPGA Development

The experimental CPU-only NTRPU simulator models distributed weights, DMA queues,
and row-sharded INT8/INT32 and Q8_0/FP32 GEMV across two virtual 512 GiB FPGA units
using sparse memory. Integer results are checked against an independent INT64 reference;
Q8_0 tests check quantization error and distributed equivalence.
It does not yet run full inference or emulate hardware timing. No CUDA toolkit is required:

```bash
cmake -S . -B build/ntrpu-sim -DNTRANSFORMER_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/ntrpu-sim --config Release --parallel
ctest --test-dir build/ntrpu-sim -C Release --output-on-failure
```

See the [simulator guide](docs/fpga/SIMULATOR.md) for execution commands,
the [GEMV contract](docs/fpga/GEMV_INT8.md) for numerical limits and GGUF considerations,
and the [FPGA recovery and architecture notes](docs/fpga/README.md) for the design.
The [HLS guide](docs/fpga/HLS.md) covers the bounded synthesis source, reproducible
Vitis flow, and experimental Q8_0 host adapter. RTL synthesis and co-simulation are pending.
The [distributed Q8_0 demo](docs/fpga/Q8_DISTRIBUTED.md) chains two operations through
the simulator queues using local weights and FP32 activation transfers.
The [Brandon FFN reference](docs/fpga/BRANDON_REFERENCE.md) adds local SwiGLU commands
and tests real model weights against NumPy and PyTorch CUDA, with a Lambda container recipe.
The CUDA inference engine remains enabled by default.

## System Setup

Running ntransformer with NVMe direct I/O requires system-level modifications. An automated setup script handles all of them:

```bash
# Full first-time setup (interactive, creates backups)
sudo ./scripts/setup_system.sh

# Check current system state (no changes)
sudo ./scripts/setup_system.sh --check

# NVMe-only (run after every reboot)
sudo ./scripts/setup_system.sh --nvme-only
```

### What the script modifies and why

| Phase | What | Why | Risk | Rollback |
|-------|------|-----|------|----------|
| 1 | Installs gcc-14, cmake, kernel headers | CUDA 13.1 is incompatible with gcc-15 (Ubuntu 25.10 default) | Low — standard packages | `apt remove` |
| 2 | Adds `amd_iommu=off` to GRUB | AMD root complex drops GPU→NVMe P2P reads if IOMMU is on. Disabling IOMMU lets posted PCIe writes (doorbells) through | **Medium** — removes hardware DMA isolation between all PCIe devices. Don't run on multi-tenant/server systems | Remove `amd_iommu=off` from `/etc/default/grub`, run `update-grub`, reboot |
| 3 | Patches NVIDIA DKMS (`os-mlock.c`) | `follow_pfn()` was removed in kernel 6.12+. Without the patch, `cudaHostRegisterIoMemory` fails and the GPU can't map NVMe BAR0 for MMIO writes | **High** — bad patch prevents GPU driver from loading (black screen on reboot). Backup `.orig` created automatically | `cp os-mlock.c.orig os-mlock.c` in DKMS source dir, `dkms remove/install nvidia/VERSION` |
| 3b | Patches CUDA header (`math_functions.h`) | glibc 2.42+ (Ubuntu 25.10) declares `rsqrt()`/`rsqrtf()` with `noexcept`. CUDA 13.1 declares without, causing build failure | Low — only affects one header, backup created | `cp math_functions.h.orig math_functions.h` |
| 4 | Loads VFIO modules (`vfio`, `vfio-pci`) | NVMe must be bound to VFIO for userspace access. Consumer GPUs (GeForce) require `enable_unsafe_noiommu_mode=1` | Low — modules unload on reboot. "Unsafe noiommu" means no IOMMU DMA protection for VFIO devices | Reboot (or `modprobe -r vfio-pci vfio`) |
| 5 | Unbinds NVMe from kernel, binds to VFIO | gpu-nvme-direct needs raw PCIe access. The NVMe disappears from `/dev/` while bound to VFIO | **High if wrong device** — never run on your boot drive. Script auto-detects and refuses boot devices | `sudo ./scripts/restore_nvme.sh` |

### BIOS settings (manual, before running the script)

- **Above 4G Decoding**: ON (required for 64-bit BAR mapping)
- **IOMMU**: OFF (or leave on — the script adds the kernel parameter)
- **Secure Boot**: OFF (required for unsigned/patched kernel module loading)

### Hardware disclaimer

> **WARNING**: This project performs low-level PCIe operations (GPU MMIO writes to NVMe controller
> registers, userspace NVMe command submission, VFIO device passthrough). While tested extensively on
> RTX 3090 + WD SN740, incorrect configuration or hardware incompatibilities could theoretically cause:
>
> - **NVMe link failure** requiring power cycle (observed during development with GPU reads)
> - **Data loss** on the NVMe device used for raw block storage
> - **System instability** from disabled IOMMU or patched kernel modules
>
> **Never use your boot drive for NVMe direct I/O.** Always use a dedicated secondary NVMe.
> The authors are not responsible for hardware damage or data loss. Use at your own risk.

### Scripts reference

| Script | Purpose | When to run |
|--------|---------|-------------|
| `scripts/setup_system.sh` | Full system configuration (7 phases) | First-time setup |
| `scripts/setup_system.sh --nvme-only` | VFIO + NVMe bind only | After every reboot |
| `scripts/setup_system.sh --check` | Verify system state | Debugging |
| `scripts/setup_nvme.sh [BDF]` | Bind single NVMe to VFIO | After reboot (standalone) |
| `scripts/restore_nvme.sh [BDF]` | Restore NVMe to kernel driver | When done with NVMe direct |

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
├── core/           # Tensor, allocator, GPU device, cross-platform file mapping
├── cuda/           # CUDA kernels: GEMV, RMSNorm, RoPE, SwiGLU, softmax
├── memory/         # SLEP layer streaming engine (NVMe + mmap backends), peer GPU cache
├── model/          # Llama transformer: config, GGUF loader (split GGUFs), attention, FFN, norms
│   ├── ds41/       # DeepSeek-V4.1-Flash: CPU reference + GPU offload, expert cache
│   └── glm5/       # GLM-5.3-Flash: CPU reference, GPU engine, fast path, gpnvme experts
├── inference/      # Tokenizer, sampler, engine
├── ntrpu/          # CPU-only NTRPU (FPGA) functional simulator
├── training/       # Experimental training CUDA ops (C ABI for PyTorch)
├── utils/          # Timer, logger
├── main.cpp        # CLI entry point (dispatches by general.architecture)
fpga/               # NTRPU HLS kernels, SystemVerilog RTL, testbenches
python/             # ntransformer_training PyTorch adapter
linux_port/         # GLM-5.3-Flash Linux runbook, LBA indexes, gpu-nvme-direct patch
scripts/
├── setup_system.sh # Full system setup (GRUB, NVIDIA patch, CUDA patch, VFIO, NVMe)
├── setup_nvme.sh   # Bind NVMe to VFIO, configure for gpu-nvme-direct
├── restore_nvme.sh # Restore NVMe to kernel driver
tests/              # Standalone tests (CTest), incl. DS41, NTRPU and training checks
tools/              # GGUF tools, DS41 benchmarks, NTRPU generators/checkers, linux_kit bridge
docs/               # Design notes, research, benchmark records (docs/benchmarks)
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

## Experimental H100 training

The optional training backend adds native CUDA RMSNorm/SwiGLU gradients and
pre-reserved peer GPU activation blocks with asynchronous DMA and backward
prefetch. It uses PyTorch for GEMM, attention, AdamW and DDP. The default uses
native RMSNorm pointwise operations with PyTorch reductions. In a paired
100-step Brandon Mini continuation it reached 334,814 global tokens/s versus
301,551 (+11.0%), reducing peak allocated memory from 18.44 to 14.94 GiB/GPU.
Peer staging and block recomputation also passed the numerical checks.
See [the training experiment](docs/H100_TRAINING_EXPERIMENT.md) for measured memory,
limitations, architecture, build instructions and the verified-instruction pilot.
The compatible RMSNorm/SwiGLU SFT pilot reproduces final weights, AdamW/RNG state
and recorded losses exactly. The older fully fused RMSNorm remains opt-in: it
reached +25.8% in pretraining but failed the fresh-optimizer SFT check.
With `--deterministic`, a separate 100-step pretraining audit also matches weights,
AdamW/RNG state and recorded losses exactly, including local and peer recomputation.
See the [numerical difference review](docs/benchmarks/h100-training-2026-10-13/differences/DIFFERENCES.md)
for attention-backward variability and the diagnostic mode's performance cost.

## License

BSD-2-Clause
