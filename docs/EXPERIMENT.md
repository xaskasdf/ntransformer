# ntransformer — Streaming Experiment Log
**Branch:** `feat/gpunvme-bdf-autodetect`  
**Main repo:** `~/workspace/ntransformer`  
**GPU-NVMe library:** `~/workspace/gpu-nvme-direct` (PR #2)

---

## Hardware

| Component | Spec |
|---|---|
| GPU | RTX 5060 Ti 16GB (sm_120, Blackwell) |
| GPU memory bus | 128-bit GDDR7 (~450 GB/s bandwidth) |
| PCIe slot | Gen 5 x8 = **31.0 GB/s** H2D bandwidth |
| RAM | 32GB DDR5 (20GB available for tier B pinned) |
| Boot NVMe | nvme1n1 (SM2268XT, `0000:03:00.0`) — **DO NOT TOUCH** |
| Experiment NVMe | nvme0n1 (MAXIO MAP1602 ARSDS 2TB, `0000:02:00.0`) — safe for experiments |

> ⚠️ PCIe ASPM (power management) downgrades the link to Gen 2 at idle.
> Detection code uses `max_link_speed` (stable at boot) to avoid bogus readings.

---

## Build

### Standard (no NVMe)
```bash
cd build && cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120" \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build . -j$(nproc)
```

### With gpu-nvme-direct (BAR1 NVMe→VRAM streaming)
```bash
cd build-gpunvme && cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120" \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DUSE_GPUNVME=ON \
  -DGPUNVME_DIR=/home/akoscz/workspace/gpu-nvme-direct
cmake --build . -j$(nproc)
```

**Requirements:** `gcc-14` (not gcc-15 — incompatible with CUDA 13.1), CUDA at `/opt/cuda/`.

---

## Pre-run checklist

```bash
# Always evict Ollama before running (frees ~9GB VRAM — qwen2.5:14b is loaded)
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:14b-instruct-q4_K_M","keep_alive":0}'
# Or: sudo systemctl stop ollama

# Verify VRAM headroom
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader

# For NVMe experiments: unbind nvme driver FIRST (not vfio-pci!)
# ⚠️ DO NOT echo 1 > .../reset while nvme driver is bound — kernel panic
echo 0000:02:00.0 | sudo tee /sys/bus/pci/drivers/nvme/unbind
```

---

## Results

### Benchmark 1 — 8B Llama Q8_0 (resident vs streaming)

| Mode | Prompt | Decode | VRAM | Notes |
|---|---|---|---|---|
| Resident | 34.3 tok/s | **31.0 tok/s** | 15.0 GB | All 32 layers in VRAM |
| Streaming (--streaming) | 34.1 tok/s | **31.0 tok/s** | 15.2 GB | Auto-promoted to 32 VRAM (model fits) |

**vs upstream baseline (RTX 3090 24GB):** 48.9 tok/s  
**Gap explained:** 3090 has 384-bit GDDR6X bus (~936 GB/s) vs our 128-bit GDDR7 (~450 GB/s).
At batch=1 GEMV decode is purely memory-bandwidth-bound → wider bus wins.

---

### Benchmark 2 — 32B Qwen2.5 Q4_K_M (streaming, tier B)

**Tier split:** 19 layers VRAM (5.5 GB) + 45 layers RAM (13.1 GB) + 0 NVMe

| Buffers | Decode | Notes |
|---|---|---|
| 2 (upstream default) | **1.7 tok/s** | Optimal for our bandwidth |
| 3 (experimental) | 1.6 tok/s | Tiny regression — see analysis |

**Output quality:** Garbage (Qwen2 architecture not fully supported). Speed numbers valid; quality not.

---

### Benchmark 3 — 8B Llama Q8_0 (NVMe BAR1 direct streaming)

**Setup:** Model dd'd to nvme0n1 at LBA 0. `gpu-nvme-direct` library with CPU doorbell fallback.

```bash
sudo env LD_LIBRARY_PATH=/home/akoscz/workspace/gpu-nvme-direct/build-hw \
  GPUNVME_PCI_BDF=0000:02:00.0 GPUNVME_GPU_BDF=0000:01:00.0 \
  GPUNVME_GGUF_LBA=0 GPUNVME_MAX_VRAM_LAYERS=8 GPUNVME_MAX_RAM_LAYERS=0 \
  ./build-gpunvme/ntransformer \
  -m ~/workspace/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --streaming -p "Hello" -n 64
```

| Mode | Decode | VRAM | NVMe throughput |
|---|---|---|---|
| All RAM → GPU (H2D) | 3.4 tok/s | 6.7 GB | — |
| 8 VRAM + 24 NVMe BAR1 | **0.9 tok/s** | 6.7 GB | **~5050 MB/s** |

**BAR1 DMA output (confirmed working):**
```
load_layer_vram: 231752192 bytes (443 cmds) in 43.7 ms — 5054.1 MB/s [BAR1→VRAM]
load_layer_vram: 231752192 bytes (443 cmds) in 44.0 ms — 5027.7 MB/s [BAR1→VRAM]
load_layer_vram: 231752192 bytes (443 cmds) in 44.1 ms — 5008.7 MB/s [BAR1→VRAM]
```

**Why 0.9 tok/s < 3.4 tok/s (RAM):**  
NVMe at 5 GB/s < PCIe H2D from RAM at ~16 GB/s. Single-token compute is only ~few ms on
sm_120 — DMA-bound. 24 layers × 44ms/layer ≈ 1056ms/token. Pipelining hides some but not all.

**NVMe streaming is useful when:** models exceed RAM+VRAM capacity (70B Q4_K_M ~40 GB, 405B).
On this machine with 32 GB RAM, up to ~70B Q2_K fits without NVMe.

---

### Pipeline depth analysis (Benchmark 2)

Why 3 buffers didn't help:

```
Layer transfer time: 297 MB / 31 GB/s = 9.6 ms
Layer compute time:  1.7 tok/s → 588ms/token / 64 layers ≈ 9.2 ms
```

Transfer ≈ compute → already perfectly pipelined with 2 buffers. A 3rd adds VRAM pressure
(1 fewer VRAM-resident layer: 18 vs 19) with no latency to hide.

**3 buffers would help when:** transfer >> compute, e.g., much lower PCIe bandwidth.
The auto-detect threshold (≥63 GB/s → 3 buffers) correctly returns 2 for Gen5 x8.

---

## A/B Test — Brain Model (Qwen2.5-14B vs DeepSeek-R1-14B)

Tested both as Ollama brain model for ren-plugin tasks (episode generation, digest, compaction).

| Model | Tokens/task | Format compliance | Notes |
|---|---|---|---|
| `qwen2.5:14b-instruct-q4_K_M` | ~325 | ✅ Follows constraints | **Winner** |
| `deepseek-r1:14b` | ~2411 | ❌ Ignores format | 7× more tokens, too verbose |

**Decision:** Keep Qwen2.5-14B as default. R1-14B stays installed but not default.

---

## Bugs Found and Fixed

### 1. CUDA architecture not native
Upstream targets `sm_80;86;89;90`. RTX 5060 Ti is sm_120. Without `sm_120`, kernels JIT.
**Fix:** Added `120` to `CMAKE_CUDA_ARCHITECTURES`.

### 2. PCIe detection using current_link_speed
ASPM idles link at Gen1/2 = 5 GT/s. Bogus bandwidth estimates between runs.
**Fix:** Use `max_link_speed` / `max_link_width` (stable at boot).

### 3. cudaHostRegisterIoMemory on NVMe BAR0 crashes Blackwell
Registering NVMe MMIO with CUDA triggers GSP firmware errors → Wayland freeze + reboot.
**Fix (gpu-nvme-direct PR #2):** CPU doorbell fallback — BAR0 is CPU-only, GPU never writes MMIO directly.

### 4. PRP Offset Invalid (cqe_status=0x0013) on BAR1 DMA
`cudaMalloc` returns 256-byte aligned VRAM; physical address may have page offset (confirmed: 0x200).
NVMe spec requires PRP2 and list entries to be 4KB-aligned — offset propagates to all entries.
**Fix (gpu-nvme-direct PR #2):** compute `second_page_phys = (chunk_phys + 4095) & ~4095`,
fill list as `second_page_phys + (p-1)*4096`. Handles aligned and misaligned chunk_phys.

### 5. NVMe FLR while nvme driver bound → kernel panic
`echo 1 > /sys/bus/.../reset` while the kernel nvme driver still owns the device = panic.
**Fix:** Always unbind from `nvme` driver first: `echo 0000:02:00.0 | sudo tee /sys/bus/pci/drivers/nvme/unbind`.
The binary's auto-FLR handles recovery internally — no manual reset needed after unbind.

---

## Features Added (ntransformer PRs #4–#11)

| PR | Feature | Branch |
|---|---|---|
| #4 | Blackwell sm_120 native CUDA support | `feat/blackwell-sm120` |
| #5 | PCIe ASPM fix (max_link_speed) | `feat/pcie-aspm` |
| #6 | Smart TierConfig (adaptive RAM reserve, PCIe detection) | `feat/smart-tier` |
| #7 | Configurable pipeline depth (`--n-buffers`, auto-detect) | `feat/configurable-pipeline` |
| #8 | PCIe detection consistency fixes | `feat/pcie-detection-consistency` |
| #9 | TMA async bulk copy (stub, documented as future work) | `feat/tma-async-bulk-copy` |
| #10 | Q2_K GEMV kernel (unlocks 70B) | `feat/q2k-gemv` |
| #11 | Benchmark script | `feat/benchmark-script` |

---

## Known Limitations / Next Steps

| Item | Priority | Notes |
|---|---|---|
| **Q2_K GEMV** | High | PR #10 open — unlocks 70B Q2_K (25 GB, fits in RAM) |
| **Qwen2 architecture** | Medium | 32B output is garbage — RoPE/embedding mismatch |
| **70B Q4_K_M** | Low | ~40 GB, needs NVMe tier — prove end-to-end with fixed Q2_K first |
| **VRAM temp buffer sizing** | Low | 753 MB temp for NVMe BAR1 eats into tier A VRAM; only allocate if n_nvme > 0 |
| **Freeze risk on 70B Q2_K** | ⚠️ Safety | Do NOT run until Q2_K GEMV PR merges + Q2_K kernel validated |

---

## Model Library

`~/workspace/models/`:

| File | Size | Format | Status |
|---|---|---|---|
| `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` | 8 GB | Llama, Q8_0 | ✅ 31 tok/s resident; 0.9 tok/s NVMe BAR1 |
| `Qwen2.5-32B-Instruct-Q4_K_M.gguf` | 19 GB | Qwen2, Q4_K_M | ⚠️ 1.7 tok/s; output garbage |
| `Meta-Llama-3.1-70B-Instruct-Q2_K.gguf` | 25 GB | Llama, Q2_K | ❌ Q2_K GEMV unsupported (PR #10) |

Ollama (separate stack, brain model):

| Model | Size | Status |
|---|---|---|
| `qwen2.5:14b-instruct-q4_K_M` | 9 GB | ✅ Default brain model — do not change |
| `deepseek-r1:14b` | 9 GB | ✅ Installed, not default (too verbose) |
