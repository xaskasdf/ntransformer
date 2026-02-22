# ntransformer — Streaming Experiment Log
**Branch:** `ren/streaming-experiment`  
**Worktree:** `~/workspace/ntransformer-experiment`  
**Main repo:** `~/workspace/ntransformer` (stays on `main`)

---

## Hardware

| Component | Spec |
|---|---|
| GPU | RTX 5060 Ti 16GB (sm_120, Blackwell) |
| GPU memory bus | 128-bit GDDR7 (~450 GB/s bandwidth) |
| PCIe slot | Gen 5 x8 = **31.0 GB/s** H2D bandwidth |
| RAM | 32GB DDR5 (20GB available for tier B pinned) |
| Boot NVMe | nvme1n1 (Viper VP4300L 1TB) — **DO NOT USE** |
| Experiment NVMe | nvme0n1 (ARSDS 2TB, PCI `0000:02:00.0`) — safe for VFIO |

> ⚠️ PCIe ASPM (power management) downgrades the link to Gen 2 at idle.
> The detection code now uses `max_link_speed` (stable at boot) to avoid bogus readings.

---

## Build

```bash
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build . -j$(nproc)
```

**Requirements:** `gcc14` package (not system gcc-15, incompatible with CUDA 13.1), CUDA at `/opt/cuda/`.

**Commits on this branch:**
- `a9ff91a` — upstream HEAD at time of fork
- `c774c9d` — experiment notes + build config
- `fe9a57a` — 8B baseline results
- `4e744c4` — feat: smart TierConfig defaults (adaptive RAM reserve + PCIe detection)
- `5fa6731` — feat: configurable pipeline depth (N-buffer with auto-detect)
- `17a915c` — fix: PCIe detection uses max_link_speed (ASPM idle fix)

---

## Results

### Pre-run checklist
```bash
# Always evict Ollama before running (frees ~9GB VRAM)
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:14b-instruct-q4_K_M","keep_alive":0}'
# Verify VRAM headroom
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
```

---

### Benchmark 1 — 8B Llama Q8_0 (resident vs streaming)

| Mode | Prompt | Decode | VRAM | Notes |
|---|---|---|---|---|
| Resident | 34.3 tok/s | **31.0 tok/s** | 15.0 GB | All 32 layers in VRAM |
| Streaming (--streaming) | 34.1 tok/s | **31.0 tok/s** | 15.2 GB | Auto-promoted to 32 VRAM (model fits) |

**vs upstream baseline (RTX 3090 24GB):** 48.9 tok/s  
**Gap explained:** 3090 has 384-bit GDDR6X bus (~936 GB/s) vs our 128-bit GDDR7 (~450 GB/s). At batch=1 GEMV decode is purely memory-bandwidth-bound → wider bus wins.

---

### Benchmark 2 — 32B Qwen2.5 Q4_K_M (streaming, tier B)

**Tier split:** 19 layers VRAM (5.5 GB) + 45 layers RAM (13.1 GB) + 0 NVMe

| Buffers | Decode | Notes |
|---|---|---|
| 2 (upstream default) | **1.7 tok/s** | Optimal for our bandwidth |
| 3 (experimental) | 1.6 tok/s | Tiny regression — see analysis |

**Output quality:** Garbage (Qwen2 architecture not fully supported — RoPE scaling / tied embeddings differ from Llama). Speed numbers are valid; quality is not.

---

### Pipeline depth analysis

Why 3 buffers didn't help:

```
Layer transfer time: 297 MB / 31 GB/s = 9.6 ms
Layer compute time:  1.7 tok/s → 588ms/token / 64 layers ≈ 9.2 ms
```

Transfer ≈ compute → they're already perfectly pipelined with 2 buffers. A 3rd buffer
adds VRAM pressure (1 fewer VRAM-resident layer: 18 vs 19) with no latency to hide.

**3 buffers would help when:** transfer >> compute, i.e., much lower PCIe bandwidth.
The auto-detect threshold (≥63 GB/s → 3 buffers) correctly returns 2 for our Gen5 x8.

---

## Bugs Found and Fixed

### 1. CUDA architecture not native (fixed in build)
Upstream targets `sm_80;86;89;90`. RTX 5060 Ti is sm_120 (Blackwell).
Without `sm_120`, kernels run via PTX JIT — extra startup latency, no Blackwell-specific opts.
**Fix:** Added `120` to `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

### 2. PCIe detection using current link speed (fixed in commit 17a915c)
`current_link_speed` fluctuates with ASPM power management (idles at Gen1/2 = 5 GT/s).
Caused wildly wrong bandwidth estimates that varied between runs.
**Fix:** Prefer `max_link_speed` / `max_link_width` (stable at boot), fall back to `current_*`.

---

## New Features Added

### `--n-buffers N` (commit 5fa6731)
Controls pipeline depth (number of GPU streaming buffers).
- `0` (default): auto-detect from PCIe bandwidth (≥63 GB/s → 3, otherwise → 2)
- `2`: upstream default, optimal for Gen5 x8 and below
- `3+`: useful on Gen5 x16 (64+ GB/s) where transfer < compute

Also controllable via env: `NT_PIPELINE_DEPTH=3 ./ntransformer ...`

### Smart TierConfig defaults (commit 4e744c4)
- **RAM reserve:** `max(4GB, total_ram × 15%)` instead of hardcoded 6GB — scales to any machine
- **PCIe detection:** reads sysfs `max_link_speed` + `max_link_width`, logs detected bandwidth
- **`TierConfig.pcie_bandwidth_gbps`:** stored for downstream use (pipeline depth, diagnostics)

---

## Known Limitations / Next Steps

| Item | Priority | Effort | Impact |
|---|---|---|---|
| **Q2_K GEMV kernel** | High | 2-3 days | Unlocks 70B (currently crashes) |
| **Qwen2 architecture fix** | Medium | Unknown | Clean output from 32B we have |
| **Freeze risk on 70B** | ⚠️ Safety | — | Do NOT run 70B Q2_K — froze desktop |
| **`--requant-q4k` test** | Low | 1 hour | Reduce Q6_K tier B transfers by 31% |
| **NVMe direct (tier C)** | Low | Half day | nvme0n1 is safe, scripts exist in repo |

### To unlock 70B properly
Get a supported quant. Options:
- `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` at **Q3_K_M** (~29GB, check if Q3_K_M GEMV exists)
- `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` at **Q4_K_M** (~40GB, needs NVMe tier — exceeds VRAM+RAM)
- Implement Q2_K GEMV in `src/cuda/gemm.cu` and use the Q2_K file we already have

---

## Model Library

Located at `~/workspace/models/`:

| File | Size | Format | Status |
|---|---|---|---|
| `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` | 8 GB | Llama, Q8_0 | ✅ Works, 31 tok/s resident |
| `Qwen2.5-32B-Instruct-Q4_K_M.gguf` | 19 GB | Qwen2, Q4_K_M | ⚠️ Speed OK (1.7 tok/s), output garbage |
| `Meta-Llama-3.1-70B-Instruct-Q2_K.gguf` | 25 GB | Llama, Q2_K | ❌ Q2_K GEMV unsupported |

Ollama (separate stack):
| Model | Size | Status |
|---|---|---|
| `qwen2.5:14b-instruct-q4_K_M` | 9 GB | ✅ Default brain model |
| `deepseek-r1:14b` | 9 GB | ✅ Available for ad-hoc reasoning |
