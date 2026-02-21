# 3-Tier Adaptive Layer Caching for 70B Streaming

## Problem Statement

70B Q6_K mmap streaming baseline: **0.006 tok/s** (page cache thrashing — 53.5 GB model > 48 GB RAM).

Root cause: every token forces re-reading all 80 layers from NVMe because mmap pages get evicted. Meanwhile, ~20 GB VRAM and ~34 GB RAM sit idle.

## Architecture

Three explicit caching tiers, auto-sized from hardware:

| Tier | Storage | Source | I/O per token | Latency |
|------|---------|--------|---------------|---------|
| **A (VRAM)** | GPU memory | `cudaMalloc` | None — zero-copy | 0 ms |
| **B (RAM)** | Pinned host | `cudaMallocHost` | H2D async DMA | ~103 ms/layer (Gen3 x8) |
| **C (NVMe)** | SSD | mmap/gpunvme | NVMe read + H2D | ~315 ms/layer |

### Auto-Parametric Tier Sizing

Tier sizes are computed automatically at init time:

```
VRAM available = cudaMemGetInfo(free) - inference_reserve
  inference_reserve = KV_cache + workspace + hidden_bufs + 256 MB safety

RAM available  = /proc/meminfo MemAvailable - 6 GB reserve

n_vram = min(n_layers, vram_avail / layer_bytes)
n_ram  = min(remaining, ram_avail / layer_bytes)
n_nvme = n_layers - n_vram - n_ram
```

Key fix: VRAM reserve is computed dynamically from model config (KV cache depends on n_layers, max_seq_len, n_kv_heads). RAM uses `/proc/meminfo` MemAvailable (not `sysinfo(freeram)`) to account for reclaimable page cache.

### Measured Results

| Model | Layers | Layer Size | Tier A | Tier B | Tier C | Tok/s | VRAM |
|-------|--------|-----------|--------|--------|--------|-------|------|
| 8B Q8_0 | 32 | 221 MB | 32 | 0 | 0 | **48.8** | 10.3 GB |
| 70B Q6_K (ctx=4096) | 80 | 669 MB | 24 | 54 | 2 | **0.2** | 23.0 GB |
| 70B Q6_K (ctx=512) | 80 | 669 MB | 29 | 51 | 0 | **0.2** | 23.1 GB |

**Hardware**: RTX 3090 (Gen3 x8, B450), 48 GB DDR4, WD SN740 512GB.

### Performance Analysis

```
70B Q6_K (ctx=512): 80 layers x 669 MB, RTX 3090 24 GB, 48 GB RAM

Tier A:  29 layers x 669 MB = 19.0 GB VRAM   -> 0 ms I/O
Tier B:  51 layers x 669 MB = 33.3 GB RAM     -> ~103 ms H2D (669 MB / 6.5 GB/s Gen3 x8)
Tier C:  0 layers (Q6_K fits in A+B)

Per-token measured: ~5.5 seconds
  Tier A compute:  29 x ~0.7ms = ~20ms (memory-bandwidth limited GeMV)
  Tier B pipeline:  51 x max(compute=0.7ms, H2D=103ms) = ~5250ms (H2D bottleneck)
  Overhead:        ~230ms (pipeline startup, final norm, LM head)
  Total:           ~5.5s -> 0.18 tok/s (measured 0.2)

Bottleneck: PCIe Gen3 x8 H2D bandwidth (~6.5 GB/s)
  - B450 runs GPU at x8 due to M.2_1 lane sharing
  - At Gen3 x16 (13 GB/s): ~51ms/layer -> ~2.6s -> 0.39 tok/s
  - At Gen4 x16 (25 GB/s): ~27ms/layer -> ~1.4s -> 0.7 tok/s (compute-bound)
```

## Implementation

### Forward Pass: `forward_tiered()`

Hybrid loop that handles both resident and streamed layers:

```
for each layer i:
    if tier A (VRAM):
        compute directly (weights permanently set at init)
    else (tier B/C):
        double-buffer pipeline (same as forward_streaming)
        slot = (i - first_stream) % 2
        wait_transfer(slot) -> set_weights -> compute -> signal_done
```

The pipeline prefetches ahead: while computing layer `i`, H2D runs for `i+1` and staging prefetches `i+2`.

### Graceful Degradation

- All layers fit VRAM -> pure resident mode (~48.8 tok/s for 8B)
- All layers need streaming -> pure streaming (same as `forward_streaming()`)
- Mix -> hybrid with optimal overlap

No user configuration needed. `--streaming` flag auto-activates tiered mode.

### Key Bug Fixes

1. **VRAM OOM**: Static 512 MB reserve was insufficient for 70B (KV cache = 2.5 GB). Fixed with dynamic reserve computed from model config.
2. **RAM undercount**: `sysinfo(freeram)` ignores reclaimable page cache (showed 2 GB instead of 42 GB). Fixed with `/proc/meminfo` `MemAvailable`.

## v2: Dynamic VRAM Swapping

Instead of static tier A, use VRAM as a ring buffer:

```
VRAM slots: [S0] [S1] [S2] [S3]

Layer 0 -> S0 (preloaded)
Layer 1 -> S1 (preloaded)
Layer 2 -> S2 (preloaded)
Layer 3 -> S3 (preloaded)
compute L0, free S0, begin H2D L4 -> S0
compute L1, free S1, begin H2D L5 -> S1
...
```

For sequential transformers: same throughput as static tier A (overlap math identical). Advantage for MoE / non-sequential access patterns where layer order isn't predictable.

Planned after static tiered caching is validated.

## Speedup Summary

| Mode | Tok/s | vs Baseline | Bottleneck |
|------|-------|-------------|------------|
| mmap streaming (baseline) | 0.006 | 1x | Page cache thrashing |
| NVMe direct streaming | 0.03 | 5x | NVMe read bandwidth |
| **3-Tier caching (Gen3 x8)** | **0.2** | **33x** | **PCIe H2D at Gen3 x8** |
| 3-Tier caching (Gen3 x16 est.) | ~0.4 | ~65x | PCIe H2D at Gen3 x16 |
| 3-Tier caching (Gen4 x16 est.) | ~0.7 | ~115x | GeMV compute |
