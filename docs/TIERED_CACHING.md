# 3-Tier Adaptive Layer Caching for 70B Streaming

## Problem Statement

70B Q6_K mmap streaming baseline: **0.006 tok/s** (page cache thrashing — 53.5 GB model > 48 GB RAM).

Root cause: every token forces re-reading all 80 layers from NVMe because mmap pages get evicted. Meanwhile, ~20 GB VRAM and ~34 GB RAM sit idle.

## Architecture

Three explicit caching tiers, auto-sized from hardware:

| Tier | Storage | Source | I/O per token | Latency |
|------|---------|--------|---------------|---------|
| **A (VRAM)** | GPU memory | `cudaMalloc` | None — zero-copy | 0 ms |
| **B (RAM)** | Pinned host | `cudaMallocHost` | H2D async DMA | ~51 ms/layer |
| **C (NVMe)** | SSD | mmap/gpunvme | NVMe read + H2D | ~315 ms/layer |

### Auto-Parametric Tier Sizing

Tier sizes are computed automatically at init time:

```
VRAM available = cudaMemGetInfo(free) - 512 MB reserve
RAM available  = sysinfo(freeram + bufferram) - 6 GB reserve

n_vram = min(n_layers, vram_avail / layer_bytes)
n_ram  = min(remaining, ram_avail / layer_bytes)
n_nvme = n_layers - n_vram - n_ram
```

### Example Configurations

| Model | Layers | Layer Size | Tier A | Tier B | Tier C | Tok/s |
|-------|--------|-----------|--------|--------|--------|-------|
| 8B Q8_0 | 32 | 875 MB | 32 | 0 | 0 | ~40 (all resident) |
| 70B Q6_K | 80 | 669 MB | ~29 | ~51 | 0 | ~0.39 |
| 70B Q8_0 | 80 | 875 MB | ~22 | ~38 | ~20 | ~0.11 |

## Performance Model

```
70B Q6_K: 80 layers x 669 MB, RTX 3090 24 GB, 48 GB RAM

Tier A:  ~29 layers x 669 MB = 19.4 GB VRAM   -> 0 ms I/O
Tier B:  ~51 layers x 669 MB = 34.1 GB RAM     -> ~51 ms H2D (669 MB / 13 GB/s)
Tier C:  0 layers (Q6_K fits in A+B)

Per-token pipeline (double-buffer overlap):
  Layers 0-28 (tier A):  29 x ~25ms compute = 725ms
  During those 725ms:    H2D ~14 tier B layers overlap
  Layers 29-79 (tier B): (51-14) x 51ms = ~1887ms (H2D bottleneck)
  Total: ~2.6s -> 0.39 tok/s
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

- All layers fit VRAM -> pure resident mode (like `forward()`)
- All layers need streaming -> pure streaming (like `forward_streaming()`)
- Mix -> hybrid with optimal overlap

No user configuration needed. `--streaming` flag auto-activates tiered mode.

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
