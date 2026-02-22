# R2: Sparse FFN Loading via Gate Prediction

## Summary

Reduce FFN weight transfer by ~33% by loading only the weight columns corresponding
to neurons that the SwiGLU gate predicts will be active. The gate projection output
is a zero-cost predictor — it's computed as part of normal FFN execution anyway.

## Motivation

In Llama's SwiGLU FFN:
```
FFN(x) = W_down * (SiLU(W_gate * x) ⊙ (W_up * x))
```

The SiLU gate acts as a soft switch: when `SiLU(W_gate * x)[i] ≈ 0`, neuron i
contributes nothing to the output regardless of `W_up * x`. Empirically, ~50% of
neurons gate to near-zero for any given input (Deja Vu, ICML 2023).

The FFN weights dominate each layer:
- Llama 70B: `ffn_gate` + `ffn_up` + `ffn_down` = 3 × [8192 × 28672] ≈ **446 MB each at Q6_K**
- Total FFN per layer: **~1338 MB** (67% of the 669 MB compressed layer — rest is attention)
- If we skip 50% of `ffn_up` and `ffn_down` columns: save **~223 MB per layer**

## Prior Art

| Paper | Approach | Difference from our proposal |
|-------|----------|------------------------------|
| PowerInfer (SOSP 2024) | Offline-profiled hot/cold neurons + MLP predictors | Uses CPU for cold neurons; we use GPU-initiated NVMe selective loading |
| LLM in a Flash (Apple, ACL 2024) | Flash memory + sparsity-aware row-column bundling | Apple silicon flash controller; we use PCIe NVMe with GPU MMIO |
| Deja Vu (ICML 2023) | Contextual sparsity + trained 2-layer MLP predictors | Assumes weights in memory; saves FLOPs not I/O |
| Mixture-of-Channels (Nov 2025) | SwiGLU gate as top-K channel selector | Requires training from scratch with MoC architecture |
| CoreInfer (Oct 2024) | Sentence-level core neuron prediction | Reuses prefill activations; no storage-level optimization |
| SCAP (NeurIPS 2024) | Post-training mode-centering for higher sparsity | Modifies activations, not I/O path |
| R-Sparse (ICLR 2025) | SVD-based rank-aware sparsity, training-free | Compute-level sparsity, not I/O-level |

**Gap**: No existing work uses the SwiGLU gate as a **per-token, zero-cost predictor**
to decide which weight columns to load from NVMe in a streaming pipeline. All prior
systems either (a) assume weights in memory, (b) use offline profiling, or (c)
require training. Our approach is online, per-token adaptive, and targets I/O bandwidth.

## Technical Design

### Two-Phase FFN Execution

```
Phase 1 — Gate computation (full load):
  Load W_gate [8192 × 28672] from NVMe/RAM          ~149 MB, 45 ms (NVMe)
  Compute: g = SiLU(W_gate * x)                      ~0.5 ms
  Generate active mask: mask[i] = |g[i]| > threshold  (top-K or magnitude threshold)
  Active neurons: K ≈ 14336 (50% of 28672)

Phase 2 — Selective load (sparse):
  Load only columns mask[i]==1 of W_up   [8192 × K]  ~74 MB, 22 ms (NVMe)
  Load only rows mask[i]==1 of W_down    [K × 8192]  ~74 MB, 22 ms (NVMe)
  Compute: out = W_down[mask,:] * (g[mask] ⊙ W_up[:,mask] * x)

Total FFN transfer: 149 + 74 + 74 = 297 MB  (vs 446 MB full = 33% reduction)
```

### NVMe Layout for Sparse Access

Standard GGUF stores weights row-major. To efficiently load arbitrary columns,
we need a **column-grouped layout** on the NVMe raw device:

```
Standard GGUF layout (row-major, bad for column selection):
  W_up = [row0_all_cols][row1_all_cols]...[row8191_all_cols]
  To load column 5: need to read byte 5 from every row → scattered reads

Column-grouped layout (good for column selection):
  W_up = [col_group_0: cols 0-255][col_group_1: cols 256-511]...
  To load columns 0-255: single contiguous 8192×256 read

Tile-grouped layout (best for NVMe block alignment):
  W_up = [tile_0: rows 0-255, cols 0-255][tile_1: rows 0-255, cols 256-511]...
  Each tile = 256×256 = 65536 weights = aligned to NVMe blocks
```

For Q6_K (6.625 bits/weight), each tile of 256×256 = 65536 weights ≈ 54 KB.
With NVMe block size 512B and MDTS 1024KB, each tile is a single NVMe command.

Total tiles per FFN matrix: (8192/256) × (28672/256) = 32 × 112 = 3584 tiles.
Loading 50% of column groups: 3584 × 0.5 = 1792 tiles × 54 KB = ~97 MB.

### Active Neuron Prediction Strategies

**Strategy A: Gate threshold (zero-cost, per-token)**
```cpp
// After computing g = SiLU(W_gate * x):
float threshold = 0.01f;  // or adaptive percentile
int K = 0;
for (int i = 0; i < ffn_dim; i++) {
    active_mask[i] = (fabsf(g[i]) > threshold);
    K += active_mask[i];
}
// Expected: K ≈ 14000-15000 (49-52% active)
```

**Strategy B: Top-K selection (fixed sparsity, per-token)**
```cpp
// Select top-K neurons by gate magnitude
// K = ffn_dim / 2 = 14336 → exactly 50% sparsity
partial_sort(indices, indices + K, indices + ffn_dim,
    [&](int a, int b) { return fabsf(g[a]) > fabsf(g[b]); });
```

**Strategy C: CoreInfer-style (sentence-level, amortized)**
```cpp
// During prefill: compute gate for all tokens, take union of active neurons
// Use this fixed mask for all decode tokens in the same sequence
// Advantage: one prediction, reused across all tokens
// Disadvantage: union is larger than per-token (~70-80% active)
```

### Sparse GeMV Kernel

```cuda
// Sparse GeMV: y = W[:,active_cols] * x[active_cols]
// W is stored in column-group tiles; we only load active tiles
__global__ void sparse_gemv_kernel(
    const void* W_tiles,        // column-grouped weight tiles
    const int* active_groups,   // which column groups are active
    int n_active_groups,
    const float* x,             // input (full, but only active entries used)
    float* y,                   // output
    int out_features,
    int group_size              // 256
) {
    // Each block handles one output row
    // Sum contributions from active column groups only
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int g = 0; g < n_active_groups; g++) {
        int col_start = active_groups[g] * group_size;
        // Load tile[row, col_start:col_start+group_size]
        // Dot product with x[col_start:col_start+group_size]
        for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
            sum += dequant(W_tiles, row, col_start + j) * x[col_start + j];
        }
    }
    // Reduce and write
    y[row] = block_reduce_sum(sum);
}
```

### Integration into ntransformer

| File | Change |
|------|--------|
| `src/memory/streamer.h` | Add `sparse_ffn_mode_`, active mask buffers |
| `src/memory/streamer.cu` | Two-phase prefetch: gate first, then selective up/down |
| `src/cuda/sparse_gemv.cu` | **New**: Sparse GeMV kernel for column-grouped tiles |
| `src/model/transformer.cpp` | Modified FFN path: gate → mask → sparse up/down |
| `tools/reformat_gguf.py` | **New**: Convert GGUF to column-grouped layout |
| `tools/write_sparse_nvme.sh` | **New**: Write reformatted model to NVMe raw device |

### Pipeline Timeline

```
Without sparse FFN (current):
  NVMe: [==== gate 149MB ====][==== up 149MB ====][==== down 149MB ====]  135 ms
  GPU:  [                                                                ][compute]

With sparse FFN:
  NVMe: [==== gate 149MB ====]                     45 ms
  GPU:  [                    ][gate compute + mask]  0.5 ms
  NVMe:                       [== up_sparse 74MB ==][== down_sparse 74MB ==]  44 ms
  GPU:                                               [      sparse compute     ]
  Total: ~90 ms (vs 135 ms) = 33% reduction in FFN I/O time
```

## Verification Plan

1. **Sparsity measurement**: Run calibration prompts, measure per-layer gate sparsity
   distribution. Verify ~50% at threshold 0.01 for Llama 70B.

2. **Quality at various sparsity levels**: Measure output quality (exact match + perplexity)
   at 30%, 50%, 70% sparsity. Find the knee of the quality curve.

3. **Column-group granularity**: Verify that group_size=256 aligns with Q6_K quantization
   blocks and NVMe block boundaries.

4. **Output correctness**: "What is the capital of France?" → "Paris" at 50% sparsity.

## Sparsity Levels in Llama 70B SwiGLU

Based on prior literature (Deja Vu, PowerInfer, TurboSparse):

| Activation function | Natural sparsity | With threshold tuning |
|---------------------|------------------|-----------------------|
| ReLU | 85-95% | N/A (exact zeros) |
| GELU | 40-60% | 50-70% with CATS |
| **SiLU/SwiGLU** (Llama) | **45-55%** | **50-60% with threshold** |
| dReLU (TurboSparse) | 90-97% | Requires fine-tuning |

SwiGLU gives us ~50% "for free" without model modification. Higher sparsity (70%+)
requires activation function replacement (dReLU) or fine-tuning (CATS, SCAP).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Column-grouped layout requires model conversion | One-time offline cost | Script converts GGUF → sparse layout in ~10 minutes |
| NVMe scattered reads for non-contiguous groups | Poor I/O efficiency | Tile-based layout with 256-column groups → sequential reads |
| 50% sparsity is modest savings | Only 33% reduction | Combine with R1 (delta) for multiplicative effect |
| Gate threshold varies across layers/tokens | Inconsistent sparsity | Per-layer calibrated thresholds or top-K (fixed budget) |
| Q6_K block structure conflicts with column granularity | Wasted reads at group boundaries | Group size = multiple of Q6_K super-block (256) |
| Two-phase execution adds pipeline bubble | Gate must complete before selective load | Gate is small (1/3 of FFN), ~45ms; total still faster |

## References

- Deja Vu: Contextual Sparsity (ICML 2023) — https://arxiv.org/abs/2310.17157
- PowerInfer (SOSP 2024) — https://arxiv.org/abs/2312.12456
- PowerInfer-2 (Jun 2024) — https://arxiv.org/abs/2406.06282
- LLM in a Flash (Apple, ACL 2024) — https://arxiv.org/abs/2312.11514
- Mixture-of-Channels (Nov 2025) — https://arxiv.org/abs/2511.09323
- CoreInfer (Oct 2024) — https://arxiv.org/abs/2410.18311
- SCAP (NeurIPS 2024) — https://arxiv.org/abs/2412.07174
- R-Sparse (ICLR 2025) — https://arxiv.org/abs/2504.19449
- TurboSparse (Jun 2024) — https://arxiv.org/abs/2406.05955
- CATS (Stanford, Jul 2024) — https://arxiv.org/abs/2404.08763
- ReLU Strikes Back (ICLR 2024) — https://arxiv.org/abs/2310.04564
