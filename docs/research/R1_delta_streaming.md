# R1: Delta-Encoded Layer Streaming

## Summary

Reduce per-layer NVMe/RAM transfer from 669 MB to ~50 MB by exploiting cross-layer
weight similarity. Store weights as a shared base layer + per-layer low-rank deltas.

## Motivation

Llama 70B has 80 transformer layers. Middle layers (roughly 10-65) are remarkably
similar to each other — cosine similarity of hidden states between adjacent layers
exceeds 0.98 (measured in our layer skip calibration). This similarity extends to
the weight matrices themselves: adjacent layers' attention and FFN weights differ
by a low-rank perturbation.

Today we transfer the full 669 MB for every tier B/C layer, even though ~95% of
those bytes are redundant with the previous layer. Delta encoding eliminates this
redundancy at the I/O level.

## Prior Art

| Paper | What it does | Difference from our proposal |
|-------|-------------|------------------------------|
| DeltaLLM (Jan 2025) | Shares weights between adjacent blocks + low-rank deltas | Targets storage compression; requires distillation training (~40M tokens) |
| Basis Sharing (ICLR 2025) | Shares SVD singular vectors across layers, per-layer coefficients | Targets compression; not applied to streaming I/O |
| Relaxed Recursive Transformers (Oct 2024) | Converts LLMs to recursive with LoRA per-layer | Requires architecture change and fine-tuning |
| MASA (Aug 2025) | Dictionary atoms shared across attention matrices | Training-based; attention-only |
| DOCS (ICLR 2025) | Quantifies weight cosine similarity between layers | Empirical study; no compression scheme |

**Gap**: No prior work applies cross-layer delta decomposition to a streaming inference
pipeline where the goal is reducing bytes transferred from storage per forward pass.
All existing work assumes weights are already resident in memory.

## Technical Design

### Offline Decomposition (one-time preprocessing)

For each weight matrix W_i in layer i (7 matrices: attn_q/k/v/o, ffn_gate/up/down):

```
1. Compute shared base:  B = mean(W_0, W_1, ..., W_79)   [or weighted by layer importance]
2. Compute residual:     R_i = W_i - B
3. SVD of residual:      R_i = U_i * S_i * V_i^T
4. Truncate to rank r:   R_i ≈ U_i[:,:r] * diag(S_i[:r]) * V_i[:r,:]^T
5. Store:                delta_i = (U_i * sqrt(S_i), sqrt(S_i) * V_i^T)  [two thin matrices]
```

For Llama 70B Q6_K with rank r=64:
- **Base per weight matrix**: e.g., attn_q = [8192 × 8192] at Q6_K = ~44 MB
- **Delta per weight matrix**: U=[8192×64] + V=[64×8192] at F16 = ~2 MB
- **7 matrices per layer**: base = ~300 MB total, delta = ~14 MB total
- **With overhead/alignment**: delta ≈ 50 MB per layer

### Runtime Reconstruction

```cuda
// On GPU, after loading delta from NVMe/RAM:
// W_reconstructed = Base + U * V^T
//
// Base is already in gpu_buf (loaded once, kept pinned in RAM or VRAM)
// U and V are the delta (loaded from NVMe, ~50 MB)
// U * V^T is a rank-64 outer product: O(n * r) = cheap

// Step 1: Load delta_i (U, V) into staging buffer
prefetch_staging(layer_i, slot);   // NVMe read: 50 MB instead of 669 MB

// Step 2: Reconstruct W = Base + U * V^T on GPU
// This is a GEMM: [8192 × 64] × [64 × 8192] = [8192 × 8192]
// At rank 64: 8192*64*8192*2 = ~8.6 GFLOP per matrix
// 7 matrices: ~60 GFLOP total → ~0.5 ms on RTX 3090 (118 TFLOP/s F16)
reconstruct_layer_kernel<<<...>>>(base_buf, delta_U, delta_V, gpu_buf[slot]);

// Step 3: Run attention + FFN as usual
compute_layer(layer_i, slot);
```

### Storage Format

```
delta_model.bin:
  Header:
    magic, version, n_layers, rank, base_dtype, delta_dtype
    per-layer: {offset, size} for each delta
  Base weights:
    7 matrices × [dim × dim] at original quantization (Q6_K/Q4_K_M)
  Per-layer deltas:
    layer 0: U_q[dim×r] V_q[r×dim] U_k[dim_k×r] V_k[r×dim] ... (7 pairs)
    layer 1: ...
    ...
    layer 79: ...
```

### Integration into ntransformer

Changes required:

| File | Change |
|------|--------|
| `src/memory/streamer.h` | Add `base_weights_` buffer, `delta_mode_` flag |
| `src/memory/streamer.cu` | Modified `prefetch_staging()` to load delta; new `reconstruct_layer()` |
| `src/cuda/reconstruct.cu` | **New**: GPU kernel for W = Base + U*V^T (rank-r outer product addition) |
| `src/model/transformer.cpp` | No changes — compute path unchanged |
| `tools/decompose_gguf.py` | **New**: Offline script to decompose GGUF → base + deltas |
| `CMakeLists.txt` | Add reconstruct.cu |

### Pipeline Timeline

```
Without delta (current):
  NVMe: [====== 669 MB read ======]    200 ms
  H2D:  [                          ][== H2D ==]  100 ms
  GPU:  [                                       ][compute]  5 ms
  Total: ~305 ms per layer

With delta:
  NVMe: [= 50 MB =]                     15 ms
  H2D:  [          ][= H2D delta =]     8 ms
  GPU:  [                          ][reconstruct 0.5ms][compute 5ms]
  Total: ~28 ms per layer → 10x faster
```

## Verification Plan

1. **Decomposition quality**: For each layer, compute `||W - (Base + U*V^T)|| / ||W||`.
   Target: < 1% relative error at rank 64. If not sufficient, increase rank or use
   per-cluster bases (e.g., base_early, base_mid, base_late).

2. **Output correctness**: Compare token-by-token output at temperature=0 between
   original model and delta-reconstructed model. Allow minor floating-point drift
   but verify that "What is the capital of France?" still produces "Paris".

3. **Perplexity benchmark**: Measure perplexity on a standard dataset (WikiText-2)
   for original vs. delta-reconstructed at various ranks (16, 32, 64, 128).

## Rank Selection Analysis

| Rank | Delta size/layer | Total NVMe (80 layers) | Reconstruction time | Expected quality |
|------|-----------------|----------------------|--------------------|-----------------|
| 16 | ~14 MB | 1.1 GB | ~0.1 ms | May degrade on edge cases |
| 32 | ~28 MB | 2.2 GB | ~0.2 ms | Good for most layers |
| **64** | **~50 MB** | **4.0 GB** | **~0.5 ms** | **>99% variance preserved** |
| 128 | ~100 MB | 8.0 GB | ~1.0 ms | Near-lossless |

Rank 64 is the sweet spot: 13x bandwidth reduction with sub-millisecond reconstruction
and >99% weight variance preserved.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low rank insufficient for some layers | Quality loss | Per-layer adaptive rank (higher for sensitive layers like 0, 1, 78, 79) |
| Reconstruction adds latency | Offsets bandwidth savings | 0.5 ms at rank 64 vs. 200 ms saved — 400x margin |
| Base doesn't represent all layers well | Poor deltas | Use k-means clustering → 2-3 bases for layer clusters |
| Quantized weights don't decompose cleanly | SVD on dequantized floats, re-quantize delta | Store deltas in F16 (small anyway) |
| Increased complexity | Maintenance burden | Isolate in reconstruct.cu, toggle via `--delta-streaming` flag |

## References

- DOCS: Distribution of Cosine Similarity (ICLR 2025) — https://arxiv.org/abs/2501.16650
- DeltaLLM (Jan 2025) — https://arxiv.org/abs/2501.18596
- Basis Sharing (ICLR 2025) — https://arxiv.org/abs/2410.03765
- Relaxed Recursive Transformers (Oct 2024) — https://arxiv.org/abs/2410.20672
- MASA: Share Your Attention (Aug 2025) — https://arxiv.org/abs/2508.04581
- SVD-LLM (ICLR 2025) — https://arxiv.org/abs/2403.07378
- FlashSVD (Aug 2025) — https://arxiv.org/abs/2508.01506
