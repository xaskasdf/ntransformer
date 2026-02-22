# R1: Delta-Encoded Layer Streaming

**Status: CLOSED — Negative result (2026-02-22)**

## Summary

Hypothesis: reduce per-layer transfer from 669 MB to ~20 MB by exploiting cross-layer
weight similarity. Store weights as a shared base + per-layer low-rank SVD deltas.

**Result**: Transformer layer weights in Llama 8B/70B are **uncorrelated matrices**
(cosine similarity ≈ 0.000). Delta encoding is fundamentally impossible for these
architectures. The infrastructure works mechanically (33× less H2D, 6× faster transfer)
but produces garbage output (50-93% reconstruction error).

---

## Why This Does Not Work

### The core assumption was wrong

The proposal assumed that adjacent Llama layers share >95% weight structure, based on
the DOCS paper (ICLR 2025) which reports high cosine similarity between layers.

**What DOCS actually measured**: cosine similarity of **hidden state activations** (the
output of each layer), NOT the weight matrices. Activations are similar because each
layer applies a small residual update. Weights are not.

### Measured weight similarity (Llama 70B Q6_K)

#### Adjacent layer cosine similarity ≈ 0

```
=== attn_q.weight: adjacent layers (every 5) ===
blk.0→blk.5:   cos_sim = -0.000107
blk.5→blk.10:  cos_sim = -0.000778
blk.10→blk.15: cos_sim = -0.000157
blk.15→blk.20: cos_sim =  0.000426
blk.20→blk.25: cos_sim = -0.000108
blk.35→blk.40: cos_sim =  0.000940
blk.60→blk.65: cos_sim = -0.001318
blk.70→blk.75: cos_sim =  0.000594

=== ffn_gate.weight ===
blk.0→blk.5:   cos_sim = -0.000008
blk.10→blk.15: cos_sim =  0.002786
blk.30→blk.35: cos_sim =  0.008974
blk.50→blk.55: cos_sim =  0.004759
blk.70→blk.75: cos_sim =  0.002793
```

Every pair has `cos_sim ≈ 0`. The weight matrices are **essentially random relative to
each other** — no more similar than two independent random matrices of the same
dimensions.

#### Residual norms > weight norms

```
=== attn_q.weight: ||R|| / ||W|| (residual = W - mean) ===
layer 0:  0.8338  (outlier — norm 816 vs ~110 for others)
layer 1:  1.4142
layer 39: 1.4312
layer 40: 1.4852
layer 78: 1.4648
layer 79: 1.6121

=== ffn_gate.weight ===
layer 0:  0.8332
layer 1:  1.3796
layer 39: 1.3179
layer 78: 1.2317

=== ffn_down.weight ===
layer 0:  0.9639
layer 1:  0.8864
layer 39: 0.9187
layer 78: 0.9291
```

`||R||/||W|| > 1.0` means the residual (difference from mean) is **larger** than the
weight itself. The mean is so far from any individual layer that subtracting it makes
things worse, not better.

#### Layer 0 is a massive outlier

```
attn_q norms:  blk.0 = 816.18,  blk.1 = 118.01,  blk.39 = 110.64
ffn_gate norms: blk.0 = 1001.34, blk.1 = 131.23,  blk.39 = 138.30
```

Layer 0 has 6-8× larger weight norms than all other layers. This single outlier
dominates the mean, pulling it away from every other layer.

#### SVD captures almost nothing

With rank-64 SVD on the residual:
```
attn_q:      SVD error 8-60%, total reconstruction error 8-87%
ffn_gate:    SVD error 11-71%, total reconstruction error 9-93%
ffn_down:    SVD error 97%, total reconstruction error 86-93%
```

The residuals are full-rank — they have no low-rank structure to exploit.

### Why this is fundamental, not fixable

The following variants would NOT fix the problem:

| Variant | Why it fails |
|---------|-------------|
| **Higher rank** (128, 256, 512) | Deltas grow proportionally; at rank ~4096 you're storing the full weight again |
| **Per-pair deltas** (W_i - W_{i-1}) | Adjacent layers have `cos_sim ≈ 0` — differences are full-rank |
| **K-means clustering** (2-3 bases) | Every layer is equidistant from every other — no clusters exist |
| **PCA across layers** | 80 layers in 67M-dimensional space — no meaningful principal components |
| **Different base** (median, weighted) | All bases are equally far from all layers |
| **Quantized deltas** (Q4/Q8) | Doesn't help when the delta IS the full weight |

The root cause is that transformer layer weights are initialized independently and
trained to perform different functions. They converge to solutions that produce
similar *activations* through the residual stream, but the weight matrices that produce
those activations are completely different linear maps.

### Analogy

Two rotation matrices can map input vectors to similar output vectors while being
completely different matrices. Layer 5's `attn_q` and layer 6's `attn_q` both produce
useful query vectors, but through entirely different linear transformations.

---

## What Was Built and Tested

### Implementation (fully functional)

| Component | Status |
|-----------|--------|
| `tools/decompose_gguf.py` — offline GGUF → NTD decomposition | Working |
| `launch_gemv_add` — accumulate GEMV kernel (y += W*x, F16) | Working |
| `streamer.cu` — `init_delta()`, delta H2D for tier B/C | Working |
| `attention.cpp` / `ffn.cpp` — `delta_gemv()` forward path | Working |
| `--delta-model` CLI flag | Working |
| NTD file format (header + Q6_K base + F16 U/V per layer) | Working |

### Test results

| Model | Config | Output | Speed | Notes |
|-------|--------|--------|-------|-------|
| 8B Q8_0 | 16V + 16 delta | Garbage | 27.6 tok/s | `||R||/||W|| = 0.72`, total error 59% |
| 70B Q6_K | 20V + 30R + 30 delta | Garbage | 1.2 tok/s | `||R||/||W|| > 1.0`, total error 50-93% |
| 70B baseline | 20V + 30R + 30 mmap | "Paris" (correct) | 0.2 tok/s | Reference |

The 6× speed improvement (1.2 vs 0.2 tok/s) confirms the pipeline works mechanically:
19.8 MB/layer delta vs 669 MB/layer full = 33× less H2D bandwidth.

### Q6_K quantization verified correct

Round-trip test (F32 → Q6_K → F32): 2% relative error. The quantization is not the
problem — the SVD decomposition is.

---

## Original Proposal

*(Preserved below for reference — the technical design is sound, only the assumption
about weight similarity was wrong.)*

### Motivation

Llama 70B has 80 transformer layers. ~~Middle layers (roughly 10-65) are remarkably
similar to each other~~ — cosine similarity of **hidden states** between adjacent layers
exceeds 0.98 (measured in our layer skip calibration). ~~This similarity extends to
the weight matrices themselves~~ — **this was the incorrect assumption**.

### Prior Art

| Paper | What it does | Relevance to our finding |
|-------|-------------|--------------------------|
| DeltaLLM (Jan 2025) | Shares weights between adjacent blocks + low-rank deltas | Requires **distillation training** to force weight sharing — confirms pre-trained weights don't share naturally |
| Basis Sharing (ICLR 2025) | Shares SVD singular vectors across layers | Works on **singular vectors** (structural), not raw weights |
| Relaxed Recursive Transformers (Oct 2024) | Converts LLMs to recursive with LoRA per-layer | Requires **architecture change** — acknowledges standard weights aren't shared |
| DOCS (ICLR 2025) | Measures cosine similarity between layers | Measures **activations**, not weights. Our experiment confirms weights don't correlate |

All prior work that achieves weight sharing does so through **training modifications**
(distillation, recursive architecture, fine-tuning). No post-hoc decomposition of
standard pre-trained weights can create sharing that doesn't exist.

### Technical Design

For each weight matrix W_i in layer i (7 matrices: attn_q/k/v/o, ffn_gate/up/down):

```
1. Compute shared base:  B = mean(W_0, W_1, ..., W_79)
2. Compute residual:     R_i = W_i - B
3. SVD of residual:      R_i = U_i * S_i * V_i^T
4. Truncate to rank r:   R_i ≈ U_i[:,:r] * diag(S_i[:r]) * V_i[:r,:]^T
5. Store:                delta_i = (U_i * sqrt(S_i), sqrt(S_i) * V_i^T)
```

Runtime: `y = Base*x + U*(V^T*x)` — three GeMVs instead of materializing the full matrix.

### NTD File Format

```
Header (64 bytes):
  magic[4] = "NTD1", rank, n_layers, hidden, intermediate,
  n_heads, n_kv_heads, head_dim, base_dtype, delta_dtype,
  base_offset, delta_offset

Base section: 7 Q6_K weight matrices (669.4 MB for 70B)
Delta section: n_layers × 14 F16 tensors (19.8 MB/layer for 70B rank-64)
```

### Rank Selection Analysis

| Rank | Delta/layer | Bandwidth reduction | Quality (measured) |
|------|------------|--------------------|--------------------|
| 64 | 19.8 MB | 33× | Unusable (50-93% error) |
| 128 | 39.6 MB | 17× | Still unusable (residuals are full-rank) |
| 4096+ | ~669 MB | 1× | Equivalent to original (no savings) |

No rank provides both meaningful compression AND acceptable quality.

## References

- DOCS: Distribution of Cosine Similarity (ICLR 2025) — https://arxiv.org/abs/2501.16650
- DeltaLLM (Jan 2025) — https://arxiv.org/abs/2501.18596
- Basis Sharing (ICLR 2025) — https://arxiv.org/abs/2410.03765
- Relaxed Recursive Transformers (Oct 2024) — https://arxiv.org/abs/2410.20672
- MASA: Share Your Attention (Aug 2025) — https://arxiv.org/abs/2508.04581
- SVD-LLM (ICLR 2025) — https://arxiv.org/abs/2403.07378
- FlashSVD (Aug 2025) — https://arxiv.org/abs/2508.01506
