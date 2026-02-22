# Active Research: Bandwidth-Aware Streaming Inference

Novel optimizations for ntransformer's SLEP pipeline, targeting the core bottleneck:
**PCIe transfer bandwidth** (not compute) dominates token latency when streaming
70B+ models through 24GB VRAM.

## The Problem

| Layer path | Bandwidth | Time per layer (669 MB) | 80 layers | tok/s |
|------------|-----------|------------------------|-----------|-------|
| Tier A (VRAM resident) | ∞ (zero I/O) | 0 ms | — | — |
| Tier B (RAM → GPU H2D) | ~6.5 GB/s | ~100 ms | 5.3 s | 0.2 |
| Tier C (NVMe → staging → H2D) | ~3.3 GB/s | ~200 ms | 16 s | 0.06 |

Every millisecond saved in transfer = direct tok/s improvement. Compute per layer
is only ~5 ms (GeMV at batch=1). The GPU is idle 95%+ of the time, waiting for data.

## Three Research Directions

Each targets a different axis of the bandwidth bottleneck:

### [R1: Delta-Encoded Layer Streaming](R1_delta_streaming.md)
**Reduce bytes transferred per layer** by exploiting cross-layer weight similarity.
Store a shared base + low-rank per-layer deltas. Transfer 50 MB instead of 669 MB.
- Expected: **10-15x bandwidth reduction** per layer
- Complexity: Medium
- Key insight: Adjacent Llama layers share >95% of weight structure (DOCS, ICLR 2025)

### [R2: Sparse FFN Loading via Gate Prediction](R2_sparse_ffn.md)
**Skip loading inactive neuron weights** using SwiGLU's gate as a zero-cost predictor.
FFN is 2/3 of each layer; ~50% of neurons gate to near-zero per token.
- Expected: **~1.5x bandwidth reduction** per layer (33% less FFN data)
- Complexity: High (custom NVMe layout + sparse kernel)
- Key insight: Gate projection output predicts which FFN columns contribute to output

### [R3: Predictive Layer Skip Before Loading](R3_predictive_skip.md)
**Eliminate I/O for entire layers** by predicting skip decisions before initiating DMA.
Current layer skip saves compute (~5 ms); predictive skip saves I/O (~200 ms).
- Expected: **4+ seconds saved per token** (20 skipped layers × 200 ms each)
- Complexity: Low (lightweight MLP predictor)
- Key insight: No existing work predicts layer skip in a streaming context where weights aren't resident

## Combined Potential

The three techniques are orthogonal and composable:

```
Baseline 70B Q6_K (30 NVMe layers):
  30 layers × 669 MB × (1/3.3 GB/s) = ~6 seconds NVMe I/O per token

R3 (predictive skip, 10 of 30 NVMe layers skipped):
  20 layers × 669 MB = ~4 seconds

R1 (delta streaming on remaining 20 layers):
  20 layers × 50 MB  = ~0.3 seconds

R1 + R2 (delta + sparse FFN):
  20 layers × 35 MB  = ~0.2 seconds

R1 + R2 + R3 (all three):
  ~0.2 seconds NVMe I/O per token → potential 30x speedup over baseline
```

## Literature Survey

See [literature_survey.md](literature_survey.md) for the full research landscape
that informed these proposals, covering 60+ papers across:
- Layer similarity and redundancy in LLMs
- Activation sparsity and contextual sparsity
- Low-rank and delta weight compression
- Bandwidth-aware and offloaded inference systems

## Hardware Context

- GPU: RTX 3090 (24 GB VRAM, PCIe Gen3 x8 = ~6.5 GB/s H2D)
- RAM: 48 GB DDR4 (pinned staging for tier B)
- NVMe: WD SN740 (Gen3 x4 on B450 = 3.35 GB/s via gpu-nvme-direct)
- Model: Llama 3.1 70B (80 layers, Q6_K = 669 MB/layer, Q4_K_M = 495 MB/layer)
- Current best: 0.5 tok/s (Q4_K_M, 36 VRAM + 44 RAM, 20 layers skipped)
