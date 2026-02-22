# Literature Survey: Bandwidth-Constrained LLM Inference

Comprehensive review of ~60 papers informing ntransformer's research directions.
Organized by topic area. Last updated: 2026-02-22.

---

## 1. Layer Similarity and Redundancy in LLMs

The empirical foundation for R1 (delta streaming) and R3 (predictive skip).

### Key findings

- **Adjacent transformer layers have high weight cosine similarity**, forming clusters
  in the middle and final layers (DOCS, ICLR 2025).
- **Middle layers are surprisingly interchangeable** — they can be skipped, reordered,
  or parallelized with graceful quality degradation (Transformer Layers as Painters, AAAI 2025).
- **A large portion of attention layers show excessive similarity** and can be pruned.
  Llama-2-70B achieves 48.4% speedup pruning half the attention layers
  (What Matters in Transformers, Jun 2024).
- **Cosine similarity between layer input/output** reliably identifies prunable layers.
  High similarity = simple mapping = replaceable by lightweight network
  (Streamlining Redundant Layers, ICLR 2025).
- **Entropy is a complementary metric** to cosine similarity — it measures information
  richness rather than directional similarity (Entropy-Based Block Pruning, Apr 2025).

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| DOCS: Distribution of Cosine Similarity | ICLR 2025 | Formal metric for weight similarity; layers form two clusters |
| Transformer Layers as Painters | AAAI 2025 | Middle layers interchangeable; graceful degradation |
| What Matters in Transformers? Not All Attention | Jun 2024 | 48.4% speedup pruning redundant attention layers |
| Streamlining Redundant Layers | ICLR 2025 | Cosine similarity identifies prunable layers |
| Entropy-Based Block Pruning | Apr 2025 | Entropy > cosine similarity for importance ranking |
| DLO: Dynamic Layer Operations | Jul 2024 | Layerwise feature similarity for routing policies |
| Sliding-Window Merging | Feb 2025 | Consecutive layers show patch-like redundancy |
| DenseFormer | Feb 2024 | Depth-weighted averaging reveals cross-layer reuse |
| Neurons in LLMs: Dead, N-gram, Positional | Sep 2023 | >70% dead neurons in some layers of OPT-66B |

**Links:**
- https://arxiv.org/abs/2501.16650 (DOCS)
- https://arxiv.org/abs/2407.09298 (Painters)
- https://arxiv.org/abs/2406.15786 (What Matters)
- https://arxiv.org/abs/2403.19135 (Streamlining)
- https://arxiv.org/abs/2504.03794 (Entropy)
- https://arxiv.org/abs/2407.11030 (DLO)
- https://arxiv.org/abs/2502.19159 (Sliding-Window)
- https://arxiv.org/abs/2402.02622 (DenseFormer)
- https://arxiv.org/abs/2309.04827 (Dead Neurons)

---

## 2. Layer Skip and Early Exit

Directly relevant to R3 (predictive skip). All assume weights resident in memory.

### Key findings

- **LayerSkip** trains with layer dropout so earlier layers become sufficient for
  direct prediction. Combined with self-speculative decoding (draft from early layers,
  verify with full model).
- **FiRST** is the closest to predictive skip — routes at prefill time, then skips
  during decode. But assumes weights in GPU memory.
- **DASH** uses Markov Decision Policies with a compensation mechanism for dynamic
  layer skipping. Input-aware but post-load.
- **Learned gating** for symmetric spans of central blocks can skip 10-40% of layers
  with minimal quality loss.

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| LayerSkip | Meta, ACL 2024 | Layer dropout training + self-speculative decoding |
| FiRST | Oct 2024 | Prefill-time routing for decode layer skip |
| DASH | May 2025 | Input-aware MDP for dynamic layer skip |
| AdaSkip | Jan 2025 | Adaptive sublayer skip for long-context |
| Learning to Skip the Middle | Jun 2025 | Learned gating for symmetric span skip |

**Links:**
- https://arxiv.org/abs/2404.16710 (LayerSkip)
- https://arxiv.org/abs/2410.12513 (FiRST)
- https://arxiv.org/abs/2505.17420 (DASH)
- https://arxiv.org/abs/2501.02336 (AdaSkip)
- https://arxiv.org/abs/2506.21103 (Skip Middle)

---

## 3. Activation Sparsity in Transformer FFN Layers

Foundation for R2 (sparse FFN loading).

### Key findings

- **The Lazy Neuron Phenomenon**: Activation maps in MLP layers are naturally sparse.
  With ReLU, sparsity is exact (~90%); with SwiGLU, soft-sparse (~50%) (Li et al., 2022).
- **Contextual sparsity is 7x more efficient** than static sparsity — the active neuron
  set varies per input but is predictable (Deja Vu, ICML 2023).
- **ReLU replacement** recovers explicit sparsity with minimal accuracy loss after
  brief fine-tuning (ReLU Strikes Back, ICLR 2024).
- **dReLU** pushes sparsity to 90%+ in dense models, 97% in MoE (TurboSparse, Jun 2024).
- **R-Sparse** exploits rank structure of input channels via SVD decomposition for
  50% sparsity without any training (ICLR 2025).
- **SCAP** applies mode-centering post-training to shift activations toward zero,
  increasing sparsity across all three SwiGLU matrices (NeurIPS 2024).
- **CoreInfer** discovers that active neuron sets are stable within a sentence,
  enabling sentence-level prediction with zero overhead (Oct 2024).
- **Batch size matters**: sparsity benefits diminish with larger batches as the union
  of active sets across inputs approaches the full set (Polar Sparsity, May 2025).
  Batch-1 decode (our use case) is the best case for sparsity exploitation.

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| The Lazy Neuron Phenomenon | Oct 2022 | Foundational observation of MLP activation sparsity |
| Deja Vu | ICML 2023 Oral | Contextual sparsity + MLP predictors; 2x speedup on OPT-175B |
| ReLU Strikes Back | ICLR 2024 | ReLU replacement for explicit sparsity |
| ReLU^2 Wins | Feb 2024 | ReLU^2 best for sparsity+prediction+hardware |
| TurboSparse | Jun 2024 | dReLU for 90%+ sparsity; 2-5x speedup |
| CATS | Stanford, Jul 2024 | Contextual thresholding; 50% sparsity, no fine-tuning |
| Spark Transformer | Google, Jun 2025 | 8% FFN activation; 2.5x FLOP reduction |
| R-Sparse | ICLR 2025 | Training-free rank-aware sparsity via SVD |
| TEAL | Aug 2024 | Magnitude-based activation sparsity, training-free |
| SCAP | NeurIPS 2024 | Post-training mode-centering for higher sparsity |
| CoreInfer | Oct 2024 | Sentence-level core neurons; 10.3x speedup |
| ShadowLLM | Jun 2024 | Improved predictor design for Deja Vu |
| WINA | May 2025 | Weight-informed neuron activation prediction |
| CHESS | Sep 2024 | Per-channel learned sparsity thresholds |
| FFSplit | Jan 2024 | Heavy-hitter vs sparse neuron partition |
| SparseInfer | DATE 2025 | Sign-bit prediction, training-free |
| Polar Sparsity | May 2025 | Sparsity degrades at larger batch sizes |

**Links:**
- https://arxiv.org/abs/2210.06313 (Lazy Neuron)
- https://arxiv.org/abs/2310.17157 (Deja Vu)
- https://arxiv.org/abs/2310.04564 (ReLU Strikes Back)
- https://arxiv.org/abs/2402.03804 (ReLU^2)
- https://arxiv.org/abs/2406.05955 (TurboSparse)
- https://arxiv.org/abs/2404.08763 (CATS)
- https://arxiv.org/abs/2506.06644 (Spark)
- https://arxiv.org/abs/2504.19449 (R-Sparse)
- https://arxiv.org/abs/2408.14690 (TEAL)
- https://arxiv.org/abs/2412.07174 (SCAP)
- https://arxiv.org/abs/2410.18311 (CoreInfer)
- https://arxiv.org/abs/2406.16635 (ShadowLLM)
- https://arxiv.org/abs/2505.19427 (WINA)
- https://arxiv.org/abs/2409.01366 (CHESS)
- https://arxiv.org/abs/2401.04044 (FFSplit)
- https://arxiv.org/abs/2411.12692 (SparseInfer)
- https://arxiv.org/abs/2505.14884 (Polar Sparsity)

---

## 4. Cross-Layer Weight Sharing and Delta Compression

Directly relevant to R1 (delta streaming).

### Key findings

- **DeltaLLM** shares weights between adjacent blocks + low-rank deltas, achieving
  12% parameter reduction with 90% performance. Requires distillation training.
- **Basis Sharing** shares SVD singular vectors across layers, keeping only unique
  coefficients per layer. Outperforms standard SVD at high compression ratios.
- **MASA** uses dictionary learning for shared attention matrix "atoms" across layers,
  reducing attention parameters by 66.7%.
- **Relaxed Recursive Transformers** convert existing LLMs to recursive (shared base
  block repeated in a loop) + per-layer LoRA. Requires fine-tuning.

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| DeltaLLM | Jan 2025 | Adjacent block weight sharing + low-rank deltas |
| Basis Sharing | ICLR 2025 | Shared SVD bases across layers |
| MASA: Share Your Attention | Aug 2025 | Dictionary atoms for cross-layer attention |
| Relaxed Recursive Transformers | Oct 2024 | Convert to recursive + depth-wise LoRA |
| Delta-CoMe | Jun 2024 | Mixed-precision delta between fine-tuned and base model |
| ImPart | Apr 2025 | Importance-aware delta sparsification |
| Per-Axis Weight Deltas | Dec 2024 | 1-bit binary sign mask delta representation |
| MiniViT | Apr 2022 | Weight multiplexing across ViT layers |
| MLKV | Jun 2024 | Multi-layer KV head sharing |

**Links:**
- https://arxiv.org/abs/2501.18596 (DeltaLLM)
- https://arxiv.org/abs/2410.03765 (Basis Sharing)
- https://arxiv.org/abs/2508.04581 (MASA)
- https://arxiv.org/abs/2410.20672 (Recursive Transformers)
- https://arxiv.org/abs/2406.08903 (Delta-CoMe)
- https://arxiv.org/abs/2504.13237 (ImPart)
- https://arxiv.org/abs/2512.19720 (Per-Axis Deltas)
- https://arxiv.org/abs/2204.07154 (MiniViT)
- https://arxiv.org/abs/2406.09297 (MLKV)

---

## 5. Low-Rank Approximation for Inference

Foundation for R1's SVD-based decomposition.

### Key findings

- **FlashSVD** is the first streaming inference framework for SVD-compressed LLMs.
  Cuts peak activation memory by 70.2%. But focuses on activation memory, not transfer.
- **SVD-LLM** provides truncation-aware SVD compression minimizing discarded value loss.
- **Generalized Fisher-Weighted SVD** incorporates off-diagonal Fisher information
  for better rank selection.

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| FlashSVD | Aug 2025 | Streaming inference for SVD-compressed LLMs |
| SVD-LLM | ICLR 2025 | Truncation-aware SVD compression |
| Generalized Fisher-Weighted SVD | May 2025 | Fisher-informed rank selection |
| FWSVD | Jul 2022 | Fisher-Weighted SVD for task-aware compression |
| PALU | ICLR 2025 | Low-rank KV cache projection |

**Links:**
- https://arxiv.org/abs/2508.01506 (FlashSVD)
- https://arxiv.org/abs/2403.07378 (SVD-LLM)
- https://arxiv.org/abs/2505.17974 (GFW-SVD)
- https://arxiv.org/abs/2207.00112 (FWSVD)

---

## 6. Bandwidth-Aware and Offloaded Inference Systems

Systems-level context for all three research directions.

### Key findings

- **LLM in a Flash** (Apple) is the closest system-level work: streams weights from
  flash to DRAM using sparsity-aware loading. 20-25x speedup. But uses Apple's flash
  controller, not PCIe NVMe with GPU-initiated I/O.
- **Endor** provides hardware-friendly sparse format for offloaded inference with
  bitmap-based compression. 2.25-2.37x speedup with SSD-to-GPU transfer.
- **PowerInfer** is a complete system exploiting hot/cold neuron partition with
  GPU-CPU hybrid computation. 11.69x faster than llama.cpp on OPT-175B.
- **PIPO** achieves 3.1x throughput improvement on consumer GPUs (RTX 3060 6GB)
  by increasing GPU utilization from 40% to 90% through pipelined offloading.
- **FlexInfer** dynamically selects execution policy based on hardware config.
- **No existing system uses GPU-initiated NVMe I/O** (our Tier 1 MMIO approach).
  All use CPU-mediated DMA. The combination of GPU-autonomous storage access with
  bandwidth-reduction techniques is entirely novel.

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| LLM in a Flash | Apple, ACL 2024 | Flash memory + sparsity-aware loading; 20-25x speedup |
| PowerInfer | SOSP 2024 | Hot/cold neuron partition; GPU-CPU hybrid; 11.69x speedup |
| PowerInfer-2 | Jun 2024 | Extended to smartphones; neuron-cluster pipelining |
| Endor | Jun 2024 | Hardware-friendly sparse format for SSD offloading |
| PIPO | Apr 2025 | Pipelined offloading on consumer GPUs |
| FlexInfer | MLSys 2025 | Dynamic execution policy selection |
| DeepSpeed ZeRO-Inference | Sep 2022 | Layer-by-layer NVMe→CPU→GPU streaming |
| InstInfer | Sep 2024 | Attention offload to Computational Storage Drives |
| CMoE | Feb 2025 | Post-training dense-to-MoE conversion; 1.5x latency reduction |
| Q-Sparse | Microsoft, Jul 2024 | Full sparse activation via top-K + STE |

**Links:**
- https://arxiv.org/abs/2312.11514 (LLM in a Flash)
- https://arxiv.org/abs/2312.12456 (PowerInfer)
- https://arxiv.org/abs/2406.06282 (PowerInfer-2)
- https://arxiv.org/abs/2406.11674 (Endor)
- https://arxiv.org/abs/2504.03664 (PIPO)
- https://arxiv.org/abs/2503.03777 (FlexInfer)
- https://arxiv.org/abs/2409.04992 (InstInfer)
- https://arxiv.org/abs/2502.04416 (CMoE)
- https://arxiv.org/abs/2407.10969 (Q-Sparse)

---

## 7. Frequency-Domain and Progressive Compression

Additional techniques explored but not selected for primary research.

### Key findings

- **SpecQuant** uses spectral decomposition (DCT/FFT) for ultra-low-bit quantization
  by retaining low-frequency weight components.
- **SVD-Free Low-Rank via DCT** shows that predefined DCT bases approximate SVD well
  for weight matrices, avoiding the computational cost of SVD.
- **Progressive Mixed-Precision Decoding (PMPD)** varies precision across the generated
  sequence (high at start, low later). Different from our multi-resolution proposal.
- **MatFormer** trains nested FFN blocks for elastic inference (582M-850M from one model).
  Requires training from scratch.

### Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| SpecQuant | Nov 2025 | DCT/FFT for ultra-low-bit quantization |
| SVD-Free Low-Rank via DCT | May 2025 | DCT bases ≈ SVD for weight projection |
| PMPD | ICLR 2025 | Progressive precision across token sequence |
| MatFormer | NeurIPS 2024 | Nested FFN for elastic inference |
| QuEPT | Feb 2026 | Multi-bitwidth switching with one-shot calibration |
| Looped Transformers | Nov 2023 | Single block reused iteratively |
| SpiralFormer | Feb 2026 | Multi-resolution recursive transformer |

**Links:**
- https://arxiv.org/abs/2511.11663 (SpecQuant)
- https://arxiv.org/abs/2505.17967 (SVD-Free DCT)
- https://arxiv.org/abs/2410.13461 (PMPD)
- https://arxiv.org/abs/2310.07707 (MatFormer)
- https://arxiv.org/abs/2602.12609 (QuEPT)
- https://arxiv.org/abs/2311.12424 (Looped Transformers)
- https://arxiv.org/abs/2602.11698 (SpiralFormer)

---

## 8. Our Unique Position

ntransformer occupies a unique position in the design space:

| Dimension | Existing systems | ntransformer |
|-----------|-----------------|-------------|
| Weight loading | CPU-mediated DMA | **GPU-initiated NVMe MMIO** |
| Bottleneck | Compute or DRAM bandwidth | **PCIe transfer bandwidth** |
| Model location | DRAM (resident) | **VRAM + RAM + NVMe (3-tier streaming)** |
| Layer execution | All at once or early exit | **One-by-one streaming with skip** |
| Target hardware | Data center or Apple silicon | **Consumer GPU (RTX 3090)** |

This unique architecture enables optimizations that are impossible or irrelevant
in other systems:
- **R1 (Delta streaming)**: Only useful when transfer bandwidth (not compute) dominates.
  In DRAM-resident systems, delta reconstruction adds overhead with no I/O savings.
- **R2 (Sparse FFN loading)**: Only useful when loading individual weight columns from
  storage is feasible. In DRAM-resident systems, all weights are already accessible.
- **R3 (Predictive skip)**: Only useful when layer loading is expensive. In DRAM-resident
  systems, skipping saves ~5 ms compute; in our pipeline, it saves ~200 ms I/O.

The common thread: **our bottleneck is bandwidth, not compute**, and our architecture
enables bandwidth-aware optimizations that no existing system has explored.
