# R3: Predictive Layer Skip Before Loading

## Summary

Predict which layers can be skipped **before initiating their weight transfer**,
saving ~200 ms of NVMe I/O per skipped layer instead of just ~5 ms of compute.
Uses a lightweight MLP predictor trained on calibration data.

## Motivation

ntransformer already implements layer skip (cosine similarity threshold = 0.98),
which skips 20 of 80 layers on Llama 70B. But the current implementation:

1. Loads the layer weights from NVMe/RAM (~200 ms)
2. Computes the layer (~5 ms)
3. Measures cosine similarity with input
4. Decides to skip (discards the computation)

Step 1 dominates. If we could predict the skip decision **before step 1**, we save
200 ms per skipped layer. For 20 skipped layers across tiers B+C, that's **4 seconds
per token** — a transformative speedup.

## Prior Art

| Paper | Approach | Difference from our proposal |
|-------|----------|------------------------------|
| LayerSkip (Meta, ACL 2024) | Train with layer dropout for early exit | Requires training; assumes weights resident |
| FiRST (Oct 2024) | Prefill-time routing decision for decode layers | Closest work; but assumes weights in memory |
| DASH (May 2025) | Markov Decision Policies for input-aware skip | Dynamic but still loads weights first |
| AdaSkip (Jan 2025) | On-the-fly similarity for sublayer skip | Real-time similarity, but post-load |
| Transformer Layers as Painters (AAAI 2025) | Middle layers are interchangeable | Empirical foundation; no prediction mechanism |
| Learning to Skip the Middle (Jun 2025) | Learned gating for symmetric spans | Training-based; assumes weights resident |

**Gap**: Every existing layer-skip method assumes weights are already in GPU memory.
In a streaming pipeline where loading is the bottleneck, predicting skip before I/O
is fundamentally different and has **zero prior work**. This gap exists because
ntransformer's streaming architecture is itself novel — no other system loads layers
one-by-one from NVMe.

## Technical Design

### Predictor Architecture

```
Input:  hidden_state h ∈ R^8192 (after last computed layer)
Output: skip_prob[i] ∈ [0,1] for each remaining layer

Architecture: 2-layer MLP
  h → Linear(8192, 256) → ReLU → Linear(256, n_remaining_layers) → Sigmoid

Parameters: 8192×256 + 256 + 256×80 + 80 ≈ 2.1M params ≈ 8.4 MB at F32
Inference:  ~0.01 ms (negligible vs 200 ms I/O)
```

The predictor is tiny relative to the model (8 MB vs 56 GB) and can be kept
permanently in VRAM.

### Training Data Collection

```python
# Calibration script (offline, one-time):
# Run 100-1000 diverse prompts through the model at temp=0
# For each token, record:
#   - hidden_state after each layer
#   - cosine_similarity between input and output of each layer
#   - skip decision (cos_sim > 0.98)

calibration_data = []
for prompt in calibration_prompts:
    for token in decode(prompt):
        for layer_idx in range(80):
            h_in = hidden_states[layer_idx]      # input to layer
            h_out = hidden_states[layer_idx + 1]  # output of layer
            cos_sim = cosine_similarity(h_in, h_out)
            skip = (cos_sim > 0.98)
            calibration_data.append({
                'hidden_state': h_in,
                'layer_idx': layer_idx,
                'skip': skip,
                'cos_sim': cos_sim
            })

# Train predictor on calibration data
# Input: hidden_state at any point in the network
# Target: binary skip decisions for all subsequent layers
```

### Inference Integration

```cpp
// In forward_tiered(), before the tier B/C loop:

// Step 1: Run tier A layers (VRAM resident, no I/O)
for (int i = 0; i < n_vram; i++) {
    compute_layer(i);
}

// Step 2: Predict skip mask for remaining layers
// hidden_state is now the output of the last tier A layer
float* skip_probs = predict_skip(hidden_state, predictor_weights);
bool skip_mask[n_layers];
for (int i = n_vram; i < n_layers; i++) {
    skip_mask[i] = (skip_probs[i] > 0.99);  // high confidence threshold
}

// Step 3: Stream only non-skipped layers
for (int i = n_vram; i < n_layers; i++) {
    if (skip_mask[i]) {
        // NO prefetch, NO H2D, NO compute — total skip
        stats.layers_skipped++;
        continue;
    }
    prefetch_staging(i, slot);    // 200 ms saved per skip
    begin_h2d(i, slot);
    compute_layer(i, slot);
    slot = 1 - slot;
}
```

### Confidence-Based Skip with Fallback

To prevent quality degradation from predictor errors:

```cpp
// Conservative: only skip when predictor is very confident
float confidence_threshold = 0.99;  // 99% sure it should be skipped

// Additional safety: limit maximum consecutive skips
int max_consecutive_skip = 5;
int consecutive_skips = 0;

for (int i = n_vram; i < n_layers; i++) {
    bool should_skip = (skip_probs[i] > confidence_threshold)
                    && (consecutive_skips < max_consecutive_skip);

    if (should_skip) {
        consecutive_skips++;
        continue;
    }

    consecutive_skips = 0;
    // Load and compute normally...

    // Optional: verify prediction with actual cosine similarity
    // If predictor was wrong (would NOT have skipped), retrain
    float actual_cos = cosine_similarity(h_in, h_out);
    if (actual_cos < 0.95 && skip_probs[i] > 0.5) {
        log_prediction_error(i, skip_probs[i], actual_cos);
    }
}
```

### Predictor Update Strategies

**Strategy A: Static predictor (simplest)**
- Train once on calibration data, use forever
- Works well if skip patterns are consistent across prompts
- Risk: may not generalize to out-of-distribution inputs

**Strategy B: Prompt-adaptive predictor**
- During prefill (all layers computed anyway), collect skip statistics
- Adjust confidence threshold per-layer based on prefill observations
- Better generalization, zero training cost at inference time

**Strategy C: Online learning**
- Periodically verify predictions against actual cosine similarity
- Update predictor weights via SGD on mispredictions
- Best quality but highest complexity

### Integration into ntransformer

| File | Change |
|------|--------|
| `src/inference/skip_predictor.h` | **New**: Predictor class definition |
| `src/inference/skip_predictor.cu` | **New**: MLP inference kernel + calibration data loader |
| `src/model/transformer.cpp` | Add predictor call before tier B/C loop |
| `tools/calibrate_skip.py` | **New**: Offline calibration data collection + predictor training |
| `CMakeLists.txt` | Add skip_predictor.cu |

### Pipeline Timeline

```
Without predictive skip (current layer skip):
  Layer 30: NVMe [=== 200ms ===] H2D [100ms] Compute [5ms] → cos_sim > 0.98 → SKIPPED
  Layer 31: NVMe [=== 200ms ===] H2D [100ms] Compute [5ms] → computed
  ...
  Wasted: 200ms + 100ms + 5ms per skipped layer

With predictive skip:
  Predictor: [0.01ms] → skip_mask = {30: skip, 31: compute, ...}
  Layer 30: SKIPPED (0 ms — no I/O, no compute)
  Layer 31: NVMe [=== 200ms ===] H2D [100ms] Compute [5ms]
  ...
  Saved: 305 ms per skipped tier C layer, 105 ms per skipped tier B layer
```

For 20 skipped layers (10 tier B + 10 tier C):
- Tier B savings: 10 × 105 ms = 1.05 s
- Tier C savings: 10 × 305 ms = 3.05 s
- **Total: ~4.1 seconds saved per token**

## Verification Plan

1. **Predictor accuracy**: On held-out calibration data, measure precision/recall
   of skip predictions. Target: >99% precision (false skips are costly), >80% recall
   (missing some skips is acceptable — falls back to current behavior).

2. **Output quality**: Compare outputs at temp=0:
   - Baseline (no skip)
   - Current layer skip (post-compute)
   - Predictive layer skip
   All three should produce identical tokens for well-calibrated predictor.

3. **Prediction latency**: Verify predictor MLP runs in <0.1 ms (negligible).

4. **Generalization**: Test on prompts not in calibration set. Measure false-skip rate.

## Skip Pattern Analysis (from existing data)

From our layer skip calibration on Llama 70B Q4_K_M:
- Layers 0-9: never skipped (early layers are unique)
- Layers 10-65: ~35% skippable (middle layers redundant)
- Layers 66-79: rarely skipped (final layers refine output)
- 20/80 total layers skipped at threshold 0.98

The skip pattern is highly structured — middle layers skip, edges don't.
This structure makes prediction easier (the predictor mainly needs to learn
"how middle-layer-like is the current hidden state?").

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Predictor wrong → layer incorrectly skipped | Quality degradation | High confidence threshold (0.99); max consecutive skip limit |
| Skip patterns vary with input | Predictor doesn't generalize | Strategy B: prompt-adaptive thresholds from prefill data |
| Predictor too conservative → few skips | Minimal speedup | Acceptable — falls back to current behavior, no regression |
| Calibration data not representative | Biased predictions | Diverse calibration set: code, math, prose, multilingual |
| First decode token has no skip history | Can't predict well | Use prefill hidden states as initial context for predictor |

## References

- LayerSkip (Meta, ACL 2024) — https://arxiv.org/abs/2404.16710
- FiRST (Oct 2024) — https://arxiv.org/abs/2410.12513
- DASH (May 2025) — https://arxiv.org/abs/2505.17420
- AdaSkip (Jan 2025) — https://arxiv.org/abs/2501.02336
- Transformer Layers as Painters (AAAI 2025) — https://arxiv.org/abs/2407.09298
- Learning to Skip the Middle (Jun 2025) — https://arxiv.org/abs/2506.21103
- Deja Vu: Contextual Sparsity (ICML 2023) — https://arxiv.org/abs/2310.17157
- DOCS (ICLR 2025) — https://arxiv.org/abs/2501.16650
