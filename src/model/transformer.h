#pragma once

#include "../core/tensor.h"
#include "config.h"
#include "loader.h"
#include "norm.h"
#include "attention.h"
#include "ffn.h"
#include "../memory/streamer.h"
#include <vector>
#include <memory>

namespace nt {

// ============================================================
// Transformer Layer
// ============================================================
struct TransformerLayer {
    RMSNorm attn_norm;
    Attention attention;
    RMSNorm ffn_norm;
    FFN ffn;
};

// ============================================================
// Complete Transformer Model
// Phase 1: All layers resident in VRAM
// Phase 2: SLEP streaming (layers loaded on demand)
// ============================================================

class Transformer {
public:
    Transformer() = default;
    ~Transformer();

    // Load model from GGUF
    // streaming: if true, use SLEP double-buffer streaming (Phase 2)
    bool load(const std::string& gguf_path, int max_context = 4096, bool streaming = false);

    // Forward pass
    // tokens: [seq_len] token IDs (on CPU)
    // start_pos: position in KV cache (0 for prefill, >0 for decode)
    // Returns logits: [vocab_size] on GPU
    float* forward(const int* tokens, int seq_len, int start_pos);

    // Forward pass returning logits at ALL positions (for speculative verification)
    // Returns pointer to GPU buffer of [seq_len * vocab_size] floats
    float* forward_verify(const int* tokens, int seq_len, int start_pos);

    const ModelConfig& config() const { return config_; }
    const GGUFVocab& vocab() const { return loader_.vocab(); }

    // For engine to access raw logits
    float* logits_ptr() { return logits_; }

    bool is_streaming() const { return streaming_mode_; }

    // Early exit: skip remaining layers when hidden state converges
    void set_early_exit(float threshold);

    // Layer skipping: skip middle layers where hidden state barely changes
    // First token calibrates which layers to skip; subsequent tokens skip them
    void set_layer_skip(float threshold);

private:
    ModelConfig config_;
    GGUFLoader loader_;

    // Model components
    std::vector<TransformerLayer> layers_;
    RMSNorm output_norm_;
    Tensor token_embedding_;    // [vocab, hidden] on CPU (mmap'd)
    Tensor output_weight_;      // [vocab, hidden] on GPU (may share with embedding)

    // KV cache
    Tensor k_cache_;  // [n_layers, max_seq, n_kv_heads, head_dim]
    Tensor v_cache_;  // [n_layers, max_seq, n_kv_heads, head_dim]

    // Workspace buffers
    Tensor workspace_;
    Tensor hidden_buf_;       // [max_seq, hidden_size]
    Tensor residual_buf_;     // [max_seq, hidden_size]
    Tensor logits_buf_;       // [vocab_size]

    float* logits_ = nullptr;

    // Positions buffer on GPU
    Tensor positions_gpu_;

    // === Phase 2: SLEP streaming ===
    bool streaming_mode_ = false;
    LayerStreamer streamer_;
    void* norm_weights_gpu_ = nullptr;   // preloaded norm weights for all layers
    size_t norm_weights_size_ = 0;

    // === Early exit ===
    float early_exit_threshold_ = 0.0f;  // 0 = disabled, 0.9999 = aggressive
    int early_exit_min_layer_ = 0;       // don't check before this layer
    void* prev_hidden_gpu_ = nullptr;    // [hidden_size] saved state for comparison
    void* cosine_result_d_ = nullptr;    // device float for kernel output
    float* cosine_result_h_ = nullptr;   // pinned host float for reading

    // === Layer skipping ===
    float skip_threshold_ = 0.0f;        // 0 = disabled; layers with cos > threshold are skipped
    std::vector<bool> skip_layer_;       // [n_layers] true = skip this layer
    bool skip_calibrated_ = false;       // true after first token calibrates skip list

    // === Speculative decoding verification ===
    float* verify_logits_ = nullptr;     // [verify_capacity_ * vocab_size] on GPU
    int verify_logits_capacity_ = 0;     // max seq_len slots allocated
    void ensure_verify_logits(int seq_len);

    // Streaming-specific forward pass
    float* forward_streaming(const int* tokens, int seq_len, int start_pos);

    // Tiered forward pass (hybrid VRAM-resident + streaming)
    float* forward_tiered(const int* tokens, int seq_len, int start_pos);

    // Internal
    void allocate_buffers();
    void load_layer(int layer_idx);
    void load_streaming();
    void load_tiered();
    void embed_tokens(const int* tokens, int seq_len, float* output, void* stream);

    // Tensor name helpers for GGUF
    std::string layer_prefix(int i) const;
};

} // namespace nt
