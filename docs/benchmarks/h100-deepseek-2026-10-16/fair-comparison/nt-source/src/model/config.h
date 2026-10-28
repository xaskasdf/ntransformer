#pragma once

#include "../core/types.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nt {

// ============================================================
// Model configuration parsed from GGUF metadata
// Supports Llama, Mistral, and similar architectures
// ============================================================

struct ModelConfig {
    // Architecture
    std::string architecture = "llama";
    std::string model_name;

    // Dimensions
    int vocab_size    = 32000;
    int hidden_size   = 4096;    // d_model
    int intermediate_size = 11008;  // FFN intermediate
    int n_layers      = 32;
    int n_heads       = 32;      // query heads
    int n_kv_heads    = 32;      // key/value heads (GQA if < n_heads)
    int head_dim      = 128;     // hidden_size / n_heads

    // Normalization
    float norm_eps    = 1e-5f;

    // RoPE
    float rope_theta  = 10000.0f;
    float rope_freq_scale = 1.0f;
    bool  rope_interleaved = false;

    // Context
    int max_seq_len   = 4096;

    // Quantization (per model, individual tensors may differ)
    DType weight_dtype = DType::Q4_0;

    // BOS/EOS tokens
    int bos_token_id  = 1;
    int eos_token_id  = 2;

    // Computed values
    int group_size() const { return n_heads / n_kv_heads; }
    bool is_gqa() const { return n_kv_heads < n_heads; }

    // Raw metadata storage
    std::unordered_map<std::string, std::variant<int, float, std::string, bool>> metadata;

    void print() const;

    // Parse from GGUF metadata key-value pairs
    void from_gguf_metadata(
        const std::unordered_map<std::string, std::variant<int, float, std::string, bool>>& kv);
};

} // namespace nt
