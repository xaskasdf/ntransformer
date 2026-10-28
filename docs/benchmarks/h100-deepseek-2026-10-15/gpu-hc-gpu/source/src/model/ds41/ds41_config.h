#pragma once

// ============================================================
// DeepSeek-V4.1 ("deepseek41") architecture support
// Reference: llama.cpp fork branch runtime/deepseek41
// (llama_model_deepseek4 / llama_model_deepseek41 graphs)
// ============================================================

#include "../loader.h"
#include <cstdint>
#include <string>
#include <vector>

namespace nt {
namespace ds41 {

struct Config {
    // dimensions
    int vocab_size      = 129280;
    int hidden          = 5120;
    int n_layers        = 40;
    int n_heads         = 64;
    int head_dim        = 512;   // attention.key_length; latent KV width
    int rope_dim        = 64;    // rope.dimension_count within head_dim
    int q_lora_rank     = 1280;
    int sliding_window  = 128;
    float norm_eps      = 1e-6f;

    // rope
    float rope_theta    = 10000.0f;
    float rope_factor   = 1.0f;      // yarn scaling factor
    int   rope_orig_ctx = 65536;
    float rope_beta_fast = 32.0f;
    float rope_beta_slow = 1.0f;
    float compress_rope_theta = 160000.0f;

    // experts / MoE
    int   n_expert          = 384;
    int   n_expert_used     = 6;
    int   n_expert_shared   = 1;
    int   n_ff_exp          = 2304;
    float expert_weights_scale = 1.5f;
    bool  expert_weights_norm  = true;
    std::vector<float> swiglu_clamp_exp;    // per layer
    std::vector<float> swiglu_clamp_shexp;  // per layer

    // indexer (sparse attention)
    int indexer_n_head   = 32;
    int indexer_head_dim = 128;
    int indexer_top_k    = 512;

    // grouped output projection
    int o_groups    = 8;
    int o_lora_rank = 1024;

    // hyper-connections
    int   hc_mult          = 4;
    int   hc_sinkhorn_iters = 20;
    float hc_eps           = 1e-6f;

    // per-layer compression ratios (0 = sliding window only)
    std::vector<int> compress_ratios;

    // engram
    std::vector<int> engram_layer_ids;
    int engram_n_head       = 8;
    int engram_key_length   = 256;
    int engram_max_ngram    = 4;
    int engram_pad_id       = 2;
    std::vector<uint64_t> engram_multipliers;  // n_engram_layers * max_ngram
    std::vector<uint64_t> engram_primes;       // n_engram_layers * n_cols
    std::vector<uint64_t> engram_offsets;      // n_engram_layers * n_cols
    std::vector<int32_t>  engram_token_map;    // vocab size

    // special tokens
    int bos_id = 0;
    int eos_id = 1;

    // derived: roles per layer (mirrors llama_model_deepseek41::load_arch_tensors)
    std::vector<int32_t> kv_source;        // layer that publishes the stream this layer reads
    std::vector<int32_t> index_key_source; // layer that published index keys
    std::vector<int32_t> topk_source;      // layer that runs the indexer for this layer

    int engram_ncols() const { return (engram_max_ngram - 1) * engram_n_head; }
    int engram_index_of_layer(int il) const {
        for (size_t e = 0; e < engram_layer_ids.size(); e++) {
            if (engram_layer_ids[e] == il) return (int)e;
        }
        return -1;
    }
    bool is_kv_compressed(int il) const { return compress_ratios[il] != 0; }
    bool is_index_source(int il) const { return topk_source[il] == il; }
};

// Populates `cfg` from loader metadata; computes the layer roles from tensor presence.
// Returns false (with message in `err`) when required keys or tensors are missing.
bool load_config(const GGUFLoader& loader, Config& cfg, std::string& err);

} // namespace ds41
} // namespace nt
