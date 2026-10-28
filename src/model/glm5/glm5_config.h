#pragma once

// ============================================================
// GLM-5.3-Flash ("glm5-next") architecture support
// Reference: llama.cpp src/models/glm5-next.cpp (PR #27754)
// Hybrid KDA (gated delta rule) + nope MLA with k-pool DSA
// indexer, mHC residual streams, DeepSeek-style MoE.
// ============================================================

#include "../loader.h"
#include <cstdint>
#include <string>
#include <vector>

namespace nt {
namespace glm5 {

struct Config {
    int n_layers = 45;          // trunk layers (block_count - nextn)
    int n_layer_all = 46;       // including the nextn/MTP block
    int n_nextn = 1;
    int hidden = 4096;
    int n_heads = 64;           // KDA heads and MLA heads
    int vocab_size = 154880;

    float norm_eps = 1e-5f;     // attention.layer_norm_rms_epsilon
    float ln_eps = 1e-7f;       // attention.layer_norm_epsilon (indexer LN)

    // dense FFN (leading blocks)
    int dense_lead = 3;
    int n_ff_dense = 12288;

    // MoE
    int n_expert = 288;
    int n_expert_used = 8;
    int n_expert_shared = 1;
    int n_ff_exp = 2048;
    int n_ff_shexp = 2048;
    float expert_weights_scale = 2.5f;
    bool expert_weights_norm = true;
    std::vector<float> swiglu_clamp_exp;   // per layer (n_layer_all)
    std::vector<float> swiglu_clamp_shexp;

    // MLA (nope-only, rope.dimension_count == 0)
    int q_lora_rank = 1536;
    int kv_lora_rank = 512;
    int qk_nope = 256;          // key_length_mla
    int v_dim = 256;            // value_length_mla

    // KDA / gated delta rule
    int kda_head_dim = 128;
    int d_conv = 4;
    float kda_gate_lower_bound = -5.0f;

    // DSA indexer with k-pool compression
    int indexer_n_head = 32;
    int indexer_head_dim = 128;
    int indexer_top_k = 2048;
    int indexer_kpool = 4;

    // mHC
    int hc_mult = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps = 1e-6f;

    // per-layer roles (trunk)
    std::vector<uint8_t> is_recr;   // 1 = KDA layer (n_head_kv == 0)

    int bos_id = 0;
    int eos_id = 1;
};

bool load_config(const GGUFLoader& loader, Config& cfg, std::string& err);

} // namespace glm5
} // namespace nt
