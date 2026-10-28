#pragma once

// ============================================================
// GLM-5.3-Flash decode fast-path (Fase A del playbook ter):
// forward completo device-residente, el host solo gestiona los
// fills de expertos entre el router y los kernels MoE.
// ============================================================

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nt {
namespace glm5 {

struct FastConfig {
    int n_layers = 45;
    int hidden = 4096;
    int n_heads = 64;
    int kda_head_dim = 128;
    int d_conv = 4;
    float kda_gate_lower_bound = -5.0f;
    int q_lora_rank = 1536;
    int kv_lora_rank = 512;
    int qk_nope = 256;
    int v_dim = 256;
    int indexer_n_head = 32;
    int indexer_head_dim = 128;
    int indexer_kpool = 4;
    int indexer_top_k = 2048;
    int hc_mult = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps = 1e-6f;
    float norm_eps = 1e-5f;
    float ln_eps = 1e-7f;
    int dense_lead = 3;
    int n_ff_dense = 12288;
    int n_expert = 288;
    int n_expert_used = 8;
    int n_ff_exp = 2048;
    float expert_weights_scale = 2.5f;
    bool expert_weights_norm = true;
    int vocab_size = 154880;
    int max_ctx = 2048;
    std::vector<uint8_t> is_recr;   // n_layers
};

// lookup("blk.3.attn_q.weight") -> host pointer (mmap, estable)
bool glm5_fast_init(const std::function<const void*(const std::string&)>& lookup,
                    const FastConfig& cfg);
void glm5_fast_reset();
bool glm5_fast_ready();

// un token de decode: escribe logits [vocab]. Sync unico al final.
bool glm5_fast_step(int token, float* logits);

void glm5_fast_stats();

} // namespace glm5
} // namespace nt
