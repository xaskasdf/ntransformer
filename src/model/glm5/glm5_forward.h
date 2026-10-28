#pragma once

// ============================================================
// GLM-5.3-Flash CPU forward pass (correctness-first reference)
// Mirrors llama.cpp src/models/glm5-next.cpp graph semantics
// ============================================================

#include "glm5_model.h"
#include <string>
#include <vector>

namespace nt {
namespace glm5 {

struct GenOptions {
    std::string prompt;
    int   max_tokens     = 64;
    float temperature    = 0.7f;
    int   top_k          = 40;
    float top_p          = 0.9f;
    float repeat_penalty = 1.1f;
    int   repeat_window  = 64;
    uint64_t seed        = 42;
    int   ctx            = 4096;
    bool  verbose        = true;

    std::vector<int> token_ids;
    std::string dump_logits;
};

// Returns -1 when the file is not a glm5-next model, otherwise 0 on success.
int run_cli(const std::string& model_path, const GenOptions& opt);

class Session {
public:
    explicit Session(Model& model, int ctx);
    void reset();

    const float* forward(int token, bool compute_logits = true);

    // Chunked prefill: processes n tokens (layer-major loop; MoE experts are
    // loaded once per layer for the whole chunk). Returns the last token's
    // logits, or nullptr when GPU chunking is unavailable.
    const float* forward_chunk(const int* tokens, size_t n);

    int pos() const { return pos_; }
    int eos_id() const { return cfg_.eos_id; }
    int vocab_size() const { return cfg_.vocab_size; }
    float* logits() { return logits_valid_ ? logits_.data() : nullptr; }
    const std::vector<int32_t>& history_ref() const { return history_; }

private:
    Model& m_;
    const Config& cfg_;
    int ctx_;

    int pos_ = 0;
    bool logits_valid_ = false;
    std::vector<int32_t> history_;

    std::vector<int> recr_index_;   // layer -> compact KDA slot
    std::vector<int> mla_index_;    // layer -> compact MLA slot
    int n_recr_ = 0;
    int n_mla_ = 0;

    // KDA state
    std::vector<std::vector<float>> conv_state_;   // [n_recr][3 qkv][d_conv-1][d_inner]
    std::vector<std::vector<float>> ssm_state_;    // [n_recr][n_heads][head_dim*head_dim]

    // MLA + indexer state
    std::vector<std::vector<float>> latent_;       // [n_mla][ctx][kv_lora]
    std::vector<std::vector<float>> idx_k_;        // [n_mla][ctx][idx_dim]
    std::vector<std::vector<float>> idx_g_;        // [n_mla][ctx][idx_dim]
    std::vector<std::vector<float>> pool_k_;       // [n_mla][ctx/kpool][idx_dim]

    // absorbed wk_b / wv_b dequantized per MLA layer (F32)
    std::vector<std::vector<float>> wkb_f32_;      // [n_mla][heads][kv_lora*qk_nope]
    std::vector<std::vector<float>> wvb_f32_;      // [n_mla][heads][v_dim*kv_lora]

    // scratch
    std::vector<float> logits_;
    std::vector<float> emb_, stream_, residual_, cur_, out_, tmp_, tmp2_;
    std::vector<float> flat_norm_, mixes_, pre_, post_, comb_;
    // KDA
    std::vector<float> xq_, xk_, xv_, cq_, ck_, cv_;
    std::vector<float> fa_, g1_, beta_, decay_, cout_, gated_, ga_;
    std::vector<float> ssm_a_, dt_b_, onorm_;
    // MLA
    std::vector<float> qr_, q_, kv_, q_abs_, iq_, ik_, ig_, wproj_;
    std::vector<float> attn_lat_, attn_out_, score_, visible_;    std::vector<float> ape_;
    // MoE / FFN
    std::vector<float> moe_logits_, moe_bias_, gate_, up_, h_;

    void rms_norm(float* x, int64_t n, const float* w, float eps) const;
    void ln_norm(float* x, int64_t n, const float* w, const float* b, float eps) const;
    void matvec_name(const std::string& name, const float* x, float* y, int64_t rows) const;
    void tensor_vec(const std::string& name, float* out) const;

    void hc_mixes(const std::string& fn_name, int il);
    void kda_layer(int il, const float* x, float* out);
    void mla_layer(int il, const float* x, int64_t t, float* out);
    void select_experts(int il, const float* x, std::vector<int>& idx, std::vector<float>& wts);
    bool run_experts(int il, const float* x, const std::vector<int>& idx, const std::vector<float>& wts, float* out);
    void moe_ffn(int il, const float* x, float* out);
    void dense_ffn(int il, const float* x, float* out);
};

} // namespace glm5
} // namespace nt
