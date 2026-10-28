#pragma once

#include "ds41_model.h"
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace nt {
namespace ds41 {

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

    // parity-testing hooks
    std::vector<int> token_ids;     // when non-empty, skip the text tokenizer
    std::string dump_logits;        // write the final prompt logits (F32 binary) here
};

// Runs a DeepSeek-V4.1 model. Returns -1 when the file is not a deepseek41
// model (caller should fall back to the Llama path), otherwise 0 on success.
int run_cli(const std::string& model_path, const GenOptions& opt);

class Session {
public:
    struct MoeProfile {
        double router_seconds = 0;
        // Host submission/wait time, excluding an overlapped CPU callback.
        double routed_seconds = 0;
        double shared_seconds = 0;
        uint64_t gpu_calls = 0, cpu_calls = 0, overlapped_calls = 0;
    };
    explicit Session(Model& model);
    void reset();

    // Updates every layer's state for one token. With compute_logits=false,
    // skips the terminal collapse/norm/head and returns nullptr. The default
    // preserves callers that consume logits at every position.
    const float* forward(int token, bool compute_logits = true);

    int pos() const { return pos_; }
    int eos_id() const { return cfg_.eos_id; }
    int vocab_size() const { return cfg_.vocab_size; }
    float* logits() { return logits_valid_ ? logits_.data() : nullptr; }
    int head_evaluations() const { return head_evaluations_; }
    const std::vector<int32_t>& history_ref() const { return history_; }

private:
    Model& m_;
    const Config& cfg_;

    // ------------------------------------------------------------------
    // state
    // ------------------------------------------------------------------
    int pos_ = 0;
    bool logits_valid_ = false;
    int head_evaluations_ = 0;
    std::vector<int32_t> history_;                 // raw token ids seen

    std::vector<float> raw_kv_;                    // [layer * sw * head_dim] ring
    std::vector<std::vector<float>> comp_rows_;    // per layer: rows of head_dim
    std::vector<std::vector<float>> idx_keys_;     // per layer: rows of idx_dim
    std::vector<std::vector<float>> pend_kv_;      // per layer: pending pool state
    std::vector<std::vector<float>> pend_score_;
    std::vector<int> pend_count_;

    // ------------------------------------------------------------------
    // scratch
    // ------------------------------------------------------------------
    std::vector<float> logits_;
    std::vector<float> stream_, residual_, cur_, tmp_, tmp2_, tmp3_;
    std::vector<float> emb_, flat_norm_, mixes_;
    std::vector<float> pre_, post_, comb_;

    struct RopeTables {
        std::vector<float> cos_t, sin_t;  // rope_dim/2 entries
    };
    RopeTables rope_plain_;
    RopeTables rope_comp_;

    // phase profiling (DS41_PROFILE=1)
    bool prof_ = false;
    bool gpu_a_ = false;
    bool moe_overlap_ = true;
    bool hc_direct_ = true;
    double prof_moe_ = 0, prof_attn_ = 0, prof_engram_ = 0, prof_hc_ = 0, prof_other_ = 0, prof_prev_sum_ = 0;
    double prof_qkv_ = 0, prof_rope_ = 0, prof_comp_ = 0, prof_loop_ = 0, prof_tail_ = 0;
    int prof_forwards_ = 0;
    MoeProfile moe_profile_;

    // stage-sum accounting for parity debugging (DS41_DUMP_STAGES=1)
    bool dbg_ = false;
    std::unordered_map<std::string, double> dbg_sums_;
    void dbg_stage(const std::string& name, const float* data, int64_t n);
    void dbg_stage_scalar(const std::string& name, double v);

public:
    bool dbg_enabled() const { return dbg_; }
    bool prof_enabled() const { return prof_; }
    double prof_moe() const { return prof_moe_; }
    double prof_attn() const { return prof_attn_; }
    double prof_engram() const { return prof_engram_; }
    double prof_hc() const { return prof_hc_; }
    double prof_other() const { return prof_other_; }
    double prof_qkv() const { return prof_qkv_; }
    double prof_rope() const { return prof_rope_; }
    double prof_comp() const { return prof_comp_; }
    double prof_loop() const { return prof_loop_; }
    double prof_tail() const { return prof_tail_; }
    int prof_forwards() const { return prof_forwards_; }
    const MoeProfile& moe_profile() const { return moe_profile_; }
    const std::unordered_map<std::string, double>& dbg_sums() const { return dbg_sums_; }
private:

    void build_rope_tables(int pos, const RopeTables& rt) const;  // no-op marker
    void rope_apply(float* v, int offset, const RopeTables& rt, bool backward) const;
    void rope_tables_for(int pos, bool compress, RopeTables& out) const;

    void rms_norm(float* x, int64_t n, const float* w, float eps) const;
    void matvec_name(const std::string& name, const float* x, float* y, int64_t rows) const;
    void tensor_vec(const std::string& name, float* out) const;  // dequant small tensor

    void hc_mixes(const std::string& prefix, int il);
    void hc_pre_collapse(const float* x, const float* w, float* out) const;
    void hc_post_mix(const float* branch, const float* post, const float* comb);

    void attention(int il, const float* cur_attn_in, int64_t t);
    void moe_ffn(int il, const float* x, float* out);
    void engram_add(int il, int token, int64_t t);

    void dot_scores(const float* q, int64_t n_heads, int64_t head_dim,
                    const std::vector<const float*>& keys,
                    const float* sinks, float kq_scale, float* out) const;
};

} // namespace ds41
} // namespace nt
