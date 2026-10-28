// ============================================================
// DeepSeek-V4.1 CPU forward pass (correctness-first reference)
// Mirrors llama.cpp branch runtime/deepseek41 graph semantics:
//   llama_model_deepseek41::graph + llama_model_deepseek4::graph
// ============================================================

#include "ds41_forward.h"
#include "ds41_hc.h"
#include "../../inference/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include "ds41_gpu.h"
#include <limits>

namespace nt {
namespace ds41 {

namespace {
constexpr float kPI = 3.14159265358979323846f;

inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}
inline float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)i0 / 2.0f - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}
} // namespace

// ============================================================
// Session
// ============================================================

Session::Session(Model& model) : m_(model), cfg_(model.cfg()) {
    dbg_ = getenv("DS41_DUMP_STAGES") != nullptr;
    { const char* ga = getenv("DS41_GPU_ATTN"); gpu_a_ = !ga || atoi(ga) != 0; }
    { const char* flag = getenv("DS41_ATTENTION_GPU"); attention_gpu_ = (!flag || atoi(flag) != 0) && !dbg_; }
    prof_ = getenv("DS41_PROFILE") != nullptr;
    const char* overlap = getenv("DS41_MOE_OVERLAP");
    moe_overlap_ = (!overlap || atoi(overlap) != 0) && !dbg_ && !getenv("DS41_GPU_TEST");
#if defined(__AVX2__)
    const char* shared_gpu = getenv("DS41_SHARED_GPU");
    shared_gpu_ = (!shared_gpu || atoi(shared_gpu) != 0) && !getenv("DS41_GPU_TEST");
#endif
    const char* hc_direct = getenv("DS41_HC_DIRECT");
    hc_direct_ = !hc_direct || atoi(hc_direct) != 0;
    reset();
}

void Session::dbg_stage(const std::string& name, const float* data, int64_t n) {
    if (!dbg_) return;
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += data[i];
    dbg_sums_[name] += s;
}

void Session::dbg_stage_scalar(const std::string& name, double v) {
    if (!dbg_) return;
    dbg_sums_[name] += v;
}

void Session::reset() {
    gpu_attention_reset(attention_state_.get());
    pos_ = 0;
    logits_valid_ = false;
    head_evaluations_ = 0;
    history_.clear();

    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int hd = cfg_.head_dim;
    const int sw = cfg_.sliding_window;

    raw_kv_.assign((size_t)cfg_.n_layers * sw * hd, 0.0f);
    comp_rows_.assign(cfg_.n_layers, {});
    idx_keys_.assign(cfg_.n_layers, {});
    pend_kv_.assign(cfg_.n_layers, {});
    pend_score_.assign(cfg_.n_layers, {});
    pend_count_.assign(cfg_.n_layers, 0);

    logits_.assign(cfg_.vocab_size, 0.0f);
    stream_.assign(hc * H, 0.0f);
    residual_.assign(hc * H, 0.0f);
    cur_.assign(H, 0.0f);
    tmp_.assign(std::max(hc * H, sw * hd), 0.0f);
    tmp2_.assign(std::max(4 * H, cfg_.o_groups * cfg_.o_lora_rank), 0.0f);
    tmp3_.assign(std::max(4 * H, cfg_.o_groups * cfg_.o_lora_rank), 0.0f);
    emb_.assign(H, 0.0f);
    flat_norm_.assign(hc * H, 0.0f);
    mixes_.assign((2 + hc) * hc, 0.0f);
    pre_.assign(hc, 0.0f);
    post_.assign(hc, 0.0f);
    comb_.assign(hc * hc, 0.0f);
}

void Session::rope_tables_for(int pos, bool compress, RopeTables& rt) const {
    const int dims = cfg_.rope_dim;
    const int half = dims / 2;
    rt.cos_t.resize(half);
    rt.sin_t.resize(half);

    const float base = compress ? cfg_.compress_rope_theta : cfg_.rope_theta;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = cfg_.rope_beta_fast;
    float beta_slow = cfg_.rope_beta_slow;
    const int n_ctx_orig = cfg_.rope_orig_ctx;

    if (compress) {
        freq_scale = (cfg_.rope_factor > 0.0f) ? 1.0f / cfg_.rope_factor : 1.0f;
        ext_factor = (cfg_.rope_factor > 1.0f) ? 1.0f : 0.0f;
        attn_factor = (ext_factor == 0.0f) ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        auto corr_dim = [&](float n_rot) {
            return (float)dims * logf((float)n_ctx_orig / (n_rot * 2.0f * kPI)) / (2.0f * logf(base));
        };
        corr0 = floorf(corr_dim(beta_fast));
        corr1 = ceilf(corr_dim(beta_slow));
    }

    float theta = (float)pos;
    const float theta_scale = powf(base, -2.0f / (float)dims);

    for (int p = 0; p < half; p++) {
        const float theta_extrap = theta;
        const float theta_interp = freq_scale * theta_extrap;
        float th = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix = rope_yarn_ramp(corr0, corr1, 2 * p) * ext_factor;
            th = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        rt.cos_t[p] = cosf(th) * mscale;
        rt.sin_t[p] = sinf(th) * mscale;
        theta *= theta_scale;
    }
}

// deepseek41 uses LLAMA_ROPE_TYPE_NORM: consecutive pairs (2k, 2k+1)
void Session::rope_apply(float* v, int offset, const RopeTables& rt, bool backward) const {
    const int half = cfg_.rope_dim / 2;
    for (int p = 0; p < half; p++) {
        const int i0 = offset + 2 * p;
        const int i1 = i0 + 1;
        const float x0 = v[i0];
        const float x1 = v[i1];
        const float c = rt.cos_t[p];
        const float s = rt.sin_t[p] * (backward ? -1.0f : 1.0f);
        v[i0] = x0 * c - x1 * s;
        v[i1] = x0 * s + x1 * c;
    }
}

void Session::rms_norm(float* x, int64_t n, const float* w, float eps) const {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) {
        for (int64_t i = 0; i < n; i++) x[i] = x[i] * scale * w[i];
    } else {
        for (int64_t i = 0; i < n; i++) x[i] *= scale;
    }
}

void Session::matvec_name(const std::string& name, const float* x, float* y, int64_t rows) const {
    TensorRef t = m_.tensor(name);
    NT_CHECK(t.valid(), ("missing tensor: " + name).c_str());
    NT_CHECK(t.rows() >= rows, "matvec rows mismatch");
    m_.matvec_slice(t, x, y, 0, rows);
}

void Session::tensor_vec(const std::string& name, float* out) const {
    TensorRef t = m_.tensor(name);
    NT_CHECK(t.valid(), ("missing tensor: " + name).c_str());
    const int64_t row_len = t.ne0();
    const int64_t rows = t.ne[1] * t.ne[2] * t.ne[3];
    for (int64_t r = 0; r < rows; r++) {
        m_.dequant_row(t, r, out + r * row_len);
    }
}

// ------------------------------------------------------------------
// hyper-connections
// ------------------------------------------------------------------
void Session::hc_mixes(const std::string& fn_name, int il) {
    (void)il;
    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int hc_dim = hc * H;
    const int mix_dim = (2 + hc) * hc;
    auto phase_start = std::chrono::steady_clock::now();

    // rms-norm the flattened stream (no weight)
    memcpy(flat_norm_.data(), stream_.data(), sizeof(float) * hc_dim);
    rms_norm(flat_norm_.data(), hc_dim, nullptr, cfg_.norm_eps);
    auto projection_start = std::chrono::steady_clock::now();
    if (prof_) hc_profile_.norm_seconds += std::chrono::duration<double>(projection_start - phase_start).count();

    TensorRef fn = m_.tensor(fn_name);
    NT_CHECK(fn.valid(), ("missing tensor: " + fn_name).c_str());
    NT_CHECK(fn.ne0() == hc_dim && fn.rows() == mix_dim, "hc_fn shape mismatch");
    static const bool gpu_projection = [] {
        const char* flag = getenv("DS41_HC_PROJECTION_GPU");
        return !flag || atoi(flag) != 0;
    }();
    static const bool serial_projection = [] {
        const char* flag = getenv("DS41_HC_PROJECTION_SERIAL");
        return flag && atoi(flag) != 0;
    }();
#if defined(__AVX2__)
    constexpr bool avx2_order = true;
#else
    constexpr bool avx2_order = false;
#endif
    const bool dense = fn.dtype == DType::F32 || fn.dtype == DType::BF16;
    if (gpu_projection && dense && gpu_hc_projection(fn.data, flat_norm_.data(), mixes_.data(),
            mix_dim, hc_dim, fn.dtype == DType::BF16, avx2_order)) {
        if (prof_) ++hc_profile_.gpu_calls;
    } else if (serial_projection) {
        // Diagnostic control: isolate pool-dispatch cost without changing dots.
        for (int row = 0; row < mix_dim; ++row) mixes_[row] = m_.dot_row(fn, row, flat_norm_.data());
        if (prof_) ++hc_profile_.cpu_serial_calls;
    } else {
        m_.matvec(fn, flat_norm_.data(), mixes_.data());
        if (prof_) ++hc_profile_.cpu_pool_calls;
    }
    auto mix_start = std::chrono::steady_clock::now();
    if (prof_) hc_profile_.projection_seconds += std::chrono::duration<double>(mix_start - projection_start).count();
    dbg_stage("hc_mixes-" + std::to_string(il), mixes_.data(), mix_dim);

    std::string base_name = fn_name.substr(0, fn_name.size() - 10) + "_base.weight";
    std::string scale_name = fn_name.substr(0, fn_name.size() - 10) + "_scale.weight";
    float base[128], scale[3];
    NT_CHECK(mix_dim <= 128, "hc mix dim too large");
    tensor_vec(base_name, base);
    tensor_vec(scale_name, scale);
    const float eps = cfg_.hc_eps;

    for (int h = 0; h < hc; h++) {
        pre_[h] = sigmoid_f(mixes_[h] * scale[0] + base[h]) + eps;
        post_[h] = 2.0f * sigmoid_f(mixes_[hc + h] * scale[1] + base[hc + h]);
    }

    // comb matrix [dst][src]
    float m2[64];
    NT_CHECK(hc * hc <= 64, "hc comb too large");
    for (int dst = 0; dst < hc; dst++) {
        for (int src = 0; src < hc; src++) {
            m2[dst * hc + src] = mixes_[2 * hc + dst + src * hc] * scale[2] + base[2 * hc + dst + src * hc];
        }
    }

    // sinkhorn: softmax over dst (per src), add eps, then alternating normalization
    for (int src = 0; src < hc; src++) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int dst = 0; dst < hc; dst++) mx = std::max(mx, m2[dst * hc + src]);
        float sum = 0.0f;
        for (int dst = 0; dst < hc; dst++) {
            m2[dst * hc + src] = expf(m2[dst * hc + src] - mx);
            sum += m2[dst * hc + src];
        }
        for (int dst = 0; dst < hc; dst++) m2[dst * hc + src] /= sum;
    }
    for (int i = 0; i < hc * hc; i++) m2[i] += eps;

    auto norm_cols = [&]() {  // divide by sum over src (per dst)
        for (int dst = 0; dst < hc; dst++) {
            float s = 0.0f;
            for (int src = 0; src < hc; src++) s += m2[dst * hc + src];
            s += eps;
            for (int src = 0; src < hc; src++) m2[dst * hc + src] /= s;
        }
    };
    auto norm_rows = [&]() {  // divide by sum over dst (per src)
        for (int src = 0; src < hc; src++) {
            float s = 0.0f;
            for (int dst = 0; dst < hc; dst++) s += m2[dst * hc + src];
            s += eps;
            for (int dst = 0; dst < hc; dst++) m2[dst * hc + src] /= s;
        }
    };

    norm_cols();
    for (int i = 1; i < cfg_.hc_sinkhorn_iters; i++) {
        norm_rows();
        norm_cols();
    }

    memcpy(comb_.data(), m2, sizeof(float) * hc * hc);
    dbg_stage("hc_pre-" + std::to_string(il), pre_.data(), hc);
    dbg_stage("hc_post-" + std::to_string(il), post_.data(), hc);
    dbg_stage("hc_comb-" + std::to_string(il), comb_.data(), hc * hc);
    if (prof_) hc_profile_.mix_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - mix_start).count();
}

void Session::hc_pre_collapse(const float* x, const float* w, float* out) const {
    detail::hc_collapse(x, w, out, cfg_.hidden, cfg_.hc_mult, hc_direct_, m_.n_threads);
}

void Session::hc_post_mix(const float* branch, const float* post, const float* comb) {
    detail::hc_post_mix(branch, residual_.data(), post, comb, stream_.data(),
                        cfg_.hidden, cfg_.hc_mult, hc_direct_, m_.n_threads);
}

// ------------------------------------------------------------------
// attention (MLA + sliding window + sparse compressed stream)
// ------------------------------------------------------------------
void Session::attention(int il, const float* x, int64_t t) {
    const int H = cfg_.hidden;
    const int hd = cfg_.head_dim;
    const int nh = cfg_.n_heads;
    const int rd = cfg_.rope_dim;
    const int sw = cfg_.sliding_window;
    const int ratio = cfg_.compress_ratios[il];
    const bool compress = ratio != 0;
    if (attention_gpu_ && gpu_a_ && !attention_state_ && il == 0 && t == 0)
        attention_state_.reset(gpu_attention_create(cfg_.n_layers, sw, nh, hd));

    const std::string p = "blk." + std::to_string(il) + ".";
    RopeTables rt;
    rope_tables_for((int)t, compress, rt);
    auto ta = std::chrono::steady_clock::now();

    // ---- query + sliding-window latent (shared K and V) ----
    std::vector<float> qr(cfg_.q_lora_rank);
    std::vector<float> q((size_t)nh * hd);
    std::vector<float> kv(hd);

    bool gpu_qkv = false;
    bool resident_qkv = false;
    if (gpu_a_) {
        TensorRef wqa = m_.tensor(p + "attn_q_a.weight");
        TensorRef qan = m_.tensor(p + "attn_q_a_norm.weight");
        TensorRef wqb = m_.tensor(p + "attn_q_b.weight");
        TensorRef wkv = m_.tensor(p + "attn_kv.weight");
        TensorRef kvn = m_.tensor(p + "attn_kv_a_norm.weight");
        if (wqa.valid() && qan.valid() && wqb.valid() && wkv.valid() && kvn.valid() &&
            wqa.dtype == DType::Q8_0 && wqb.dtype == DType::Q8_0 && wkv.dtype == DType::Q8_0 &&
            qan.dtype == DType::F32 && kvn.dtype == DType::F32) {
            auto* state = gpu_attention_ready(attention_state_.get(), il, t) ? attention_state_.get() : nullptr;
            gpu_qkv = nt::ds41::gpu_attn_prologue(x, H, wqa.data, qan.data, wqb.data, wkv.data, kvn.data,
                    cfg_.q_lora_rank, hd, nh, rd, cfg_.norm_eps,
                    rt.cos_t.data(), rt.sin_t.data(), qr.data(), q.data(), kv.data(), state, il, t);
            resident_qkv = gpu_qkv && state;
        }
    }

    if (!gpu_qkv) {
        if (gpu_a_ && il == 0 && getenv("DS41_GPU_TIMING")) fprintf(stderr, "[attn] gpu prologue failed at layer %d, falling back to CPU\n", il);
        matvec_name(p + "attn_q_a.weight", x, qr.data(), cfg_.q_lora_rank);
        {
            std::vector<float> w(cfg_.q_lora_rank);
            tensor_vec(p + "attn_q_a_norm.weight", w.data());
            rms_norm(qr.data(), cfg_.q_lora_rank, w.data(), cfg_.norm_eps);
        }
        {
            TensorRef wq_b = m_.tensor(p + "attn_q_b.weight");
            NT_CHECK(wq_b.valid(), "missing attn_q_b");
            m_.matvec(wq_b, qr.data(), q.data());
        }
        for (int h = 0; h < nh; h++) rope_apply(q.data() + (size_t)h * hd, hd - rd, rt, false);

        matvec_name(p + "attn_kv.weight", x, kv.data(), hd);
        {
            std::vector<float> w(hd);
            tensor_vec(p + "attn_kv_a_norm.weight", w.data());
            rms_norm(kv.data(), hd, w.data(), cfg_.norm_eps);
        }
        rope_apply(kv.data(), hd - rd, rt, false);
    }
    dbg_stage("q-" + std::to_string(il), q.data(), (int64_t)nh * hd);
    prof_qkv_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
    auto tb = std::chrono::steady_clock::now();

    float* raw_layer = raw_kv_.data() + (size_t)il * sw * hd;
    memcpy(raw_layer + (size_t)(t % sw) * hd, kv.data(), sizeof(float) * hd);
    dbg_stage("kv-" + std::to_string(il), kv.data(), hd);

    // ---- compressed stream ----
    std::vector<const float*> comp_keys;
    std::vector<int> comp_ids;
    int comp_source = -1;
    int n_visible = 0;

    if (compress) {
        const int src = cfg_.kv_source[il];
        comp_source = src;
        NT_CHECK(src >= 0, "layer has no KV source");

        if (il == src) {
            std::vector<float> skv(hd), sscore(hd);
            bool gpu_cs = false;
            if (gpu_a_) {
                TensorRef ckw = m_.tensor(p + "attn_compressor_kv.weight");
                TensorRef cgt = m_.tensor(p + "attn_compressor_gate.weight");
                if (ckw.valid() && ckw.dtype == DType::BF16 && (!cgt.valid() || cgt.dtype == DType::BF16)) {
                    gpu_cs = nt::ds41::gpu_comp_state(x, H, hd, ckw.data,
                                cgt.valid() ? cgt.data : nullptr,
                                skv.data(), cgt.valid() ? sscore.data() : nullptr);
                    if (gpu_cs && !cgt.valid()) sscore = skv;
                }
            }
            if (!gpu_cs) {
                matvec_name(p + "attn_compressor_kv.weight", x, skv.data(), hd);
                if (m_.tensor(p + "attn_compressor_gate.weight").valid()) {
                    matvec_name(p + "attn_compressor_gate.weight", x, sscore.data(), hd);
                } else {
                    sscore = skv;
                }
            }

            auto& pend_kv = pend_kv_[il];
            auto& pend_sc = pend_score_[il];
            pend_kv.insert(pend_kv.end(), skv.begin(), skv.end());
            pend_sc.insert(pend_sc.end(), sscore.begin(), sscore.end());
            pend_count_[il]++;

            if (pend_count_[il] == ratio) {
                std::vector<float> pooled(hd, 0.0f);
                // softmax over the ratio tokens, per channel
                for (int c = 0; c < hd; c++) {
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int r = 0; r < ratio; r++) mx = std::max(mx, pend_sc[r * hd + c]);
                    float sum = 0.0f;
                    for (int r = 0; r < ratio; r++) sum += expf(pend_sc[r * hd + c] - mx);
                    float acc = 0.0f;
                    for (int r = 0; r < ratio; r++) {
                        acc += (expf(pend_sc[r * hd + c] - mx) / sum) * pend_kv[r * hd + c];
                    }
                    pooled[c] = acc;
                }

                {
                    std::vector<float> w(hd);
                    tensor_vec(p + "attn_compressor_norm.weight", w.data());
                    rms_norm(pooled.data(), hd, w.data(), cfg_.norm_eps);
                }

                std::vector<float> latent_pre = pooled;

                // rows are rotated at the start position of their block
                const int comp_pos = (int)(t + 1 - ratio);
                RopeTables rtc;
                rope_tables_for(comp_pos, true, rtc);
                rope_apply(pooled.data(), hd - rd, rtc, false);

                comp_rows_[il].insert(comp_rows_[il].end(), pooled.begin(), pooled.end());
                dbg_stage("comp_kv-" + std::to_string(il), pooled.data(), hd);

                // index keys derive from the pre-rope latent
                if (il == cfg_.index_key_source[il]) {
                    std::vector<float> idxk(cfg_.indexer_head_dim);
                    TensorRef ikw = m_.tensor(p + "indexer.attn_k.weight");
                    if (!(gpu_a_ && ikw.valid() && ikw.dtype == DType::BF16 &&
                          nt::ds41::gpu_indexer_gemv(ikw.data, latent_pre.data(), hd, cfg_.indexer_head_dim, 1, idxk.data()))) {
                        matvec_name(p + "indexer.attn_k.weight", latent_pre.data(), idxk.data(), cfg_.indexer_head_dim);
                    }
                    std::vector<float> w(cfg_.indexer_head_dim);
                    tensor_vec(p + "indexer.k_norm.weight", w.data());
                    rms_norm(idxk.data(), cfg_.indexer_head_dim, w.data(), cfg_.norm_eps);
                    rope_apply(idxk.data(), cfg_.indexer_head_dim - rd, rtc, false);
                    idx_keys_[il].insert(idx_keys_[il].end(), idxk.begin(), idxk.end());
                    dbg_stage("idx_k_new-" + std::to_string(il), idxk.data(), cfg_.indexer_head_dim);
                }

                pend_kv.clear();
                pend_sc.clear();
                pend_count_[il] = 0;
            }
        }

        n_visible = (int)((t + 1) / ratio);
        const int src_rows = (int)(comp_rows_[src].size() / hd);
        if (n_visible > src_rows) n_visible = src_rows;

        // indexer top-k (only on layers that run it)
        std::vector<int> selected;
        if (il == cfg_.topk_source[il] && n_visible > 0) {
            const int owner = cfg_.index_key_source[il];
            const int idh = cfg_.indexer_head_dim;
            const int inh = cfg_.indexer_n_head;
            NT_CHECK(owner >= 0, "no index key source");

            std::vector<float> idxq((size_t)inh * idh);
            TensorRef iqb = m_.tensor(p + "indexer.attn_q_b.weight");
            if (!(gpu_a_ && iqb.valid() && iqb.dtype == DType::Q8_0 &&
                  nt::ds41::gpu_indexer_gemv(iqb.data, qr.data(), cfg_.q_lora_rank, inh * idh, 0, idxq.data()))) {
                matvec_name(p + "indexer.attn_q_b.weight", qr.data(), idxq.data(), inh * idh);
            }
            for (int h = 0; h < inh; h++) {
                rope_apply(idxq.data() + (size_t)h * idh, idh - rd, rt, false);
            }
            std::vector<float> idxw(inh);
            TensorRef ipj = m_.tensor(p + "indexer.proj.weight");
            if (!(gpu_a_ && ipj.valid() && ipj.dtype == DType::BF16 &&
                  nt::ds41::gpu_indexer_gemv(ipj.data, x, H, inh, 1, idxw.data()))) {
                matvec_name(p + "indexer.proj.weight", x, idxw.data(), inh);
            }
            const float wscale = 1.0f / sqrtf((float)(idh * inh));
            for (int h = 0; h < inh; h++) idxw[h] *= wscale;
            dbg_stage("idx_weights-" + std::to_string(il), idxw.data(), inh);

            const float* idxk_all = idx_keys_[owner].data();
            const int n_idxk = (int)(idx_keys_[owner].size() / idh);
            const int n_score = std::min({n_visible, n_idxk, (int)(idx_keys_[owner].size() / idh)});

            std::vector<std::pair<float, int>> scores(n_score);
            parallel_for(n_score, m_.n_threads, [&](int64_t j, int) {
                float s = 0.0f;
                for (int h = 0; h < inh; h++) {
                    const float* qh = idxq.data() + (size_t)h * idh;
                    const float* kh = idxk_all + ((size_t)j * idh);
                    float dot = 0.0f;
                    for (int d = 0; d < idh; d++) dot += qh[d] * kh[d];
                    if (dot > 0.0f) s += dot * idxw[h];
                }
                scores[j] = { s, (int)j };
            });

            const int k = std::min(cfg_.indexer_top_k, n_score);
            std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            selected.resize(k);
            for (int i = 0; i < k; i++) selected[i] = scores[i].second;
        }

        const float* rows = comp_rows_[src].data();
        if (!selected.empty()) {
            for (int j : selected) {
                if (j < n_visible) {
                    comp_keys.push_back(rows + (size_t)j * hd);
                    comp_ids.push_back(j);
                }
            }
        } else {
            for (int j = 0; j < n_visible; j++) {
                comp_keys.push_back(rows + (size_t)j * hd);
                comp_ids.push_back(j);
            }
        }
    }

    prof_comp_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
    auto tcc = std::chrono::steady_clock::now();

    std::vector<float> sinks(nh);
    tensor_vec(p + "attn_sinks.weight", sinks.data());
    if (resident_qkv) {
        TensorRef wa = m_.tensor(p + "attn_output_a.weight"), wb = m_.tensor(p + "attn_output_b.weight");
        AttentionRows rows{comp_source, n_visible ? comp_rows_[comp_source].data() : nullptr,
            n_visible, comp_ids.data(), (int)comp_ids.size()};
        if (wa.valid() && wb.valid() && wa.dtype == DType::Q8_0 && wb.dtype == DType::Q8_0 &&
            wa.ne0() == (nh / cfg_.o_groups) * hd && wa.rows() == cfg_.o_groups * cfg_.o_lora_rank &&
            wb.ne0() == cfg_.o_groups * cfg_.o_lora_rank && wb.rows() == H &&
            gpu_attention_tail(attention_state_.get(), il, t, rows, sinks.data(), wa.data, wb.data,
                               cfg_.o_groups, cfg_.o_lora_rank, H, tmp3_.data())) {
            prof_resident_attention_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tcc).count();
            memcpy(cur_.data(), tmp3_.data(), sizeof(float) * H);
            return;
        }
        // The CPU KV mirror is current. Recover the original projected Q if
        // device allocation/shape limits require this layer's ordinary path.
        NT_CHECK(gpu_attention_read_query(attention_state_.get(), q.data()), "resident attention query recovery failed");
    }

    // raw window keys (oldest first)
    std::vector<const float*> keys;
    const int64_t first = std::max<int64_t>(0, t - sw + 1);
    keys.reserve((size_t)(t - first + 1) + comp_keys.size());
    for (int64_t pp = first; pp <= t; pp++) {
        keys.push_back(raw_layer + (size_t)(pp % sw) * hd);
    }
    for (const float* ck : comp_keys) keys.push_back(ck);

    // attention per head; output = weighted sum of value vectors (v == k)
    const float kq_scale = 1.0f / sqrtf((float)hd);
    std::vector<float> attn_out((size_t)nh * hd);
    const float* qp = q.data();
    parallel_for(nh, m_.n_threads, [&](int64_t h, int) {
        const float* qh = qp + (size_t)h * hd;
        std::vector<float> acc(hd, 0.0f);
        const int nk = (int)keys.size();
        std::vector<float> sc(nk);
        float mx = sinks[h];
        for (int j = 0; j < nk; j++) {
            const float* kj = keys[j];
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) dot += qh[d] * kj[d];
            dot *= kq_scale;
            sc[j] = dot;
            if (dot > mx) mx = dot;
        }
        float denom = expf(sinks[h] - mx);
        for (int j = 0; j < nk; j++) {
            float pr = expf(sc[j] - mx);
            denom += pr;
            const float* kj = keys[j];
            for (int d = 0; d < hd; d++) acc[d] += pr * kj[d];
        }
        float inv = 1.0f / denom;
        for (int d = 0; d < hd; d++) attn_out[(size_t)h * hd + d] = acc[d] * inv;
    });

    dbg_stage("attn_out_raw-" + std::to_string(il), attn_out.data(), (int64_t)nh * hd);
    // de-rope the output (inverse rotation at the current position)
    for (int h = 0; h < nh; h++) {
        rope_apply(attn_out.data() + (size_t)h * hd, hd - rd, rt, true);
    }
    dbg_stage("attn_derope-" + std::to_string(il), attn_out.data(), (int64_t)nh * hd);

    prof_loop_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tcc).count();
    auto td = std::chrono::steady_clock::now();

    // ---- grouped output projection ----
    const int npg = nh / cfg_.o_groups;          // heads per group
    const int gdim = npg * hd;                   // 4096
    std::vector<float> oa((size_t)cfg_.o_groups * cfg_.o_lora_rank);
    TensorRef wo_a = m_.tensor(p + "attn_output_a.weight");
    NT_CHECK(wo_a.valid(), "missing attn_output_a");
    NT_CHECK(wo_a.ne0() == gdim && wo_a.rows() == cfg_.o_groups * cfg_.o_lora_rank, "wo_a shape");
    bool gpu_tail = false;
    if (gpu_a_) {
        TensorRef wo_b = m_.tensor(p + "attn_output_b.weight");
        if (wo_b.valid() && wo_b.dtype == DType::Q8_0 && wo_a.dtype == DType::Q8_0) {
            gpu_tail = nt::ds41::gpu_attn_tail(attn_out.data(), hd, nh, wo_a.data, wo_b.data,
                        cfg_.o_groups, cfg_.o_lora_rank, H, tmp3_.data());
        }
    }
    for (int g = 0; !gpu_tail && g < cfg_.o_groups; g++) {
        // concat heads of the group
        float* gin = tmp_.data();
        for (int hi = 0; hi < npg; hi++) {
            memcpy(gin + (size_t)hi * hd, attn_out.data() + (size_t)(g * npg + hi) * hd, sizeof(float) * hd);
        }
        m_.matvec_slice(wo_a, gin, oa.data() + (size_t)g * cfg_.o_lora_rank,
                        (size_t)g * cfg_.o_lora_rank, cfg_.o_lora_rank);
    }

    dbg_stage("attn_wo_a-" + std::to_string(il), oa.data(), (int64_t)cfg_.o_groups * cfg_.o_lora_rank);
    if (!gpu_tail) matvec_name(p + "attn_output_b.weight", oa.data(), tmp3_.data(), H);
    prof_tail_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
    memcpy(cur_.data(), tmp3_.data(), sizeof(float) * H);
    dbg_stage("attn_out-" + std::to_string(il), cur_.data(), H);
}

// ------------------------------------------------------------------
// MoE FFN (+ shared expert)
// ------------------------------------------------------------------
void Session::moe_ffn(int il, const float* x, float* out) {
    const auto router_start = std::chrono::steady_clock::now();
    const int H = cfg_.hidden;
    const int ne = cfg_.n_expert;
    const int used = cfg_.n_expert_used;
    const int ff = cfg_.n_ff_exp;
    const std::string p = "blk." + std::to_string(il) + ".";

    TensorRef gate_inp = m_.tensor(p + "ffn_gate_inp.weight");
    NT_CHECK(gate_inp.valid(), "missing ffn_gate_inp");

    std::vector<float> logits(ne);
    static const bool router_gpu = [] {
        const char* flag = getenv("DS41_ROUTER_GPU");
        return !flag || atoi(flag) != 0;
    }();
#if defined(__AVX2__)
    constexpr bool avx2_order = true;
#else
    constexpr bool avx2_order = false;
#endif
    const bool dense_router = gate_inp.dtype == DType::F32 || gate_inp.dtype == DType::BF16;
    if (!(router_gpu && dense_router && gate_inp.rows() == ne && gate_inp.ne0() == H &&
          gpu_router(gate_inp.data, x, logits.data(), ne, H, gate_inp.dtype == DType::BF16, avx2_order))) {
        m_.matvec(gate_inp, x, logits.data());
    }

    std::vector<float> bias(ne, 0.0f);
    if (m_.tensor(p + "exp_probs_b.bias").valid()) {
        tensor_vec(p + "exp_probs_b.bias", bias.data());
    }

    std::vector<float> probs(ne);
    for (int e = 0; e < ne; e++) probs[e] = sqrtf(softplus_f(logits[e]));
    dbg_stage("ffn_moe_logits-" + std::to_string(il), logits.data(), ne);
    dbg_stage("ffn_moe_probs-" + std::to_string(il), probs.data(), ne);

    std::vector<int> idx(ne);
    for (int e = 0; e < ne; e++) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + used, idx.end(),
        [&](int a, int b) { return (probs[a] + bias[a]) > (probs[b] + bias[b]); });

    std::vector<float> wts(used);
    float wsum = 0.0f;
    for (int k = 0; k < used; k++) {
        wts[k] = probs[idx[k]];
        wsum += wts[k];
    }
    if (cfg_.expert_weights_norm) {
        wsum = std::max(wsum, 6.103515625e-5f);
        for (int k = 0; k < used; k++) wts[k] /= wsum;
    }
    if (cfg_.expert_weights_scale != 0.0f && cfg_.expert_weights_scale != 1.0f) {
        for (int k = 0; k < used; k++) wts[k] *= cfg_.expert_weights_scale;
    }

    const float limit = (il < (int)cfg_.swiglu_clamp_exp.size()) ? cfg_.swiglu_clamp_exp[il] : 0.0f;

    TensorRef gate_exps = m_.tensor(p + "ffn_gate_exps.weight");
    TensorRef up_exps   = m_.tensor(p + "ffn_up_exps.weight");
    TensorRef down_exps = m_.tensor(p + "ffn_down_exps.weight");
    NT_CHECK(gate_exps.valid() && up_exps.valid() && down_exps.valid(), "missing expert tensors");

    const auto routed_start = std::chrono::steady_clock::now();
    if (prof_) moe_profile_.router_seconds += std::chrono::duration<double>(routed_start - router_start).count();

    std::fill(out, out + H, 0.0f);

    // Shared and routed experts depend only on x. Keep their outputs separate
    // until the same ordered final addition as the serial schedule.
    std::vector<float> shared_out;
    bool shared_ready = false;
    bool shared_on_gpu = false;
    double shared_seconds = 0;
    const auto compute_shared = [&]() {
        const auto start = std::chrono::steady_clock::now();
        const float slim = (il < (int)cfg_.swiglu_clamp_shexp.size()) ? cfg_.swiglu_clamp_shexp[il] : limit;
        const int ffs = ff * cfg_.n_expert_shared;
        TensorRef gs = m_.tensor(p + "ffn_gate_shexp.weight");
        TensorRef us = m_.tensor(p + "ffn_up_shexp.weight");
        TensorRef ds = m_.tensor(p + "ffn_down_shexp.weight");
        NT_CHECK(gs.valid() && us.valid() && ds.valid(), "missing shared expert tensors");
        std::vector<float> gate(ffs), up(ffs), h(ffs);
        shared_out.resize(H);
        m_.matvec(gs, x, gate.data());
        m_.matvec(us, x, up.data());
        for (int i = 0; i < ffs; i++) {
            float g = gate[i];
            float u = up[i];
            if (slim > 1e-6f) {
                g = std::min(g, slim);
                u = std::max(-slim, std::min(u, slim));
            }
            h[i] = (g / (1.0f + expf(-g))) * u;
        }
        m_.matvec(ds, h.data(), shared_out.data());
        shared_ready = true;
        if (prof_) ++moe_profile_.shared_cpu_calls;
        shared_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    std::function<void()> host_work;
    if (moe_overlap_ && cfg_.n_expert_shared > 0) host_work = compute_shared;

    bool gpu_ok = false;
    if (nt::ds41::gpu_experts_enabled() &&
        gate_exps.dtype == DType::MXFP4 && up_exps.dtype == DType::MXFP4 && down_exps.dtype == DType::MXFP4) {
        if (shared_gpu_ && cfg_.n_expert_shared > 0) {
            TensorRef gs = m_.tensor(p + "ffn_gate_shexp.weight");
            TensorRef us = m_.tensor(p + "ffn_up_shexp.weight");
            TensorRef ds = m_.tensor(p + "ffn_down_shexp.weight");
            const int ffs = ff * cfg_.n_expert_shared;
            if (gs.valid() && us.valid() && ds.valid() && gs.dtype == DType::Q8_0 &&
                us.dtype == DType::Q8_0 && ds.dtype == DType::Q8_0 &&
                gs.ne0() == H && gs.rows() == ffs && us.ne0() == H && us.rows() == ffs &&
                ds.ne0() == ffs && ds.rows() == H) {
                const float slim = il < (int)cfg_.swiglu_clamp_shexp.size() ? cfg_.swiglu_clamp_shexp[il] : limit;
                SharedExpert shared{gs.data, us.data, ds.data, ffs, slim};
                shared_out.resize(H);
                gpu_ok = gpu_moe_experts_shared(x, H, gate_exps.data, up_exps.data, down_exps.data,
                    ff, idx.data(), wts.data(), used, limit, out, shared, shared_out.data());
                if (gpu_ok) {
                    shared_ready = shared_on_gpu = true;
                    if (prof_) ++moe_profile_.shared_gpu_calls;
                }
            }
        }
        if (!gpu_ok) gpu_ok = nt::ds41::gpu_moe_experts(x, H, gate_exps.data, up_exps.data, down_exps.data,
                                           ff, idx.data(), wts.data(), used, limit, out, host_work);
    }
    const bool shared_overlapped = shared_ready && !shared_on_gpu;

    std::vector<float> gpu_out;
    if (gpu_ok && getenv("DS41_GPU_TEST")) {
        gpu_out.assign(out, out + H);
        std::fill(out, out + H, 0.0f);
        gpu_ok = false;   // fall through to the CPU path for the comparison
    }

    if (!gpu_ok) {
        // GPU success does not need these CPU fallback buffers.
        std::vector<std::vector<float>> partials(used, std::vector<float>(H, 0.0f));
        std::vector<std::vector<float>> hbuf(used, std::vector<float>(ff));
        for (int k = 0; k < used; k++) {
            const int e = idx[k];
            std::vector<float> gate(ff), up(ff);
            m_.matvec_slice(gate_exps, x, gate.data(), (size_t)e * ff, ff);
            m_.matvec_slice(up_exps, x, up.data(), (size_t)e * ff, ff);
            for (int i = 0; i < ff; i++) {
                float g = gate[i];
                float u = up[i];
                if (limit > 1e-6f) {
                    g = std::min(g, limit);
                    u = std::max(-limit, std::min(u, limit));
                }
                hbuf[k][i] = (g / (1.0f + expf(-g))) * u;
            }
            m_.matvec_slice(down_exps, hbuf[k].data(), partials[k].data(), (size_t)e * H, H);
        }
        for (int k = 0; k < used; k++) {
            const float w = wts[k];
            for (int i = 0; i < H; i++) out[i] += w * partials[k][i];
        }
    }

    if (!gpu_out.empty()) {
        std::vector<float> gpu2(H, 0.0f);
        nt::ds41::gpu_moe_experts(x, H, gate_exps.data, up_exps.data, down_exps.data, ff,
                                  idx.data(), wts.data(), used, limit, gpu2.data());
        float m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
        int a1 = -1;
        for (int i = 0; i < H; i++) {
            float d1 = fabsf(out[i] - gpu_out[i]);
            float d2 = fabsf(out[i] - gpu2[i]);
            float d3 = fabsf(gpu_out[i] - gpu2[i]);
            if (d1 > m1) { m1 = d1; a1 = i; }
            m2 = std::max(m2, d2);
            m3 = std::max(m3, d3);
        }
        fprintf(stderr, "[gpu-test] layer %d: |gpu1-cpu|=%.7f (at %d) |gpu2-cpu|=%.7f |gpu1-gpu2|=%.7f  cpu=%.6f gpu1=%.6f gpu2=%.6f x0=%.6f\n",
                il, m1, a1, m2, m3, out[a1], gpu_out[a1], gpu2[a1], x[0]);
    }

    dbg_stage("ffn_moe_weights-" + std::to_string(il), wts.data(), used);
    dbg_stage("ffn_moe_out-" + std::to_string(il), out, H);

    const auto shared_start = std::chrono::steady_clock::now();
    if (prof_) {
        moe_profile_.routed_seconds += std::chrono::duration<double>(shared_start - routed_start).count() - shared_seconds;
        if (gpu_ok) ++moe_profile_.gpu_calls;
        else ++moe_profile_.cpu_calls;
        if (shared_overlapped) ++moe_profile_.overlapped_calls;
    }

    // shared expert
    if (cfg_.n_expert_shared > 0) {
        if (!shared_ready) compute_shared();
        dbg_stage("ffn_shexp-" + std::to_string(il), shared_out.data(), H);
        for (int i = 0; i < H; i++) out[i] += shared_out[i];
    }
    dbg_stage("ffn_out-" + std::to_string(il), out, H);
    if (prof_) moe_profile_.shared_seconds += shared_seconds;
}

// ------------------------------------------------------------------
// engram n-gram lookup
// ------------------------------------------------------------------
void Session::engram_add(int il, int token, int64_t t) {
    const int eg = cfg_.engram_index_of_layer(il);
    if (eg < 0) return;

    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int ncols = cfg_.engram_ncols();
    const int key_len = cfg_.engram_key_length;
    const int ngram = cfg_.engram_max_ngram;
    const int nhead = cfg_.engram_n_head;

    const uint64_t* mult = cfg_.engram_multipliers.data() + (size_t)eg * ngram;
    const uint64_t* prime = cfg_.engram_primes.data() + (size_t)eg * ncols;
    const uint64_t* offset = cfg_.engram_offsets.data() + (size_t)eg * ncols;

    auto map_of = [&](int64_t tk) -> uint64_t {
        if (tk < 0 || tk >= (int64_t)cfg_.engram_token_map.size()) return (uint64_t)cfg_.engram_pad_id;
        return (uint64_t)cfg_.engram_token_map[tk];
    };

    uint64_t ctx[8] = {0};
    ctx[0] = map_of(token);
    bool blocked = false;
    for (int s = 1; s < ngram; s++) {
        int64_t prev_pos = t - s;
        int64_t prev_tok = -1;
        if (prev_pos >= 0 && prev_pos < (int64_t)history_.size()) prev_tok = history_[prev_pos];
        if (prev_tok < 0) blocked = true;
        ctx[s] = blocked ? (uint64_t)cfg_.engram_pad_id : map_of(prev_tok);
    }

    std::vector<int64_t> rows(ncols);
    uint64_t rolling = ctx[0] * mult[0];
    for (int s = 1; s < ngram; s++) {
        rolling ^= ctx[s] * mult[s];
        for (int h = 0; h < nhead; h++) {
            const int b = (s - 1) * nhead + h;
            rows[b] = (int64_t)(rolling % prime[b] + offset[b]);
        }
    }

    // gather the table rows and flatten (buckets slowest)
    TensorRef tab = m_.tensor("blk." + std::to_string(il) + ".engram_embd.weight");
    NT_CHECK(tab.valid(), "missing engram_embd");
    std::vector<float> emb((size_t)ncols * key_len);
    for (int b = 0; b < ncols; b++) {
        m_.dequant_row(tab, rows[b], emb.data() + (size_t)b * key_len);
    }

    dbg_stage("engram_embd-" + std::to_string(il), emb.data(), (int64_t)ncols * key_len);
    // kv = wkv @ emb -> [(hc+1)*H];  key (hc copies) then value (shared)
    std::vector<float> kvv((size_t)(hc + 1) * H);
    matvec_name("blk." + std::to_string(il) + ".engram_wkv.weight", emb.data(), kvv.data(), (hc + 1) * H);
    dbg_stage("engram_kv-" + std::to_string(il), kvv.data(), (int64_t)(hc + 1) * H);
    const float* value = kvv.data() + (size_t)hc * H;

    std::vector<float> wq(hc * H), wk(hc * H);
    tensor_vec("blk." + std::to_string(il) + ".engram_q.weight", wq.data());
    tensor_vec("blk." + std::to_string(il) + ".engram_k.weight", wk.data());

    const float sq = sqrtf((float)H);
    for (int h = 0; h < hc; h++) {
        std::vector<float> k(H), qq(H);
        memcpy(k.data(), kvv.data() + (size_t)h * H, sizeof(float) * H);
        memcpy(qq.data(), stream_.data() + (size_t)h * H, sizeof(float) * H);

        rms_norm(k.data(), H, wk.data() + (size_t)h * H, cfg_.norm_eps);
        rms_norm(qq.data(), H, wq.data() + (size_t)h * H, cfg_.norm_eps);

        float s = 0.0f;
        for (int d = 0; d < H; d++) s += k[d] * qq[d];
        s /= sq;

        const float mag = sqrtf(std::max(fabsf(s), 1e-6f));
        const float gate = sigmoid_f((s >= 0.0f ? 1.0f : -1.0f) * mag);
        dbg_stage_scalar("engram_gate-" + std::to_string(il), gate);
        for (int d = 0; d < H; d++) stream_[(size_t)h * H + d] += gate * value[d];
    }
    dbg_stage("engram_out-" + std::to_string(il), stream_.data(), (int64_t)hc * H);
}

// ------------------------------------------------------------------
// forward
// ------------------------------------------------------------------
const float* Session::forward(int token, bool compute_logits) {
    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int64_t t = pos_;
    logits_valid_ = false;

    auto tf0 = std::chrono::steady_clock::now();
    m_.embed(token, emb_.data());
    for (int h = 0; h < hc; h++) {
        memcpy(stream_.data() + (size_t)h * H, emb_.data(), sizeof(float) * H);
    }
    dbg_stage("hc_init", stream_.data(), (int64_t)hc * H);

    // layer 0 consumes a one-hot pre-mix selecting the first copy
    std::vector<float> pre_mix(hc, 0.0f);
    pre_mix[0] = 1.0f;

    for (int il = 0; il < cfg_.n_layers; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";

        auto t0 = std::chrono::steady_clock::now();
        if (cfg_.engram_index_of_layer(il) >= 0) {
            engram_add(il, token, t);
        }
        auto t1 = std::chrono::steady_clock::now();
        prof_engram_ += std::chrono::duration<double>(t1 - t0).count();

        // ---- attention sublayer ----
        memcpy(residual_.data(), stream_.data(), sizeof(float) * hc * H);

        t0 = std::chrono::steady_clock::now();
        hc_mixes(p + "hc_attn_fn.weight", il);   // attn pre is consumed by this layer's FFN
        std::vector<float> attn_pre = pre_;
        std::vector<float> post_a = post_, comb_a = comb_;

        hc_pre_collapse(stream_.data(), pre_mix.data(), cur_.data());
        dbg_stage("hc_attn_pre-" + std::to_string(il), cur_.data(), H);
        {
            std::vector<float> w(H);
            tensor_vec(p + "attn_norm.weight", w.data());
            rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
        }
        dbg_stage("attn_norm-" + std::to_string(il), cur_.data(), H);

        t1 = std::chrono::steady_clock::now();
        prof_hc_ += std::chrono::duration<double>(t1 - t0).count();
        t0 = std::chrono::steady_clock::now();
        attention(il, cur_.data(), t);           // writes cur_ with the attention output
        t1 = std::chrono::steady_clock::now();
        prof_attn_ += std::chrono::duration<double>(t1 - t0).count();

        hc_post_mix(cur_.data(), post_a.data(), comb_a.data());
        dbg_stage("hc_attn_post-" + std::to_string(il), stream_.data(), (int64_t)hc * H);

        memcpy(residual_.data(), stream_.data(), sizeof(float) * hc * H);

        // ---- FFN sublayer ----
        hc_mixes(p + "hc_ffn_fn.weight", il);    // ffn pre becomes the next layer's pre-mix
        std::vector<float> next_pre = pre_;
        std::vector<float> post_f = post_, comb_f = comb_;

        hc_pre_collapse(stream_.data(), attn_pre.data(), cur_.data());
        dbg_stage("hc_ffn_pre-" + std::to_string(il), cur_.data(), H);
        {
            std::vector<float> w(H);
            tensor_vec(p + "ffn_norm.weight", w.data());
            rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
        }
        dbg_stage("ffn_norm-" + std::to_string(il), cur_.data(), H);

        t0 = std::chrono::steady_clock::now();
        moe_ffn(il, cur_.data(), tmp2_.data());
        t1 = std::chrono::steady_clock::now();
        prof_moe_ += std::chrono::duration<double>(t1 - t0).count();

        hc_post_mix(tmp2_.data(), post_f.data(), comb_f.data());
        dbg_stage("l_last-" + std::to_string(il), stream_.data(), (int64_t)hc * H);

        pre_mix = std::move(next_pre);
    }

    if (compute_logits) {
        // This terminal branch does not publish KV, compressed or Engram state.
        // The last layer's un-consumed FFN pre-mix collapses the stream.
        hc_pre_collapse(stream_.data(), pre_mix.data(), cur_.data());
        dbg_stage("hc_out", cur_.data(), H);
        {
            std::vector<float> w(H);
            tensor_vec("output_norm.weight", w.data());
            rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
        }
        dbg_stage("result_norm", cur_.data(), H);

        TensorRef outw = m_.tensor("output.weight");
        NT_CHECK(outw.valid(), "missing output.weight");
        if (!nt::ds41::gpu_lm_head(cur_.data(), logits_.data(), cfg_.vocab_size, H)) {
            m_.matvec(outw, cur_.data(), logits_.data());
        }
        dbg_stage("result_output", logits_.data(), cfg_.vocab_size);
        logits_valid_ = true;
        ++head_evaluations_;
    }

    if (prof_) {
        auto tf1 = std::chrono::steady_clock::now();
        prof_other_ += std::chrono::duration<double>(tf1 - tf0).count()
                     - (prof_moe_ + prof_attn_ + prof_engram_ + prof_hc_ - prof_prev_sum_);
        prof_prev_sum_ = prof_moe_ + prof_attn_ + prof_engram_ + prof_hc_;
        prof_forwards_++;
    }
    history_.push_back(token);
    pos_++;
    return logits_valid_ ? logits_.data() : nullptr;
}

// ============================================================
// sampling + CLI
// ============================================================

static int sample_token(const float* logits, int n, const std::vector<int32_t>& history,
                        const GenOptions& opt, std::mt19937_64& rng,
                        std::vector<std::pair<float, int>>& scratch) {
    scratch.resize(n);
    for (int i = 0; i < n; i++) scratch[i] = { logits[i], i };

    // repetition penalty over the recent window
    if (opt.repeat_penalty > 1.0f && !history.empty()) {
        const int w = std::min<int>((int)history.size(), std::max(1, opt.repeat_window));
        for (int k = (int)history.size() - w; k < (int)history.size(); k++) {
            const int tok = history[k];
            if (tok < 0 || tok >= n) continue;
            float& l = scratch[tok].first;
            l = (l > 0.0f) ? (l / opt.repeat_penalty) : (l * opt.repeat_penalty);
        }
    }

    if (opt.temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (scratch[i].first > scratch[best].first) best = i;
        return best;
    }

    // top-k
    int k = (opt.top_k > 0) ? std::min(opt.top_k, n) : n;
    std::partial_sort(scratch.begin(), scratch.begin() + k, scratch.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    scratch.resize(k);

    // temperature + softmax
    const float inv_t = 1.0f / opt.temperature;
    float mx = scratch[0].first;
    double sum = 0.0;
    for (auto& s : scratch) {
        s.first = expf((s.first - mx) * inv_t);
        sum += s.first;
    }

    // top-p
    double acc = 0.0;
    int keep = k;
    for (int i = 0; i < k; i++) {
        acc += scratch[i].first / sum;
        if (acc >= opt.top_p) { keep = i + 1; break; }
    }

    double r = std::generate_canonical<double, 53>(rng) * (acc > 0 ? acc : 1.0);
    double c = 0.0;
    for (int i = 0; i < keep; i++) {
        c += scratch[i].first / sum;
        if (r <= c) return scratch[i].second;
    }
    return scratch[0].second;
}

int run_cli(const std::string& model_path, const GenOptions& opt) {
    // quick architecture check before committing to the full model load
    {
        GGUFLoader peek;
        if (!peek.load(model_path)) {
            return -1;
        }
        if (peek.kv_string("general.architecture", "") != "deepseek41") {
            return -1;
        }
    }

    Model model;
    std::string err;
    fprintf(stderr, "Loading DeepSeek-V4.1 model: %s\n", model_path.c_str());
    if (!model.load(model_path, err)) {
        fprintf(stderr, "ds41: failed to load model: %s\n", err.c_str());
        return 1;
    }

    const Config& cfg = model.cfg();
    fprintf(stderr, "deepseek41: %d layers, hidden %d, heads %d, head_dim %d, experts %d/%d, "
            "engram layers %zu, hc %d\n",
            cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.n_expert_used, cfg.n_expert,
            cfg.engram_layer_ids.size(), cfg.hc_mult);

    Tokenizer tok;
    tok.init(model.loader().vocab(), cfg.bos_id, cfg.eos_id);

    {
        TensorRef ow = model.tensor("output.weight");
        if (ow.valid() && ow.dtype == DType::BF16) {
            nt::ds41::gpu_lm_head_init(ow.data, ow.rows(), ow.ne0());
        }
    }

    Session session(model);

    std::vector<int> tokens;
    if (!opt.token_ids.empty()) {
        tokens = opt.token_ids;
    } else {
        tokens = tok.encode(opt.prompt, /*add_bos=*/true);
    }
    if (tokens.empty()) tokens.push_back(cfg.bos_id);
    if (opt.ctx > 0 && (int)tokens.size() >= opt.ctx) {
        tokens.resize(opt.ctx - 1);
    }

    if (getenv("DS41_DUMP_TOKENS")) {
        fprintf(stderr, "tokens:");
        for (int tk : tokens) fprintf(stderr, " %d", tk);
        fprintf(stderr, "\ndecoded:");
        for (int tk : tokens) fprintf(stderr, "[%s]", tok.decode_token(tk).c_str());
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "prompt: %zu tokens\n", tokens.size());

    // Preserve full stage-sum diagnostics unless the caller explicitly tests
    // the Session API. DS41_PREFILL_ALL_LOGITS=1 selects the old CLI schedule.
    const char* all_logits_env = getenv("DS41_PREFILL_ALL_LOGITS");
    const bool all_logits = session.dbg_enabled() ||
                            (all_logits_env && atoi(all_logits_env) != 0);
    const auto prefill_start = std::chrono::steady_clock::now();
    const float* logits = nullptr;
    for (size_t i = 0; i < tokens.size(); i++) {
        logits = session.forward(tokens[i], all_logits || i + 1 == tokens.size());
    }
    const double prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_start).count();

    if (session.prof_enabled()) {
        fprintf(stderr, "[prefill] tokens=%zu heads=%d mode=%s seconds=%.9f\n",
                tokens.size(), session.head_evaluations(), all_logits ? "all" : "last", prefill_seconds);
        fprintf(stderr, "[prof] moe=%.2fs attn=%.2fs engram=%.2fs hc=%.2fs other=%.2fs forwards=%d",
                session.prof_moe(), session.prof_attn(), session.prof_engram(), session.prof_hc(),
                session.prof_other(), session.prof_forwards());
        fprintf(stderr, "[prof-attn] qkv=%.2fs rope=%.2fs comp=%.2fs loop=%.2fs tail=%.2fs\n",
                session.prof_qkv(), session.prof_rope(), session.prof_comp(), session.prof_loop(), session.prof_tail());
        const auto& moe = session.moe_profile();
        fprintf(stderr, "[prof-moe] router=%.3fs routed_host=%.3fs shared_cpu=%.3fs gpu=%llu cpu=%llu overlapped=%llu\n",
                moe.router_seconds, moe.routed_seconds, moe.shared_seconds,
                (unsigned long long)moe.gpu_calls, (unsigned long long)moe.cpu_calls,
                (unsigned long long)moe.overlapped_calls);
    }

    if (session.dbg_enabled()) {
        std::vector<std::pair<std::string, double>> sums(session.dbg_sums().begin(), session.dbg_sums().end());
        std::sort(sums.begin(), sums.end());
        for (const auto& kv : sums) fprintf(stderr, "[stage] %s sum=%.6f\n", kv.first.c_str(), kv.second);
    }

    if (!opt.dump_logits.empty() && logits) {
        FILE* f = fopen(opt.dump_logits.c_str(), "wb");
        if (f) {
            fwrite(logits, sizeof(float), cfg.vocab_size, f);
            fclose(f);
            fprintf(stderr, "wrote logits (%d floats) to %s\n", cfg.vocab_size, opt.dump_logits.c_str());
        }
        std::vector<std::pair<float, int>> top(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; i++) top[i] = { logits[i], i };
        std::partial_sort(top.begin(), top.begin() + 8, top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < 8; i++) {
            fprintf(stderr, "  top%d: id=%d logit=%.6f token=%s\n", i, top[i].second, top[i].first,
                    tok.decode_token(top[i].second).c_str());
        }
    }

    if (getenv("DS41_GPU_SELFTEST2")) {
        TensorRef g = model.tensor("blk.0.ffn_gate_exps.weight");
        TensorRef u = model.tensor("blk.0.ffn_up_exps.weight");
        TensorRef d = model.tensor("blk.0.ffn_down_exps.weight");
        const int64_t H2 = cfg.hidden, ff2 = cfg.n_ff_exp;
        std::vector<float> xs(H2);
        for (int i = 0; i < H2; i++) xs[i] = sinf((float)i * 0.007f) * 0.5f;
        const int ex[2] = {3, 17};
        const float wt[2] = {0.7f, 0.3f};
        std::vector<float> y_gpu(H2, 0.0f), y_cpu(H2, 0.0f);
        nt::ds41::gpu_moe_experts(xs.data(), H2, g.data, u.data, d.data, ff2,
                                  ex, wt, 2, 7.0f, y_gpu.data());
        for (int k = 0; k < 2; k++) {
            std::vector<float> gate(ff2), up(ff2), h(ff2), down(H2);
            model.matvec_slice(g, xs.data(), gate.data(), (size_t)ex[k] * ff2, ff2);
            model.matvec_slice(u, xs.data(), up.data(), (size_t)ex[k] * ff2, ff2);
            for (int i = 0; i < ff2; i++) {
                float gv = std::min(gate[i], 7.0f);
                float uv = std::max(-7.0f, std::min(up[i], 7.0f));
                h[i] = (gv / (1.0f + expf(-gv))) * uv;
            }
            model.matvec_slice(d, h.data(), down.data(), (size_t)ex[k] * H2, H2);
            for (int i = 0; i < H2; i++) y_cpu[i] += wt[k] * down[i];
            fprintf(stderr, "[selftest2] expert %d: gate0=%.6f up0=%.6f h0=%.6f down0=%.6f\n",
                    ex[k], gate[0], up[0], h[0], down[0]);
        }
        float maxd = 0.0f;
        int argd = -1;
        for (int i = 0; i < H2; i++) {
            float dd = fabsf(y_cpu[i] - y_gpu[i]);
            if (dd > maxd) { maxd = dd; argd = i; }
        }
        fprintf(stderr, "[selftest2] full moe: max|d|=%.7f at %d  cpu=%.6f gpu=%.6f\n",
                maxd, argd, y_cpu[argd], y_gpu[argd]);
    }
    if (getenv("DS41_GPU_SELFTEST")) {
        TensorRef g = model.tensor("blk.0.ffn_gate_exps.weight");
        std::vector<float> xs(cfg.hidden), y_cpu(2304), y_gpu(2304, 0.0f);
        for (int i = 0; i < cfg.hidden; i++) xs[i] = sinf((float)i * 0.013f);
        model.matvec_slice(g, xs.data(), y_cpu.data(), 0, 2304);
        {
            // one-hot probes: y[0] should equal dequant(W[0][k])
            std::vector<float> tmp(cfg.hidden);
            std::vector<float> xo(cfg.hidden, 0.0f), yo(2304, 0.0f), yc(2304, 0.0f);
            model.dequant_row(g, 0, tmp.data());
            for (int probe : {0, 1, 15, 16, 31, 32, 33, 5119}) {
                xo.assign(cfg.hidden, 0.0f);
                xo[probe] = 1.0f;
                nt::ds41::gpu_gemv_expert(g.data, 0, 2304, cfg.hidden, xo.data(), yo.data());
                model.matvec_slice(g, xo.data(), yc.data(), 0, 2304);
                fprintf(stderr, "[selftest] onehot k=%d: cpu=%.6f gpu=%.6f ref=%.6f\n",
                        probe, yc[0], yo[0], tmp[probe]);
            }
        }
        {
            // Q8_0 + BF16 dot vs dequant reference on real tensors
            TensorRef qa = model.tensor("blk.0.attn_q_a.weight");
            std::vector<float> yq(1280), tmpr(qa.ne0());
            model.matvec_slice(qa, xs.data(), yq.data(), 0, 1280);
            float maxq = 0.0f;
            for (int r = 0; r < 256; r++) {
                model.dequant_row(qa, r, tmpr.data());
                float dot = 0.0f;
                for (int64_t i2 = 0; i2 < qa.ne0(); i2++) dot += tmpr[i2] * xs[i2];
                maxq = std::max(maxq, fabsf(dot - yq[r]));
            }
            fprintf(stderr, "[selftest] q8_0 dot max|d|=%.8f\n", maxq);

            TensorRef gi = model.tensor("blk.0.ffn_gate_inp.weight");
            std::vector<float> yg(384), tmpb(gi.ne0());
            model.matvec_slice(gi, xs.data(), yg.data(), 0, 384);
            float maxb = 0.0f;
            for (int r = 0; r < 384; r++) {
                model.dequant_row(gi, r, tmpb.data());
                float dot = 0.0f;
                for (int64_t i2 = 0; i2 < gi.ne0(); i2++) dot += tmpb[i2] * xs[i2];
                maxb = std::max(maxb, fabsf(dot - yg[r]));
            }
            fprintf(stderr, "[selftest] bf16 dot max|d|=%.8f\n", maxb);
        }
        if (!nt::ds41::gpu_gemv_expert(g.data, 0, 2304, cfg.hidden, xs.data(), y_gpu.data())) {
            fprintf(stderr, "[selftest] gpu_gemv_expert failed\n");
        } else {
            float maxd = 0.0f; double sumd = 0.0;
            for (int i = 0; i < 2304; i++) { float d = fabsf(y_cpu[i] - y_gpu[i]); maxd = std::max(maxd, d); sumd += d; }
            fprintf(stderr, "[selftest] gemv expert0: max|d|=%.8f mean|d|=%.8f  cpu0=%.6f gpu0=%.6f\n",
                    maxd, sumd / 2304, y_cpu[0], y_gpu[0]);
        }
    }
    if (getenv("DS41_MICROBENCH")) {
        // time a single 2304x5120 MXFP4 expert slice matvec repeatedly
        TensorRef g = model.tensor("blk.0.ffn_gate_exps.weight");
        std::vector<float> x(cfg.hidden, 0.01f), y(2304);
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 20; rep++) {
            model.matvec_slice(g, x.data(), y.data(), 0, 2304);
        }
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 20;
        fprintf(stderr, "[bench] mxpf4 2304x5120 matvec: %.1f us (%.1f GFLOP/s)\n",
                us, 2.0 * 2304 * 5120 / us / 1e3);
        std::vector<float> xf(cfg.hidden, 0.01f), yf(32768);
        TensorRef qb = model.tensor("blk.0.attn_q_b.weight");
        t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 20; rep++) {
            model.matvec_slice(qb, xf.data(), yf.data(), 0, 32768);
        }
        t1 = std::chrono::steady_clock::now();
        us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 20;
        fprintf(stderr, "[bench] q8_0 32768x1280 matvec: %.1f us (%.1f GFLOP/s)\n",
                us, 2.0 * 32768 * 1280 / us / 1e3);
    }

    std::mt19937_64 rng(opt.seed);
    std::vector<std::pair<float, int>> scratch;
    std::vector<int> generated;

    for (int step = 0; step < opt.max_tokens; step++) {
        if (!logits) break;
        int next = sample_token(logits, cfg.vocab_size, session.history_ref(), opt, rng, scratch);

        if (next == cfg.eos_id) break;
        generated.push_back(next);

        std::string piece = tok.decode_token(next);
        fputs(piece.c_str(), stdout);
        fflush(stdout);

        logits = session.forward(next);
    }
    fputs("\n", stdout);

    nt::ds41::gpu_print_stats();
    fprintf(stderr, "\ngenerated %zu tokens\n", generated.size());
    return 0;
}

} // namespace ds41
} // namespace nt
