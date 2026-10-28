// ============================================================
// GLM-5.3-Flash ("glm5-next") CPU forward pass.
// Mirrors llama.cpp src/models/glm5-next.cpp:
//   - mHC residual streams (hc=4, Sinkhorn comb)
//   - KDA layers: causal conv1d + gated delta rule (per-dim decay)
//   - MLA layers: nope-only MLA with absorbed wk_b/wv_b, latent cache
//   - DSA indexer with k-pool compression (top_k/kpool pools)
//   - MoE: sigmoid gating + bias selection, shared expert, swiglu clamp
// ============================================================

#include "glm5_forward.h"
#include "glm5_gpu.h"
#include "glm5_gpu2.h"
#include "../ds41/ds41_hc.h"
#include "../../inference/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <limits>
#include <mutex>
#include <random>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nt {
namespace glm5 {

namespace {

inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

float dot_plain_f32(const float* w, const float* x, int64_t n) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i),     _mm256_loadu_ps(x + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
    }
    __m256 a = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float acc = _mm_cvtss_f32(s);
    for (; i < n; i++) acc += w[i] * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) acc += w[i] * x[i];
    return acc;
#endif
}

} // namespace

// ============================================================
// Session
// ============================================================

Session::Session(Model& model, int ctx) : m_(model), cfg_(model.cfg()) {
    ctx_ = ctx > 0 ? ctx : 4096;

    recr_index_.assign(cfg_.n_layers, -1);
    mla_index_.assign(cfg_.n_layers, -1);
    for (int il = 0; il < cfg_.n_layers; il++) {
        if (cfg_.is_recr[il]) {
            recr_index_[il] = n_recr_++;
        } else {
            mla_index_[il] = n_mla_++;
        }
    }

    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int HD = cfg_.kda_head_dim;
    const int NH = cfg_.n_heads;
    const int DI = NH * HD;
    const int KL = cfg_.kv_lora_rank;

    conv_state_.assign(n_recr_, std::vector<float>(3 * (cfg_.d_conv - 1) * DI, 0.0f));
    ssm_state_.assign(n_recr_, std::vector<float>((size_t)NH * HD * HD, 0.0f));

    latent_.assign(n_mla_, std::vector<float>((size_t)ctx_ * KL, 0.0f));
    idx_k_.assign(n_mla_, std::vector<float>((size_t)ctx_ * cfg_.indexer_head_dim, 0.0f));
    idx_g_.assign(n_mla_, std::vector<float>((size_t)ctx_ * cfg_.indexer_head_dim, 0.0f));
    pool_k_.assign(n_mla_, std::vector<float>((size_t)(ctx_ / cfg_.indexer_kpool + 2) * cfg_.indexer_head_dim, 0.0f));

    wkb_f32_.assign(n_mla_, std::vector<float>());
    wvb_f32_.assign(n_mla_, std::vector<float>());

    logits_.assign(cfg_.vocab_size, 0.0f);
    emb_.assign(H, 0.0f);
    stream_.assign((size_t)hc * H, 0.0f);
    residual_.assign((size_t)hc * H, 0.0f);
    cur_.assign(H, 0.0f);
    out_.assign(H, 0.0f);
    tmp_.assign(H, 0.0f);
    tmp2_.assign(H, 0.0f);
    flat_norm_.assign((size_t)hc * H, 0.0f);
    mixes_.assign((2 + hc) * hc, 0.0f);
    pre_.assign(hc, 0.0f);
    post_.assign(hc, 0.0f);
    comb_.assign(hc * hc, 0.0f);

    xq_.assign(DI, 0.0f); xk_.assign(DI, 0.0f); xv_.assign(DI, 0.0f);
    cq_.assign(DI, 0.0f); ck_.assign(DI, 0.0f); cv_.assign(DI, 0.0f);
    fa_.assign(HD, 0.0f); g1_.assign(DI, 0.0f);
    beta_.assign(NH, 0.0f); decay_.assign(DI, 0.0f);
    cout_.assign(DI, 0.0f); gated_.assign(DI, 0.0f); ga_.assign(HD, 0.0f);
    ssm_a_.assign(NH, 0.0f); dt_b_.assign(DI, 0.0f); onorm_.assign(HD, 0.0f);

    qr_.assign(cfg_.q_lora_rank, 0.0f);
    q_.assign((size_t)NH * cfg_.qk_nope, 0.0f);
    kv_.assign(KL, 0.0f);
    q_abs_.assign((size_t)NH * KL, 0.0f);
    iq_.assign((size_t)cfg_.indexer_n_head * cfg_.indexer_head_dim, 0.0f);
    ik_.assign(cfg_.indexer_head_dim, 0.0f);
    ig_.assign(cfg_.indexer_head_dim, 0.0f);
    wproj_.assign(cfg_.indexer_n_head, 0.0f);
    attn_lat_.assign((size_t)NH * KL, 0.0f);
    attn_out_.assign((size_t)NH * cfg_.v_dim, 0.0f);
    score_.assign(ctx_ / cfg_.indexer_kpool + 2, 0.0f);
    visible_.assign(ctx_, 0.0f);
    ape_.assign((size_t)cfg_.indexer_head_dim * cfg_.indexer_kpool, 0.0f);

    moe_logits_.assign(cfg_.n_expert, 0.0f);
    moe_bias_.assign(cfg_.n_expert, 0.0f);
    gate_.assign(std::max(cfg_.n_ff_exp, cfg_.n_ff_dense), 0.0f);
    up_.assign(std::max(cfg_.n_ff_exp, cfg_.n_ff_dense), 0.0f);
    h_.assign(std::max(cfg_.n_ff_exp, cfg_.n_ff_dense), 0.0f);

    // dequantize absorbed wk_b / wv_b once per MLA layer
    for (int il = 0; il < cfg_.n_layers; il++) {
        int mi = mla_index_[il];
        if (mi < 0) continue;
        const std::string p = "blk." + std::to_string(il) + ".";
        TensorRef wkb = m_.tensor(p + "attn_k_b.weight");
        TensorRef wvb = m_.tensor(p + "attn_v_b.weight");
        NT_CHECK(wkb.valid() && wvb.valid(), "missing attn_k_b/attn_v_b");
        wkb_f32_[mi].assign((size_t)wkb.ne1() * wkb.ne[2] * wkb.ne0(), 0.0f);
        wvb_f32_[mi].assign((size_t)wvb.ne1() * wvb.ne[2] * wvb.ne0(), 0.0f);
        const int64_t rows_k = wkb.ne1() * wkb.ne[2];   // (c*H + h) rows of qk_nope
        for (int64_t r = 0; r < rows_k; r++) {
            m_.dequant_row(wkb, r, wkb_f32_[mi].data() + r * wkb.ne0());
        }
        const int64_t rows_v = wvb.ne1() * wvb.ne[2];   // (v*H + h) rows of kv_lora
        for (int64_t r = 0; r < rows_v; r++) {
            m_.dequant_row(wvb, r, wvb_f32_[mi].data() + r * wvb.ne0());
        }
    }
}

void Session::reset() {
    pos_ = 0;
    logits_valid_ = false;
    history_.clear();
    for (auto& v : conv_state_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : ssm_state_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : latent_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : idx_k_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : idx_g_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : pool_k_) std::fill(v.begin(), v.end(), 0.0f);
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

void Session::ln_norm(float* x, int64_t n, const float* w, const float* b, float eps) const {
    float mean = 0.0f;
    for (int64_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int64_t i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= (float)n;
    const float rstd = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; i++) x[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

void Session::matvec_name(const std::string& name, const float* x, float* y, int64_t rows) const {
    TensorRef t = m_.tensor(name);
    NT_CHECK(t.valid(), ("missing tensor: " + name).c_str());
    NT_CHECK(t.rows() >= rows, "matvec rows mismatch");
    if (glm5_gpu_enabled()) {
        if (t.dtype == DType::Q8_0 && glm5_q8_proj(t.data, x, y, rows, t.ne0())) return;
        if (t.dtype == DType::F32 && glm5_f32_proj(t.data, x, y, rows, t.ne0())) return;
    }
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
// mHC (same math as DeepSeek-V4.1 mHC; pre applied immediately)
// ------------------------------------------------------------------
void Session::hc_mixes(const std::string& fn_name, int il) {
    (void)il;
    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int hc_dim = hc * H;
    const int mix_dim = (2 + hc) * hc;

    memcpy(flat_norm_.data(), stream_.data(), sizeof(float) * hc_dim);
    rms_norm(flat_norm_.data(), hc_dim, nullptr, cfg_.norm_eps);

    TensorRef fn = m_.tensor(fn_name);
    NT_CHECK(fn.valid(), ("missing tensor: " + fn_name).c_str());
    NT_CHECK(fn.ne0() == hc_dim && fn.rows() == mix_dim, "hc_fn shape mismatch");
    m_.matvec(fn, flat_norm_.data(), mixes_.data());

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

    nt::ds41::detail::hc_sinkhorn(mixes_.data() + 2 * hc, base + 2 * hc, scale[2],
                                  comb_.data(), hc, eps, cfg_.hc_sinkhorn_iters);
}

// ------------------------------------------------------------------
// KDA layer: causal conv1d + gated delta rule (Kimi Delta Attention)
// ------------------------------------------------------------------
void Session::kda_layer(int il, const float* x, float* out) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const int H = cfg_.hidden;
    const int HD = cfg_.kda_head_dim;
    const int NH = cfg_.n_heads;
    const int DI = NH * HD;
    const int cs = recr_index_[il];

    {
        // q/k/v + f_a + g_a + beta share the input x: one GPU sync for all six
        const void* mats[6];
        int64_t rows[6];
        TensorRef tq = m_.tensor(p + "attn_q.weight");
        TensorRef tk = m_.tensor(p + "attn_k.weight");
        TensorRef tv = m_.tensor(p + "attn_v.weight");
        TensorRef tfa = m_.tensor(p + "ssm_f_a.weight");
        TensorRef tga = m_.tensor(p + "ssm_g_a.weight");
        TensorRef tb = m_.tensor(p + "ssm_beta.weight");
        mats[0] = tq.data; rows[0] = tq.rows();
        mats[1] = tk.data; rows[1] = tk.rows();
        mats[2] = tv.data; rows[2] = tv.rows();
        mats[3] = tfa.data; rows[3] = tfa.rows();
        mats[4] = tga.data; rows[4] = tga.rows();
        mats[5] = tb.data; rows[5] = tb.rows();
        std::vector<float> all((size_t)rows[0] + rows[1] + rows[2] + rows[3] + rows[4] + rows[5]);
        if (glm5_gpu_enabled() && tq.dtype == DType::Q8_0 && tk.dtype == DType::Q8_0 &&
            tv.dtype == DType::Q8_0 && tfa.dtype == DType::Q8_0 && tga.dtype == DType::Q8_0 &&
            tb.dtype == DType::Q8_0 &&
            glm5_q8_proj_many(mats, rows, 6, H, x, all.data())) {
            memcpy(xq_.data(), all.data(), sizeof(float) * DI);
            memcpy(xk_.data(), all.data() + DI, sizeof(float) * DI);
            memcpy(xv_.data(), all.data() + 2 * DI, sizeof(float) * DI);
            memcpy(fa_.data(), all.data() + 3 * DI, sizeof(float) * HD);
            memcpy(ga_.data(), all.data() + 3 * DI + HD, sizeof(float) * HD);
            memcpy(beta_.data(), all.data() + 3 * DI + 2 * HD, sizeof(float) * NH);
        } else {
            m_.matvec(m_.tensor(p + "attn_q.weight"), x, xq_.data());
            m_.matvec(m_.tensor(p + "attn_k.weight"), x, xk_.data());
            m_.matvec(m_.tensor(p + "attn_v.weight"), x, xv_.data());
            matvec_name(p + "ssm_f_a.weight", x, fa_.data(), HD);
            matvec_name(p + "ssm_g_a.weight", x, ga_.data(), HD);
            matvec_name(p + "ssm_beta.weight", x, beta_.data(), NH);
        }
    }

    // causal depthwise conv1d + SiLU, with rolling (d_conv-1) state
    {
        const float* srcs[3] = { xq_.data(), xk_.data(), xv_.data() };
        float*       dsts[3] = { cq_.data(), ck_.data(), cv_.data() };
        const char* names[3] = { "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight" };
        for (int qkv = 0; qkv < 3; qkv++) {
            TensorRef cw = m_.tensor(p + names[qkv]);
            NT_CHECK(cw.valid() && cw.dtype == DType::F32, "ssm_conv1d must be F32");
            const float* w = reinterpret_cast<const float*>(cw.data);   // [DI channels][d_conv] (window fastest)
            float* st = conv_state_[cs].data() + (size_t)qkv * (cfg_.d_conv - 1) * DI;
            const float* src = srcs[qkv];
            float* dst = dsts[qkv];
            const int P = cfg_.d_conv - 1;
            const int DC = cfg_.d_conv;
            parallel_for(DI, m_.n_threads, [&](int64_t c, int) {
                float acc = w[c * DC + P] * src[c];
                for (int s = 0; s < P; s++) acc += w[(size_t)c * DC + s] * st[(size_t)s * DI + c];
                dst[c] = acc / (1.0f + expf(-acc));
            });
            for (int s = 0; s + 1 < P; s++) {
                memcpy(st + (size_t)s * DI, st + (size_t)(s + 1) * DI, sizeof(float) * DI);
            }
            if (P > 0) memcpy(st + (size_t)(P - 1) * DI, src, sizeof(float) * DI);
        }
    }

    // decay gate g1: sigmoid(-(f_b(f_a(x)) + dt_b) * A) * lower_bound
    {
        matvec_name(p + "ssm_f_b.weight", fa_.data(), g1_.data(), DI);
        tensor_vec(p + "ssm_dt.bias", dt_b_.data());
        tensor_vec(p + "ssm_a", ssm_a_.data());
        for (int i = 0; i < DI; i++) g1_[i] += dt_b_[i];
        const float lb = cfg_.kda_gate_lower_bound;
        if (lb > -std::numeric_limits<float>::infinity()) {
            for (int h = 0; h < NH; h++) {
                const float a = ssm_a_[h];
                for (int i = 0; i < HD; i++) {
                    const float g = sigmoid_f(-(g1_[(size_t)h * HD + i] * a)) * lb;
                    decay_[(size_t)h * HD + i] = expf(g);
                }
            }
        } else {
            for (int h = 0; h < NH; h++) {
                const float a = ssm_a_[h];
                for (int i = 0; i < HD; i++) {
                    const float g = log1pf(expf(g1_[(size_t)h * HD + i])) * a;
                    decay_[(size_t)h * HD + i] = expf(g);
                }
            }
        }
        for (int h = 0; h < NH; h++) beta_[h] = sigmoid_f(beta_[h]);
    }

    // l2 norm per head (FLA style: x / sqrt(||x||^2 + eps))
    {
        constexpr float l2_eps = 1e-6f;
        auto norm_heads = [&](float* v) {
            for (int h = 0; h < NH; h++) {
                float* vh = v + (size_t)h * HD;
                float ss = 0.0f;
                for (int i = 0; i < HD; i++) ss += vh[i] * vh[i];
                const float s = 1.0f / sqrtf(ss + l2_eps);
                for (int i = 0; i < HD; i++) vh[i] *= s;
            }
        };
        norm_heads(cq_.data());
        norm_heads(ck_.data());
    }

    // gated delta rule; state S[i][j] stored as s[j*HD + i]
    {
        float* S = ssm_state_[cs].data();
        const float oosqrt = 1.0f / sqrtf((float)HD);
        parallel_for(NH, m_.n_threads, [&](int64_t h, int) {
            const float* qh = cq_.data() + (size_t)h * HD;
            const float* kh = ck_.data() + (size_t)h * HD;
            const float* vh = cv_.data() + (size_t)h * HD;
            const float* dec = decay_.data() + (size_t)h * HD;
            float* sh = S + (size_t)h * HD * HD;
            float* oh = cout_.data() + (size_t)h * HD;
            float delta[256];
            for (int j = 0; j < HD; j++) {
                float* row = sh + (size_t)j * HD;
                for (int i = 0; i < HD; i++) row[i] *= dec[i];
            }
            for (int j = 0; j < HD; j++) {
                float* row = sh + (size_t)j * HD;
                float sum = 0.0f;
                for (int i = 0; i < HD; i++) sum += row[i] * kh[i];
                delta[j] = (vh[j] - sum) * beta_[h];
            }
            for (int j = 0; j < HD; j++) {
                float* row = sh + (size_t)j * HD;
                const float dj = delta[j];
                for (int i = 0; i < HD; i++) row[i] += kh[i] * dj;
            }
            for (int j = 0; j < HD; j++) {
                float* row = sh + (size_t)j * HD;
                float sum = 0.0f;
                for (int i = 0; i < HD; i++) sum += row[i] * qh[i];
                oh[j] = sum * oosqrt;
            }
        });
    }

    // output gate: RMSNorm(o) * sigmoid(g2), then wo
    {
        matvec_name(p + "ssm_g_b.weight", ga_.data(), g1_.data(), DI);  // reuse g1_ as g2 buffer
        tensor_vec(p + "ssm_norm.weight", onorm_.data());
        for (int h = 0; h < NH; h++) {
            float* oh = cout_.data() + (size_t)h * HD;
            rms_norm(oh, HD, onorm_.data(), cfg_.norm_eps);
            for (int i = 0; i < HD; i++) {
                oh[i] *= sigmoid_f(g1_[(size_t)h * HD + i]);
            }
        }
        matvec_name(p + "attn_output.weight", cout_.data(), out, H);
    }
}

// ------------------------------------------------------------------
// MLA (nope-only) layer with k-pool DSA indexer
// ------------------------------------------------------------------
void Session::mla_layer(int il, const float* x, int64_t t, float* out) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const int H = cfg_.hidden;
    const int NH = cfg_.n_heads;
    const int KL = cfg_.kv_lora_rank;
    const int QK = cfg_.qk_nope;
    const int VD = cfg_.v_dim;
    const int mi = mla_index_[il];
    const int idx_dim = cfg_.indexer_head_dim;
    const int idx_heads = cfg_.indexer_n_head;
    const int kpool = cfg_.indexer_kpool;

    // query latent and heads
    matvec_name(p + "attn_q_a.weight", x, qr_.data(), cfg_.q_lora_rank);
    {
        std::vector<float> w(cfg_.q_lora_rank);
        tensor_vec(p + "attn_q_a_norm.weight", w.data());
        rms_norm(qr_.data(), cfg_.q_lora_rank, w.data(), cfg_.norm_eps);
    }
    m_.matvec(m_.tensor(p + "attn_q_b.weight"), qr_.data(), q_.data());

    // compressed latent (k-only cache, absorbed weights) + indexer key/gate/proj in one sync
    matvec_name(p + "attn_kv_a_mqa.weight", x, kv_.data(), KL);
    {
        std::vector<float> w(KL);
        tensor_vec(p + "attn_kv_a_norm.weight", w.data());
        rms_norm(kv_.data(), KL, w.data(), cfg_.norm_eps);
    }
    memcpy(latent_[mi].data() + (size_t)t * KL, kv_.data(), sizeof(float) * KL);

    // absorbed query: q_abs[h][c] = sum_k wk_b[k,c,h] * q[h*QK+k]
    {
        bool have_qabs = false;
        if (glm5_gpu_enabled()) {
            TensorRef wkb_t = m_.tensor(p + "attn_k_b.weight");
            if (wkb_t.valid() && wkb_t.dtype == DType::Q8_0) {
                have_qabs = glm5_q8_proj_grouped(wkb_t.data, q_.data(), q_abs_.data(), KL, QK, NH);
            }
        }
        if (!have_qabs) {
            const float* wkb = wkb_f32_[mi].data();   // [h][c*QK + k]
            for (int h = 0; h < NH; h++) {
                const float* qh = q_.data() + (size_t)h * QK;
                float* qa = q_abs_.data() + (size_t)h * KL;
                const float* wh = wkb + (size_t)h * (KL * QK);
                for (int c = 0; c < KL; c++) {
                    qa[c] = dot_plain_f32(wh + (size_t)c * QK, qh, QK);
                }
            }
        }
    }

    // indexer keys / gate / query / weights
    matvec_name(p + "indexer.attn_k.weight", x, ik_.data(), idx_dim);
    {
        std::vector<float> w(idx_dim), b(idx_dim);
        tensor_vec(p + "indexer.k_norm.weight", w.data());
        tensor_vec(p + "indexer.k_norm.bias", b.data());
        ln_norm(ik_.data(), idx_dim, w.data(), b.data(), cfg_.ln_eps);
    }
    matvec_name(p + "indexer_compressor_gate.weight", x, ig_.data(), idx_dim);
    m_.matvec(m_.tensor(p + "indexer.attn_q_b.weight"), qr_.data(), iq_.data());
    matvec_name(p + "indexer.proj.weight", x, wproj_.data(), idx_heads);
    {
        const float s = 1.0f / sqrtf((float)(idx_dim * idx_heads));
        for (int h = 0; h < idx_heads; h++) wproj_[h] *= s;
    }

    memcpy(idx_k_[mi].data() + (size_t)t * idx_dim, ik_.data(), sizeof(float) * idx_dim);
    memcpy(idx_g_[mi].data() + (size_t)t * idx_dim, ig_.data(), sizeof(float) * idx_dim);

    // k-pool: softmax(gate + ape) over the kpool members, weighted sum of keys
    if ((t + 1) % kpool == 0) {
        const int64_t pi = (t + 1) / kpool - 1;
        if ((size_t)(pi + 1) * idx_dim <= pool_k_[mi].size()) {
            tensor_vec(p + "indexer_compressor_ape.weight", ape_.data());
            const int64_t base = pi * kpool;
            float* pooled = pool_k_[mi].data() + (size_t)pi * idx_dim;
            for (int d = 0; d < idx_dim; d++) {
                float mx = -std::numeric_limits<float>::infinity();
                float lg[8];
                for (int m = 0; m < kpool; m++) {
                    lg[m] = idx_g_[mi][(size_t)(base + m) * idx_dim + d] + ape_[(size_t)m * idx_dim + d];
                    mx = std::max(mx, lg[m]);
                }
                float sum = 0.0f;
                float pr[8];
                for (int m = 0; m < kpool; m++) { pr[m] = expf(lg[m] - mx); sum += pr[m]; }
                float acc = 0.0f;
                for (int m = 0; m < kpool; m++) {
                    acc += (pr[m] / sum) * idx_k_[mi][(size_t)(base + m) * idx_dim + d];
                }
                pooled[d] = acc;
            }
        }
    }

    // pool scores and selection
    const int64_t n_pool_struct = t / kpool + 1;
    const int64_t n_top_pool = std::min<int64_t>(n_pool_struct, cfg_.indexer_top_k / kpool);
    {
        const float* pk = pool_k_[mi].data();
        for (int64_t pi = 0; pi < n_pool_struct; pi++) {
            const float* key = pk + (size_t)pi * idx_dim;
            float s = 0.0f;
            for (int h = 0; h < idx_heads; h++) {
                float d = dot_plain_f32(iq_.data() + (size_t)h * idx_dim, key, idx_dim);
                if (d < 0.0f) continue;
                s += d * wproj_[h];
            }
            score_[pi] = s;
        }
        std::vector<int64_t> order(n_pool_struct);
        for (int64_t i = 0; i < n_pool_struct; i++) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + n_top_pool, order.end(),
                          [&](int64_t a, int64_t b) {
                              if (score_[a] != score_[b]) return score_[a] > score_[b];
                              return a < b;
                          });
        std::fill(visible_.begin(), visible_.begin() + t + 1, 0.0f);
        for (int64_t k = 0; k < n_top_pool; k++) {
            const int64_t pi = order[k];
            const int64_t first = pi * kpool;
            for (int m = 0; m < kpool; m++) {
                const int64_t cell = first + m;
                if (cell <= t) visible_[cell] = 1.0f;
            }
        }
    }

    // sparse attention over the latents, wv_b applied after
    {
        const float scale = 1.0f / sqrtf((float)QK);
        const float* lat = latent_[mi].data();
        const float* wvb = wvb_f32_[mi].data();   // [h][v*KL + c]
        parallel_for(NH, m_.n_threads, [&](int64_t h, int) {
            const float* qa = q_abs_.data() + (size_t)h * KL;
            float mx = -std::numeric_limits<float>::infinity();
            std::vector<float> sc(t + 1);
            for (int64_t tt = 0; tt <= t; tt++) {
                float s = -std::numeric_limits<float>::infinity();
                if (visible_[tt] > 0.0f) {
                    s = dot_plain_f32(qa, lat + (size_t)tt * KL, KL) * scale;
                    mx = std::max(mx, s);
                }
                sc[tt] = s;
            }
            float sum = 0.0f;
            for (int64_t tt = 0; tt <= t; tt++) {
                if (visible_[tt] > 0.0f) { sc[tt] = expf(sc[tt] - mx); sum += sc[tt]; }
            }
            const float inv = 1.0f / sum;
            float* al = attn_lat_.data() + (size_t)h * KL;
            std::fill(al, al + KL, 0.0f);
            for (int64_t tt = 0; tt <= t; tt++) {
                if (visible_[tt] > 0.0f) {
                    const float wgt = sc[tt] * inv;
                    const float* row = lat + (size_t)tt * KL;
                    for (int c = 0; c < KL; c++) al[c] += wgt * row[c];
                }
            }
            const float* wh = wvb + (size_t)h * (VD * KL);
            float* oh = attn_out_.data() + (size_t)h * VD;
            for (int v = 0; v < VD; v++) {
                oh[v] = dot_plain_f32(wh + (size_t)v * KL, al, KL);
            }
        });
    }

    matvec_name(p + "attn_output.weight", attn_out_.data(), out, H);
}

// ------------------------------------------------------------------
// FFN: dense SwiGLU (leading blocks) and sigmoid-gated MoE + shared
// ------------------------------------------------------------------
void Session::dense_ffn(int il, const float* x, float* out) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const int H = cfg_.hidden;
    const int ff = cfg_.n_ff_dense;
    const float lim = cfg_.swiglu_clamp_shexp[il];

    matvec_name(p + "ffn_gate.weight", x, gate_.data(), ff);
    matvec_name(p + "ffn_up.weight", x, up_.data(), ff);
    for (int i = 0; i < ff; i++) {
        float g = gate_[i];
        float u = up_[i];
        if (lim > 1e-6f) {
            g = std::min(g, lim);
            u = std::max(-lim, std::min(u, lim));
        }
        h_[i] = (g / (1.0f + expf(-g))) * u;
    }
    matvec_name(p + "ffn_down.weight", h_.data(), out, H);
}

void Session::select_experts(int il, const float* x, std::vector<int>& idx, std::vector<float>& wts) {
    const int ne = cfg_.n_expert;
    const int used = cfg_.n_expert_used;
    const std::string p = "blk." + std::to_string(il) + ".";

    TensorRef gate_inp = m_.tensor(p + "ffn_gate_inp.weight");
    NT_CHECK(gate_inp.valid(), "missing ffn_gate_inp");

    std::vector<float>& logits = moe_logits_;
    if (glm5_gpu_enabled() && gate_inp.dtype == DType::F32) {
        if (!glm5_f32_proj(gate_inp.data, x, logits.data(), ne, gate_inp.ne0())) {
            m_.matvec(gate_inp, x, logits.data());
        }
    } else {
        m_.matvec(gate_inp, x, logits.data());
    }

    std::vector<float>& bias = moe_bias_;
    if (m_.tensor(p + "exp_probs_b.bias").valid()) {
        tensor_vec(p + "exp_probs_b.bias", bias.data());
    } else {
        std::fill(bias.begin(), bias.end(), 0.0f);
    }

    std::vector<float> probs(ne);
    for (int e = 0; e < ne; e++) probs[e] = sigmoid_f(logits[e]);

    idx.resize(ne);
    for (int e = 0; e < ne; e++) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + used, idx.end(),
        [&](int a, int b) { return (probs[a] + bias[a]) > (probs[b] + bias[b]); });

    {
        static FILE* trace_file = []() -> FILE* {
            const char* path = getenv("GLM5_TRACE_EXPERTS");
            return path ? fopen(path, "a") : nullptr;
        }();
        if (trace_file) {
            for (int k = 0; k < used; k++) fprintf(trace_file, "%d(%.4f) ", idx[k], wts[k]);
            fprintf(trace_file, "\n");
            fflush(trace_file);
        }
    }

    wts.resize(used);
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
    idx.resize(used);

    {
        static const bool dbg_ids = getenv("GLM5_DBG_IDS") != nullptr;
        if (dbg_ids) {
            fprintf(stderr, "[cpu-ids] il=%d:", il);
            for (int k = 0; k < used; k++) fprintf(stderr, " %d(%.4f)", idx[k], wts[k]);
            fprintf(stderr, "\n");
        }
    }
}

bool Session::run_experts(int il, const float* x, const std::vector<int>& idx,
                          const std::vector<float>& wts, float* out) {
    const int H = cfg_.hidden;
    const int ff = cfg_.n_ff_exp;
    const int used = cfg_.n_expert_used;
    const std::string p = "blk." + std::to_string(il) + ".";
    const float limit = cfg_.swiglu_clamp_exp[il];

    TensorRef gate_exps = m_.tensor(p + "ffn_gate_exps.weight");
    TensorRef up_exps   = m_.tensor(p + "ffn_up_exps.weight");
    TensorRef down_exps = m_.tensor(p + "ffn_down_exps.weight");
    NT_CHECK(gate_exps.valid() && up_exps.valid() && down_exps.valid(), "missing expert tensors");

    const float slim = cfg_.swiglu_clamp_shexp[il];
    const int ffs = ff * cfg_.n_expert_shared;

    const auto compute_shared = [&]() {
        TensorRef gs = m_.tensor(p + "ffn_gate_shexp.weight");
        TensorRef us = m_.tensor(p + "ffn_up_shexp.weight");
        TensorRef ds = m_.tensor(p + "ffn_down_shexp.weight");
        NT_CHECK(gs.valid() && us.valid() && ds.valid(), "missing shared expert tensors");
        m_.matvec(gs, x, gate_.data());
        m_.matvec(us, x, up_.data());
        for (int i = 0; i < ffs; i++) {
            float g = gate_[i];
            float u = up_[i];
            if (slim > 1e-6f) {
                g = std::min(g, slim);
                u = std::max(-slim, std::min(u, slim));
            }
            h_[i] = (g / (1.0f + expf(-g))) * u;
        }
        m_.matvec(ds, h_.data(), tmp_.data());
    };

    static const bool moe_gpu_off = getenv("GLM5_GPU_MOE_OFF") != nullptr;
    if (!moe_gpu_off && glm5_gpu_enabled() && gate_exps.dtype == DType::Q8_0 &&
        up_exps.dtype == DType::Q8_0 && down_exps.dtype == DType::Q8_0 &&
        glm5_gpu_experts(x, H, gate_exps.data, up_exps.data, down_exps.data,
                         ff, idx.data(), wts.data(), used, limit, tmp2_.data(),
                         compute_shared)) {
        for (int i = 0; i < H; i++) out[i] = tmp2_[i] + tmp_[i];
        if (getenv("GLM5_DBG_IDS") && il == 3) {
            double ss = 0;
            for (int i = 0; i < H; i++) ss += (double)tmp2_[i] * tmp2_[i];
            fprintf(stderr, "[cpu-moe] routed il=%d ss=%.6f head: %.6f %.6f %.6f\n", il, ss, tmp2_[0], tmp2_[1], tmp2_[2]);
            double sss = 0;
            for (int i = 0; i < H; i++) sss += (double)tmp_[i] * tmp_[i];
            fprintf(stderr, "[cpu-moe] shared il=%d ss=%.6f head: %.6f %.6f %.6f\n", il, sss, tmp_[0], tmp_[1], tmp_[2]);
        }
        return true;
    }

    std::fill(out, out + H, 0.0f);

    for (int k = 0; k < used; k++) {
        const int e = idx[k];
        m_.matvec_slice(gate_exps, x, gate_.data(), (size_t)e * ff, ff);
        m_.matvec_slice(up_exps, x, up_.data(), (size_t)e * ff, ff);
        for (int i = 0; i < ff; i++) {
            float g = gate_[i];
            float u = up_[i];
            if (limit > 1e-6f) {
                g = std::min(g, limit);
                u = std::max(-limit, std::min(u, limit));
            }
            h_[i] = (g / (1.0f + expf(-g))) * u;
        }
        m_.matvec_slice(down_exps, h_.data(), tmp_.data(), (size_t)e * H, H);
        for (int i = 0; i < H; i++) out[i] += wts[k] * tmp_[i];
    }

    // shared expert
    compute_shared();
    for (int i = 0; i < H; i++) out[i] += tmp_[i];
    return true;
}

void Session::moe_ffn(int il, const float* x, float* out) {
    std::vector<int> idx;
    std::vector<float> wts;
    select_experts(il, x, idx, wts);
    run_experts(il, x, idx, wts, out);
}

// ------------------------------------------------------------------
// forward
// ------------------------------------------------------------------
const float* Session::forward(int token, bool compute_logits) {
    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int64_t t = pos_;
    logits_valid_ = false;

    m_.embed(token, emb_.data());
    for (int h = 0; h < hc; h++) {
        memcpy(stream_.data() + (size_t)h * H, emb_.data(), sizeof(float) * H);
    }

    for (int il = 0; il < cfg_.n_layers; il++) {
        static const int max_layers = [] {
            const char* e = getenv("GLM5_MAX_LAYERS");
            return e ? atoi(e) : 1000;
        }();
        if (il >= max_layers) break;
        const std::string p = "blk." + std::to_string(il) + ".";
        static const bool dbg_layers = getenv("GLM5_DBG_LAYERS") != nullptr;

        // attention sublayer
        memcpy(residual_.data(), stream_.data(), sizeof(float) * hc * H);
        hc_mixes(p + "hc_attn_fn.weight", il);
        std::vector<float> post_a = post_, comb_a = comb_;
        nt::ds41::detail::hc_collapse(stream_.data(), pre_.data(), cur_.data(),
                                      H, hc, true, m_.n_threads);
        {
            std::vector<float> w(H);
            tensor_vec(p + "attn_norm.weight", w.data());
            rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
        }
        if (cfg_.is_recr[il]) {
            kda_layer(il, cur_.data(), out_.data());
        } else {
            mla_layer(il, cur_.data(), t, out_.data());
        }
        if (dbg_layers && il == 3) {
            double ss = 0.0;
            for (int i = 0; i < H; i++) ss += (double)out_[i] * out_[i];
            fprintf(stderr, "[cpu-cp] mla_out t=%lld ss=%.6f head: %.6f %.6f %.6f\n",
                    (long long)t, ss, out_[0], out_[1], out_[2]);
        }
        nt::ds41::detail::hc_post_mix(out_.data(), residual_.data(), post_a.data(), comb_a.data(),
                                      stream_.data(), H, hc, true, m_.n_threads);

        // layer-ahead prefetch: router of layer il+1 on the post-attention stream
        static const int prefetch_on = [] {
            const char* e = getenv("GLM5_PREFETCH");
            return e ? atoi(e) : 1;   // 1 = page-cache warming, 2 = + ring prefetch
        }();
        if (prefetch_on >= 1 && glm5_gpu_enabled() && il + 1 < cfg_.n_layers && il + 1 >= cfg_.dense_lead) {
            const std::string pnext = "blk." + std::to_string(il + 1) + ".";
            TensorRef gi = m_.tensor(pnext + "ffn_gate_inp.weight");
            TensorRef ge = m_.tensor(pnext + "ffn_gate_exps.weight");
            if (gi.valid() && gi.dtype == DType::F32 && ge.valid() &&
                ge.dtype == DType::Q8_0 && ge.ne[2] >= 8) {
                for (int d = 0; d < H; d++) {
                    float acc = 0.0f;
                    for (int h = 0; h < hc; h++) acc += stream_[(size_t)h * H + d];
                    tmp2_[d] = acc / (float)hc;
                }
                m_.matvec(gi, tmp2_.data(), moe_logits_.data());
                std::vector<float> pb(cfg_.n_expert, 0.0f);
                if (m_.tensor(pnext + "exp_probs_b.bias").valid()) {
                    tensor_vec(pnext + "exp_probs_b.bias", pb.data());
                }
                std::vector<int> ids(cfg_.n_expert);
                for (int e = 0; e < cfg_.n_expert; e++) ids[e] = e;
                std::partial_sort(ids.begin(), ids.begin() + cfg_.n_expert_used, ids.end(),
                    [&](int a, int b) {
                        return (sigmoid_f(moe_logits_[a]) + pb[a]) > (sigmoid_f(moe_logits_[b]) + pb[b]);
                    });
                TensorRef ue = m_.tensor(pnext + "ffn_up_exps.weight");
                TensorRef de = m_.tensor(pnext + "ffn_down_exps.weight");
                const int64_t ff = cfg_.n_ff_exp;
                const int64_t slice = std::max<int64_t>(ff * (H / 32) * 34, H * (ff / 32) * 34);
                glm5_warm_slices(ge.data, ue.data, de.data, slice, ids.data(), cfg_.n_expert_used);
                if (prefetch_on >= 2) {
                    glm5_prefetch_layer(ge.data, ue.data, de.data, slice, ids.data(), cfg_.n_expert_used);
                }
            }
        }

        // FFN sublayer
        memcpy(residual_.data(), stream_.data(), sizeof(float) * hc * H);
        hc_mixes(p + "hc_ffn_fn.weight", il);
        std::vector<float> post_f = post_, comb_f = comb_;
        nt::ds41::detail::hc_collapse(stream_.data(), pre_.data(), cur_.data(),
                                      H, hc, true, m_.n_threads);
        {
            std::vector<float> w(H);
            tensor_vec(p + "ffn_norm.weight", w.data());
            rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
        }
        if (il < cfg_.dense_lead) {
            dense_ffn(il, cur_.data(), out_.data());
        } else {
            moe_ffn(il, cur_.data(), out_.data());
        }
        nt::ds41::detail::hc_post_mix(out_.data(), residual_.data(), post_f.data(), comb_f.data(),
                                      stream_.data(), H, hc, true, m_.n_threads);
        if (dbg_layers) {
            double ss = 0.0;
            for (int i = 0; i < hc * H; i++) ss += (double)stream_[i] * stream_[i];
            fprintf(stderr, "[glm5-dbg] t=%lld il=%-2d ss=%.6f head:", (long long)t, il, ss);
            for (int i = 0; i < 8; i++) fprintf(stderr, " %.5f", stream_[i]);
            fprintf(stderr, "\n");
        }
    }

    if (compute_logits) {
        for (int d = 0; d < H; d++) {
            float acc = 0.0f;
            for (int h = 0; h < hc; h++) acc += stream_[(size_t)h * H + d];
            cur_[d] = acc / (float)hc;
        }
        {
            std::vector<float> w(H);
            tensor_vec("output_norm.weight", w.data());
            rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
        }
        TensorRef outw = m_.tensor("output.weight");
        NT_CHECK(outw.valid(), "missing output.weight");
        bool got = false;
        if (glm5_gpu_enabled() && outw.dtype == DType::Q8_0) {
            static bool lm_inited = glm5_lm_head_init(outw.data, outw.rows(), outw.ne0());
            if (lm_inited) {
                got = glm5_lm_head(cur_.data(), logits_.data(), outw.rows(), outw.ne0());
            }
        }
        if (!got) {
            m_.matvec(outw, cur_.data(), logits_.data());
        }
        logits_valid_ = true;
    }

    history_.push_back(token);
    pos_++;
    return logits_valid_ ? logits_.data() : nullptr;
}

// ------------------------------------------------------------------
// chunked prefill (layer-major): batched projections per chunk, one
// expert load per layer, CPU state work per token
// ------------------------------------------------------------------
const float* Session::forward_chunk(const int* tokens, size_t n) {
    if (!glm5_gpu_enabled() || n == 0) return nullptr;
    static const bool dbg = getenv("GLM5_DBG_CHUNK") != nullptr;
    static const bool prof = getenv("GLM5_PROFILE") != nullptr;
    double prof_attn = 0, prof_union = 0, prof_run = 0;
    const auto t_start = std::chrono::steady_clock::now();
    const int H = cfg_.hidden;
    const int hc = cfg_.hc_mult;
    const int HD = cfg_.kda_head_dim;
    const int NH = cfg_.n_heads;
    const int DI = NH * HD;
    const size_t T = n;
    const size_t base = (size_t)pos_;

    std::vector<std::vector<float>> streams(n, std::vector<float>((size_t)hc * H));
    std::vector<float> x_all((size_t)T * H);
    std::vector<float> y_all((size_t)T * H);
    std::vector<float> a_all, b_all;

    for (size_t t = 0; t < n; t++) {
        m_.embed(tokens[t], emb_.data());
        for (int h = 0; h < hc; h++) {
            memcpy(streams[t].data() + (size_t)h * H, emb_.data(), sizeof(float) * H);
        }
        history_.push_back(tokens[t]);
    }

    // batched projection helpers (GPU first, CPU fallback)
    auto proj_t = [&](const std::string& name, const float* x, float* y, int64_t rows) -> bool {
        TensorRef tr = m_.tensor(name);
        if (!tr.valid() || tr.rows() < rows) return false;
        if (tr.dtype == DType::Q8_0 && glm5_q8_proj_tokens(tr.data, x, y, rows, tr.ne0(), (int)T)) return true;
        if (tr.dtype == DType::F32 && glm5_f32_proj_tokens(tr.data, x, y, rows, tr.ne0(), (int)T)) return true;
        for (size_t t = 0; t < T; t++) {
            m_.matvec_slice(tr, x + t * tr.ne0(), y + t * rows, 0, rows);
        }
        return true;
    };

    for (int il = 0; il < cfg_.n_layers; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        if (dbg) fprintf(stderr, "[chunk] layer %d\n", il);
        const auto t_attn0 = std::chrono::steady_clock::now();

        // ---- attention sublayer ----
        // hc mixes: flatten + rmsnorm per token (CPU), one batched projection
        a_all.assign((size_t)T * hc * H, 0.0f);
        {
            TensorRef fn = m_.tensor(p + "hc_attn_fn.weight");
            NT_CHECK(fn.valid(), "missing hc_attn_fn");
            std::vector<float> mixes((size_t)T * (2 + hc) * hc);
            for (size_t t = 0; t < T; t++) {
                memcpy(flat_norm_.data(), streams[t].data(), sizeof(float) * hc * H);
                rms_norm(flat_norm_.data(), (size_t)hc * H, nullptr, cfg_.norm_eps);
                memcpy(a_all.data() + (size_t)t * hc * H, flat_norm_.data(), sizeof(float) * hc * H);
            }
            if (fn.dtype == DType::Q8_0) {
                glm5_q8_proj_tokens(fn.data, a_all.data(), mixes.data(), fn.rows(), fn.ne0(), (int)T);
            } else {
                for (size_t t = 0; t < T; t++) m_.matvec(fn, a_all.data() + t * hc * H, mixes.data() + t * fn.rows());
            }
            // per-token pre/post/comb + collapse (CPU, tiny)
            std::vector<float> base_v, scale_v;
            base_v.resize((2 + hc) * hc);
            scale_v.resize(3);
            tensor_vec(p + "hc_attn_base.weight", base_v.data());
            tensor_vec(p + "hc_attn_scale.weight", scale_v.data());
            for (size_t t = 0; t < T; t++) {
                const float* mx = mixes.data() + t * (2 + hc) * hc;
                for (int h = 0; h < hc; h++) {
                    pre_[h] = sigmoid_f(mx[h] * scale_v[0] + base_v[h]) + cfg_.hc_eps;
                    post_[h] = 2.0f * sigmoid_f(mx[hc + h] * scale_v[1] + base_v[hc + h]);
                }
                nt::ds41::detail::hc_sinkhorn(mx + 2 * hc, base_v.data() + 2 * hc, scale_v[2],
                                              comb_.data(), hc, cfg_.hc_eps, cfg_.hc_sinkhorn_iters);
                // stash post/comb into the stream tail for after the attention
                streams[t].resize((size_t)hc * H + hc + hc * hc);
                memcpy(streams[t].data() + (size_t)hc * H, post_.data(), sizeof(float) * hc);
                memcpy(streams[t].data() + (size_t)hc * H + hc, comb_.data(), sizeof(float) * hc * hc);
                // collapse with pre into x_all
                nt::ds41::detail::hc_collapse(streams[t].data(), pre_.data(), x_all.data() + t * H,
                                              H, hc, true, m_.n_threads);
            }
        }
        {
            std::vector<float> w(H);
            tensor_vec(p + "attn_norm.weight", w.data());
            for (size_t t = 0; t < T; t++) {
                rms_norm(x_all.data() + t * H, H, w.data(), cfg_.norm_eps);
            }
        }

        // attention proper: batched projections + per-token state
        if (cfg_.is_recr[il]) {
            a_all.assign((size_t)T * DI, 0.0f);
            b_all.assign((size_t)T * DI, 0.0f);
            std::vector<float> xq((size_t)T * DI), xk((size_t)T * DI), xv((size_t)T * DI);
            std::vector<float> fa((size_t)T * HD), g1((size_t)T * DI), ga((size_t)T * HD), g2((size_t)T * DI);
            std::vector<float> betav((size_t)T * NH);
            proj_t(p + "attn_q.weight", x_all.data(), xq.data(), DI);
            proj_t(p + "attn_k.weight", x_all.data(), xk.data(), DI);
            proj_t(p + "attn_v.weight", x_all.data(), xv.data(), DI);
            proj_t(p + "ssm_f_a.weight", x_all.data(), fa.data(), HD);
            proj_t(p + "ssm_f_b.weight", fa.data(), g1.data(), DI);
            proj_t(p + "ssm_g_a.weight", x_all.data(), ga.data(), HD);
            proj_t(p + "ssm_g_b.weight", ga.data(), g2.data(), DI);
            proj_t(p + "ssm_beta.weight", x_all.data(), betav.data(), NH);
            tensor_vec(p + "ssm_dt.bias", dt_b_.data());
            tensor_vec(p + "ssm_a", ssm_a_.data());
            tensor_vec(p + "ssm_norm.weight", onorm_.data());
            const int cs = recr_index_[il];
            const float lb = cfg_.kda_gate_lower_bound;
            // per-token: conv + gdn; v conv lands in `gated`, later replaced by the out gate
            std::vector<float> gated((size_t)T * DI);
            {
                const float* srcs3[3] = { xq.data(), xk.data(), xv.data() };
                const char* names3[3] = { "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight" };
                float* dsts3[3] = { a_all.data(), b_all.data(), gated.data() };
                for (size_t t = 0; t < T; t++) {
                    for (int qkv = 0; qkv < 3; qkv++) {
                        TensorRef cw = m_.tensor(p + names3[qkv]);
                        const float* w = reinterpret_cast<const float*>(cw.data);
                        float* st = conv_state_[cs].data() + (size_t)qkv * (cfg_.d_conv - 1) * DI;
                        const float* src = srcs3[qkv] + t * DI;
                        float* dst = dsts3[qkv] + t * DI;
                        const int P = cfg_.d_conv - 1;
                        const int DC = cfg_.d_conv;
                        parallel_for(DI, m_.n_threads, [&](int64_t c, int) {
                            float acc = w[(size_t)c * DC + P] * src[c];
                            for (int s = 0; s < P; s++) acc += w[(size_t)c * DC + s] * st[(size_t)s * DI + c];
                            dst[c] = acc / (1.0f + expf(-acc));
                        });
                    }
                    for (int qkv = 0; qkv < 3; qkv++) {
                        float* st = conv_state_[cs].data() + (size_t)qkv * (cfg_.d_conv - 1) * DI;
                        const float* src = srcs3[qkv] + t * DI;
                        const int P = cfg_.d_conv - 1;
                        for (int s = 0; s + 1 < P; s++) {
                            memcpy(st + (size_t)s * DI, st + (size_t)(s + 1) * DI, sizeof(float) * DI);
                        }
                        memcpy(st + (size_t)(P - 1) * DI, src, sizeof(float) * DI);
                    }
                }
            }
            {
                float* S = ssm_state_[cs].data();
                const float oosqrt = 1.0f / sqrtf((float)HD);
                for (size_t t = 0; t < T; t++) {
                    float* g1t = g1.data() + t * DI;
                    for (int i = 0; i < DI; i++) g1t[i] += dt_b_[i];
                    for (int h = 0; h < NH; h++) {
                        const float a = ssm_a_[h];
                        for (int i = 0; i < HD; i++) {
                            const float g = sigmoid_f(-(g1t[(size_t)h * HD + i] * a)) * lb;
                            decay_[(size_t)h * HD + i] = expf(g);
                        }
                    }
                    float beta_l[64];
                    for (int h = 0; h < NH; h++) beta_l[h] = sigmoid_f(betav[t * NH + h]);
                    float* cqt = a_all.data() + t * DI;
                    float* ckt = b_all.data() + t * DI;
                    float* vt = gated.data() + t * DI;
                    // l2norm q/k
                    for (int h = 0; h < NH; h++) {
                        float* vh1 = cqt + (size_t)h * HD;
                        float ss = 0.0f;
                        for (int i = 0; i < HD; i++) ss += vh1[i] * vh1[i];
                        const float sc = 1.0f / sqrtf(ss + 1e-6f);
                        for (int i = 0; i < HD; i++) vh1[i] *= sc;
                        float* vh2 = ckt + (size_t)h * HD;
                        ss = 0.0f;
                        for (int i = 0; i < HD; i++) ss += vh2[i] * vh2[i];
                        const float sc2 = 1.0f / sqrtf(ss + 1e-6f);
                        for (int i = 0; i < HD; i++) vh2[i] *= sc2;
                    }
                    // v source: conv output in `gated`
                    // GDN into cout_ (per head)
                    float* coutp = cout_.data();
                    for (int h = 0; h < NH; h++) {
                        const float* qh = cqt + (size_t)h * HD;
                        const float* kh = ckt + (size_t)h * HD;
                        const float* vh = vt + (size_t)h * HD;
                        const float* dec = decay_.data() + (size_t)h * HD;
                        float* sh = S + (size_t)h * HD * HD;
                        float* oh = coutp + (size_t)h * HD;
                        float delta[256];
                        for (int j = 0; j < HD; j++) {
                            float* row = sh + (size_t)j * HD;
                            for (int i = 0; i < HD; i++) row[i] *= dec[i];
                        }
                        for (int j = 0; j < HD; j++) {
                            float* row = sh + (size_t)j * HD;
                            float sum = 0.0f;
                            for (int i = 0; i < HD; i++) sum += row[i] * kh[i];
                            delta[j] = (vh[j] - sum) * beta_l[h];
                        }
                        for (int j = 0; j < HD; j++) {
                            float* row = sh + (size_t)j * HD;
                            const float dj = delta[j];
                            for (int i = 0; i < HD; i++) row[i] += kh[i] * dj;
                        }
                        for (int j = 0; j < HD; j++) {
                            float* row = sh + (size_t)j * HD;
                            float sum = 0.0f;
                            for (int i = 0; i < HD; i++) sum += row[i] * qh[i];
                            oh[j] = sum * oosqrt;
                        }
                    }
                    // out gate
                    float* g2t = g2.data() + t * DI;
                    float* gout = gated.data() + t * DI;
                    for (int h = 0; h < NH; h++) {
                        float* oh = cout_.data() + (size_t)h * HD;
                        rms_norm(oh, HD, onorm_.data(), cfg_.norm_eps);
                        for (int i = 0; i < HD; i++) {
                            oh[i] *= sigmoid_f(g2t[(size_t)h * HD + i]);
                        }
                    }
                    memcpy(gout, cout_.data(), sizeof(float) * DI);
                }
            }
            proj_t(p + "attn_output.weight", gated.data(), y_all.data(), H);
        } else {
            // MLA: batched projections + per-token latent/indexer/attention
            const int mi = mla_index_[il];
            const int KL = cfg_.kv_lora_rank;
            const int QK = cfg_.qk_nope;
            const int VD = cfg_.v_dim;
            const int IDX = cfg_.indexer_head_dim;
            const int IH = cfg_.indexer_n_head;
            std::vector<float> qr((size_t)T * cfg_.q_lora_rank);
            std::vector<float> q((size_t)T * NH * QK);
            std::vector<float> kv((size_t)T * KL);
            std::vector<float> q_abs((size_t)T * NH * KL);
            std::vector<float> ik((size_t)T * IDX), ig((size_t)T * IDX), iq((size_t)T * IH * IDX), wp((size_t)T * IH);
            proj_t(p + "attn_q_a.weight", x_all.data(), qr.data(), cfg_.q_lora_rank);
            {
                std::vector<float> w(cfg_.q_lora_rank);
                tensor_vec(p + "attn_q_a_norm.weight", w.data());
                for (size_t t = 0; t < T; t++) rms_norm(qr.data() + t * cfg_.q_lora_rank, cfg_.q_lora_rank, w.data(), cfg_.norm_eps);
            }
            proj_t(p + "attn_q_b.weight", qr.data(), q.data(), (int64_t)NH * QK);
            proj_t(p + "attn_kv_a_mqa.weight", x_all.data(), kv.data(), KL);
            {
                std::vector<float> w(KL);
                tensor_vec(p + "attn_kv_a_norm.weight", w.data());
                for (size_t t = 0; t < T; t++) rms_norm(kv.data() + t * KL, KL, w.data(), cfg_.norm_eps);
            }
            {
                TensorRef wkb_t = m_.tensor(p + "attn_k_b.weight");
                if (wkb_t.dtype == DType::Q8_0) {
                    glm5_q8_proj_grouped_t(wkb_t.data, q.data(), q_abs.data(), KL, QK, NH, (int)T);
                } else {
                    for (size_t t = 0; t < T; t++) {
                        const float* wkb = wkb_f32_[mi].data();
                        for (int h = 0; h < NH; h++) {
                            const float* qh = q.data() + t * NH * QK + (size_t)h * QK;
                            float* qa = q_abs.data() + t * NH * KL + (size_t)h * KL;
                            const float* wh = wkb + (size_t)h * (KL * QK);
                            for (int c = 0; c < KL; c++) {
                                qa[c] = dot_plain_f32(wh + (size_t)c * QK, qh, QK);
                            }
                        }
                    }
                }
            }
            proj_t(p + "indexer.attn_k.weight", x_all.data(), ik.data(), IDX);
            proj_t(p + "indexer_compressor_gate.weight", x_all.data(), ig.data(), IDX);
            proj_t(p + "indexer.attn_q_b.weight", qr.data(), iq.data(), (int64_t)IH * IDX);
            proj_t(p + "indexer.proj.weight", x_all.data(), wp.data(), IH);
            {
                const float s = 1.0f / sqrtf((float)(IDX * IH));
                for (size_t i = 0; i < (size_t)T * IH; i++) wp[i] *= s;
            }
            // per-token state: latents, pooling, selection, attention
            std::vector<float> attn_out((size_t)T * NH * VD);
            {
                std::vector<float> wln(IDX), bln(IDX);
                tensor_vec(p + "indexer.k_norm.weight", wln.data());
                tensor_vec(p + "indexer.k_norm.bias", bln.data());
                std::vector<float> ape((size_t)IDX * cfg_.indexer_kpool);
                tensor_vec(p + "indexer_compressor_ape.weight", ape.data());
                const int kpool = cfg_.indexer_kpool;
                float* lat = latent_[mi].data();
                float* pk = pool_k_[mi].data();
                for (size_t t = 0; t < T; t++) {
                    const int64_t pos = (int64_t)(base + t);
                    memcpy(lat + (size_t)pos * KL, kv.data() + t * KL, sizeof(float) * KL);
                    float* ikt = ik.data() + t * IDX;
                    ln_norm(ikt, IDX, wln.data(), bln.data(), cfg_.ln_eps);
                    memcpy(idx_k_[mi].data() + (size_t)pos * IDX, ikt, sizeof(float) * IDX);
                    memcpy(idx_g_[mi].data() + (size_t)pos * IDX, ig.data() + t * IDX, sizeof(float) * IDX);
                    if ((pos + 1) % kpool == 0) {
                        const int64_t pi = (pos + 1) / kpool - 1;
                        const int64_t bs = pi * kpool;
                        float* pooled = pk + (size_t)pi * IDX;
                        for (int d = 0; d < IDX; d++) {
                            float mx = -std::numeric_limits<float>::infinity();
                            float lg[8];
                            for (int m = 0; m < kpool; m++) {
                                lg[m] = idx_g_[mi][(size_t)(bs + m) * IDX + d] + ape[(size_t)m * IDX + d];
                                mx = std::max(mx, lg[m]);
                            }
                            float sum = 0.0f, pr[8];
                            for (int m = 0; m < kpool; m++) { pr[m] = expf(lg[m] - mx); sum += pr[m]; }
                            float acc = 0.0f;
                            for (int m = 0; m < kpool; m++) {
                                acc += (pr[m] / sum) * idx_k_[mi][(size_t)(bs + m) * IDX + d];
                            }
                            pooled[d] = acc;
                        }
                    }
                    const int64_t n_pool_struct = pos / kpool + 1;
                    std::fill(visible_.begin(), visible_.begin() + pos + 1, 0.0f);
                    const int64_t n_top_pool = std::min<int64_t>(n_pool_struct, cfg_.indexer_top_k / kpool);
                    {
                        std::vector<int64_t> order(n_pool_struct);
                        for (int64_t i2 = 0; i2 < n_pool_struct; i2++) order[i2] = i2;
                        std::vector<float> sc2(n_pool_struct);
                        for (int64_t pi = 0; pi < n_pool_struct; pi++) {
                            const float* key = pk + (size_t)pi * IDX;
                            float s2 = 0.0f;
                            for (int h = 0; h < IH; h++) {
                                float d2 = dot_plain_f32(iq.data() + t * IH * IDX + (size_t)h * IDX, key, IDX);
                                if (d2 > 0.0f) s2 += d2 * wp[t * IH + h];
                            }
                            sc2[pi] = s2;
                        }
                        std::partial_sort(order.begin(), order.begin() + n_top_pool, order.end(),
                                          [&](int64_t aa, int64_t bb) {
                                              if (sc2[aa] != sc2[bb]) return sc2[aa] > sc2[bb];
                                              return aa < bb;
                                          });
                        for (int64_t k2 = 0; k2 < n_top_pool; k2++) {
                            const int64_t first = order[k2] * kpool;
                            for (int m = 0; m < kpool; m++) {
                                const int64_t cell = first + m;
                                if (cell <= pos) visible_[cell] = 1.0f;
                            }
                        }
                    }
                    const float scale = 1.0f / sqrtf((float)QK);
                    std::vector<float> sc(pos + 1);
                    for (int h = 0; h < NH; h++) {
                        const float* qa = q_abs.data() + t * NH * KL + (size_t)h * KL;
                        float mx = -std::numeric_limits<float>::infinity();
                        for (int64_t tt = 0; tt <= pos; tt++) {
                            float s2 = -std::numeric_limits<float>::infinity();
                            if (visible_[tt] > 0.0f) {
                                s2 = dot_plain_f32(qa, lat + (size_t)tt * KL, KL) * scale;
                                mx = std::max(mx, s2);
                            }
                            sc[tt] = s2;
                        }
                        float sum = 0.0f;
                        for (int64_t tt = 0; tt <= pos; tt++) {
                            if (visible_[tt] > 0.0f) { sc[tt] = expf(sc[tt] - mx); sum += sc[tt]; }
                        }
                        const float inv = 1.0f / sum;
                        std::fill(attn_lat_.data(), attn_lat_.data() + KL, 0.0f);
                        for (int64_t tt = 0; tt <= pos; tt++) {
                            if (visible_[tt] > 0.0f) {
                                const float wgt = sc[tt] * inv;
                                const float* row = lat + (size_t)tt * KL;
                                for (int c = 0; c < KL; c++) attn_lat_[c] += wgt * row[c];
                            }
                        }
                        const float* wh = wvb_f32_[mi].data() + (size_t)h * (VD * KL);
                        float* oh = attn_out.data() + t * NH * VD + (size_t)h * VD;
                        for (int v = 0; v < VD; v++) {
                            oh[v] = dot_plain_f32(wh + (size_t)v * KL, attn_lat_.data(), KL);
                        }
                    }
                }
            }
            proj_t(p + "attn_output.weight", attn_out.data(), y_all.data(), H);
        }

        // post-mix attention with stashed post/comb (residual copied aside: hc_post_mix
        // cannot alias its residual and output)
        for (size_t t = 0; t < T; t++) {
            const float* postp = streams[t].data() + (size_t)hc * H;
            const float* combp = postp + hc;
            memcpy(stream_.data(), streams[t].data(), sizeof(float) * hc * H);
            nt::ds41::detail::hc_post_mix(y_all.data() + t * H, stream_.data(), postp, combp,
                                          streams[t].data(), H, hc, true, m_.n_threads);
        }
        const auto t_attn1 = std::chrono::steady_clock::now();
        prof_attn += std::chrono::duration<double>(t_attn1 - t_attn0).count();

        // ---- FFN sublayer ----
        {
            TensorRef fn = m_.tensor(p + "hc_ffn_fn.weight");
            NT_CHECK(fn.valid(), "missing hc_ffn_fn");
            std::vector<float> mixes((size_t)T * (2 + hc) * hc);
            a_all.assign((size_t)T * hc * H, 0.0f);
            for (size_t t = 0; t < T; t++) {
                memcpy(flat_norm_.data(), streams[t].data(), sizeof(float) * hc * H);
                rms_norm(flat_norm_.data(), (size_t)hc * H, nullptr, cfg_.norm_eps);
                memcpy(a_all.data() + t * hc * H, flat_norm_.data(), sizeof(float) * hc * H);
            }
            if (fn.dtype == DType::Q8_0) {
                glm5_q8_proj_tokens(fn.data, a_all.data(), mixes.data(), fn.rows(), fn.ne0(), (int)T);
            } else {
                for (size_t t = 0; t < T; t++) m_.matvec(fn, a_all.data() + t * hc * H, mixes.data() + t * fn.rows());
            }
            std::vector<float> base_v, scale_v;
            base_v.resize((2 + hc) * hc);
            scale_v.resize(3);
            tensor_vec(p + "hc_ffn_base.weight", base_v.data());
            tensor_vec(p + "hc_ffn_scale.weight", scale_v.data());
            for (size_t t = 0; t < T; t++) {
                const float* mx = mixes.data() + t * (2 + hc) * hc;
                for (int h = 0; h < hc; h++) {
                    pre_[h] = sigmoid_f(mx[h] * scale_v[0] + base_v[h]) + cfg_.hc_eps;
                    post_[h] = 2.0f * sigmoid_f(mx[hc + h] * scale_v[1] + base_v[hc + h]);
                }
                nt::ds41::detail::hc_sinkhorn(mx + 2 * hc, base_v.data() + 2 * hc, scale_v[2],
                                              comb_.data(), hc, cfg_.hc_eps, cfg_.hc_sinkhorn_iters);
                streams[t].resize((size_t)hc * H + hc + hc * hc);
                memcpy(streams[t].data() + (size_t)hc * H, post_.data(), sizeof(float) * hc);
                memcpy(streams[t].data() + (size_t)hc * H + hc, comb_.data(), sizeof(float) * hc * hc);
                nt::ds41::detail::hc_collapse(streams[t].data(), pre_.data(), x_all.data() + t * H,
                                              H, hc, true, m_.n_threads);
            }
        }
        {
            std::vector<float> w(H);
            tensor_vec(p + "ffn_norm.weight", w.data());
            for (size_t t = 0; t < T; t++) {
                rms_norm(x_all.data() + t * H, H, w.data(), cfg_.norm_eps);
            }
        }

        if (il < cfg_.dense_lead) {
            const int ff = cfg_.n_ff_dense;
            std::vector<float> gt((size_t)T * ff), up((size_t)T * ff), hbuf((size_t)T * ff);
            proj_t(p + "ffn_gate.weight", x_all.data(), gt.data(), ff);
            proj_t(p + "ffn_up.weight", x_all.data(), up.data(), ff);
            const float lim = cfg_.swiglu_clamp_shexp[il];
            for (size_t t = 0; t < T; t++) {
                for (int i = 0; i < ff; i++) {
                    float g = gt[t * ff + i];
                    float u = up[t * ff + i];
                    if (lim > 1e-6f) {
                        g = std::min(g, lim);
                        u = std::max(-lim, std::min(u, lim));
                    }
                    hbuf[t * ff + i] = (g / (1.0f + expf(-g))) * u;
                }
            }
            proj_t(p + "ffn_down.weight", hbuf.data(), y_all.data(), H);
        } else {
            // router sweep + union load + per-token expert compute (all hits)
            std::vector<std::vector<int>> sels(n);
            std::vector<std::vector<float>> wtss(n);
            std::vector<int> union_ids;
            {
                std::vector<char> seen(cfg_.n_expert, 0);
                for (size_t t = 0; t < n; t++) {
                    select_experts(il, x_all.data() + t * H, sels[t], wtss[t]);
                    for (int e : sels[t]) {
                        if (!seen[e]) { seen[e] = 1; union_ids.push_back(e); }
                    }
                }
            }
            TensorRef ge = m_.tensor(p + "ffn_gate_exps.weight");
            TensorRef ue = m_.tensor(p + "ffn_up_exps.weight");
            TensorRef de = m_.tensor(p + "ffn_down_exps.weight");
            if (ge.dtype == DType::Q8_0 && ue.dtype == DType::Q8_0 && de.dtype == DType::Q8_0) {
                glm5_gpu_load_experts(ge.data, ue.data, de.data, cfg_.n_ff_exp,
                                      union_ids.data(), (int)union_ids.size());
            }
            const auto t_union1 = std::chrono::steady_clock::now();
            prof_union += std::chrono::duration<double>(t_union1 - t_attn1).count();
            const auto t_run0 = std::chrono::steady_clock::now();
            for (size_t t = 0; t < n; t++) {
                run_experts(il, x_all.data() + t * H, sels[t], wtss[t], y_all.data() + t * H);
            }
            prof_run += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run0).count();
        }

        // post-mix FFN (residual aside)
        for (size_t t = 0; t < T; t++) {
            const float* postp = streams[t].data() + (size_t)hc * H;
            const float* combp = postp + hc;
            memcpy(stream_.data(), streams[t].data(), sizeof(float) * hc * H);
            nt::ds41::detail::hc_post_mix(y_all.data() + t * H, stream_.data(), postp, combp,
                                          streams[t].data(), H, hc, true, m_.n_threads);
        }
    }

    // terminal: last token logits
    const std::vector<float>& last = streams[n - 1];
    for (int d = 0; d < H; d++) {
        float acc = 0.0f;
        for (int h = 0; h < hc; h++) acc += last[(size_t)h * H + d];
        cur_[d] = acc / (float)hc;
    }
    {
        std::vector<float> w(H);
        tensor_vec("output_norm.weight", w.data());
        rms_norm(cur_.data(), H, w.data(), cfg_.norm_eps);
    }
    TensorRef outw = m_.tensor("output.weight");
    NT_CHECK(outw.valid(), "missing output.weight");
    bool got = false;
    if (outw.dtype == DType::Q8_0) {
        static bool lm_inited = glm5_lm_head_init(outw.data, outw.rows(), outw.ne0());
        if (lm_inited) got = glm5_lm_head(cur_.data(), logits_.data(), outw.rows(), outw.ne0());
    }
    if (!got) m_.matvec(outw, cur_.data(), logits_.data());
    logits_valid_ = true;
    pos_ += (int)n;
    if (prof) {
        const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "[chunk-prof] n=%zu total=%.1fs attn=%.1fs union+load=%.1fs moe_run=%.1fs\n",
                n, total, prof_attn, prof_union, prof_run);
    }
    return logits_.data();
}

// ============================================================
// sampling + CLI
// ============================================================

namespace {

int sample_token(const float* logits, int n, const std::vector<int32_t>& history,
                 const GenOptions& opt, std::mt19937_64& rng,
                 std::vector<std::pair<float, int>>& scratch) {
    scratch.resize(n);
    for (int i = 0; i < n; i++) scratch[i] = { logits[i], i };

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

    int k = (opt.top_k > 0) ? std::min(opt.top_k, n) : n;
    std::partial_sort(scratch.begin(), scratch.begin() + k, scratch.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    scratch.resize(k);

    const float inv_t = 1.0f / opt.temperature;
    float mx = scratch[0].first;
    double sum = 0.0;
    for (auto& s : scratch) {
        s.first = expf((s.first - mx) * inv_t);
        sum += s.first;
    }

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

} // namespace

int run_cli(const std::string& model_path, const GenOptions& opt) {
    {
        GGUFLoader peek;
        if (!peek.load(model_path)) {
            return -1;
        }
        if (peek.kv_string("general.architecture", "") != "glm5-next") {
            return -1;
        }
    }

    Model model;
    std::string err;
    fprintf(stderr, "Loading GLM-5.3-Flash (glm5-next) model: %s\n", model_path.c_str());
    if (!model.load(model_path, err)) {
        fprintf(stderr, "glm5: failed to load model: %s\n", err.c_str());
        return 1;
    }

    const Config& cfg = model.cfg();
    const int n_mla = (int)std::count(cfg.is_recr.begin(), cfg.is_recr.end(), 0);
    fprintf(stderr, "glm5-next: %d layers (%d KDA + %d MLA), hidden %d, heads %d, "
            "experts %d/%d, hc %d, ctx %d\n",
            cfg.n_layers, cfg.n_layers - n_mla, n_mla, cfg.hidden, cfg.n_heads,
            cfg.n_expert_used, cfg.n_expert, cfg.hc_mult, opt.ctx);

    if (glm5_gpu_enabled()) {
        const GGUFLoader& L = model.loader();
        for (int i = 0; i < L.split_count(); i++) {
            glm5_register_shard(L.shard_data_base(i), L.shard_data_length(i),
                                L.shard_path(i).c_str(), L.shard_data_offset(i));
        }
    }

    Tokenizer tok;
    tok.init(model.loader().vocab(), cfg.bos_id, cfg.eos_id);

    Session session(model, opt.ctx);

    std::vector<int> tokens;
    if (!opt.token_ids.empty()) {
        tokens = opt.token_ids;
    } else {
        tokens = tok.encode(opt.prompt, /*add_bos=*/false);
    }
    if (tokens.empty()) tokens.push_back(0);
    if (opt.ctx > 0 && (int)tokens.size() >= opt.ctx) {
        tokens.resize(opt.ctx - 1);
    }

    if (getenv("GLM5_DUMP_TOKENS")) {
        fprintf(stderr, "tokens:");
        for (int tk : tokens) fprintf(stderr, " %d", tk);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "prompt: %zu tokens\n", tokens.size());

    const auto prefill_start = std::chrono::steady_clock::now();
    const float* logits = nullptr;
    static const int chunk_sz = [] {
        const char* e = getenv("GLM5_CHUNK");
        return e ? atoi(e) : 512;
    }();
    bool chunked = false;
    if (glm5_gpu_enabled() && chunk_sz > 1) {
        chunked = true;
        for (size_t i = 0; i < tokens.size(); i += (size_t)chunk_sz) {
            const size_t n = std::min((size_t)chunk_sz, tokens.size() - i);
            logits = session.forward_chunk(&tokens[i], n);
            if (!logits) { chunked = false; break; }
        }
    }
    if (!chunked) {
        for (size_t i = 0; i < tokens.size(); i++) {
            logits = session.forward(tokens[i], i + 1 == tokens.size());
        }
    }
    const double prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_start).count();
    fprintf(stderr, "[prefill] tokens=%zu seconds=%.3f (%.3f s/tok, %s)\n",
            tokens.size(), prefill_seconds, prefill_seconds / std::max<size_t>(1, tokens.size()),
            chunked ? "chunked" : "token");

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

    std::mt19937_64 rng(opt.seed);
    std::vector<std::pair<float, int>> scratch;
    std::vector<int> generated;

    // fast-path decode (Fase A): forward completo en GPU
    static const bool fast_on = [] {
        const char* e = getenv("GLM5_FAST");
        return e ? atoi(e) != 0 : false;
    }();
    if (fast_on && glm5_gpu_enabled()) {
        FastConfig fc;
        fc.n_layers = cfg.n_layers;
        fc.hidden = cfg.hidden;
        fc.n_heads = cfg.n_heads;
        fc.kda_head_dim = cfg.kda_head_dim;
        fc.d_conv = cfg.d_conv;
        fc.kda_gate_lower_bound = cfg.kda_gate_lower_bound;
        fc.q_lora_rank = cfg.q_lora_rank;
        fc.kv_lora_rank = cfg.kv_lora_rank;
        fc.qk_nope = cfg.qk_nope;
        fc.v_dim = cfg.v_dim;
        fc.indexer_n_head = cfg.indexer_n_head;
        fc.indexer_head_dim = cfg.indexer_head_dim;
        fc.indexer_kpool = cfg.indexer_kpool;
        fc.indexer_top_k = cfg.indexer_top_k;
        fc.hc_mult = cfg.hc_mult;
        fc.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters;
        fc.hc_eps = cfg.hc_eps;
        fc.norm_eps = cfg.norm_eps;
        fc.ln_eps = cfg.ln_eps;
        fc.dense_lead = cfg.dense_lead;
        fc.n_ff_dense = cfg.n_ff_dense;
        fc.n_expert = cfg.n_expert;
        fc.n_expert_used = cfg.n_expert_used;
        fc.n_ff_exp = cfg.n_ff_exp;
        fc.expert_weights_scale = cfg.expert_weights_scale;
        fc.expert_weights_norm = cfg.expert_weights_norm;
        fc.vocab_size = cfg.vocab_size;
        fc.max_ctx = std::min(opt.ctx, 2048);
        fc.is_recr = cfg.is_recr;
        auto lookup = [&](const std::string& name) -> const void* {
            return model.tensor(name).data;
        };
        if (!glm5_fast_init(lookup, fc)) {
            fprintf(stderr, "glm5-fast: init failed, falling back\n");
        } else {
            fprintf(stderr, "glm5-fast: device-resident decode active (ctx<=%d)\n", fc.max_ctx);
            glm5_fast_reset();
            const auto gen_start2 = std::chrono::steady_clock::now();
            std::vector<float> flogits(cfg.vocab_size);
            for (int tk : tokens) {
                if (!glm5_fast_step(tk, flogits.data())) {
                    fprintf(stderr, "glm5-fast: step failed\n");
                    return 1;
                }
            }
            const double pre2 = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gen_start2).count();
            fprintf(stderr, "[fast-prefill] tokens=%zu seconds=%.2f (%.2f s/tok)\n",
                    tokens.size(), pre2, pre2 / std::max<size_t>(1, tokens.size()));

            const auto gen_start = std::chrono::steady_clock::now();
            const float* fl = flogits.data();
            for (int step = 0; step < opt.max_tokens; step++) {
                if (!fl) break;
                int next = sample_token(fl, cfg.vocab_size, session.history_ref(), opt, rng, scratch);
                if (next == cfg.eos_id) break;
                generated.push_back(next);
                std::string piece = tok.decode_token(next);
                fputs(piece.c_str(), stdout);
                fflush(stdout);
                fl = glm5_fast_step(next, flogits.data()) ? flogits.data() : nullptr;
            }
            const double gen_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gen_start).count();
            fputs("\n", stdout);
            if (glm5_gpu_enabled()) glm5_gpu_stats();
            glm5_fast_stats();
            fprintf(stderr, "\ngenerated %zu tokens in %.2fs (%.2f tok/s)\n",
                    generated.size(), gen_seconds, generated.size() / std::max(0.001, gen_seconds));
            return 0;
        }
    }

    const auto gen_start = std::chrono::steady_clock::now();
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
    const double gen_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gen_start).count();
    fputs("\n", stdout);

    if (glm5_gpu_enabled()) glm5_gpu_stats();
    fprintf(stderr, "\ngenerated %zu tokens in %.2fs (%.2f tok/s)\n",
            generated.size(), gen_seconds, generated.size() / std::max(0.001, gen_seconds));
    return 0;
}

} // namespace glm5
} // namespace nt

