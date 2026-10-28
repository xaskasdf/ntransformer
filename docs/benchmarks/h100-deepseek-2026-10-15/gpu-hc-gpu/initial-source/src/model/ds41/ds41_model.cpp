#include "ds41_model.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nt {
namespace ds41 {

// ============================================================
// Persistent thread pool
// (previous implementation spawned O(cores) threads per matvec,
//  which dominated the per-token time at ~1300 matvecs/token)
// ============================================================
namespace {
thread_local bool t_in_pool = false;

class ThreadPool {
public:
    static ThreadPool& get() {
        static ThreadPool pool;
        return pool;
    }

    void run(int64_t n, const std::function<void(int64_t)>& fn) {
        if (n <= 0) return;
        const int nw = (int)workers_.size();
        if (nw == 0 || n == 1) {
            for (int64_t i = 0; i < n; i++) fn(i);
            return;
        }
        if (t_in_pool) {
            // nested call from a worker or participating main thread: run serially
            for (int64_t i = 0; i < n; i++) fn(i);
            return;
        }
        {
            std::unique_lock<std::mutex> lk(m_);
            fn_ = &fn;
            n_ = n;
            // Claim adjacent rows together. Keep several chunks per participant
            // for load balancing, without changing any row's reduction order.
            grain_ = fixed_grain_ > 0 ? fixed_grain_ :
                std::clamp<int64_t>(n / (4 * (nw + 1)), 1, 64);
            next_.store(0, std::memory_order_relaxed);
            done_ = 0;
            ++gen_;
            cv_start_.notify_all();
        }
        // main thread participates too
        t_in_pool = true;
        consume(n, grain_, fn);
        t_in_pool = false;
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return done_ == nw; });
        fn_ = nullptr;
    }

private:
    ThreadPool() {
        // Read once at pool creation. A grain of 1 restores the original
        // per-row dispatch for controlled comparisons.
        if (const char* value = std::getenv("DS41_CPU_GRAIN")) {
            char* end = nullptr;
            errno = 0;
            long long parsed = std::strtoll(value, &end, 10);
            if (errno == 0 && end != value && *end == '\0' && parsed >= 1 && parsed <= 1048576) {
                fixed_grain_ = parsed;
            } else {
                std::fprintf(stderr, "ds41-cpu: invalid DS41_CPU_GRAIN; using adaptive chunks\n");
            }
        }
        int hw = (int)std::thread::hardware_concurrency();
        int nw = (hw > 1) ? hw - 1 : 0;
        for (int t = 0; t < nw; t++) {
            workers_.emplace_back([this] { worker(); });
        }
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto& w : workers_) w.join();
    }

    void consume(int64_t n, int64_t grain, const std::function<void(int64_t)>& fn) {
        for (;;) {
            const int64_t begin = next_.fetch_add(grain, std::memory_order_relaxed);
            if (begin >= n) break;
            const int64_t end = begin + std::min(grain, n - begin);
            for (int64_t i = begin; i < end; ++i) fn(i);
        }
    }

    void worker() {
        t_in_pool = true;
        int64_t last_gen = 0;
        for (;;) {
            const std::function<void(int64_t)>* fn = nullptr;
            int64_t n = 0, grain = 1;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_start_.wait(lk, [&] { return stop_ || gen_ != last_gen; });
                if (stop_) return;
                last_gen = gen_;
                fn = fn_;
                n = n_;
                grain = grain_;
            }
            consume(n, grain, *fn);
            {
                std::unique_lock<std::mutex> lk(m_);
                if (++done_ == (int64_t)workers_.size()) cv_done_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int64_t)>* fn_ = nullptr;
    int64_t n_ = 0;
    int64_t grain_ = 1;
    int64_t fixed_grain_ = 0;
    std::atomic<int64_t> next_{0};
    int64_t gen_ = 0;
    int64_t done_ = 0;
    bool stop_ = false;
};

} // namespace

void parallel_for(int64_t n, int n_threads, const std::function<void(int64_t, int)>& fn) {
    (void)n_threads;
    ThreadPool::get().run(n, [&](int64_t i) { fn(i, 0); });
}

int64_t TensorRef::row_bytes() const {
    int64_t be = block_elems();
    return (ne[0] / be) * (int64_t)dtype_size(dtype);
}

int64_t TensorRef::block_elems() const {
    return (int64_t)dtype_block_size(dtype);
}

// ============================================================
// Fused dequant-dot helpers (row of a quantized tensor vs float vector)
// ============================================================

static float dot_f32(const float* w, const float* x, int64_t n) {
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

static float dot_f16(const uint16_t* w, const float* x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) acc += fp16_to_fp32_host(w[i]) * x[i];
    return acc;
}

static float dot_bf16(const uint16_t* w, const float* x, int64_t n) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    const __m256i sh = _mm256_set1_epi32(16);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i w0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
        __m256i w1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8)));
        w0 = _mm256_slli_epi32(w0, 16);
        w1 = _mm256_slli_epi32(w1, 16);
        a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(w0), _mm256_loadu_ps(x + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(w1), _mm256_loadu_ps(x + i + 8), a1);
    }
    (void)sh;
    __m256 a = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float acc = _mm_cvtss_f32(s);
    for (; i < n; i++) acc += bf16_to_fp32_host(w[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) acc += bf16_to_fp32_host(w[i]) * x[i];
    return acc;
#endif
}

static float dot_q8_0(const BlockQ8_0* blocks, const float* x, int64_t n) {
    float acc = 0.0f;
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const float d = fp16_to_fp32_host(blocks[b].d);
        const int8_t* q = blocks[b].qs;
        const float* xb = x + b * 32;
#if defined(__AVX2__)
        __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        __m128i lo128 = _mm256_castsi256_si128(q8);
        __m128i hi128 = _mm256_extracti128_si256(q8, 1);
        __m256 qf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128));
        __m256 qf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo128, 8)));
        __m256 qf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128));
        __m256 qf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi128, 8)));
        __m256 p = _mm256_mul_ps(qf0, _mm256_loadu_ps(xb));
        p = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(xb + 8),  p);
        p = _mm256_fmadd_ps(qf2, _mm256_loadu_ps(xb + 16), p);
        p = _mm256_fmadd_ps(qf3, _mm256_loadu_ps(xb + 24), p);
        __m128 slo = _mm256_castps256_ps128(p);
        __m128 shi = _mm256_extractf128_ps(p, 1);
        __m128 sr = _mm_add_ps(slo, shi);
        sr = _mm_hadd_ps(sr, sr);
        sr = _mm_hadd_ps(sr, sr);
        acc += d * _mm_cvtss_f32(sr);
#else
        float s = 0.0f;
        for (int i = 0; i < 32; i++) s += (float)q[i] * xb[i];
        acc += d * s;
#endif
    }
    return acc;
}

static float dot_q4_0(const BlockQ4_0* blocks, const float* x, int64_t n) {
    float acc = 0.0f;
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const float d = fp16_to_fp32_host(blocks[b].d);
        const float* xb = x + b * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; j++) {
            const int lo = blocks[b].qs[j] & 0x0F;
            const int hi = blocks[b].qs[j] >> 4;
            s += (float)(lo - 8) * xb[j];
            s += (float)(hi - 8) * xb[j + 16];
        }
        acc += d * s;
    }
    return acc;
}

static float dot_mxfp4(const BlockMXFP4* blocks, const float* x, int64_t n) {
    float acc = 0.0f;
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const float d = e8m0_to_fp32_mxfp4(blocks[b].e);
        const float* xb = x + b * 32;
#if defined(__AVX2__)
        const __m128i mag_tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12);
        const __m128i b16  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[b].qs));
        const __m128i mask = _mm_set1_epi8(0x0F);
        const __m128i lo   = _mm_and_si128(b16, mask);
        const __m128i hi   = _mm_and_si128(_mm_srli_epi16(b16, 4), mask);
        // two's-complement negation where the nibble's high bit is set
        const __m128i seven = _mm_set1_epi8(7);
        const __m128i neglo = _mm_cmpgt_epi8(lo, seven);
        const __m128i neghi = _mm_cmpgt_epi8(hi, seven);
        const __m128i vlo = _mm_sub_epi8(_mm_xor_si128(_mm_shuffle_epi8(mag_tab, lo), neglo), neglo);
        const __m128i vhi = _mm_sub_epi8(_mm_xor_si128(_mm_shuffle_epi8(mag_tab, hi), neghi), neghi);

        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vlo));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vlo, 8)));
        __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vhi));
        __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vhi, 8)));

        __m256 p = _mm256_mul_ps(f0, _mm256_loadu_ps(xb));
        p = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xb + 8),  p);
        p = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xb + 16), p);
        p = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xb + 24), p);
        __m128 lo128 = _mm256_castps256_ps128(p);
        __m128 hi128 = _mm256_extractf128_ps(p, 1);
        __m128 s = _mm_add_ps(lo128, hi128);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        acc += d * _mm_cvtss_f32(s);
#else
        float s = 0.0f;
        for (int j = 0; j < 16; j++) {
            const int lo = blocks[b].qs[j] & 0x0F;
            const int hi = blocks[b].qs[j] >> 4;
            s += (float)kMXFP4Values[lo] * xb[j];
            s += (float)kMXFP4Values[hi] * xb[j + 16];
        }
        acc += d * s;
#endif
    }
    return acc;
}
// ============================================================
// Model
// ============================================================

bool Model::load(const std::string& path, std::string& err) {
    if (!loader_.load(path)) {
        err = "failed to load GGUF " + path;
        return false;
    }
    if (!load_config(loader_, cfg_, err)) {
        return false;
    }
    if (!loader_.vocab().tokens.empty()) {
        cfg_.vocab_size = (int)loader_.vocab().tokens.size();
    }
    n_threads = (int)std::thread::hardware_concurrency();
    return true;
}

TensorRef Model::tensor(const std::string& name) const {
    TensorRef r;
    const GGUFTensorInfo* ti = loader_.tensor_info(name);
    if (!ti) return r;
    r.data = static_cast<const uint8_t*>(loader_.tensor_data(name));
    r.dtype = ggml_to_dtype(ti->ggml_type);
    for (int i = 0; i < 4; i++) r.ne[i] = (i < (int)ti->shape.size()) ? ti->shape[i] : 1;
    r.name = name;
    return r;
}

float Model::dot_row(const TensorRef& w, int64_t row, const float* x) const {
    const uint8_t* p = w.data + row * w.row_bytes();
    switch (w.dtype) {
        case DType::F32:   return dot_f32(reinterpret_cast<const float*>(p), x, w.ne0());
        case DType::F16:   return dot_f16(reinterpret_cast<const uint16_t*>(p), x, w.ne0());
        case DType::BF16:  return dot_bf16(reinterpret_cast<const uint16_t*>(p), x, w.ne0());
        case DType::Q8_0:  return dot_q8_0(reinterpret_cast<const BlockQ8_0*>(p), x, w.ne0());
        case DType::Q4_0:  return dot_q4_0(reinterpret_cast<const BlockQ4_0*>(p), x, w.ne0());
        case DType::MXFP4: return dot_mxfp4(reinterpret_cast<const BlockMXFP4*>(p), x, w.ne0());
        default:
            NT_CHECK(false, "dot_row: unsupported dtype");
            return 0.0f;
    }
}

void Model::dequant_row(const TensorRef& w, int64_t row, float* out) const {
    const uint8_t* p = w.data + row * w.row_bytes();
    const int64_t n = w.ne0();
    switch (w.dtype) {
        case DType::F32:
            memcpy(out, p, (size_t)n * sizeof(float));
            break;
        case DType::F16: {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
            for (int64_t i = 0; i < n; i++) out[i] = fp16_to_fp32_host(q[i]);
            break;
        }
        case DType::BF16: {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
            for (int64_t i = 0; i < n; i++) out[i] = bf16_to_fp32_host(q[i]);
            break;
        }
        case DType::Q8_0: {
            const BlockQ8_0* q = reinterpret_cast<const BlockQ8_0*>(p);
            for (int64_t b = 0; b < n / 32; b++) {
                const float d = fp16_to_fp32_host(q[b].d);
                for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (float)q[b].qs[i];
            }
            break;
        }
        case DType::MXFP4: {
            const BlockMXFP4* q = reinterpret_cast<const BlockMXFP4*>(p);
            for (int64_t b = 0; b < n / 32; b++) {
                const float d = e8m0_to_fp32_mxfp4(q[b].e);
                for (int j = 0; j < 16; j++) {
                    out[b * 32 + j]      = d * (float)kMXFP4Values[q[b].qs[j] & 0x0F];
                    out[b * 32 + j + 16] = d * (float)kMXFP4Values[q[b].qs[j] >> 4];
                }
            }
            break;
        }
        default:
            fprintf(stderr, "dequant_row: unsupported dtype %s for tensor %s\n", dtype_name(w.dtype), w.name.c_str());
            abort();
    }
}

void Model::embed(int token, float* out) const {
    TensorRef e = tensor("token_embd.weight");
    NT_CHECK(e.valid(), "token_embd.weight missing");
    dequant_row(e, token, out);
}

void Model::matvec(const TensorRef& w, const float* x, float* y) const {
    matvec_slice(w, x, y, 0, w.rows());
}

void Model::matvec_slice(const TensorRef& w, const float* x, float* y, int64_t row0, int64_t nrows) const {
    parallel_for(nrows, n_threads, [&](int64_t r, int) {
        y[r] = dot_row(w, row0 + r, x);
    });
}

} // namespace ds41
} // namespace nt
