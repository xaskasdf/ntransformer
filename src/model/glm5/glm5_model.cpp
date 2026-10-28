#include "glm5_model.h"
#include "glm5_gpu.h"
#include "../ds41/ds41_model.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <thread>
#include <atomic>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nt {
namespace glm5 {

void parallel_for(int64_t n, int n_threads, const std::function<void(int64_t, int)>& fn) {
    nt::ds41::parallel_for(n, n_threads, fn);
}

int64_t TensorRef::row_bytes() const {
    int64_t be = block_elems();
    return (ne[0] / be) * (int64_t)dtype_size(dtype);
}

int64_t TensorRef::block_elems() const {
    return (int64_t)dtype_block_size(dtype);
}

namespace {

inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

float dot_f32(const float* w, const float* x, int64_t n) {
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

float dot_f16(const uint16_t* w, const float* x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) acc += fp16_to_fp32_host(w[i]) * x[i];
    return acc;
}

float dot_bf16(const uint16_t* w, const float* x, int64_t n) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i w0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
        __m256i w1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + 8)));
        w0 = _mm256_slli_epi32(w0, 16);
        w1 = _mm256_slli_epi32(w1, 16);
        a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(w0), _mm256_loadu_ps(x + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(w1), _mm256_loadu_ps(x + i + 8), a1);
    }
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

float dot_q8_0(const BlockQ8_0* blocks, const float* x, int64_t n) {
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

float dot_q4_0(const BlockQ4_0* blocks, const float* x, int64_t n) {
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

float dot_mxfp4(const BlockMXFP4* blocks, const float* x, int64_t n) {
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
            s += (float)kMXFP4Values[blocks[b].qs[j] & 0x0F] * xb[j];
            s += (float)kMXFP4Values[blocks[b].qs[j] >> 4] * xb[j + 16];
        }
        acc += d * s;
#endif
    }
    return acc;
}

} // namespace

bool Model::load(const std::string& path, std::string& err) {
    if (!loader_.load(path)) {
        err = "failed to load GGUF " + path;
        return false;
    }
    if (!load_config(loader_, cfg_, err)) {
        return false;
    }
    const char* pack_dir = getenv("GLM5_BACKBONE_PACK");
    if (pack_dir && pack_dir[0]) {
        // Non-expert tensors from the backbone pack; experts from the raw
        // NVMe store through gpu-nvme-direct (see glm5_gpu.cu).
        FileMap bm;
        if (!map_file_read(std::string(pack_dir) + "/glm5_backbone.bin", bm) || !bm.ptr) {
            err = "backbone pack: cannot map glm5_backbone.bin";
            return false;
        }
        backbone_base_ = static_cast<const uint8_t*>(bm.ptr);
        backbone_size_ = bm.size;

        FILE* f = fopen((std::string(pack_dir) + "/glm5_backbone_manifest.bin").c_str(), "rb");
        if (!f) { err = "backbone pack: missing glm5_backbone_manifest.bin"; return false; }
        char magic[4]; uint32_t n = 0;
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MBN1", 4) != 0 || fread(&n, 4, 1, f) != 1) {
            err = "backbone pack: bad manifest header"; fclose(f); return false;
        }
        for (uint32_t i = 0; i < n; i++) {
            uint16_t nl = 0;
            if (fread(&nl, 2, 1, f) != 1) break;
            std::string name(nl, '\0');
            if (nl && fread(&name[0], 1, nl, f) != nl) break;
            PackTensor pt{};
            uint8_t nd = 0; uint64_t dims[4] = {0, 0, 0, 0}; uint32_t gt = 0;
            if (fread(&pt.offset, 8, 1, f) != 1 || fread(&pt.nbytes, 8, 1, f) != 1 ||
                fread(&nd, 1, 1, f) != 1 || fread(dims, 8, 4, f) != 4 || fread(&gt, 4, 1, f) != 1) break;
            pack_.emplace(std::move(name), pt);
        }
        fclose(f);

        const char* lba = getenv("GLM5_LBA_BIN");
        const char* g = getenv("GLM5_GPNVME");
        if (!lba || !g || atoi(g) == 0 || !glm5_gpnvme_init(lba, getenv("GPUNVME_PCI_BDF"))) {
            err = "backbone pack requires GLM5_GPNVME=1 and a valid GLM5_LBA_BIN";
            return false;
        }
        pack_mode_ = true;
        fprintf(stderr, "pack: %zu backbone tensors (%.1f GB mapped); experts via gpu-nvme-direct\n",
                pack_.size(), backbone_size_ / 1e9);
    }
    n_threads = (int)std::thread::hardware_concurrency();
    return true;
}

TensorRef Model::tensor(const std::string& name) const {
    TensorRef r;
    const GGUFTensorInfo* ti = loader_.tensor_info(name);
    if (!ti) return r;
    r.dtype = ggml_to_dtype(ti->ggml_type);
    for (int i = 0; i < 4; i++) r.ne[i] = (i < (int)ti->shape.size()) ? ti->shape[i] : 1;
    r.name = name;
    if (pack_mode_) {
        auto it = pack_.find(name);
        if (it != pack_.end()) {
            r.data = backbone_base_ + it->second.offset;
            return r;
        }
        // expert tensor living in the raw store: deterministic synthetic base,
        // never dereferenced on the host (fills go through gpu-nvme-direct).
        // name: "blk.<il>.ffn_{gate,up,down}_exps.weight"
        if (name.size() > 8 && name.compare(0, 4, "blk.") == 0) {
            const size_t dot1 = name.find('.', 4);
            if (dot1 != std::string::npos && name.find("_exps.weight") != std::string::npos) {
                const int il = atoi(name.c_str() + 4);
                int kind = -1;
                if (name.compare(dot1 + 1, 9, "ffn_gate_") == 0) kind = 0;
                else if (name.compare(dot1 + 1, 7, "ffn_up_") == 0) kind = 1;
                else if (name.compare(dot1 + 1, 9, "ffn_down_") == 0) kind = 2;
                if (il >= 0 && kind >= 0) {
                    r.data = (const uint8_t*)glm5_gpnvme_fake_base(il, kind);
                    static std::atomic<int> once{0};
                    if (once.fetch_add(1) < 3)
                        fprintf(stderr, "tensor: %s -> fake %p\n", name.c_str(), r.data);
                    return r;
                }
            }
        }
        static std::atomic<int> miss{0};
        if (miss.fetch_add(1) < 3)
            fprintf(stderr, "tensor: %s -> NOT FOUND in pack (no data)\n", name.c_str());
        r.data = nullptr;  // unknown tensor in pack mode: fail loudly
        return r;
    }
    r.data = static_cast<const uint8_t*>(loader_.tensor_data(name));
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

} // namespace glm5
} // namespace nt
