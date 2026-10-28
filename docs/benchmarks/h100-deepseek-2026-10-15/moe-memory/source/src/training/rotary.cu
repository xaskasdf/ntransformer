// First-order Brandon RoPE. Frequencies are the original complex64 table.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace nt::training {

struct RotaryStrides {
    int64_t batch, token, head, column;
};

template<class T> __device__ T rotary_cast(float value) { return T(value); }
template<> __device__ __nv_bfloat16 rotary_cast(float value) {
    return __float2bfloat16_rn(value);
}

template<class T>
__global__ void rotary_pair(T* out_q, T* out_k, const T* q, const T* k,
        const float2* freqs, int64_t q_pairs, int64_t total_pairs, int64_t length,
        int64_t q_heads, int64_t k_heads, int64_t half_dim,
        RotaryStrides qs, RotaryStrides ks, bool backward) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= total_pairs) return;
    const bool query = i < q_pairs;
    if (!query) i -= q_pairs;
    const T* input = query ? q : k;
    T* output = query ? out_q : out_k;
    const auto strides = query ? qs : ks;
    const int64_t heads = query ? q_heads : k_heads;
    const int64_t column = i % half_dim;
    const int64_t row = i / half_dim;
    const int64_t head = row % heads;
    const int64_t token = (row / heads) % length;
    const int64_t batch = (row / heads) / length;
    const int64_t offset = batch * strides.batch + token * strides.token
                         + head * strides.head + 2 * column * strides.column;
    const float a = float(input[offset]), b = float(input[offset + strides.column]);
    const float2 frequency = freqs[token * half_dim + column];
    const float c = frequency.x, d = backward ? -frequency.y : frequency.y;
    // Preserve ATen complex multiplication's two contractions, including in
    // backward with conjugate frequencies. The other products round first.
    output[2 * i] = rotary_cast<T>(fmaf(a, c, -(b * d)));
    output[2 * i + 1] = rotary_cast<T>(fmaf(a, d, b * c));
}

} // namespace nt::training

extern "C" int nt_train_rotary(void* out_q, void* out_k, const void* q, const void* k,
        const void* freqs, int64_t batch, int64_t length, int64_t q_heads, int64_t k_heads,
        int64_t dim, int64_t qb, int64_t qt, int64_t qh, int64_t qc,
        int64_t kb, int64_t kt, int64_t kh, int64_t kc, int dtype, int backward, void* stream) {
    if (!out_q || !out_k || !q || !k || !freqs || batch <= 0 || length <= 0
        || q_heads <= 0 || k_heads <= 0 || dim <= 0 || dim % 2 || qb < 0 || qt < 0
        || qh < 0 || qc < 0 || kb < 0 || kt < 0 || kh < 0 || kc < 0
        || (dtype != 0 && dtype != 1) || (backward != 0 && backward != 1))
        return cudaErrorInvalidValue;
    const int64_t q_pairs = batch * length * q_heads * (dim / 2);
    const int64_t total = q_pairs + batch * length * k_heads * (dim / 2);
    const auto s = static_cast<cudaStream_t>(stream);
    const nt::training::RotaryStrides qs{qb, qt, qh, qc}, ks{kb, kt, kh, kc};
    if (dtype == 0)
        nt::training::rotary_pair<<<(total + 255) / 256, 256, 0, s>>>(
            static_cast<float*>(out_q), static_cast<float*>(out_k),
            static_cast<const float*>(q), static_cast<const float*>(k),
            static_cast<const float2*>(freqs), q_pairs, total, length,
            q_heads, k_heads, dim / 2, qs, ks, bool(backward));
    else
        nt::training::rotary_pair<<<(total + 255) / 256, 256, 0, s>>>(
            static_cast<__nv_bfloat16*>(out_q), static_cast<__nv_bfloat16*>(out_k),
            static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
            static_cast<const float2*>(freqs), q_pairs, total, length,
            q_heads, k_heads, dim / 2, qs, ks, bool(backward));
    return cudaGetLastError();
}
