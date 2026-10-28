// Target-only log softmax for the BF16 Brandon training head.
// The 8192-column reduction follows ATen's register softmax ordering:
// eight strided values/thread, 1024 threads, then two warp reductions.
// Target validation must remain active in Release builds, including replay.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cfloat>
#include <cstdint>

namespace nt::training {

template<bool Maximum>
__device__ float row_reduce(float value, float* scratch) {
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = Maximum ? fmaxf(value, other) : value + other;
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 32 ? scratch[threadIdx.x] : (Maximum ? -FLT_MAX : 0.f);
    if (threadIdx.x < 32) {
        for (int offset = 16; offset; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, value, offset);
            value = Maximum ? fmaxf(value, other) : value + other;
        }
    }
    if (threadIdx.x == 0) scratch[0] = value;
    __syncthreads();
    return scratch[0];
}

struct LabelLayout {
    int length, shift;
    int64_t batch_stride, token_stride;
};

__device__ int64_t read_target(const int64_t* targets, int row, LabelLayout layout) {
    const int token = row % layout.length + layout.shift;
    return token < layout.length
        ? targets[int64_t(row / layout.length) * layout.batch_stride + token * layout.token_stride]
        : -100;
}

template<bool PrepareLabels>
__global__ void target_log_probability(float* output, float* stats, int64_t* proxy,
        const __nv_bfloat16* logits, const int64_t* targets,
        int rows, int columns, int first, int vocabulary, LabelLayout layout) {
    __shared__ float scratch[32];
    const int row = blockIdx.x;
    const int64_t target = PrepareLabels ? read_target(targets, row, layout) : targets[row];
    if constexpr (PrepareLabels) assert(target == -100 || (target >= 0 && target < vocabulary));
    float values[8];
    float maximum = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int column = threadIdx.x + i * 1024;
        values[i] = column < columns ? float(logits[int64_t(row) * columns + column]) : -FLT_MAX;
        maximum = fmaxf(maximum, values[i]);
    }
    maximum = row_reduce<true>(maximum, scratch);
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (threadIdx.x + i * 1024 < columns) sum += expf(values[i] - maximum);
    }
    sum = row_reduce<false>(sum, scratch);
    if (threadIdx.x == 0) {
        if constexpr (PrepareLabels) proxy[row] = target == -100 ? -100 : 0;
        bool present = target >= first && target < first + columns;
        float selected = present ? float(logits[int64_t(row) * columns + target - first]) : 0.f;
        stats[row] = maximum;
        stats[rows + row] = sum;
        stats[2 * rows + row] = selected;
        // Installed PyTorch computes log_softmax in BF16, then NLL in FP32.
        output[row] = present ? float(__float2bfloat16_rn((selected - maximum) - logf(sum))) : 0.f;
    }
}

template<bool PrepareLabels>
__global__ void target_log_probability_backward(__nv_bfloat16* dz,
        const __nv_bfloat16* logits, const float* stats, const int64_t* targets,
        const float* gradient, int rows, int columns, int first, LabelLayout layout) {
    const int row = blockIdx.x;
    const float maximum = stats[row], log_sum = logf(stats[rows + row]);
    const float g = float(__float2bfloat16_rn(gradient[row]));
    const int64_t target = PrepareLabels ? read_target(targets, row, layout) : targets[row];
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int64_t index = int64_t(row) * columns + column;
        float log_probability = float(__float2bfloat16_rn((float(logits[index]) - maximum) - log_sum));
        // ATen log_softmax backward contracts this product/subtraction.
        float value = fmaf(-expf(log_probability), g, target == first + column ? g : 0.f);
        dz[index] = __float2bfloat16_rn(value);
    }
}

} // namespace nt::training

extern "C" int nt_train_target_logp(void* output, void* stats, const void* logits,
        const void* targets, int rows, int columns, int first, int vocabulary, void* stream) {
    if (!output || !stats || !logits || !targets || rows <= 0 || columns <= 0
        || columns > 8192 || first < 0 || first + columns > vocabulary)
        return cudaErrorInvalidValue;
    nt::training::target_log_probability<false><<<rows, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(output), static_cast<float*>(stats), nullptr,
        static_cast<const __nv_bfloat16*>(logits), static_cast<const int64_t*>(targets),
        rows, columns, first, vocabulary, {});
    return cudaGetLastError();
}

extern "C" int nt_train_target_logp_backward(void* dz, const void* logits,
        const void* stats, const void* targets, const void* gradient,
        int rows, int columns, int first, void* stream) {
    if (!dz || !logits || !stats || !targets || !gradient || rows <= 0
        || columns <= 0 || columns > 8192 || first < 0) return cudaErrorInvalidValue;
    nt::training::target_log_probability_backward<false><<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(dz), static_cast<const __nv_bfloat16*>(logits),
        static_cast<const float*>(stats), static_cast<const int64_t*>(targets),
        static_cast<const float*>(gradient), rows, columns, first, {});
    return cudaGetLastError();
}

extern "C" int nt_train_target_logp_labels(void* output, void* stats, void* proxy,
        const void* logits, const void* targets, int rows, int columns, int length,
        int shift, int64_t batch_stride, int64_t token_stride, void* stream) {
    if (!output || !stats || !proxy || !logits || !targets || rows <= 0 || columns != 8192
        || length <= 0 || rows % length || (shift != 0 && shift != 1)
        || batch_stride < 0 || token_stride < 0) return cudaErrorInvalidValue;
    nt::training::LabelLayout layout{length, shift, batch_stride, token_stride};
    nt::training::target_log_probability<true><<<rows, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(output), static_cast<float*>(stats), static_cast<int64_t*>(proxy),
        static_cast<const __nv_bfloat16*>(logits), static_cast<const int64_t*>(targets),
        rows, columns, 0, columns, layout);
    return cudaGetLastError();
}

extern "C" int nt_train_target_logp_labels_backward(void* dz, const void* logits,
        const void* stats, const void* targets, const void* gradient, int rows,
        int columns, int length, int shift, int64_t batch_stride, int64_t token_stride,
        void* stream) {
    if (!dz || !logits || !stats || !targets || !gradient || rows <= 0 || columns != 8192
        || length <= 0 || rows % length || (shift != 0 && shift != 1)
        || batch_stride < 0 || token_stride < 0) return cudaErrorInvalidValue;
    nt::training::LabelLayout layout{length, shift, batch_stride, token_stride};
    nt::training::target_log_probability_backward<true><<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(dz), static_cast<const __nv_bfloat16*>(logits),
        static_cast<const float*>(stats), static_cast<const int64_t*>(targets),
        static_cast<const float*>(gradient), rows, columns, 0, layout);
    return cudaGetLastError();
}
