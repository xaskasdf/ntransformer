// Experimental first-order training primitives. No dependency on LibTorch's ABI.
// The caller owns all allocations and passes its current CUDA stream.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace nt::cuda {
void launch_rmsnorm(float*, const float*, const float*, int, int, float, void*);
}

namespace nt::training {

__device__ float block_sum(float value) {
    __shared__ float partial[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset; offset /= 2)
            value += __shfl_down_sync(0xffffffff, value, offset);
    }
    __syncthreads();
    return value;
}

__global__ void rms_backward_input(float* dx, float* inverse,
        const float* dy, const float* x, const float* w, int dim, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int64_t base = int64_t(row) * dim;
    float squares = 0, dot = 0;
    for (int col = tid; col < dim; col += 256) {
        const float v = x[base + col];
        squares += v * v;
        dot += dy[base + col] * w[col] * v;
    }
    squares = block_sum(squares);
    __shared__ float inv, correction;
    if (tid == 0) inv = rsqrtf(squares / dim + eps);
    dot = block_sum(dot);
    if (tid == 0) {
        inverse[row] = inv;
        correction = dot * inv * inv / dim;
    }
    __syncthreads();
    for (int col = tid; col < dim; col += 256)
        dx[base + col] = inv * (dy[base + col] * w[col] - x[base + col] * correction);
}

// Deterministic two-stage reduction: no floating-point atomics on norm weights.
__global__ void rms_backward_weight_partial(float* partial, const float* dy,
        const float* x, const float* inverse, int rows, int dim) {
    const int col = blockIdx.x * 128 + threadIdx.x;
    if (col >= dim) return;
    const int first = blockIdx.y * 128;
    float sum = 0;
    for (int row = first; row < min(first + 128, rows); ++row) {
        const int64_t index = int64_t(row) * dim + col;
        sum += dy[index] * (x[index] * inverse[row]);
    }
    partial[int64_t(blockIdx.y) * dim + col] = sum;
}

__global__ void rms_backward_weight_finish(float* dw, const float* partial, int groups, int dim) {
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (col >= dim) return;
    float sum = 0;
    for (int group = 0; group < groups; ++group)
        sum += partial[int64_t(group) * dim + col];
    dw[col] = sum;
}

template<class T> __device__ float as_float(T value) { return float(value); }
template<class T> __device__ T cast(float value) { return T(value); }
template<> __device__ __nv_bfloat16 cast(float value) { return __float2bfloat16_rn(value); }

template<class T>
__global__ void swiglu_forward(T* out, const T* gate, const T* up, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= count) return;
    const float g = as_float(gate[i]);
    // Preserve the BF16 rounding between PyTorch's silu and multiply operations.
    const T silu = cast<T>(g / (1.0f + expf(-g)));
    out[i] = cast<T>(as_float(silu) * as_float(up[i]));
}

template<class T>
__global__ void swiglu_backward(T* dg, T* du, const T* dy,
        const T* gate, const T* up, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= count) return;
    const float g = as_float(gate[i]), grad = as_float(dy[i]);
    const float denominator = 1.0f + expf(-g);
    const float sigmoid = 1.0f / denominator;
    const T silu = cast<T>(g / denominator);
    const T grad_silu = cast<T>(grad * as_float(up[i]));
    du[i] = cast<T>(grad * as_float(silu));
    // Match ATen SiLU's contracted multiply/add. This target also needs precise
    // expf: fast-math's intrinsic changes BF16 ties for small real gradients
    // (e.g. gate=-2.09375), despite matching many random unit-scale probes.
    dg[i] = cast<T>(as_float(grad_silu) * sigmoid * fmaf(g, 1.0f - sigmoid, 1.0f));
}
} // namespace nt::training

extern "C" {
const char* nt_train_error(int status) { return cudaGetErrorString(cudaError_t(status)); }

int nt_train_rms_forward(float* y, const float* x, const float* w,
                        int rows, int dim, float eps, void* stream) {
    if (!y || !x || !w || rows <= 0 || dim <= 0 || !(eps > 0)) return cudaErrorInvalidValue;
    nt::cuda::launch_rmsnorm(y, x, w, rows, dim, eps, stream);
    return cudaGetLastError();
}

int nt_train_rms_backward(float* dx, float* dw, float* inverse, float* partial,
        const float* dy, const float* x, const float* w,
        int rows, int dim, float eps, void* stream) {
    if (!dx || !dw || !inverse || !partial || !dy || !x || !w || rows <= 0 || dim <= 0 || !(eps > 0))
        return cudaErrorInvalidValue;
    const auto s = static_cast<cudaStream_t>(stream);
    const int groups = (rows + 127) / 128;
    nt::training::rms_backward_input<<<rows, 256, 0, s>>>(dx, inverse, dy, x, w, dim, eps);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    nt::training::rms_backward_weight_partial<<<dim3((dim + 127) / 128, groups), 128, 0, s>>>(
        partial, dy, x, inverse, rows, dim);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    nt::training::rms_backward_weight_finish<<<(dim + 255) / 256, 256, 0, s>>>(dw, partial, groups, dim);
    return cudaGetLastError();
}

int nt_train_swiglu_forward(void* y, const void* gate, const void* up,
                           int64_t count, int dtype, void* stream) {
    if (!y || !gate || !up || count <= 0 || (dtype != 0 && dtype != 1)) return cudaErrorInvalidValue;
    const auto s = static_cast<cudaStream_t>(stream);
    const int blocks = int((count + 255) / 256);
    if (dtype == 0)
        nt::training::swiglu_forward<<<blocks, 256, 0, s>>>(
            static_cast<float*>(y), static_cast<const float*>(gate), static_cast<const float*>(up), count);
    else
        nt::training::swiglu_forward<<<blocks, 256, 0, s>>>(
            static_cast<__nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(gate),
            static_cast<const __nv_bfloat16*>(up), count);
    return cudaGetLastError();
}

int nt_train_swiglu_backward(void* dg, void* du, const void* dy, const void* gate,
                            const void* up, int64_t count, int dtype, void* stream) {
    if (!dg || !du || !dy || !gate || !up || count <= 0 || (dtype != 0 && dtype != 1))
        return cudaErrorInvalidValue;
    const auto s = static_cast<cudaStream_t>(stream);
    const int blocks = int((count + 255) / 256);
    if (dtype == 0)
        nt::training::swiglu_backward<<<blocks, 256, 0, s>>>(static_cast<float*>(dg),
            static_cast<float*>(du), static_cast<const float*>(dy), static_cast<const float*>(gate),
            static_cast<const float*>(up), count);
    else
        nt::training::swiglu_backward<<<blocks, 256, 0, s>>>(static_cast<__nv_bfloat16*>(dg),
            static_cast<__nv_bfloat16*>(du), static_cast<const __nv_bfloat16*>(dy),
            static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up), count);
    return cudaGetLastError();
}

int nt_train_enable_peer(int peer) {
    int current = -1, capable = 0;
    auto error = cudaGetDevice(&current);
    if (error != cudaSuccess) return error;
    if (peer == current) return cudaErrorInvalidDevice;
    error = cudaDeviceCanAccessPeer(&capable, current, peer);
    if (error != cudaSuccess) return error;
    if (!capable) return cudaErrorPeerAccessUnsupported;
    error = cudaDeviceEnablePeerAccess(peer, 0);
    if (error == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return cudaSuccess;
    }
    return error;
}

int nt_train_peer_copy(void* dst, int dst_device, const void* src, int src_device,
                       size_t bytes, void* stream) {
    if (!dst || !src || !bytes || dst_device == src_device) return cudaErrorInvalidValue;
    return cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, static_cast<cudaStream_t>(stream));
}
}
