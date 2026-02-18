#include "../src/core/tensor.h"
#include "../src/core/device.h"
#include "../src/cuda/kernels.h"
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

using namespace nt;

// Compare with tolerance
bool approx_eq(float a, float b, float tol = 1e-3f) {
    return fabsf(a - b) < tol;
}

void test_gemv_f32() {
    fprintf(stderr, "test_gemv_f32...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    // W: [4, 3], x: [3], y: [4]
    // Simple matrix-vector multiply
    int M = 4, K = 3;

    float W_cpu[] = {
        1, 2, 3,   // row 0
        4, 5, 6,   // row 1
        7, 8, 9,   // row 2
        10, 11, 12  // row 3
    };
    float x_cpu[] = {1, 1, 1};
    float expected[] = {6, 15, 24, 33};

    // Upload to GPU
    auto W_gpu = Tensor::empty({M, K}, DType::F32, Device::CUDA);
    auto x_gpu = Tensor::empty({K}, DType::F32, Device::CUDA);
    auto y_gpu = Tensor::zeros({M}, DType::F32, Device::CUDA);

    nt_cuda_memcpy_h2d(W_gpu.data(), W_cpu, sizeof(W_cpu));
    nt_cuda_memcpy_h2d(x_gpu.data(), x_cpu, sizeof(x_cpu));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_gemv(y_gpu.data_as<float>(), W_gpu.data(), x_gpu.data_as<float>(),
        M, K, DType::F32, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    // Download result
    float y_result[4];
    nt_cuda_memcpy_d2h(y_result, y_gpu.data(), sizeof(y_result));

    for (int i = 0; i < M; i++) {
        if (!approx_eq(y_result[i], expected[i])) {
            fprintf(stderr, "  FAIL: y[%d] = %f, expected %f\n", i, y_result[i], expected[i]);
            return;
        }
    }

    fprintf(stderr, "  PASS\n");
}

void test_gemv_q4_0() {
    fprintf(stderr, "test_gemv_q4_0...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    // Create a simple Q4_0 quantized matrix
    // W: [2, 32] (minimum: 1 block per row)
    int M = 2, K = 32;

    // Create Q4_0 blocks manually
    // Each block: half scale + 16 bytes of nibbles (32 weights)
    BlockQ4_0 blocks[2];  // 2 rows, 1 block each

    // Row 0: all weights = 1.0 (quantized: scale=1/8, quant_val=8+1=9)
    // Q4_0 stores vals 0-15, subtracts 8 -> range [-8,7]
    // For val=1.0 with scale d: quant = round(1.0/d) + 8
    // Let d=0.5, so quant = 2+8 = 10, lo=10&0xF=10, hi=10
    {
        float scale = 0.5f;
        uint16_t h;
        // Convert float to half
        uint32_t f;
        memcpy(&f, &scale, 4);
        uint32_t sign = (f >> 31) & 1;
        int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (f >> 13) & 0x3FF;
        if (exp <= 0) { exp = 0; mant = 0; }
        else if (exp >= 31) { exp = 31; mant = 0; }
        h = (sign << 15) | (exp << 10) | mant;
        blocks[0].d = h;

        // All weights = (10-8)*0.5 = 1.0
        for (int j = 0; j < 16; j++) {
            blocks[0].qs[j] = (10 << 4) | 10;  // hi=10, lo=10
        }
    }

    // Row 1: all weights = -0.5 (quant = round(-0.5/0.5)+8 = 7)
    {
        float scale = 0.5f;
        uint16_t h;
        uint32_t f;
        memcpy(&f, &scale, 4);
        uint32_t sign = (f >> 31) & 1;
        int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (f >> 13) & 0x3FF;
        if (exp <= 0) { exp = 0; mant = 0; }
        else if (exp >= 31) { exp = 31; mant = 0; }
        h = (sign << 15) | (exp << 10) | mant;
        blocks[1].d = h;

        // All weights = (7-8)*0.5 = -0.5
        for (int j = 0; j < 16; j++) {
            blocks[1].qs[j] = (7 << 4) | 7;  // hi=7, lo=7
        }
    }

    // x = all 1.0
    float x_cpu[32];
    for (int i = 0; i < 32; i++) x_cpu[i] = 1.0f;

    // Expected: row0 = 32 * 1.0 = 32.0, row1 = 32 * (-0.5) = -16.0
    float expected[] = {32.0f, -16.0f};

    // Upload
    auto W_gpu = Tensor::empty({(int64_t)(sizeof(blocks))}, DType::Q4_0, Device::CUDA);
    // Actually, we need to allocate raw bytes for the quantized data
    void* w_ptr = nt_cuda_malloc(sizeof(blocks));
    nt_cuda_memcpy_h2d(w_ptr, blocks, sizeof(blocks));

    auto x_gpu = Tensor::empty({K}, DType::F32, Device::CUDA);
    auto y_gpu = Tensor::zeros({M}, DType::F32, Device::CUDA);
    nt_cuda_memcpy_h2d(x_gpu.data(), x_cpu, sizeof(x_cpu));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_gemv(y_gpu.data_as<float>(), w_ptr, x_gpu.data_as<float>(),
        M, K, DType::Q4_0, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    float y_result[2];
    nt_cuda_memcpy_d2h(y_result, y_gpu.data(), sizeof(y_result));

    for (int i = 0; i < M; i++) {
        fprintf(stderr, "  y[%d] = %f (expected %f)\n", i, y_result[i], expected[i]);
        if (!approx_eq(y_result[i], expected[i], 0.1f)) {
            fprintf(stderr, "  FAIL\n");
            nt_cuda_free(w_ptr);
            return;
        }
    }

    nt_cuda_free(w_ptr);
    fprintf(stderr, "  PASS\n");
}

void test_silu_mul() {
    fprintf(stderr, "test_silu_mul...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    int N = 4;
    float gate_cpu[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float up_cpu[] = {1.0f, 1.0f, 1.0f, 1.0f};

    auto gate_gpu = Tensor::empty({N}, DType::F32, Device::CUDA);
    auto up_gpu = Tensor::empty({N}, DType::F32, Device::CUDA);
    auto out_gpu = Tensor::empty({N}, DType::F32, Device::CUDA);

    nt_cuda_memcpy_h2d(gate_gpu.data(), gate_cpu, N * sizeof(float));
    nt_cuda_memcpy_h2d(up_gpu.data(), up_cpu, N * sizeof(float));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_silu_mul(out_gpu.data_as<float>(), gate_gpu.data_as<float>(),
        up_gpu.data_as<float>(), N, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    float result[4];
    nt_cuda_memcpy_d2h(result, out_gpu.data(), N * sizeof(float));

    // SiLU(0)*1 = 0, SiLU(1)*1 ≈ 0.731, SiLU(-1)*1 ≈ -0.269, SiLU(2)*1 ≈ 1.762
    float expected[] = {0.0f, 0.731f, -0.269f, 1.762f};
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "  silu_mul[%d] = %f (expected ~%f)\n", i, result[i], expected[i]);
        if (!approx_eq(result[i], expected[i], 0.01f)) {
            fprintf(stderr, "  FAIL\n");
            return;
        }
    }

    fprintf(stderr, "  PASS\n");
}

void test_rmsnorm() {
    fprintf(stderr, "test_rmsnorm...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    int hidden = 4;
    float input_cpu[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight_cpu[] = {1.0f, 1.0f, 1.0f, 1.0f};

    auto input_gpu = Tensor::empty({1, hidden}, DType::F32, Device::CUDA);
    auto weight_gpu = Tensor::empty({hidden}, DType::F32, Device::CUDA);
    auto output_gpu = Tensor::empty({1, hidden}, DType::F32, Device::CUDA);

    nt_cuda_memcpy_h2d(input_gpu.data(), input_cpu, hidden * sizeof(float));
    nt_cuda_memcpy_h2d(weight_gpu.data(), weight_cpu, hidden * sizeof(float));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_rmsnorm(output_gpu.data_as<float>(), input_gpu.data_as<float>(),
        weight_gpu.data_as<float>(), 1, hidden, 1e-5f, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    float result[4];
    nt_cuda_memcpy_d2h(result, output_gpu.data(), hidden * sizeof(float));

    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // output = input / RMS * weight
    float rms = sqrtf(7.5f);
    for (int i = 0; i < hidden; i++) {
        float expected = input_cpu[i] / rms;
        fprintf(stderr, "  rmsnorm[%d] = %f (expected %f)\n", i, result[i], expected);
        if (!approx_eq(result[i], expected, 0.01f)) {
            fprintf(stderr, "  FAIL\n");
            return;
        }
    }

    fprintf(stderr, "  PASS\n");
}

int main() {
    fprintf(stderr, "=== GEMM/Kernel Tests ===\n");

    test_gemv_f32();
    test_gemv_q4_0();
    test_silu_mul();
    test_rmsnorm();

    fprintf(stderr, "\nAll kernel tests passed!\n");
    return 0;
}
