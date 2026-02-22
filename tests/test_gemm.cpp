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

// Helper: float to FP16 bits
static uint16_t float_to_half(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);
    uint32_t sign = (f >> 31) & 1;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0) { exp = 0; mant = 0; }
    else if (exp >= 31) { exp = 31; mant = 0; }
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

void test_gemv_q6_k() {
    fprintf(stderr, "test_gemv_q6_k...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    // W: [2, 256] Q6_K (1 block per row, 2 rows)
    int M = 2, K = 256;

    BlockQ6_K blocks[2];
    memset(blocks, 0, sizeof(blocks));

    // Row 0: all weights = 1.0
    // q6 = 33 → lower4 = 1, upper2 = 2 → dequant = 1.0 * 1 * (33-32) = 1.0
    // ql packing: for each l in 0..31:
    //   ql[l] = (lower4[l+64] << 4) | lower4[l] = (1<<4)|1 = 0x11
    //   ql[l+32] = (lower4[l+96] << 4) | lower4[l+32] = 0x11
    // qh packing: for each l in 0..31:
    //   qh[l] = (upper2[l+96]<<6)|(upper2[l+64]<<4)|(upper2[l+32]<<2)|upper2[l]
    //         = (2<<6)|(2<<4)|(2<<2)|2 = 0xAA
    blocks[0].d = float_to_half(1.0f);
    for (int i = 0; i < 16; i++) blocks[0].scales[i] = 1;
    for (int i = 0; i < 128; i++) blocks[0].ql[i] = 0x11;
    for (int i = 0; i < 64; i++) blocks[0].qh[i] = 0xAA;

    // Row 1: all weights = -1.0
    // q6 = 31 → lower4 = 15 (0xF), upper2 = 1 → dequant = 1.0 * 1 * (31-32) = -1.0
    // ql[l] = (0xF<<4)|0xF = 0xFF
    // qh[l] = (1<<6)|(1<<4)|(1<<2)|1 = 0x55
    blocks[1].d = float_to_half(1.0f);
    for (int i = 0; i < 16; i++) blocks[1].scales[i] = 1;
    for (int i = 0; i < 128; i++) blocks[1].ql[i] = 0xFF;
    for (int i = 0; i < 64; i++) blocks[1].qh[i] = 0x55;

    // x = all 1.0
    std::vector<float> x_cpu(K, 1.0f);

    // Expected: row0 = 256 * 1.0 = 256.0, row1 = 256 * (-1.0) = -256.0
    float expected[] = {256.0f, -256.0f};

    // Upload to GPU
    void* w_ptr = nt_cuda_malloc(sizeof(blocks));
    nt_cuda_memcpy_h2d(w_ptr, blocks, sizeof(blocks));

    auto x_gpu = Tensor::empty({K}, DType::F32, Device::CUDA);
    auto y_gpu = Tensor::zeros({M}, DType::F32, Device::CUDA);
    nt_cuda_memcpy_h2d(x_gpu.data(), x_cpu.data(), K * sizeof(float));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_gemv(y_gpu.data_as<float>(), w_ptr, x_gpu.data_as<float>(),
        M, K, DType::Q6_K, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    float y_result[2];
    nt_cuda_memcpy_d2h(y_result, y_gpu.data(), sizeof(y_result));

    for (int i = 0; i < M; i++) {
        fprintf(stderr, "  y[%d] = %f (expected %f)\n", i, y_result[i], expected[i]);
        if (!approx_eq(y_result[i], expected[i], 0.5f)) {
            fprintf(stderr, "  FAIL\n");
            nt_cuda_free(w_ptr);
            return;
        }
    }

    nt_cuda_free(w_ptr);
    fprintf(stderr, "  PASS\n");
}

void test_gemv_q2_k() {
    fprintf(stderr, "test_gemv_q2_k...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    // W: [2, 256] Q2_K (1 super-block per row, 2 rows)
    // BlockQ2_K layout: scales[16] + qs[64] + d(FP16) + dmin(FP16) = 84 bytes
    int M = 2, K = 256;

    BlockQ2_K blocks[2];
    memset(blocks, 0, sizeof(blocks));

    // Row 0: all weights = 1.0
    // d=1.0, dmin=0.0, sc_val=1 (scales[s] low nibble), min_val=0 (high nibble)
    // q=1 for all weights: qs byte = 0x55 (bit pairs: 01 01 01 01)
    // w = d * sc_val * q - dmin * min_val = 1.0 * 1 * 1 - 0.0 * 0 = 1.0
    // x=1.0 → sum = 256 * 1.0 = 256.0
    {
        blocks[0].d    = float_to_half(1.0f);
        blocks[0].dmin = float_to_half(0.0f);
        for (int s = 0; s < 16; s++) blocks[0].scales[s] = 0x01;  // sc=1, min=0
        for (int i = 0; i < 64; i++) blocks[0].qs[i] = 0x55;      // q=1 for all 4 weights per byte
    }

    // Row 1: all weights = -3.0
    // d=1.0, dmin=1.0, sc_val=2 (scales[s] low nibble), min_val=3 (high nibble)
    // q=0 for all weights: qs byte = 0x00
    // w = d * sc_val * q - dmin * min_val = 1.0 * 2 * 0 - 1.0 * 3 = -3.0
    // x=1.0 → sum = 256 * (-3.0) = -768.0
    {
        blocks[1].d    = float_to_half(1.0f);
        blocks[1].dmin = float_to_half(1.0f);
        for (int s = 0; s < 16; s++) blocks[1].scales[s] = 0x32;  // sc=2 (low nibble), min=3 (high nibble)
        for (int i = 0; i < 64; i++) blocks[1].qs[i] = 0x00;      // q=0 for all weights
    }

    std::vector<float> x_cpu(K, 1.0f);
    float expected[] = {256.0f, -768.0f};

    void* w_ptr = nt_cuda_malloc(sizeof(blocks));
    nt_cuda_memcpy_h2d(w_ptr, blocks, sizeof(blocks));

    auto x_gpu = Tensor::empty({K}, DType::F32, Device::CUDA);
    auto y_gpu = Tensor::zeros({M}, DType::F32, Device::CUDA);
    nt_cuda_memcpy_h2d(x_gpu.data(), x_cpu.data(), K * sizeof(float));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_gemv(y_gpu.data_as<float>(), w_ptr, x_gpu.data_as<float>(),
        M, K, DType::Q2_K, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    float y_result[2];
    nt_cuda_memcpy_d2h(y_result, y_gpu.data(), sizeof(y_result));

    for (int i = 0; i < M; i++) {
        fprintf(stderr, "  y[%d] = %f (expected %f)\n", i, y_result[i], expected[i]);
        if (!approx_eq(y_result[i], expected[i], 0.5f)) {
            fprintf(stderr, "  FAIL\n");
            nt_cuda_free(w_ptr);
            return;
        }
    }

    nt_cuda_free(w_ptr);
    fprintf(stderr, "  PASS\n");
}

// Test Q6_K GEMV with large in_features (forces USE_SMEM=false path)
void test_gemv_q6_k_large() {
    fprintf(stderr, "test_gemv_q6_k_large (no-smem path)...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    // W: [2, 256*128=32768] Q6_K → 128 blocks per row, smem=128KB > 64KB
    int M = 2, K = 256 * 128;
    int blocks_per_row = K / 256;  // 128

    size_t total_blocks = M * blocks_per_row;
    std::vector<BlockQ6_K> blocks(total_blocks);
    memset(blocks.data(), 0, total_blocks * sizeof(BlockQ6_K));

    // Row 0: all weights = 1.0 (same encoding as small test)
    for (int b = 0; b < blocks_per_row; b++) {
        auto& blk = blocks[0 * blocks_per_row + b];
        blk.d = float_to_half(1.0f);
        for (int i = 0; i < 16; i++) blk.scales[i] = 1;
        for (int i = 0; i < 128; i++) blk.ql[i] = 0x11;
        for (int i = 0; i < 64; i++) blk.qh[i] = 0xAA;
    }

    // Row 1: all weights = -1.0
    for (int b = 0; b < blocks_per_row; b++) {
        auto& blk = blocks[1 * blocks_per_row + b];
        blk.d = float_to_half(1.0f);
        for (int i = 0; i < 16; i++) blk.scales[i] = 1;
        for (int i = 0; i < 128; i++) blk.ql[i] = 0xFF;
        for (int i = 0; i < 64; i++) blk.qh[i] = 0x55;
    }

    // x = all 1.0
    std::vector<float> x_cpu(K, 1.0f);

    // Expected: row0 = 32768 * 1.0, row1 = 32768 * (-1.0)
    float expected[] = {(float)K, -(float)K};

    size_t w_bytes = total_blocks * sizeof(BlockQ6_K);
    void* w_ptr = nt_cuda_malloc(w_bytes);
    nt_cuda_memcpy_h2d(w_ptr, blocks.data(), w_bytes);

    auto x_gpu = Tensor::empty({K}, DType::F32, Device::CUDA);
    auto y_gpu = Tensor::zeros({M}, DType::F32, Device::CUDA);
    nt_cuda_memcpy_h2d(x_gpu.data(), x_cpu.data(), K * sizeof(float));

    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    cuda::launch_gemv(y_gpu.data_as<float>(), w_ptr, x_gpu.data_as<float>(),
        M, K, DType::Q6_K, stream);
    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);

    float y_result[2];
    nt_cuda_memcpy_d2h(y_result, y_gpu.data(), sizeof(y_result));

    for (int i = 0; i < M; i++) {
        fprintf(stderr, "  y[%d] = %f (expected %f)\n", i, y_result[i], expected[i]);
        if (!approx_eq(y_result[i], expected[i], 1.0f)) {
            fprintf(stderr, "  FAIL\n");
            nt_cuda_free(w_ptr);
            return;
        }
    }

    nt_cuda_free(w_ptr);
    fprintf(stderr, "  PASS\n");
}

int main() {
    fprintf(stderr, "=== GEMM/Kernel Tests ===\n");

    test_gemv_f32();
    test_gemv_q4_0();
    test_gemv_q2_k();
    test_gemv_q6_k();
    test_gemv_q6_k_large();
    test_silu_mul();
    test_rmsnorm();

    fprintf(stderr, "\nAll kernel tests passed!\n");
    return 0;
}
