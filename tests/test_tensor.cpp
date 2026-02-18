#include "../src/core/tensor.h"
#include "../src/core/device.h"
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>

using namespace nt;

void test_tensor_basic() {
    fprintf(stderr, "test_tensor_basic...\n");

    // CPU tensor creation
    auto t = Tensor::empty({3, 4}, DType::F32, Device::CPU);
    assert(t.ndim() == 2);
    assert(t.size(0) == 3);
    assert(t.size(1) == 4);
    assert(t.numel() == 12);
    assert(t.nbytes() == 48);
    assert(t.dtype() == DType::F32);
    assert(t.device() == Device::CPU);
    assert(t.is_contiguous());

    fprintf(stderr, "  %s\n", t.to_string().c_str());
    fprintf(stderr, "  PASS\n");
}

void test_tensor_zeros() {
    fprintf(stderr, "test_tensor_zeros...\n");

    auto t = Tensor::zeros({2, 3}, DType::F32, Device::CPU);
    float* data = t.data_as<float>();
    for (int i = 0; i < 6; i++) {
        assert(data[i] == 0.0f);
    }

    fprintf(stderr, "  PASS\n");
}

void test_tensor_from_ptr() {
    fprintf(stderr, "test_tensor_from_ptr...\n");

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_ptr(data, {2, 3}, DType::F32, Device::CPU);

    assert(t.is_view());
    assert(t.data_as<float>()[0] == 1.0f);
    assert(t.data_as<float>()[5] == 6.0f);

    fprintf(stderr, "  PASS\n");
}

void test_tensor_view() {
    fprintf(stderr, "test_tensor_view...\n");

    auto t = Tensor::empty({2, 3, 4}, DType::F32, Device::CPU);
    float* data = t.data_as<float>();
    for (int i = 0; i < 24; i++) data[i] = (float)i;

    // Reshape
    auto v = t.view({6, 4});
    assert(v.size(0) == 6);
    assert(v.size(1) == 4);
    assert(v.data_as<float>()[0] == 0.0f);
    assert(v.data_as<float>()[23] == 23.0f);

    // Reshape with -1
    auto v2 = t.view({-1, 4});
    assert(v2.size(0) == 6);

    fprintf(stderr, "  PASS\n");
}

void test_tensor_move() {
    fprintf(stderr, "test_tensor_move...\n");

    auto t1 = Tensor::empty({4, 4}, DType::F32, Device::CPU);
    float* orig_data = t1.data_as<float>();
    orig_data[0] = 42.0f;

    // Move constructor
    Tensor t2 = std::move(t1);
    assert(t2.data_as<float>()[0] == 42.0f);
    assert(t1.data() == nullptr);

    // Move assignment
    Tensor t3;
    t3 = std::move(t2);
    assert(t3.data_as<float>()[0] == 42.0f);

    fprintf(stderr, "  PASS\n");
}

void test_tensor_gpu() {
    fprintf(stderr, "test_tensor_gpu...\n");

    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "  SKIP (no GPU)\n");
        return;
    }

    // Create CPU tensor with data
    auto cpu = Tensor::empty({4}, DType::F32, Device::CPU);
    float* data = cpu.data_as<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;

    // Transfer to GPU
    auto gpu = cpu.to(Device::CUDA);
    assert(gpu.device() == Device::CUDA);
    assert(gpu.numel() == 4);

    // Transfer back to CPU
    auto cpu2 = gpu.to(Device::CPU);
    float* data2 = cpu2.data_as<float>();
    for (int i = 0; i < 4; i++) {
        assert(data2[i] == data[i]);
    }

    fprintf(stderr, "  PASS\n");
}

void test_tensor_dtype_sizes() {
    fprintf(stderr, "test_tensor_dtype_sizes...\n");

    assert(dtype_size(DType::F32) == 4);
    assert(dtype_size(DType::F16) == 2);
    assert(dtype_size(DType::Q4_0) == 18);
    assert(dtype_size(DType::Q8_0) == 36);

    assert(dtype_block_size(DType::F32) == 1);
    assert(dtype_block_size(DType::Q4_0) == 32);
    assert(dtype_block_size(DType::Q8_0) == 32);

    // 1024 elements in Q4_0 = 1024/32 * 18 = 576 bytes
    assert(dtype_row_size(DType::Q4_0, 1024) == 576);

    fprintf(stderr, "  PASS\n");
}

int main() {
    fprintf(stderr, "=== Tensor Tests ===\n");

    test_tensor_basic();
    test_tensor_zeros();
    test_tensor_from_ptr();
    test_tensor_view();
    test_tensor_move();
    test_tensor_dtype_sizes();
    test_tensor_gpu();

    fprintf(stderr, "\nAll tensor tests passed!\n");
    return 0;
}
