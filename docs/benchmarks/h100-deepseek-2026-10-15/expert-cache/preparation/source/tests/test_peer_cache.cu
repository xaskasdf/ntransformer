#include "memory/peer_cache.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) { \
    fprintf(stderr, "FAIL: %s: %s\n", #call, cudaGetErrorString(error)); return 1; } } while (0)
#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL: %s\n", #condition); return 1; } } while (0)

__global__ void consume(const uint32_t* source, uint32_t* output, size_t count, uint32_t salt) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = source[index] ^ salt;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        fprintf(stderr, "SKIP: peer cache requires two visible CUDA GPUs\n");
        return 77;
    }
    int capable = 0;
    CHECK(cudaDeviceCanAccessPeer(&capable, 0, 1));
    if (!capable) {
        fprintf(stderr, "SKIP: native CUDA P2P access unavailable\n");
        return 77;
    }
    CHECK(cudaSetDevice(0));
    constexpr size_t count = 1ULL << 18;
    constexpr size_t bytes = count * sizeof(uint32_t);
    nt::PeerCache cache;
    REQUIRE(!cache.init(0, 0, bytes));
    REQUIRE(!cache.init(0, devices, bytes));
    REQUIRE(cache.init(0, 1, bytes));
    REQUIRE(cache.allocate(3) == 3);
    REQUIRE(cache.allocated_bytes() == 3 * bytes);
    int current = -1;
    CHECK(cudaGetDevice(&current));
    REQUIRE(current == 0);
    std::vector<uint32_t> source(count);
    for (int block = 0; block < 3; ++block) {
        for (size_t i = 0; i < count; ++i) source[i] = static_cast<uint32_t>(i * 17) ^ (0x12340000u + block);
        // Split upload also checks nonzero offsets in each fixed-size block.
        REQUIRE(cache.upload(block, 0, source.data(), bytes / 2));
        REQUIRE(cache.upload(block, bytes / 2, source.data() + count / 2, bytes / 2));
    }
    REQUIRE(!cache.upload(3, 0, source.data(), bytes));
    REQUIRE(!cache.upload(0, bytes - 1, source.data(), bytes));
    CHECK(cudaGetDevice(&current));
    REQUIRE(current == 0);

    cudaStream_t compute, transfers[2];
    cudaEvent_t copied[2], consumed[2];
    uint32_t* staging[2];
    uint32_t* output[2];
    CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    for (int slot = 0; slot < 2; ++slot) {
        CHECK(cudaStreamCreateWithFlags(&transfers[slot], cudaStreamNonBlocking));
        CHECK(cudaEventCreateWithFlags(&copied[slot], cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&consumed[slot], cudaEventDisableTiming));
        CHECK(cudaMalloc(&staging[slot], bytes));
        CHECK(cudaMalloc(&output[slot], bytes));
        CHECK(cudaEventRecord(consumed[slot], compute));
    }
    REQUIRE(!cache.copy_to_compute(0, staging[0], bytes + 1, transfers[0]));
    std::vector<uint32_t> result(count);
    for (int pass = 0; pass < 8; ++pass) {
        for (int block = 0; block < 3; ++block) {
            int slot = block % 2;
            uint32_t salt = 0xabc00000u + pass;
            CHECK(cudaStreamWaitEvent(transfers[slot], consumed[slot], 0));
            REQUIRE(cache.copy_to_compute(block, staging[slot], bytes, transfers[slot]));
            CHECK(cudaEventRecord(copied[slot], transfers[slot]));
            CHECK(cudaStreamWaitEvent(compute, copied[slot], 0));
            consume<<<(count + 255) / 256, 256, 0, compute>>>(staging[slot], output[slot], count, salt);
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(consumed[slot], compute));
            CHECK(cudaStreamSynchronize(compute));
            CHECK(cudaMemcpy(result.data(), output[slot], bytes, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < count; ++i) {
                uint32_t expected = static_cast<uint32_t>(i * 17) ^ (0x12340000u + block) ^ salt;
                REQUIRE(result[i] == expected);
            }
        }
    }
    REQUIRE(cache.transfer_count() == 24);
    REQUIRE(cache.transferred_bytes() == 24 * bytes);
    // Teardown must preserve the caller's current device and allow reinitialization.
    CHECK(cudaSetDevice(1));
    cache.shutdown();
    CHECK(cudaGetDevice(&current));
    REQUIRE(current == 1);
    REQUIRE(cache.block_count() == 0);
    REQUIRE(cache.init(0, 1, bytes));
    CHECK(cudaGetDevice(&current));
    REQUIRE(current == 1);
    REQUIRE(cache.allocate(1) == 1);
    cache.shutdown();
    CHECK(cudaSetDevice(0));
    for (int slot = 0; slot < 2; ++slot) {
        CHECK(cudaFree(staging[slot]));
        CHECK(cudaFree(output[slot]));
        CHECK(cudaEventDestroy(copied[slot]));
        CHECK(cudaEventDestroy(consumed[slot]));
        CHECK(cudaStreamDestroy(transfers[slot]));
    }
    CHECK(cudaStreamDestroy(compute));
    fprintf(stderr, "PASS: native P2P allocation, upload, copies, numerical output and teardown\n");
    return 0;
}
