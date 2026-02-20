/*
 * test_nvme_layer.cu — Standalone test for gpu-nvme-direct Layer Loader
 *
 * Reads layer 0 of a GGUF model from NVMe and compares with mmap'd data.
 *
 * Usage: sudo ./test_nvme_layer <gguf_path> <pci_bdf>
 * Example: sudo ./test_nvme_layer /path/to/model.gguf 0000:0b:00.0
 *
 * Assumes the GGUF file has been dd'd to the NVMe starting at LBA 0.
 */

#include <gpunvme/layer_loader.h>
#include "../src/model/loader.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gguf_path> <pci_bdf>\n", argv[0]);
        fprintf(stderr, "Example: sudo %s /path/to/model.gguf 0000:0b:00.0\n", argv[0]);
        return 1;
    }

    const char* gguf_path = argv[1];
    const char* pci_bdf = argv[2];

    fprintf(stderr, "=== NVMe Layer Loader Test ===\n");

    // 1. Parse GGUF header via mmap (still needed for metadata)
    nt::GGUFLoader loader;
    if (!loader.load(gguf_path)) {
        fprintf(stderr, "Failed to load GGUF: %s\n", gguf_path);
        return 1;
    }
    loader.print_info();

    // 2. Compute layer 0 offset and size
    uint64_t layer0_offset = loader.tensor_file_offset("blk.0.attn_q.weight");
    fprintf(stderr, "Layer 0 file offset: %lu (0x%lx)\n", layer0_offset, layer0_offset);

    const char* names[] = {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"
    };

    size_t total_bytes = 0;
    for (auto& n : names) {
        auto* info = loader.tensor_info(n);
        if (!info) {
            fprintf(stderr, "Missing tensor: %s\n", n);
            return 1;
        }
        fprintf(stderr, "  %s: %zu bytes\n", n, info->nbytes);
        total_bytes += info->nbytes;
    }
    fprintf(stderr, "Layer 0 total: %zu bytes (%.1f MB)\n\n", total_bytes, total_bytes / 1e6);

    // 3. Init Layer Loader
    gpunvme_layer_loader_t nvme;
    gpunvme_err_t err = gpunvme_layer_loader_init(&nvme, pci_bdf, total_bytes, 32);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Layer loader init failed (err=%d)\n", err);
        return 1;
    }

    uint32_t block_size = gpunvme_layer_loader_block_size(&nvme);
    fprintf(stderr, "NVMe block_size=%u, MDTS=%uK\n",
            block_size, gpunvme_layer_loader_max_transfer(&nvme) / 1024);

    // 4. Allocate destination buffer and read layer 0
    // Align up to block size for NVMe read
    size_t read_bytes = (total_bytes + block_size - 1) & ~(size_t)(block_size - 1);

    void* buf;
    cudaError_t cerr = cudaMallocHost(&buf, read_bytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(cerr));
        gpunvme_layer_loader_destroy(&nvme);
        return 1;
    }
    memset(buf, 0xDE, read_bytes);

    // GGUF starts at LBA 0 on the NVMe
    uint64_t start_lba = layer0_offset / block_size;
    size_t block_offset = layer0_offset % block_size;

    fprintf(stderr, "Reading: start_lba=%lu, size=%zu bytes, block_offset=%zu\n",
            start_lba, total_bytes, block_offset);

    err = gpunvme_load_layer(&nvme, start_lba, read_bytes, buf);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "FAIL: gpunvme_load_layer failed (err=%d)\n", err);
        cudaFreeHost(buf);
        gpunvme_layer_loader_destroy(&nvme);
        return 1;
    }

    // 5. Compare first tensor with mmap'd data
    const uint8_t* mmap_data = (const uint8_t*)loader.tensor_data("blk.0.attn_q.weight");
    const uint8_t* nvme_data = (const uint8_t*)buf + block_offset;

    auto* first_info = loader.tensor_info("blk.0.attn_q.weight");
    size_t cmp_bytes = first_info->nbytes;
    if (cmp_bytes > total_bytes) cmp_bytes = total_bytes;

    int mismatch = 0;
    for (size_t i = 0; i < cmp_bytes; i++) {
        if (nvme_data[i] != mmap_data[i]) {
            if (mismatch < 10) {
                fprintf(stderr, "MISMATCH at byte %zu: nvme=0x%02x mmap=0x%02x\n",
                        i, nvme_data[i], mmap_data[i]);
            }
            mismatch++;
        }
    }

    if (mismatch == 0) {
        fprintf(stderr, "PASS: NVMe data matches mmap (%zu bytes verified)\n", cmp_bytes);
    } else {
        fprintf(stderr, "FAIL: %d mismatches in %zu bytes\n", mismatch, cmp_bytes);
    }

    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&nvme);
    return mismatch > 0 ? 1 : 0;
}
