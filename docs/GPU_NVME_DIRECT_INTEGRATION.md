# gpu-nvme-direct Integration for ntransformer

## Summary

Integrate **gpu-nvme-direct** as an I/O backend in ntransformer, removing
the CPU from the layer streaming data path.

**gpu-nvme-direct project**: `../gpu-nvme-direct`
**Status**: Layer Loader API ready (`gpunvme_layer_loader_init` / `gpunvme_load_layer` / `gpunvme_layer_loader_destroy`).
GPU reads 8.6GB @ 3.35 GB/s from SN740 (PCIe 4.0) via MMIO doorbells, without CPU intervention.

### Current pipeline (CPU bottleneck)

```
NVMe → page cache → mmap → CPU memcpy → staging → H2D DMA → GPU compute
                            (worker thread)   (pinned)    (PCIe)
```

Result: ~0.02 tok/s on 70B Q6_K. The worker thread memcpy is the bottleneck.

### Target pipeline (GPU-autonomous)

```
GPU doorbell write → NVMe DMA → host pinned buffer → GPU compute
  (MMIO to BAR0)     (autonomous)  (no CPU memcpy)    (reads directly)
```

**Note**: In Tier 1, data arrives in host pinned memory (not directly to VRAM).
The GPU reads from pinned memory, which is efficient via PCIe UVA.

---

## Layer Loader API (new)

The Layer Loader API encapsulates all the BAR0 mapping, controller init,
PRP building, and kernel launch boilerplate in 3 calls:

```c
#include <gpunvme/layer_loader.h>

gpunvme_layer_loader_t loader;

// Init: opens BAR0, registers CUDA, inits controller, creates I/O queue, pre-allocs PRP pool
gpunvme_layer_loader_init(&loader, "0000:01:00.0", max_layer_bytes, /*pipeline_depth=*/32);

// Load: rebuilds PRPs for dest, launches GPU kernel, synchronizes
gpunvme_load_layer(&loader, start_lba, size_bytes, dest_pinned);

// Destroy: full cleanup
gpunvme_layer_loader_destroy(&loader);
```

**Helpers**:
- `gpunvme_layer_loader_block_size(&loader)` — NVMe block size (512)
- `gpunvme_layer_loader_max_transfer(&loader)` — MDTS in bytes (1024K SN740, 512K SN530)
- `gpunvme_layer_loader_ns_blocks(&loader)` — total namespace capacity

**Queue state rolls naturally** between calls to `gpunvme_load_layer()` — there is
no reset; CIDs, sq_tail, cq_head, and phase bit continue from where they left off.

**Source code**:
- Header: `gpu-nvme-direct/include/gpunvme/layer_loader.h`
- Impl: `gpu-nvme-direct/src/host/layer_loader.cu`
- Test: `gpu-nvme-direct/tests/test_layer_loader.cu`

---

## Hardware requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA with cudaHostRegisterIoMemory support (RTX 3090 tested) |
| NVMe | Any NVMe on VFIO (WD SN740 PCIe 4.0 x4 tested, SN530 Gen3 also) |
| CPU | AMD Zen 3 tested (Intel should work, P2P reads too) |
| OS | Linux (kernel 6.12+, needs nvidia DKMS patch for follow_pfn) |
| IOMMU | OFF (`amd_iommu=off` in GRUB) |

---

## Integration architecture

### What does NOT change

- `forward_streaming()` in `transformer.cpp` — the pipeline loop remains identical
- `LayerWeightPtrs`, `get_weights()` — the GPU still reads weights from `gpu_buf_[slot]`
- CUDA events, streams, double-buffering — all synchronization is preserved
- `GGUFLoader` — GGUF parsing, metadata, vocab

### What changes

| Component | Before | After |
|---|---|---|
| **Data source** | mmap of GGUF file | Direct NVMe DMA via Layer Loader |
| **CPU worker thread** | memcpy mmap→staging | **Eliminated** (GPU initiates reads) |
| **staging_buf_[]** | 2 pinned buffers for memcpy | **Reused** as NVMe DMA destination |
| **prefetch_staging()** | Queue work to worker thread | `gpunvme_load_layer()` directly to staging |
| **New dependency** | CUDA only | CUDA + libgpunvme_layer_loader + VFIO setup |

---

## Detailed changes by file

### 1. `CMakeLists.txt` — Build system

```cmake
# Add at the top:
option(USE_GPUNVME "Enable gpu-nvme-direct backend for NVMe streaming" OFF)

if(USE_GPUNVME)
    set(GPUNVME_DIR "${CMAKE_SOURCE_DIR}/../gpu-nvme-direct")

    # Include the pre-compiled library (build-hw must exist)
    # Option A: Link against pre-built static libraries
    add_library(gpunvme_layer_loader STATIC IMPORTED)
    set_target_properties(gpunvme_layer_loader PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_layer_loader.a)

    add_library(gpunvme_host STATIC IMPORTED)
    set_target_properties(gpunvme_host PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_host.a)

    add_library(gpunvme_device STATIC IMPORTED)
    set_target_properties(gpunvme_device PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_device.a)

    target_include_directories(ntransformer_lib PRIVATE ${GPUNVME_DIR}/include)
    target_link_libraries(ntransformer_lib PRIVATE
        gpunvme_layer_loader gpunvme_host gpunvme_device)
    target_compile_definitions(ntransformer_lib PRIVATE USE_GPUNVME=1)
endif()
```

### 2. `src/memory/streamer.h` — New members

```cpp
// Add at the top of the file:
#ifdef USE_GPUNVME
#include <gpunvme/layer_loader.h>
#endif

// Add to the LayerStreamer class (private section):
#ifdef USE_GPUNVME
    gpunvme_layer_loader_t nvme_loader_ = {};
    bool nvme_initialized_ = false;

    // Per-layer: start LBA and byte size for NVMe reads
    struct NvmeLayerInfo {
        uint64_t start_lba;     // LBA of first byte of this layer's tensor data
        size_t   total_bytes;   // total bytes for all 7 tensors
    };
    std::vector<NvmeLayerInfo> nvme_layers_;
    uint64_t gguf_start_lba_ = 0;
    uint32_t nvme_block_size_ = 512;
#endif
```

### 3. `src/memory/streamer.cu` — init / prefetch / shutdown

#### 3a. `init()` — after building `layers_[]`

```cpp
#ifdef USE_GPUNVME
    // Read NVMe parameters from environment variables
    const char* nvme_bdf = getenv("GPUNVME_PCI_BDF");
    const char* nvme_lba_str = getenv("GPUNVME_GGUF_LBA");

    if (nvme_bdf && nvme_lba_str) {
        uint64_t gguf_start_lba = strtoull(nvme_lba_str, NULL, 0);

        gpunvme_err_t err = gpunvme_layer_loader_init(
            &nvme_loader_, nvme_bdf, buf_size_, /*pipeline_depth=*/32);

        if (err == GPUNVME_OK) {
            nvme_initialized_ = true;
            gguf_start_lba_ = gguf_start_lba;
            nvme_block_size_ = gpunvme_layer_loader_block_size(&nvme_loader_);

            // Pre-compute per-layer LBAs
            uint64_t data_offset = loader.file_data_offset();
            nvme_layers_.resize(n_layers);
            for (int i = 0; i < n_layers; i++) {
                std::string first = "blk." + std::to_string(i) + ".attn_q.weight";
                uint64_t byte_offset = loader.tensor_file_offset(first);
                nvme_layers_[i].start_lba = gguf_start_lba + (byte_offset / nvme_block_size_);
                nvme_layers_[i].total_bytes = layer_transfer_size(i);
            }

            fprintf(stderr, "LayerStreamer: NVMe backend OK (MDTS=%uK, block=%u)\n",
                    gpunvme_layer_loader_max_transfer(&nvme_loader_) / 1024,
                    nvme_block_size_);
        } else {
            fprintf(stderr, "LayerStreamer: NVMe init failed (%s), fallback to mmap\n",
                    gpunvme_err_str(err));
        }
    }
#endif
```

#### 3b. `prefetch_staging()` — replace body

```cpp
void LayerStreamer::prefetch_staging(int layer_idx, int slot) {
    if (mmap_pinned_) return;

#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        const auto& nlay = nvme_layers_[layer_idx];
        gpunvme_err_t err = gpunvme_load_layer(
            &nvme_loader_, nlay.start_lba, nlay.total_bytes, staging_buf_[slot]);

        if (err == GPUNVME_OK) {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            staging_ready_[slot] = true;
            staging_ready_cv_.notify_all();
            return;
        }
        fprintf(stderr, "LayerStreamer: NVMe read failed for layer %d, fallback\n", layer_idx);
    }
#endif

    // Fallback: CPU worker thread memcpy
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        staging_ready_[slot] = false;
        worker_request_ = {layer_idx, slot};
        worker_has_work_ = true;
    }
    worker_cv_.notify_one();
}
```

#### 3c. `shutdown()` — add cleanup

```cpp
void LayerStreamer::shutdown() {
#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        gpunvme_layer_loader_destroy(&nvme_loader_);
        nvme_initialized_ = false;
    }
#endif
    // ... rest of existing shutdown ...
}
```

### 4. `src/model/loader.h` — Already has the necessary methods

The methods `tensor_file_offset()` and `file_data_offset()` are already implemented
(lines 75-80). No additional changes are needed.

---

## NVMe setup (after each reboot)

```bash
# Single command — loads VFIO, binds, sets power D0, enables Memory+BusMaster:
sudo ./scripts/setup_nvme.sh 0000:01:00.0

# Or manually:
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo modprobe vfio-pci
sudo bash ../gpu-nvme-direct/scripts/setup_vfio.sh 0000:01:00.0
sudo sh -c 'echo on > /sys/bus/pci/devices/0000:01:00.0/power/control'
sudo setpci -s 0000:01:00.0 0x84.W=0x0008
sudo setpci -s 0000:01:00.0 COMMAND=0x0006
```

### Copy the GGUF to the NVMe (one time only)

The GGUF file must be on the raw NVMe (no filesystem), starting at LBA 0:

```bash
# BEFORE binding to VFIO (needs native NVMe driver)
# WARNING: this destroys any data on the NVMe
sudo dd if=/path/to/model.gguf of=/dev/nvme0n1 bs=1M oflag=direct status=progress

# Verify:
ls -la /path/to/model.gguf
# 57,398,476,800 bytes → starts at LBA 0
```

**NOTE**: The NVMe used for gpu-nvme-direct must NOT have a filesystem or be
mounted. It is accessed as a raw block device via VFIO.

---

## Development and testing guide

### Prerequisites

1. **gpu-nvme-direct compiled** with hardware build:
   ```bash
   cd ~/gpu-nvme-direct/build-hw
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF \
     -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
     -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
     -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build . -j$(nproc)
   ```

2. **NVMe setup** (see previous section)

3. **Layer Loader test** passing:
   ```bash
   cd ~/gpu-nvme-direct/build-hw
   sudo ./test_layer_loader 0000:01:00.0       # 4MB, 3/3 tests
   sudo ./test_layer_loader 0000:01:00.0 669   # full layer, 3/3 tests
   ```

### Incremental development flow

#### Step 1: Verify Layer Loader in isolation

Before touching ntransformer, verify that the Layer Loader works with the
exact layer sizes of the target model:

```bash
# Typical layer sizes (70B Llama):
#   Q6_K: ~669 MB per layer (80 layers)
#   Q8_0: ~875 MB per layer (80 layers)
cd ~/gpu-nvme-direct/build-hw
sudo ./test_layer_loader 0000:01:00.0 669   # Q6_K layer size
sudo ./test_layer_loader 0000:01:00.0 875   # Q8_0 layer size
```

Expected: 3/3 tests pass, throughput ~3.3 GB/s (SN740) or ~2.1 GB/s (SN530).

#### Step 2: Standalone integration test

Create a minimal test in ntransformer that uses the Layer Loader to read a
real layer from the GGUF on the NVMe:

```cpp
// tests/test_nvme_layer.cu

#include <gpunvme/layer_loader.h>
#include "../src/model/loader.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gguf_path> <pci_bdf>\n", argv[0]);
        return 1;
    }

    // 1. Parse GGUF header (still need mmap for metadata)
    nt::GGUFLoader loader;
    loader.load(argv[1]);
    loader.print_info();

    // 2. Find layer 0's first tensor offset and total size
    uint64_t data_offset = loader.file_data_offset();
    uint64_t layer0_offset = loader.tensor_file_offset("blk.0.attn_q.weight");
    printf("GGUF data_offset=%lu, layer0_offset=%lu\n", data_offset, layer0_offset);

    // Get all layer 0 tensor sizes
    const char* names[] = {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"
    };
    size_t total_bytes = 0;
    for (auto& n : names) {
        auto* info = loader.tensor_info(n);
        if (info) total_bytes += info->nbytes;
    }
    printf("Layer 0 total: %zu bytes (%.1f MB)\n", total_bytes, total_bytes / 1e6);

    // 3. Init Layer Loader
    gpunvme_layer_loader_t nvme;
    gpunvme_err_t err = gpunvme_layer_loader_init(&nvme, argv[2], total_bytes, 32);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Layer loader init failed: %s\n", gpunvme_err_str(err));
        return 1;
    }

    // 4. Alloc dest buffer and read layer 0
    void* buf;
    cudaMallocHost(&buf, total_bytes);
    memset(buf, 0xDE, total_bytes);

    uint32_t block_size = gpunvme_layer_loader_block_size(&nvme);
    uint64_t start_lba = layer0_offset / block_size;  // GGUF starts at LBA 0

    err = gpunvme_load_layer(&nvme, start_lba, total_bytes, buf);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Load failed: %s\n", gpunvme_err_str(err));
        gpunvme_layer_loader_destroy(&nvme);
        return 1;
    }

    // 5. Compare with mmap'd data
    const uint8_t* mmap_data = (const uint8_t*)loader.tensor_data("blk.0.attn_q.weight");
    const uint8_t* nvme_data = (const uint8_t*)buf;

    // The offset within the NVMe block may differ from the offset within the GGUF
    // if layer0_offset is not a multiple of block_size. Adjust:
    size_t block_offset = layer0_offset % block_size;

    int mismatch = 0;
    for (size_t i = 0; i < 4096 && i < total_bytes; i++) {
        if (nvme_data[block_offset + i] != mmap_data[i]) {
            printf("MISMATCH at byte %zu: nvme=0x%02x mmap=0x%02x\n",
                   i, nvme_data[block_offset + i], mmap_data[i]);
            mismatch++;
            if (mismatch > 10) break;
        }
    }

    if (mismatch == 0) printf("PASS: NVMe data matches mmap (first 4KB)\n");
    else printf("FAIL: %d mismatches\n", mismatch);

    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&nvme);
    return mismatch > 0 ? 1 : 0;
}
```

Compile:
```bash
cd ~/ntransformer/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86

# Compile the test manually (before integrating into CMake):
nvcc -std=c++20 -O2 --compiler-bindir=/usr/bin/gcc-14 -arch=sm_86 \
  -I ~/gpu-nvme-direct/include -I ~/ntransformer/src \
  tests/test_nvme_layer.cu ~/ntransformer/src/model/loader.cpp \
  ~/ntransformer/src/model/config.cpp ~/ntransformer/src/core/tensor.cpp \
  ~/ntransformer/src/core/allocator.cpp \
  -L ~/gpu-nvme-direct/build-hw \
  -lgpunvme_layer_loader -lgpunvme_host -lgpunvme_device \
  -lcudart -lstdc++ -lm -o test_nvme_layer

# Run:
sudo ./test_nvme_layer /path/to/model.gguf 0000:01:00.0
```

#### Step 3: Integrate into LayerStreamer

Once the standalone test passes:
1. Add `USE_GPUNVME` to CMakeLists.txt (see previous section)
2. Add NVMe members to `streamer.h`
3. Modify `init()`, `prefetch_staging()`, `shutdown()`
4. Compile with `-DUSE_GPUNVME=ON`

#### Step 4: End-to-end test with ntransformer

```bash
# Ensure GGUF is on the NVMe (see setup section)
# Build ntransformer with NVMe backend
cd ~/ntransformer/build
cmake .. -DUSE_GPUNVME=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)

# Run benchmark with NVMe backend
sudo GPUNVME_PCI_BDF=0000:01:00.0 GPUNVME_GGUF_LBA=0 \
     ./ntransformer -m /path/to/model.gguf --streaming --benchmark -n 8

# Compare with mmap backend (without env vars)
./ntransformer -m /path/to/model.gguf --streaming --benchmark -n 8
```

Verify:
- Same output (bit-identical tokens)
- stderr shows "LayerStreamer: NVMe backend OK"
- stderr shows throughput ~3.3 GB/s (SN740) per layer read

### Troubleshooting

| Symptom | Probable cause | Fix |
|---------|---------------|-----|
| `layer_loader: failed to open resource0` | VFIO not configured | Run NVMe setup |
| `cudaHostRegisterIoMemory failed` | nvidia driver not patched | Patch os-mlock.c (see gpu-nvme-direct docs) |
| `controller init failed: timeout` | NVMe in D3 / link down | `setpci` commands + power/control |
| `read failed: timeout` at cmd N | PRP list not page-aligned | Verify that `dest_pinned` is page-aligned |
| `NVMe init failed, fallback to mmap` | Any init error | Check stderr, run test_layer_loader first |
| Data mismatch vs mmap | LBA offset miscalculated | Verify `tensor_file_offset()` and block alignment |
| `CSTS.CFS=1` | NVMe fatal error | Power cycle the NVMe (unplug/replug or reboot) |

### Alignment considerations

**GGUF tensor alignment**: Tensors in GGUF are aligned to 32 bytes by
default (GGUF v3). This does NOT match the NVMe block size (512B).

Options:
1. **Read full blocks**: start_lba = floor(byte_offset / 512), read extra
   bytes at the beginning. The offset within the block is applied when parsing the tensors.
   The staging_buf_[] are already large enough.

2. **Align the GGUF to 512B**: Use `gguf-split` or similar to force tensor
   alignment to 512 bytes. This simplifies LBA calculation.

Option 1 is simpler and does not require modifying the GGUF. The overhead of reading
extra bytes is negligible (<512B per layer).

### Performance profiling

To measure where time is spent:

```bash
# 1. NVMe read only (no compute)
# The layer_loader prints throughput to stderr:
# "layer_loader: read 669000000 bytes (1306 cmds) in 315.2 ms — 2023.4 MB/s"

# 2. nsight systems profile
sudo GPUNVME_PCI_BDF=0000:01:00.0 GPUNVME_GGUF_LBA=0 \
     nsys profile -o nvme_streaming \
     ./ntransformer -m /path/to/model.gguf --streaming --benchmark -n 4

# 3. Verify that compute overlap works
# In nsys, the GEMV kernels should overlap with NVMe reads.
# If there are gaps between layers → NVMe read is the pure bottleneck.
```

---

## Known limitations

1. **Linux only**: gpu-nvme-direct requires VFIO, /proc/self/pagemap, etc.
2. **Root required**: for VFIO bind and pagemap reads.
3. **Dedicated NVMe**: the NVMe cannot have a filesystem while used with VFIO.
4. **AMD: writes only**: GPU reads from NVMe BAR0 fail on AMD (CmpltTO). Tier 1
   only needs writes (doorbells); data arrives via NVMe DMA to host memory.
5. **Throughput**: 3.35 GB/s on SN740 (Gen4 via B550, downgraded 8GT/s). 2.1 GB/s on SN530 (Gen3).
6. **gcc-14 required**: gcc-15 is incompatible with CUDA 13.1.

---

## Measured performance numbers

| Metric | Current (mmap+memcpy) | gpu-nvme-direct (SN740) | gpu-nvme-direct (SN530) |
|---|---|---|---|
| I/O throughput | ~1.5-2 GB/s | **3.35 GB/s measured** | 2.1 GB/s measured |
| 1 layer (669MB Q6_K) | ~400ms | **~200ms** | ~315ms |
| 80 layers | ~32s | **~16s** | ~25s |
| tok/s (70B Q6_K) | 0.03 | **0.06** | 0.04 |
| CPU utilization | 100% (1 core memcpy) | ~0% (GPU autonomous) | ~0% |
| 8.6GB sustained | — | **3350 MB/s** | — |

**The main gain is NOT just throughput** — it is removing the CPU from the data path.
This frees CPU cores for other processes and eliminates the synchronous
worker thread bottleneck.

---

## TODO (implementation order)

### Phase 1: Basic integration (uses Layer Loader API)
- [ ] Add `USE_GPUNVME` option to CMakeLists.txt
- [ ] Add `gpunvme_layer_loader_t` to `LayerStreamer`
- [ ] In `init()`: call `gpunvme_layer_loader_init()`, pre-compute per-layer LBAs
- [ ] In `prefetch_staging()`: call `gpunvme_load_layer()` when NVMe is available
- [ ] In `shutdown()`: call `gpunvme_layer_loader_destroy()`
- [ ] Standalone test: read layer 0 from NVMe, compare with mmap
- [ ] End-to-end: `--streaming` with NVMe, verify identical output

### Phase 2: Optimization
- [ ] Eliminate staging buffer: NVMe DMA → gpu_buf_[] directly (the staging
      buffers are host pinned, which is exactly what NVMe DMA needs)
- [ ] Pre-compute PRP lists in init (avoid rebuilding per layer)
- [ ] Launch GPU kernel in transfer stream (not default stream) for overlap

### Phase 3: Eliminate mmap dependency
- [ ] Pure NVMe mode: no need to open the GGUF file with mmap
- [ ] Parse GGUF header by reading first blocks from NVMe
- [ ] Only need BDF + start_lba + model config

---

## Flow diagram: Layer streaming with gpu-nvme-direct

```
Token N forward pass:
                                                            time →
CPU:      [idle]──────────────────────────────────────────────────
GPU SM:   [compute L0][compute L1][compute L2]... [compute L79][norm+head]
GPU MMIO: [doorbell L1 ][doorbell L2 ][doorbell L3 ]...
NVMe DMA: [      DMA L1→staging1    ][DMA L2→staging0   ]...
PCIe H2D: [stg0→gpu0  ][stg1→gpu1  ][stg0→gpu0  ]...
           └──prefill──┘

Note: In Phase 2, staging is eliminated — NVMe DMA goes directly to gpu_buf_[slot]
      (which is already in host pinned memory in Tier 1).
```
