# Blackwell (sm_120) Optimizations

NVIDIA Blackwell GPUs (RTX 5060 Ti, 5080, 5090, B100, B200) and Hopper (H100, H200)
introduce hardware features that can improve streaming inference performance beyond
what's available on older architectures.

This document tracks implemented and planned optimizations for sm_90+ hardware.

---

## 1. Native sm_120 Codegen ✅ (implemented)

**Branch:** `feat/blackwell-sm120`

Without `120` in `CMAKE_CUDA_ARCHITECTURES`, Blackwell falls back to PTX JIT
compilation at first launch. JIT adds ~2-5 seconds of startup latency and misses
architecture-specific instruction scheduling.

**Fix:** Add `120` to the CUDA architectures list (see `CMakeLists.txt`).

**Required build toolchain:**
- gcc-14 / g++-14 (gcc-15 is incompatible with CUDA 13.1)
- CUDA Toolkit 13.1+
- CMake flag: `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"`

---

## 2. In-Kernel cp.async.bulk for GEMV Input Vector 💡 (planned)

**Branch:** `feat/tma-async-bulk-copy` (stub infrastructure only)

### What TMA (cp.async.bulk) actually is

Hopper (sm_90) and Blackwell (sm_120) introduce `cp.async.bulk` PTX instructions.
These are **in-kernel** operations — a warp issues the instruction and continues
executing while the hardware DMA unit completes the copy asynchronously:

```ptx
cp.async.bulk.global.shared::cta.bulk_group [smem_dst], [gmem_src], nbytes;
cp.async.bulk.commit_group;
// ... warp does other work ...
cp.async.bulk.wait_group 0;   // wait for all pending bulk copies
fence.proxy.async;
```

**Important:** `cp.async.bulk` is a kernel-side instruction (global → shared memory
within a running kernel). It is **not** a replacement for CPU-initiated
`cudaMemcpyAsync` H2D transfers.

### What cudaMemcpyAsync already does

`cudaMemcpyAsync(dev_dst, pinned_src, n, cudaMemcpyHostToDevice, stream)` uses the
GPU's **Copy Engine (CE)** — a dedicated DMA unit separate from the SM array. No SM
warps are consumed during pinned H2D transfers. `cudaMemcpyAsync` is already the
correct and optimal primitive for the staging-buffer → GPU-buffer path in
`LayerStreamer::begin_h2d()`.

### Where cp.async.bulk can help: GEMV shared-memory prefetch

In the GEMV kernels (`src/cuda/gemm.cu`), the `USE_SMEM=true` path loads the input
vector `x[]` into shared memory with a scalar for loop:

```cpp
// Current: scalar loads, one element per thread per iteration
for (int i = flat_id; i < in_features; i += nthreads) {
    sx[i] = x[i];   // ld.global.ca + st.shared — warp stalls on each load
}
__syncthreads();
```

On sm_90+, `cp.async.bulk` can submit the entire vector load as a single async
operation, allowing warps to begin computing the first super-block's partial sum
while the DMA unit fills the rest of shared memory:

```ptx
// Planned: bulk async load, warp continues immediately
cp.async.bulk.global.shared::cta.bulk_group [sx], [x], in_features * 4;
cp.async.bulk.commit_group;
// compute partial sum for super-block 0 using warp-local x values
// ...
cp.async.bulk.wait_group 0;   // ensure all of sx[] is ready before full use
fence.proxy.async;
```

**Estimated benefit:** 10-20% GEMV throughput improvement on sm_90+ for large
`in_features` (e.g. 8192 for 70B hidden dim, 28672 for FFN projections).

**Implementation steps:**
1. Add `cp.async.bulk` PTX wrapper in `src/cuda/gemm.cu` guarded by
   `__CUDA_ARCH__ >= 900`
2. Replace the scalar load loop in `gemv_q2_k_kernel`, `gemv_q4_k_kernel`, and
   `gemv_q6_k_kernel` with the bulk async path
3. Ensure alignment: `in_features * sizeof(float)` must be a multiple of 16 bytes
   (guaranteed when `in_features` is a multiple of 4, which all supported models satisfy)
4. Benchmark with and without `--streaming` on Llama 3.1 70B Q2_K

---

## 3. D2D Scatter Optimization after BAR1 Bulk Read 💡 (planned)

**Context:** The BAR1 NVMe path (`gpunvme_load_layer_vram`) reads a full layer span
into a VRAM temp buffer, then scatters 7 per-tensor chunks to their final GPU
addresses via `cudaMemcpyAsync DeviceToDevice`:

```cpp
// In worker_loop() after gpunvme_load_layer_vram:
for (int t = 0; t < nlay.n_tensors; t++) {
    cudaMemcpyAsync(gpu_dst[t], vram_temp + offset[t], nbytes[t],
                    cudaMemcpyDeviceToDevice, stream);
}
```

`cudaMemcpyDeviceToDevice` for small tensors (2–256 MB) **does use SM warps** via
the `__cudaMemcpy` kernel. For 7 tensors per layer × 80 layers = 560 small D2D
copies per token, this adds measurable SM occupancy pressure.

**Planned optimization:** Replace per-tensor D2D copies with a single in-kernel
gather-scatter using `cp.async.bulk` (global→global on sm_90+) or a custom
CUDA kernel that writes all tensors in one pass. This frees SM warps during
the scatter phase and can overlap with the next layer's NVMe read.

**Implementation steps:**
1. Write a `scatter_kernel<<<n_tensors, 256>>>` that reads from `vram_temp` and
   writes to each tensor's final GPU address using coalesced 128-byte stores
2. On sm_90+, investigate `cp.async.bulk` global→global variant (if available)
3. Benchmark: scatter kernel vs 7× `cudaMemcpyAsync D2D` on 70B Q6_K

---

## 4. Warp Specialization 💡 (concept)

Blackwell's warp group instructions allow dedicated **transfer warps** and
**compute warps** within a single kernel. In the streaming GEMV context:

- Transfer warps issue `cp.async.bulk` for the next layer's weights to shared mem
- Compute warps execute GEMV on the current layer's weights in shared mem

This eliminates the pipeline event overhead (~5-15 μs per layer) and achieves
tighter overlap than the current stream-based double-buffer approach.

**Status:** Concept only. Requires persistent-kernel architecture (fused
layer-load + GEMV in a single long-running kernel). Not scheduled.

---

## Summary

| Optimization | Status | Benefit | Location |
|---|---|---|---|
| Native sm_120 codegen | ✅ Done | -2-5s startup, better scheduling | `CMakeLists.txt` |
| H2D via CE DMA | ✅ Already correct | `cudaMemcpyAsync` already CE | `streamer.cu` |
| GEMV cp.async.bulk (global→shared) | 💡 Planned | ~10-20% GEMV throughput | `gemm.cu` |
| D2D scatter kernel (post-BAR1) | 💡 Planned | Frees SM during scatter | `streamer.cu` |
| Warp specialization | 💡 Concept | Eliminates event overhead | Requires refactor |
