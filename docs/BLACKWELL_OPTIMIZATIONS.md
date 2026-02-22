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

## 2. TMA Async Bulk Copy 🔧 (stub — implementation pending)

**Branch:** `feat/tma-async-bulk-copy`
**Stub:** `src/cuda/tma_copy.cuh`
**Call site:** `src/memory/streamer.cu:begin_h2d()` (marked with `TODO(tma)`)

### What is TMA?

Blackwell and Hopper expose a **Tensor Memory Accelerator** (TMA), a hardware unit
that handles bulk data movement between memory tiers independently of warp execution.

Standard `cudaMemcpyAsync` uses warp threads to drive the copy. Under the hood this
means SM resources are occupied by the transfer — warps on the transfer stream are
spinning or stalling on memory ops. TMA replaces this with:

```
cp.async.bulk.global.shared::cta.bulk_group [dst], [src], nbytes;
cp.async.bulk.commit_group;
...
cp.async.bulk.wait_group 0;   // wait for all pending TMA transfers
```

These PTX instructions offload the copy to dedicated DMA hardware, freeing the warp
to issue compute or retire.

### Why it matters for streaming inference

The streaming pipeline spends most of its time (60-70%) on H2D transfers:

```
Per-token timing (32B Q4_K_M, Gen5 x8 PCIe, 18 VRAM + 46 RAM layers):
  Tier A (18 layers):  ~18 × 0.7ms = 12ms   (pure compute, VRAM resident)
  Tier B (46 layers):  ~46 × 8.5ms = 391ms  (8ms H2D + 0.5ms compute)
  Total: ~403ms → 2.5 tok/s theoretical ceiling
```

With TMA:
- H2D warp stalls reduced → more overlap between transfer and compute
- Estimated improvement: 15-25% throughput on Gen4/Gen5 x8

### Implementation plan

1. Write the `cp.async.bulk` PTX wrapper in `src/cuda/tma_copy.cuh`
   - Requires `__CUDA_ARCH__ >= 900` (Hopper) or `>= 1200` (Blackwell native)
   - Use `cuda::ptx::cp_async_bulk` from CUDA 12.4+ CCCL if available
2. Replace `dev.memcpy_h2d_async(...)` in `begin_h2d()` with `nt::tma::tma_h2d_async(...)`
3. Replace `dev.record_event(transfer_done_[slot], xfer)` with `tma_sync_wait()` + event
4. Ensure 128-byte alignment of `gpu_buf_[slot]` allocations (cudaMalloc guarantees 256-byte)
5. Benchmark: compare `--n-buffers 2` and `--n-buffers 3` before and after

### Current status

Infrastructure is in place:
- `src/cuda/tma_copy.cuh` — stub header with `tma_h2d_async()` and `tma_sync_wait()`
  that fall back to `cudaMemcpyAsync` / `cudaStreamSynchronize`
- `src/memory/streamer.cu:begin_h2d()` — marked with `TODO(tma)` at the copy site
- `NTRANSFORMER_TMA_AVAILABLE` compile-time flag for conditional dispatch

**The stub compiles and runs correctly on all hardware** — it just uses standard
async copies. No performance change until the PTX wrapper is filled in.

### References

- CUDA PTX ISA: Tensor Memory Accelerator (TMA) instructions
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
- CUDA 12.4 CCCL `cuda::ptx::cp_async_bulk`:
  https://nvidia.github.io/cccl/libcudacxx/extended_api/ptx.html
- Hopper Architecture Whitepaper — TMA section
  https://resources.nvidia.com/en-us-tensor-core

---

## 3. Warp Specialization for GEMV 💡 (concept)

Blackwell's warp group instructions (`wgmma`) allow dedicated warp groups for
transfer and compute. In the context of the streaming GEMV:

- **Transfer warps**: issue TMA bulk copies for the next layer's weights
- **Compute warps**: execute GEMV on the current layer's weights

This eliminates the pipeline bubble between H2D and compute entirely. Currently
the pipeline uses separate CUDA streams + CUDA events for serialization, which
incurs ~5-15μs event overhead per layer on the critical path.

**Status:** Concept only. Not scheduled for implementation.
