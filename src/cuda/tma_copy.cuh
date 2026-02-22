#pragma once
// Blackwell / Hopper async memory acceleration — infrastructure stub
//
// == What TMA (Tensor Memory Accelerator) actually is ==
//
// Hopper (sm_90) and Blackwell (sm_120) introduce cp.async.bulk PTX
// instructions. These are IN-KERNEL operations for copying data between
// global memory and shared memory while the issuing warp continues
// executing. They are NOT a replacement for cudaMemcpyAsync H2D transfers.
//
//   cp.async.bulk.global.shared::cta [smem_dst], [gmem_src], nbytes;
//   ^--- kernel instruction, warp-issued, dst = shared mem, src = GPU global
//
// == What cudaMemcpyAsync already does for pinned H2D ==
//
// cudaMemcpyAsync(dev_dst, pinned_src, n, cudaMemcpyHostToDevice, stream)
// uses the GPU's Copy Engine (CE) — a dedicated DMA unit separate from the
// SM array. No SM warps are consumed during the transfer. This is already
// the optimal H2D primitive for our staging-buffer → GPU-buffer path.
//
// == Where Blackwell actually helps in this codebase ==
//
// 1. D2D scatter after BAR1 bulk read (streamer.cu):
//    gpunvme_load_layer_vram() + per-tensor cudaMemcpyAsync DeviceToDevice
//    D2D copies DO use SM warps. Replacing with in-kernel cp.async.bulk
//    (global→global) or warp specialization would free those warps.
//    See: docs/BLACKWELL_OPTIMIZATIONS.md §3 (D2D scatter optimization).
//
// 2. In-kernel input-vector prefetch in GEMV (gemm.cu):
//    The USE_SMEM=true path loads x[] with a scalar for loop:
//      for (int i = flat_id; i < in_features; i += nthreads) sx[i] = x[i];
//    On sm_90+, this could use cp.async.bulk (global→shared) to submit
//    the entire vector load asynchronously while warps begin computing
//    the first sub-block. See: docs/BLACKWELL_OPTIMIZATIONS.md §2.
//
// STATUS: Stub — the wrapper below currently calls cudaMemcpyAsync as a
//         placeholder. The real opportunity is in (1) and (2) above.
//
// See docs/BLACKWELL_OPTIMIZATIONS.md for design notes and implementation plan.

#include <cuda_runtime.h>

// Minimum architecture for cp.async.bulk support (Hopper = 900, Blackwell = 1200)
#define NTRANSFORMER_TMA_MIN_ARCH 900

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= NTRANSFORMER_TMA_MIN_ARCH
#  define NTRANSFORMER_TMA_AVAILABLE 1
#else
#  define NTRANSFORMER_TMA_AVAILABLE 0
#endif

namespace nt {
namespace tma {

// ---------------------------------------------------------------------------
// tma_h2d_async — STUB
//
// Currently wraps cudaMemcpyAsync HostToDevice, which already uses the GPU
// Copy Engine (CE) hardware — no SM warps consumed. This function exists as
// a call-site hook for future investigation of D2D or UVA-based alternatives
// that might reduce CPU overhead for very large transfers on Blackwell.
//
// NOTE: cp.async.bulk cannot replace this call directly. That instruction
// is kernel-side (global→shared) and not accessible from CPU code.
// A potential replacement would be a persistent kernel that uses zero-copy
// UVA access to read from pinned host memory via cp.async.bulk → shared,
// but this is experimental and likely slower than CE DMA for large transfers.
//
// Requirements for any future implementation:
//   - dst must be 128-byte aligned (cudaMalloc guarantees 256-byte)
//   - src must be pinned host memory (cudaMallocHost or cudaHostRegister)
// ---------------------------------------------------------------------------
inline cudaError_t tma_h2d_async(
    void*        dst,     // device pointer (128-byte aligned recommended)
    const void*  src,     // pinned host pointer
    size_t       nbytes,  // transfer size in bytes
    cudaStream_t stream   // CUDA stream for sequencing
) {
    // CE DMA path — already hardware-accelerated on all architectures.
    // On Hopper/Blackwell, the CE is improved but the primitive is the same.
    return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream);
}

// ---------------------------------------------------------------------------
// tma_sync_wait — STUB
//
// Waits for the async copy issued by tma_h2d_async to complete.
// For the CE DMA path, a stream event (already in place via
// dev.record_event(transfer_done_[slot], xfer)) is more efficient than a
// full stream sync. This function exists to keep the call site symmetric.
// ---------------------------------------------------------------------------
inline void tma_sync_wait(cudaStream_t stream) {
    // Event-based synchronisation in LayerStreamer::begin_h2d() handles this.
    // A full cudaStreamSynchronize here would block the CPU thread unnecessarily.
    (void)stream;
}

// ---------------------------------------------------------------------------
// tma_available() — runtime check
//
// Returns true if the current device supports cp.async.bulk (sm_90+).
// Useful for guarding in-kernel TMA usage without recompilation.
// ---------------------------------------------------------------------------
inline bool tma_available() {
    int major = 0, device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    return major >= (NTRANSFORMER_TMA_MIN_ARCH / 100);
}

} // namespace tma
} // namespace nt
