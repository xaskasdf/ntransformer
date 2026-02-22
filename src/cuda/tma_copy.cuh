#pragma once
// TMA (Tensor Memory Accelerator) async bulk copy — Blackwell sm_120 / Hopper sm_90
//
// NVIDIA Hopper (sm_90) and Blackwell (sm_120) introduce hardware Tensor Memory
// Accelerators (TMA) with cp.async.bulk PTX instructions. Unlike warp-based
// cudaMemcpyAsync, TMA uses dedicated DMA engines that:
//
//   1. Don't consume warp execution slots during transfer
//   2. Allow tighter async overlap between H2D DMA and GPU compute
//   3. Support 128-byte aligned bulk transfers with lower CPU overhead
//
// STATUS: Stub — cp.async.bulk requires sm_90a/sm_120a PTX compilation
//         with __CUDA_ARCH__ ≥ 900. Full implementation is gated behind the
//         NTRANSFORMER_TMA_AVAILABLE compile-time flag.
//
// See docs/BLACKWELL_OPTIMIZATIONS.md for design notes and implementation plan.

#include <cuda_runtime.h>

// Minimum architecture for TMA support (Hopper = 900, Blackwell = 1200)
#define NTRANSFORMER_TMA_MIN_ARCH 900

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= NTRANSFORMER_TMA_MIN_ARCH
#  define NTRANSFORMER_TMA_AVAILABLE 1
#else
#  define NTRANSFORMER_TMA_AVAILABLE 0
#endif

namespace nt {
namespace tma {

// ---------------------------------------------------------------------------
// tma_h2d_async
//
// Initiate an async bulk H2D copy using TMA (when available) or fall back
// to standard cudaMemcpyAsync. The fallback preserves correctness on all
// architectures while the TMA path unlocks hardware-accelerated transfers
// on sm_90+ GPUs.
//
// Requirements for TMA path (not yet implemented):
//   - dst must be 128-byte aligned (use cudaMalloc, which guarantees this)
//   - src must be pinned host memory (cudaMallocHost or cudaHostRegister)
//   - nbytes should be a multiple of 128 bytes for best performance
//
// TODO(tma): implement using cp.async.bulk PTX intrinsic:
//   asm volatile (
//     "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
//     : : "l"(dst_gmem), "l"(src_smem), "r"(nbytes)
//   );
//   asm volatile ("cp.async.bulk.commit_group;");
// ---------------------------------------------------------------------------
inline cudaError_t tma_h2d_async(
    void*        dst,     // device pointer (128-byte aligned recommended)
    const void*  src,     // pinned host pointer
    size_t       nbytes,  // transfer size in bytes
    cudaStream_t stream   // CUDA stream for sequencing
) {
#if NTRANSFORMER_TMA_AVAILABLE
    // TODO(tma): replace with cp.async.bulk when PTX wrapper is implemented
    // For now, fall through to standard async copy
#endif
    return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream);
}

// ---------------------------------------------------------------------------
// tma_sync_wait
//
// Wait for all pending TMA bulk operations to complete on the given stream.
//
// TODO(tma): replace with cp.async.bulk.wait_group 0 to wait only for the
//            TMA group, avoiding a full stream synchronisation:
//   asm volatile ("cp.async.bulk.wait_group 0;");
//   asm volatile ("fence.proxy.async;");
// ---------------------------------------------------------------------------
inline void tma_sync_wait(cudaStream_t stream) {
#if NTRANSFORMER_TMA_AVAILABLE
    // TODO(tma): cp.async.bulk.wait_group 0 + fence.proxy.async
#endif
    cudaStreamSynchronize(stream);
}

// ---------------------------------------------------------------------------
// tma_available() — runtime check
//
// Returns true if the current device supports TMA (compute capability ≥ 9.0).
// Useful for runtime dispatch without recompilation.
// ---------------------------------------------------------------------------
inline bool tma_available() {
    int major = 0, minor = 0, device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return (major * 100 + minor * 10) >= NTRANSFORMER_TMA_MIN_ARCH;
}

} // namespace tma
} // namespace nt
