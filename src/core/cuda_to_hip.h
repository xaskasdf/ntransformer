#pragma once

// ============================================================
// CUDA -> HIP compatibility shim (ROCm/AMD build)
//
// The NVIDIA build is unchanged; on HIP this header is
// force-included ahead of every translation unit (host .cpp and device .cu)
// so the project keeps its CUDA spelling and these aliases redirect the
// runtime + fp16 surface to HIP. On NVIDIA this header is a no-op include
// of <cuda_runtime.h>. The symbol set below is exactly what the project
// uses (see plan.md CUDA surface inventory); HIP spellings are 1:1.
// ============================================================

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Device / stream / event
#define cudaSetDevice                       hipSetDevice
#define cudaGetDevice                       hipGetDevice
#define cudaGetDeviceCount                  hipGetDeviceCount
#define cudaGetDeviceProperties             hipGetDeviceProperties
#define cudaDeviceProp                      hipDeviceProp_t
#define cudaDeviceSynchronize               hipDeviceSynchronize
#define cudaStream_t                        hipStream_t
#define cudaStreamCreate                    hipStreamCreate
#define cudaStreamCreateWithFlags           hipStreamCreateWithFlags
#define cudaStreamNonBlocking               hipStreamNonBlocking
#define cudaStreamDestroy                   hipStreamDestroy
#define cudaStreamSynchronize               hipStreamSynchronize
#define cudaStreamWaitEvent                 hipStreamWaitEvent
#define cudaEvent_t                         hipEvent_t
#define cudaEventCreate                     hipEventCreate
#define cudaEventCreateWithFlags            hipEventCreateWithFlags
#define cudaEventDisableTiming              hipEventDisableTiming
#define cudaEventDestroy                    hipEventDestroy
#define cudaEventRecord                     hipEventRecord
#define cudaEventSynchronize                hipEventSynchronize
#define cudaEventElapsedTime                hipEventElapsedTime
#define cudaEventQuery                      hipEventQuery

// Memory
#define cudaMalloc                          hipMalloc
#define cudaFree                            hipFree
#define cudaMallocHost                      hipHostMalloc
#define cudaFreeHost                        hipHostFree
#define cudaHostAlloc                       hipHostAlloc
#define cudaHostAllocDefault                hipHostAllocDefault
#define cudaMemcpy                          hipMemcpy
#define cudaMemcpyAsync                     hipMemcpyAsync
#define cudaMemset                          hipMemset
#define cudaMemsetAsync                     hipMemsetAsync
#define cudaMemGetInfo                      hipMemGetInfo
#define cudaMemcpyHostToDevice              hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice            hipMemcpyDeviceToDevice
#define cudaHostRegister                    hipHostRegister
#define cudaHostRegisterReadOnly            hipHostRegisterReadOnly
#define cudaHostUnregister                  hipHostUnregister

// Peer access (PEER tier, memory/peer_cache.cu)
#define cudaDeviceCanAccessPeer             hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess          hipDeviceEnablePeerAccess
#define cudaMemcpyPeerAsync                 hipMemcpyPeerAsync

// Function attributes
#define cudaFuncSetAttribute                hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
        hipFuncAttributeMaxDynamicSharedMemorySize

// Error handling
#define cudaError_t                         hipError_t
#define cudaSuccess                         hipSuccess
#define cudaErrorMemoryAllocation           hipErrorOutOfMemory
#define cudaGetErrorString                  hipGetErrorString
#define cudaGetLastError                    hipGetLastError
#define cudaErrorPeerAccessAlreadyEnabled   hipErrorPeerAccessAlreadyEnabled
#define cudaErrorNotReady                   hipErrorNotReady

// Math constants (<math_constants.h> resolves here through hip_compat/)
#define CUDART_INF_F                        HIP_INF_F

// Warp shuffles. CUDA's mask is a 32-bit unsigned; ROCm's __shfl_*_sync
// static_asserts a 64-bit mask type. Widening the mask at the call keeps the
// project's CUDA spelling (0xffffffffu, 0xffu << k, NT_WARP_MASK) unchanged.
// Zero-extension is exact on wave32 (gfx10/11/12), where lane i of a CUDA warp
// is lane i of the wavefront and ROCm masks the high half off anyway. A
// function-like macro is not re-expanded inside its own replacement, so the
// inner name reaches ROCm's template.
#define __shfl_sync(mask, ...)      __shfl_sync(static_cast<unsigned long long>(mask), __VA_ARGS__)
#define __shfl_up_sync(mask, ...)   __shfl_up_sync(static_cast<unsigned long long>(mask), __VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down_sync(static_cast<unsigned long long>(mask), __VA_ARGS__)
#define __shfl_xor_sync(mask, ...)  __shfl_xor_sync(static_cast<unsigned long long>(mask), __VA_ARGS__)

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
// CUDA's __dp4a(int, int, int): signed 4 x int8 dot product plus accumulator,
// wrapping (not saturating). __ockl_sdot4 picks the target's dot instruction
// (v_dot4_i32_iu8 on gfx11/12) or an exact emulation.
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    using v4i8 = HIP_vector_base<char, 4>::Native_vec_;
    return __ockl_sdot4(__builtin_bit_cast(v4i8, a), __builtin_bit_cast(v4i8, b), c, false);
}
#endif

#else

#include <cuda_runtime.h>

#endif // USE_HIP || __HIP_PLATFORM_AMD__
