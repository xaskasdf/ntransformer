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
#define cudaGetDeviceCount                  hipGetDeviceCount
#define cudaGetDeviceProperties             hipGetDeviceProperties
#define cudaDeviceProp                      hipDeviceProp_t
#define cudaDeviceSynchronize               hipDeviceSynchronize
#define cudaStream_t                        hipStream_t
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

// Memory
#define cudaMalloc                          hipMalloc
#define cudaFree                            hipFree
#define cudaMallocHost                      hipHostMalloc
#define cudaFreeHost                        hipHostFree
#define cudaMemcpy                          hipMemcpy
#define cudaMemcpyAsync                     hipMemcpyAsync
#define cudaMemset                          hipMemset
#define cudaMemGetInfo                      hipMemGetInfo
#define cudaMemcpyHostToDevice              hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice            hipMemcpyDeviceToDevice
#define cudaHostRegister                    hipHostRegister
#define cudaHostRegisterReadOnly            hipHostRegisterReadOnly

// Function attributes
#define cudaFuncSetAttribute                hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
        hipFuncAttributeMaxDynamicSharedMemorySize

// Error handling
#define cudaError_t                         hipError_t
#define cudaSuccess                         hipSuccess
#define cudaErrorMemoryAllocation           hipErrorOutOfMemory
#define cudaGetErrorString                  hipGetErrorString

#else

#include <cuda_runtime.h>

#endif // USE_HIP || __HIP_PLATFORM_AMD__
