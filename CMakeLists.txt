cmake_minimum_required(VERSION 3.24)
project(ntransformer LANGUAGES CXX)

# ============================================================
# NTransformer - High-Efficiency LLM Inference Engine
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NTRANSFORMER_ENABLE_CUDA "Build the GPU inference engine (CUDA, or HIP with USE_HIP)" ON)
# Backend selection: USE_HIP=ON targets ROCm/AMD (the .cu kernels are compiled
# as HIP); OFF keeps the original CUDA/NVIDIA build unchanged.
option(USE_HIP "Build the GPU kernels with HIP for AMD ROCm targets" OFF)
option(NTRANSFORMER_BUILD_NTRPU_SIM "Build the CPU-only NTRPU simulator" ON)
include(CTest)

if(NTRANSFORMER_BUILD_NTRPU_SIM)
    add_subdirectory(src/ntrpu)
endif()

if(NOT NTRANSFORMER_ENABLE_CUDA)
    message(STATUS "GPU inference engine: disabled (no CUDA/HIP toolkit required)")
    return()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Redistributable builds (CI artifacts) turn these OFF/ON; local builds keep the
# defaults: host code tuned for the build machine, shared CUDA runtime.
option(NTRANSFORMER_NATIVE_ARCH "Tune host code for the build CPU (-march=native); OFF targets x86-64-v3 (AVX2)" ON)
option(NTRANSFORMER_STATIC_CUDART "Link the CUDA runtime statically (no cudart DLL/.so needed at run time)" OFF)
if(NTRANSFORMER_NATIVE_ARCH)
    set(NT_HOST_MARCH "-march=native")
else()
    set(NT_HOST_MARCH "-march=x86-64-v3")
endif()

if(USE_HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    # enable_language(HIP) honors an explicit -DCMAKE_HIP_ARCHITECTURES,
    # otherwise auto-detects the host GPU(s), and otherwise errors on a build
    # host with no AMD GPU.
    if(WIN32)
        # clang-cl (MSVC frontend) uses /O2 style flags; -march is not accepted
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${NT_HOST_MARCH}")
    endif()
    set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
    # Same rule as CUDA below: targets with NTRANSFORMER_PRECISE_MATH opt out.
    add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:Release>,$<NOT:$<BOOL:$<TARGET_PROPERTY:NTRANSFORMER_PRECISE_MATH>>>>:-ffast-math>")
    enable_language(HIP)
    find_package(hip REQUIRED)
else()
    # CUDA 13.1 supports C++20 with gcc-14
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA architecture - target SM 86 (RTX 3090) and above
    # Also include SM 80 (A100) for compatibility
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
    endif()

    # Optimization flags (gcc on Linux, MSVC on Windows)
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${NT_HOST_MARCH}")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler=${NT_HOST_MARCH}")
    endif()
    # Fast transcendental intrinsics must not leak from inference into training.
    # Overriding fmad/ftz/div/sqrt alone does not undo --use_fast_math's expf mapping.
    add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>,$<NOT:$<BOOL:$<TARGET_PROPERTY:NTRANSFORMER_PRECISE_MATH>>>>:--use_fast_math>")

    # Force gcc-14 as CUDA host compiler (gcc-15 is incompatible with CUDA 13.1)
    if(NOT WIN32 AND NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(GCC14 gcc-14)
        if(GCC14)
            set(CMAKE_CUDA_HOST_COMPILER ${GCC14})
        endif()
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ============================================================
# Source files
# ============================================================

# Core C++ sources (compiled with g++)
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/model/config.cpp
    src/model/loader.cpp
    src/model/norm.cpp
    src/model/attention.cpp
    src/model/ffn.cpp
    src/model/transformer.cpp
    src/model/ds41/ds41_config.cpp
    src/model/ds41/ds41_model.cpp
    src/model/ds41/ds41_forward.cpp
    src/model/glm5/glm5_config.cpp
    src/model/glm5/glm5_model.cpp
    src/model/glm5/glm5_forward.cpp
    src/inference/tokenizer.cpp
    src/inference/sampler.cpp
    src/inference/engine.cpp
    src/utils/profiler.cpp
)

# CUDA kernel sources (compiled with nvcc)
set(CUDA_SOURCES
    src/core/device.cu
    src/cuda/rmsnorm.cu
    src/cuda/rotary.cu
    src/cuda/softmax.cu
    src/cuda/gemm.cu
    src/cuda/attention.cu
    src/cuda/elementwise.cu
    src/memory/streamer.cu
    src/memory/peer_cache.cu
    src/model/ds41/ds41_gpu.cu
    src/model/glm5/glm5_gpu.cu
    src/model/glm5/glm5_gpu2.cu
)

# ============================================================
# gpu-nvme-direct integration (optional)
# ============================================================
option(USE_GPUNVME "Enable gpu-nvme-direct backend for NVMe streaming" OFF)
option(NTRANSFORMER_BUILD_TRAINING "Build experimental training CUDA operations" OFF)

if(USE_GPUNVME AND USE_HIP)
    message(FATAL_ERROR "USE_GPUNVME (gpu-nvme-direct) is NVIDIA-only; it cannot be combined with USE_HIP")
endif()
if(NTRANSFORMER_BUILD_TRAINING AND USE_HIP)
    message(FATAL_ERROR "NTRANSFORMER_BUILD_TRAINING is CUDA-only; it cannot be combined with USE_HIP")
endif()
if(NTRANSFORMER_BUILD_TRAINING)
    add_library(ntransformer_training SHARED
        src/training/ops.cu src/training/rotary.cu src/training/cross_entropy.cu src/cuda/rmsnorm.cu)
    target_link_libraries(ntransformer_training PRIVATE CUDA::cudart)
    target_compile_options(ntransformer_training PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true>)
    set_target_properties(ntransformer_training PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON
        LINKER_LANGUAGE CXX NTRANSFORMER_PRECISE_MATH ON)
    target_link_options(ntransformer_training PRIVATE -Wl,--no-undefined)
    if(BUILD_TESTING)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME test_training_queue
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_training_queue.py)
        add_test(NAME test_training_measurement
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_training_measurement.py)
    endif()
endif()

# ============================================================
# Main library (static)
# ============================================================
set(NT_MOE_ENGINES ON)
if(USE_HIP)
    # The DS41/GLM5 MoE kernels assume 32-lane warps (lane = tid & 31, 16..1
    # shuffle reductions), which holds on RDNA (gfx10/11/12, wave32) only. On
    # wave64 targets (gfx9: GCN/CDNA) the build keeps the Llama path and
    # moe_unavailable.cpp stands in for the MoE run_cli entry points.
    foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
        if(arch MATCHES "^gfx9")
            set(NT_MOE_ENGINES OFF)
        endif()
    endforeach()
    if(NOT NT_MOE_ENGINES)
        message(STATUS "DS41/GLM5 MoE engines: disabled (wave64 HIP target in ${CMAKE_HIP_ARCHITECTURES})")
        list(FILTER CORE_SOURCES EXCLUDE REGEX "^src/model/(ds41|glm5)/")
        list(FILTER CUDA_SOURCES EXCLUDE REGEX "^src/model/(ds41|glm5)/")
        list(APPEND CORE_SOURCES src/model/moe_unavailable.cpp)
    endif()
    # Compile the .cu kernels as HIP without renaming any source file.
    set_source_files_properties(${CUDA_SOURCES} tests/test_peer_cache.cu
        src/model/ds41/ds41_shared.cu src/model/ds41/ds41_attention.cu
        PROPERTIES LANGUAGE HIP)
    # nvcc's --use_fast_math keeps IEEE infinities; clang's -ffast-math also
    # implies -ffinite-math-only, under which the MoE kernels' -INFINITY
    # sentinels (top-k selection, indexer padding, pooled softmax max) are UB.
    set_source_files_properties(src/model/ds41/ds41_gpu.cu src/model/glm5/glm5_gpu.cu
        src/model/glm5/glm5_gpu2.cu
        PROPERTIES COMPILE_OPTIONS -fno-finite-math-only)
endif()

add_library(ntransformer_lib STATIC ${CORE_SOURCES} ${CUDA_SOURCES})

if(NT_MOE_ENGINES)
    # Shared-expert arithmetic retains CPU Q8 reduction order and uses the precise
    # CUDA math library; do not inherit inference's --use_fast_math expf mapping.
    add_library(ntransformer_ds41_shared OBJECT
        src/model/ds41/ds41_shared.cu src/model/ds41/ds41_attention.cu)
    set_target_properties(ntransformer_ds41_shared PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF NTRANSFORMER_PRECISE_MATH ON)
    if(USE_HIP)
        # Same -ffast-math opt-out (NTRANSFORMER_PRECISE_MATH); the compat
        # headers must shadow <cuda_runtime_api.h> here as in ntransformer_lib.
        target_include_directories(ntransformer_ds41_shared BEFORE PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/hip_compat)
        target_link_libraries(ntransformer_ds41_shared PRIVATE hip::host)
        set_target_properties(ntransformer_ds41_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    target_sources(ntransformer_lib PRIVATE $<TARGET_OBJECTS:ntransformer_ds41_shared>)
endif()

target_include_directories(ntransformer_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

if(USE_HIP)
    # The compat-shim directory satisfies the project's literal
    # `#include <cuda_runtime.h>` / `<cuda_fp16.h>` with HIP aliases; it is on
    # the path only for this build, so the NVIDIA path is untouched.
    target_include_directories(ntransformer_lib BEFORE PUBLIC
        ${CMAKE_SOURCE_DIR}/src/core/hip_compat
    )
    target_link_libraries(ntransformer_lib PUBLIC hip::host)
    set_target_properties(ntransformer_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
else()
    if(NTRANSFORMER_STATIC_CUDART)
        target_link_libraries(ntransformer_lib PUBLIC CUDA::cudart_static)
    else()
        target_link_libraries(ntransformer_lib PUBLIC CUDA::cudart)
    endif()

    # Enable position-independent code for potential shared library
    set_target_properties(ntransformer_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

if(USE_GPUNVME)
    set(GPUNVME_DIR "${CMAKE_SOURCE_DIR}/../gpu-nvme-direct")

    add_library(gpunvme_layer_loader STATIC IMPORTED)
    set_target_properties(gpunvme_layer_loader PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_layer_loader.a)

    add_library(gpunvme_host STATIC IMPORTED)
    set_target_properties(gpunvme_host PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_host.a)

    add_library(gpunvme_device STATIC IMPORTED)
    set_target_properties(gpunvme_device PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_device.a)

    target_include_directories(ntransformer_lib PUBLIC ${GPUNVME_DIR}/include)
    target_link_libraries(ntransformer_lib PRIVATE
        gpunvme_layer_loader gpunvme_host gpunvme_device)
    target_compile_definitions(ntransformer_lib PUBLIC USE_GPUNVME=1)

    message(STATUS "gpu-nvme-direct: ENABLED (${GPUNVME_DIR})")
endif()

# ============================================================
# Main executable
# ============================================================
add_executable(ntransformer src/main.cpp)
target_link_libraries(ntransformer PRIVATE ntransformer_lib)

# ============================================================
# Tests
# ============================================================
if(BUILD_TESTING)
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE ntransformer_lib)
    add_test(NAME test_tensor COMMAND test_tensor)

    add_executable(test_gemm tests/test_gemm.cpp)
    target_link_libraries(test_gemm PRIVATE ntransformer_lib)
    add_test(NAME test_gemm COMMAND test_gemm)

    add_executable(test_peer_cache tests/test_peer_cache.cu)
    target_link_libraries(test_peer_cache PRIVATE ntransformer_lib)
    add_test(NAME test_peer_cache COMMAND test_peer_cache)
    set_tests_properties(test_peer_cache PROPERTIES SKIP_RETURN_CODE 77)
endif()

# DS41 tests need the MoE engines, which the HIP build leaves out.
if(BUILD_TESTING AND NOT USE_HIP)
    add_executable(test_ds41_cache tests/test_ds41_cache.cpp)
    target_include_directories(test_ds41_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME test_ds41_cache COMMAND test_ds41_cache)
    foreach(placement IN ITEMS 0 copy execute)
        add_test(NAME test_ds41_cache_slru_${placement} COMMAND ${CMAKE_COMMAND} -E env
            DS41_CACHE_POLICY=slru DS41_PEER_MODE=${placement} DS41_MOE_TRACE=1
            DS41_VRAM_GB=1 DS41_PEER_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
        set_tests_properties(test_ds41_cache_slru_${placement} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()

    add_executable(test_ds41_q8 tests/test_ds41_q8.cu)
    target_link_libraries(test_ds41_q8 PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_q8 COMMAND test_ds41_q8)
    set_tests_properties(test_ds41_q8 PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_ds41_backbone tests/test_ds41_backbone.cpp)
    add_executable(test_ds41_router tests/test_ds41_router.cpp)
    add_executable(test_ds41_attention tests/test_ds41_attention.cpp)
    target_link_libraries(test_ds41_attention PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_attention COMMAND test_ds41_attention)
    set_tests_properties(test_ds41_attention PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_attention_disabled COMMAND ${CMAKE_COMMAND} -E env
        DS41_GPU=0 $<TARGET_FILE:test_ds41_attention> disabled)
    target_link_libraries(test_ds41_router PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_router COMMAND test_ds41_router)
    set_tests_properties(test_ds41_router PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_router_disabled COMMAND ${CMAKE_COMMAND} -E env
        DS41_GPU=0 $<TARGET_FILE:test_ds41_router> disabled)
    target_link_libraries(test_ds41_backbone PRIVATE ntransformer_lib)
    foreach(mode IN ITEMS prologue tail compressor indexer resident_indexer)
        add_test(NAME test_ds41_backbone_${mode} COMMAND test_ds41_backbone ${mode})
        set_tests_properties(test_ds41_backbone_${mode} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    add_test(NAME test_ds41_backbone_tail_serial COMMAND ${CMAKE_COMMAND} -E env
        DS41_GROUPED_TAIL=0 $<TARGET_FILE:test_ds41_backbone> tail)
    set_tests_properties(test_ds41_backbone_tail_serial PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_backbone_prologue_serial COMMAND ${CMAKE_COMMAND} -E env
        DS41_QKV_READBACK_BATCH=0 $<TARGET_FILE:test_ds41_backbone> prologue)
    set_tests_properties(test_ds41_backbone_prologue_serial PROPERTIES SKIP_RETURN_CODE 77)

    foreach(mode IN ITEMS prologue tail indexer resident_indexer)
        add_test(NAME test_ds41_backbone_cooperative_${mode} COMMAND ${CMAKE_COMMAND} -E env
            DS41_Q8_COOPERATIVE=1 $<TARGET_FILE:test_ds41_backbone> ${mode})
        set_tests_properties(test_ds41_backbone_cooperative_${mode} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()

    add_executable(test_ds41_prefill tests/test_ds41_prefill.cpp)
    target_link_libraries(test_ds41_prefill PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_prefill COMMAND test_ds41_prefill)
    set_tests_properties(test_ds41_prefill PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_ds41_moe_overlap tests/test_ds41_moe_overlap.cpp)
    target_link_libraries(test_ds41_moe_overlap PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_moe_overlap COMMAND test_ds41_moe_overlap)
    set_tests_properties(test_ds41_moe_overlap PROPERTIES SKIP_RETURN_CODE 77
        ENVIRONMENT "DS41_VRAM_GB=1")
    add_test(NAME test_ds41_moe_overlap_disabled COMMAND ${CMAKE_COMMAND} -E env
        DS41_GPU=0 $<TARGET_FILE:test_ds41_moe_overlap> disabled)
    add_test(NAME test_ds41_moe_overlap_serial COMMAND ${CMAKE_COMMAND} -E env
        DS41_MOE_BATCH=0 DS41_MOE_TRACE=1 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
    set_tests_properties(test_ds41_moe_overlap_serial PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_moe_overlap_trace COMMAND ${CMAKE_COMMAND} -E env
        DS41_MOE_BATCH=1 DS41_MOE_TRACE=1 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
    set_tests_properties(test_ds41_moe_overlap_trace PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_moe_overlap_flow_experimental COMMAND ${CMAKE_COMMAND} -E env
        DS41_STAGING_POOL=1 DS41_MXFP4_REGISTER_LUT=1 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
    set_tests_properties(test_ds41_moe_overlap_flow_experimental PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME test_ds41_moe_overlap_cooperative COMMAND ${CMAKE_COMMAND} -E env
        DS41_MXFP4_COOPERATIVE=1 DS41_MXFP4_REGISTER_LUT=0 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
    set_tests_properties(test_ds41_moe_overlap_cooperative PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_moe_overlap_legacy COMMAND ${CMAKE_COMMAND} -E env
        DS41_MXFP4_COOPERATIVE=0 DS41_MXFP4_REGISTER_LUT=0 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
    set_tests_properties(test_ds41_moe_overlap_legacy PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(test_ds41_mxfp4 tests/test_ds41_mxfp4.cpp)
    target_link_libraries(test_ds41_mxfp4 PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_mxfp4 COMMAND test_ds41_mxfp4)
    set_tests_properties(test_ds41_mxfp4 PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "DS41_VRAM_GB=1")
    add_test(NAME test_ds41_mxfp4_legacy COMMAND ${CMAKE_COMMAND} -E env
        DS41_MXFP4_COOPERATIVE=0 DS41_MXFP4_REGISTER_LUT=0 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_mxfp4> wide)
    set_tests_properties(test_ds41_mxfp4_legacy PROPERTIES SKIP_RETURN_CODE 77)
    foreach(shape IN ITEMS small unaligned wide)
        add_test(NAME test_ds41_mxfp4_cooperative_${shape} COMMAND ${CMAKE_COMMAND} -E env
            DS41_MXFP4_COOPERATIVE=1 DS41_MXFP4_REGISTER_LUT=0 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_mxfp4> ${shape})
        set_tests_properties(test_ds41_mxfp4_cooperative_${shape} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    add_test(NAME test_ds41_mxfp4_cooperative_register COMMAND ${CMAKE_COMMAND} -E env
        DS41_MXFP4_COOPERATIVE=1 DS41_MXFP4_REGISTER_LUT=1 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_mxfp4> wide)
    set_tests_properties(test_ds41_mxfp4_cooperative_register PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_ds41_shared tests/test_ds41_shared.cpp)
    foreach(compute IN ITEMS fp32 f16 q8)
        foreach(shape IN ITEMS small unaligned wide)
            add_test(NAME test_ds41_precision_${compute}_${shape} COMMAND ${CMAKE_COMMAND} -E env
                DS41_MXFP4_COMPUTE=${compute} DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_mxfp4> ${shape})
            set_tests_properties(test_ds41_precision_${compute}_${shape} PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        foreach(peer IN ITEMS copy execute)
            add_test(NAME test_ds41_peer_${peer}_${compute} COMMAND ${CMAKE_COMMAND} -E env
                DS41_PEER_MODE=${peer} DS41_MXFP4_COMPUTE=${compute} DS41_MOE_TRACE=1
                DS41_VRAM_GB=1 DS41_PEER_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
            set_tests_properties(test_ds41_peer_${peer}_${compute} PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
    endforeach()
    add_test(NAME test_ds41_peer_local_only COMMAND ${CMAKE_COMMAND} -E env
        DS41_PEER_MODE=execute DS41_TEST_LOCAL_ONLY=1 DS41_VRAM_GB=1 DS41_PEER_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
    set_tests_properties(test_ds41_peer_local_only PROPERTIES SKIP_RETURN_CODE 77)
    target_link_libraries(test_ds41_shared PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_shared COMMAND test_ds41_shared)
    set_tests_properties(test_ds41_shared PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "DS41_VRAM_GB=2")
    add_test(NAME test_ds41_shared_late COMMAND ${CMAKE_COMMAND} -E env
        DS41_SHARED_EARLY=0 DS41_VRAM_GB=2 $<TARGET_FILE:test_ds41_shared>)
    set_tests_properties(test_ds41_shared_late PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test_ds41_shared_disabled COMMAND ${CMAKE_COMMAND} -E env
        DS41_GPU=0 $<TARGET_FILE:test_ds41_shared> disabled)

    add_executable(test_ds41_parallel tests/test_ds41_parallel.cpp)
    target_link_libraries(test_ds41_parallel PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_parallel_adaptive COMMAND ${CMAKE_COMMAND} -E env
        --unset=DS41_CPU_GRAIN $<TARGET_FILE:test_ds41_parallel>)
    foreach(grain 1 32 1048576 invalid)
        add_test(NAME test_ds41_parallel_${grain} COMMAND ${CMAKE_COMMAND} -E env
            DS41_CPU_GRAIN=${grain} $<TARGET_FILE:test_ds41_parallel>)
    endforeach()

    add_executable(test_ds41_hc tests/test_ds41_hc.cpp)
    target_link_libraries(test_ds41_hc PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_hc COMMAND test_ds41_hc)
endif()

if(BUILD_TESTING)
    if(USE_GPUNVME)
        add_executable(test_nvme_layer tests/test_nvme_layer.cu)
        target_link_libraries(test_nvme_layer PRIVATE ntransformer_lib)
        target_include_directories(test_nvme_layer PRIVATE
            ${CMAKE_SOURCE_DIR}/../gpu-nvme-direct/include)
    endif()
endif()

# ============================================================
# Install
# ============================================================
install(TARGETS ntransformer RUNTIME DESTINATION bin)
install(TARGETS ntransformer_lib ARCHIVE DESTINATION lib)
install(FILES include/ntransformer.h DESTINATION include)

# ============================================================
# Print configuration
# ============================================================
message(STATUS "=== NTransformer Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(USE_HIP)
    message(STATUS "GPU backend: HIP (ROCm)")
    message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
    message(STATUS "GPU backend: CUDA")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
endif()
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "gpu-nvme-direct: ${USE_GPUNVME}")
