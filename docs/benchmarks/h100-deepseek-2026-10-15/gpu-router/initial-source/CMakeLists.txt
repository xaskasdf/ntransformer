cmake_minimum_required(VERSION 3.24)
project(ntransformer LANGUAGES CXX)

# ============================================================
# NTransformer - High-Efficiency LLM Inference Engine
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NTRANSFORMER_ENABLE_CUDA "Build the CUDA inference engine" ON)
option(NTRANSFORMER_BUILD_NTRPU_SIM "Build the CPU-only NTRPU simulator" ON)
include(CTest)

if(NTRANSFORMER_BUILD_NTRPU_SIM)
    add_subdirectory(src/ntrpu)
endif()

if(NOT NTRANSFORMER_ENABLE_CUDA)
    message(STATUS "CUDA inference engine: disabled (no CUDA toolkit required)")
    return()
endif()

# CUDA 13.1 supports C++20 with gcc-14
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - target SM 86 (RTX 3090) and above
# Also include SM 80 (A100) for compatibility
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Optimization flags (Linux/gcc-14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler=-march=native")
# Fast transcendental intrinsics must not leak from inference into training.
# Overriding fmad/ftz/div/sqrt alone does not undo --use_fast_math's expf mapping.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>,$<NOT:$<BOOL:$<TARGET_PROPERTY:NTRANSFORMER_PRECISE_MATH>>>>:--use_fast_math>")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# CUDA Toolkit
# ============================================================
# Force gcc-14 as CUDA host compiler (gcc-15 is incompatible with CUDA 13.1)
if(NOT CMAKE_CUDA_HOST_COMPILER)
    find_program(GCC14 gcc-14)
    if(GCC14)
        set(CMAKE_CUDA_HOST_COMPILER ${GCC14})
    endif()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

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
)

# ============================================================
# gpu-nvme-direct integration (optional)
# ============================================================
option(USE_GPUNVME "Enable gpu-nvme-direct backend for NVMe streaming" OFF)
option(NTRANSFORMER_BUILD_TRAINING "Build experimental training CUDA operations" OFF)

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
add_library(ntransformer_lib STATIC ${CORE_SOURCES} ${CUDA_SOURCES})

# Shared-expert arithmetic retains CPU Q8 reduction order and uses the precise
# CUDA math library; do not inherit inference's --use_fast_math expf mapping.
add_library(ntransformer_ds41_shared OBJECT src/model/ds41/ds41_shared.cu)
set_target_properties(ntransformer_ds41_shared PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF NTRANSFORMER_PRECISE_MATH ON)
target_sources(ntransformer_lib PRIVATE $<TARGET_OBJECTS:ntransformer_ds41_shared>)

target_include_directories(ntransformer_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(ntransformer_lib PUBLIC
    CUDA::cudart
)

# Enable position-independent code for potential shared library
set_target_properties(ntransformer_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

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

    add_executable(test_ds41_backbone tests/test_ds41_backbone.cpp)
    target_link_libraries(test_ds41_backbone PRIVATE ntransformer_lib)
    foreach(mode IN ITEMS prologue tail compressor indexer)
        add_test(NAME test_ds41_backbone_${mode} COMMAND test_ds41_backbone ${mode})
        set_tests_properties(test_ds41_backbone_${mode} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    add_test(NAME test_ds41_backbone_tail_serial COMMAND ${CMAKE_COMMAND} -E env
        DS41_GROUPED_TAIL=0 $<TARGET_FILE:test_ds41_backbone> tail)
    set_tests_properties(test_ds41_backbone_tail_serial PROPERTIES SKIP_RETURN_CODE 77)

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

    add_executable(test_ds41_shared tests/test_ds41_shared.cpp)
    target_link_libraries(test_ds41_shared PRIVATE ntransformer_lib)
    add_test(NAME test_ds41_shared COMMAND test_ds41_shared)
    set_tests_properties(test_ds41_shared PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "DS41_VRAM_GB=2")
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
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "gpu-nvme-direct: ${USE_GPUNVME}")

add_test(NAME test_ds41_backbone_prologue_serial COMMAND ${CMAKE_COMMAND} -E env DS41_QKV_READBACK_BATCH=0 $<TARGET_FILE:test_ds41_backbone> prologue)
add_test(NAME test_ds41_moe_overlap_flow_control COMMAND ${CMAKE_COMMAND} -E env DS41_STAGING_POOL=0 DS41_MXFP4_REGISTER_LUT=0 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
add_test(NAME test_ds41_shared_late COMMAND ${CMAKE_COMMAND} -E env DS41_SHARED_EARLY=0 DS41_VRAM_GB=2 $<TARGET_FILE:test_ds41_shared>)

add_executable(test_ds41_router tests/test_ds41_router.cpp)
target_link_libraries(test_ds41_router PRIVATE ntransformer_lib)
add_test(NAME test_ds41_router COMMAND test_ds41_router)
add_test(NAME test_ds41_router_disabled COMMAND ${CMAKE_COMMAND} -E env DS41_GPU=0 $<TARGET_FILE:test_ds41_router> disabled)
