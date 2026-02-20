cmake_minimum_required(VERSION 3.24)
project(ntransformer LANGUAGES CXX CUDA)

# ============================================================
# NTransformer - High-Efficiency LLM Inference Engine
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CUDA 13.1 supports C++20 with gcc-14
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - target SM 86 (RTX 3090) and above
# Also include SM 80 (A100) for compatibility
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Optimization flags (Linux/gcc-14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -Xcompiler=-march=native")

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
)

# ============================================================
# Main library (static)
# ============================================================
add_library(ntransformer_lib STATIC ${CORE_SOURCES} ${CUDA_SOURCES})

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

# ============================================================
# Main executable
# ============================================================
add_executable(ntransformer src/main.cpp)
target_link_libraries(ntransformer PRIVATE ntransformer_lib)

# ============================================================
# Tests
# ============================================================
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE ntransformer_lib)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE ntransformer_lib)
add_test(NAME test_gemm COMMAND test_gemm)

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
