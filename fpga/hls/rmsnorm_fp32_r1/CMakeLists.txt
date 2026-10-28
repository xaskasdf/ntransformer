cmake_minimum_required(VERSION 3.24)
project(ntrpu_rmsnorm_reference LANGUAGES CXX)
enable_testing()
set(NTRPU_RMSNORM_FIXTURE "" CACHE FILEPATH "Pinned Llama norm fixture")
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
add_executable(test_ntrpu_hls_rmsnorm_r1 ntrpu_rmsnorm_fp32_r1.cpp
    "${ROOT}/tests/test_ntrpu_hls_rmsnorm_r1.cpp")
target_compile_features(test_ntrpu_hls_rmsnorm_r1 PRIVATE cxx_std_14)
target_include_directories(test_ntrpu_hls_rmsnorm_r1 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../swiglu")
if(MSVC)
    target_compile_options(test_ntrpu_hls_rmsnorm_r1 PRIVATE /fp:strict /W4 /wd4068)
else()
    target_compile_options(test_ntrpu_hls_rmsnorm_r1 PRIVATE -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off)
endif()
if(NTRPU_RMSNORM_FIXTURE)
    add_test(NAME ntrpu_rmsnorm_real_weights
        COMMAND test_ntrpu_hls_rmsnorm_r1 "${NTRPU_RMSNORM_FIXTURE}")
endif()
