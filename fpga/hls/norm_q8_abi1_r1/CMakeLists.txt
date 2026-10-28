cmake_minimum_required(VERSION 3.24)
project(ntrpu_norm_q8_abi1 LANGUAGES CXX)
enable_testing()
set(NTRPU_NORM_Q8_FIXTURE "" CACHE FILEPATH "Pinned norm/Q8 fixture")
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
add_executable(test_ntrpu_hls_norm_q8_r1 ntrpu_norm_q8_abi1_r1.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../rmsnorm_fp32_r1/ntrpu_rmsnorm_fp32_r1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../quantize_q8_0/ntrpu_quantize_q8_0.cpp"
    "${ROOT}/tests/test_ntrpu_hls_norm_q8_r1.cpp")
target_compile_features(test_ntrpu_hls_norm_q8_r1 PRIVATE cxx_std_14)
target_include_directories(test_ntrpu_hls_norm_q8_r1 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rmsnorm_fp32_r1" "${CMAKE_CURRENT_SOURCE_DIR}/../swiglu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../quantize_q8_0")
if(MSVC)
    target_compile_options(test_ntrpu_hls_norm_q8_r1 PRIVATE /fp:strict /W4 /wd4068)
else()
    target_compile_options(test_ntrpu_hls_norm_q8_r1 PRIVATE -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off -fno-fast-math)
endif()
if(NTRPU_NORM_Q8_FIXTURE)
    add_test(NAME ntrpu_norm_q8_abi1_fixture COMMAND test_ntrpu_hls_norm_q8_r1 "${NTRPU_NORM_Q8_FIXTURE}")
endif()
