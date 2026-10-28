cmake_minimum_required(VERSION 3.24)
project(ntrpu_quantize_abi1 LANGUAGES CXX)
enable_testing()
set(NTRPU_ATTENTION_QUANTIZE_FIXTURE "" CACHE FILEPATH "Pinned norm/Q8 fixture")
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
add_executable(test_ntrpu_hls_attention_quantize_r1 ntrpu_quantize_abi1_r1.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../quantize_q8_0/ntrpu_quantize_q8_0.cpp"
    "${ROOT}/tests/test_ntrpu_hls_attention_quantize_r1.cpp")
target_compile_features(test_ntrpu_hls_attention_quantize_r1 PRIVATE cxx_std_14)
target_include_directories(test_ntrpu_hls_attention_quantize_r1 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../quantize_q8_0")
if(MSVC)
    target_compile_options(test_ntrpu_hls_attention_quantize_r1 PRIVATE /fp:strict /W4 /wd4068)
else()
    target_compile_options(test_ntrpu_hls_attention_quantize_r1 PRIVATE -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off -fno-fast-math)
endif()
if(NTRPU_ATTENTION_QUANTIZE_FIXTURE)
    add_test(NAME ntrpu_quantize_abi1_fixture COMMAND test_ntrpu_hls_attention_quantize_r1 "${NTRPU_ATTENTION_QUANTIZE_FIXTURE}")
endif()
