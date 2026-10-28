cmake_minimum_required(VERSION 3.24)
project(ntrpu_attention_tile_reference LANGUAGES CXX)
enable_testing()
set(NTRPU_ATTENTION_FIXTURE "" CACHE FILEPATH "Pinned attention fixture")
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
add_executable(test_ntrpu_hls_attention_state_r1 ntrpu_attention_tile_fp32_r1.cpp
    "${ROOT}/tests/test_ntrpu_hls_attention_state_r1.cpp")
target_compile_features(test_ntrpu_hls_attention_state_r1 PRIVATE cxx_std_14)
target_include_directories(test_ntrpu_hls_attention_state_r1 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../swiglu")
if(MSVC)
    target_compile_options(test_ntrpu_hls_attention_state_r1 PRIVATE /fp:strict /W4 /wd4068)
else()
    target_compile_options(test_ntrpu_hls_attention_state_r1 PRIVATE
        -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off -fno-fast-math)
endif()
add_test(NAME ntrpu_attention_state COMMAND test_ntrpu_hls_attention_state_r1)
add_executable(test_ntrpu_hls_attention_fixture_r1 ntrpu_attention_tile_fp32_r1.cpp
    "${ROOT}/tests/test_ntrpu_hls_attention_fixture_r1.cpp")
target_compile_features(test_ntrpu_hls_attention_fixture_r1 PRIVATE cxx_std_14)
target_include_directories(test_ntrpu_hls_attention_fixture_r1 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../swiglu")
if(MSVC)
    target_compile_options(test_ntrpu_hls_attention_fixture_r1 PRIVATE /fp:strict /W4 /wd4068)
else()
    target_compile_options(test_ntrpu_hls_attention_fixture_r1 PRIVATE
        -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off -fno-fast-math)
endif()
if(NTRPU_ATTENTION_FIXTURE)
    add_test(NAME ntrpu_attention_fixture COMMAND test_ntrpu_hls_attention_fixture_r1 "${NTRPU_ATTENTION_FIXTURE}")
endif()
