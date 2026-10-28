// Shared by ordinary C++ builds and Vitis C simulation / C-RTL co-simulation.
#include "ntrpu_gemv_int8.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ntrpu::hls;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

static uint32_t random_state = 719;
static int8_t next_value() {
    random_state = random_state * 1664525U + 1013904223U;
    return static_cast<int8_t>(static_cast<int>((random_state >> 16) & 255U) - 128);
}

static void check_shape(uint32_t rows, uint32_t columns, int pattern = 0) {
    // Full interface-depth buffers also permit the RTL test harness to marshal ports.
    std::vector<int8_t> weights(MAX_WEIGHTS), input(MAX_COLUMNS);
    std::vector<int32_t> output(MAX_ROWS, 0x12345678);
    for (auto& value : weights) value = pattern == 0 ? next_value() : int8_t{-128};
    for (auto& value : input) {
        value = pattern == 0 ? next_value() : pattern == 1 ? int8_t{-128} :
                pattern == 2 ? int8_t{127} : int8_t{0};
    }
    const auto original_weights = weights;
    const auto original_input = input;
    CHECK(ntrpu_gemv_int8(weights.data(), input.data(), output.data(), rows, columns) == OK);
    for (uint32_t row = 0; row < rows; ++row) {
        int64_t expected = 0;
        for (uint32_t column = 0; column < columns; ++column) {
            expected += int64_t{weights[row * columns + column]} * input[column];
        }
        CHECK(output[row] == expected);
    }
    for (uint32_t row = rows; row < MAX_ROWS; ++row) CHECK(output[row] == 0x12345678);
    CHECK(weights == original_weights && input == original_input);
}

static void test_shapes() {
    for (uint32_t columns : {1U, 7U, 31U, 32U, 33U, 65U, 1023U, 1024U}) {
        for (uint32_t rows : {1U, 7U, 32U}) check_shape(rows, columns);
    }
}

static void test_extremes() {
    for (int pattern : {1, 2, 3}) check_shape(MAX_ROWS, MAX_COLUMNS, pattern);
}

static void test_invalid() {
    std::vector<int8_t> weights(MAX_WEIGHTS, 127), input(MAX_COLUMNS, -128);
    std::vector<int32_t> output(MAX_ROWS, -918273);
    const uint32_t shapes[][2] = {{0, 32}, {1, 0}, {33, 1}, {1, 1025},
        {std::numeric_limits<uint32_t>::max(), 1}, {1, std::numeric_limits<uint32_t>::max()}};
    for (const auto& shape : shapes) {
        CHECK(ntrpu_gemv_int8(weights.data(), input.data(), output.data(), shape[0], shape[1])
              == INVALID_SHAPE);
        for (auto value : output) CHECK(value == -918273);
    }
}

static void test_repeated() {
    // Different lengths on the same instance expose stale tail/accumulator state.
    for (uint32_t columns : {1024U, 1U, 33U, 32U, 1023U, 7U}) {
        check_shape(5, columns);
        check_shape(3, columns, 3);
    }
}

int main(int argc, char** argv) {
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {{"shapes", test_shapes}, {"extremes", test_extremes},
                          {"invalid", test_invalid}, {"repeated", test_repeated}};
    size_t ran = 0, failed = 0;
    if (argc > 2) return 2;
    for (const auto& test : cases) {
        if (argc == 2 && std::string(argv[1]) != test.name) continue;
        ++ran;
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    return ran != 0 && failed == 0 ? 0 : 1;
}
