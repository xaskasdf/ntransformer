#include "ntrpu/gemv.h"
#include "ntrpu/numeric_io.h"
#include "ntrpu/q8_0.h"
#include "ntrpu/simulator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace nt::ntrpu;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Function>
static void expect(Status expected, Function function) {
    try { function(); }
    catch (const Error& error) { CHECK(error.status() == expected); return; }
    throw std::runtime_error("expected simulator error");
}

static Completion complete(Fabric& fabric, uint64_t sequence, Status status = Status::Ok) {
    auto result = fabric.poll();
    CHECK(result && result->sequence == sequence && result->status == status);
    return *result;
}

static std::vector<std::byte> read_bytes(Fabric& fabric, BufferSlice slice, size_t size) {
    std::vector<std::byte> output(size);
    fabric.read(slice, output);
    return output;
}

static std::vector<std::byte> weights(size_t rows, size_t columns, std::mt19937& random) {
    std::vector<std::byte> result(rows * (columns / 32) * 34);
    const uint16_t scales[] = {0x3400, 0x3800, 0x4000};
    for (size_t offset = 0; offset < result.size(); offset += 34) {
        const auto scale = scales[(offset / 34) % 3];
        result[offset] = static_cast<std::byte>(scale & 255U);
        result[offset + 1] = static_cast<std::byte>(scale >> 8);
        for (size_t j = 0; j < 32; ++j) result[offset + 2 + j] = static_cast<std::byte>(random() % 256);
    }
    return result;
}

static std::vector<std::byte> expected_bytes(std::span<const std::byte> w,
                                            std::span<const float> x, size_t rows) {
    const auto values = q8_0_gemv_hls_model(w, x, rows);
    std::vector<float> output(rows);
    for (size_t i = 0; i < rows; ++i) output[i] = static_cast<float>(values[i]);
    return pack_f32_le(output);
}

static void test_encoding() {
    const uint32_t patterns[] = {0, 0x80000000U, 0x3f800000U, 0xc0200000U, 1,
                                 0x7f800000U, 0x7fc12345U};
    std::vector<float> values;
    std::vector<std::byte> golden;
    for (uint32_t pattern : patterns) {
        values.push_back(std::bit_cast<float>(pattern));
        for (size_t j = 0; j < 4; ++j) golden.push_back(static_cast<std::byte>((pattern >> (8 * j)) & 255U));
    }
    CHECK(pack_f32_le(values) == golden);
    const auto restored = unpack_f32_le(golden);
    for (size_t i = 0; i < values.size(); ++i) CHECK(std::bit_cast<uint32_t>(restored[i]) == patterns[i]);
    try { unpack_f32_le(std::span<const std::byte>(golden).first(3)); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("accepted incomplete FP32 value");
}

static void test_distributed() {
    std::mt19937 random(20261001);
    constexpr size_t rows = 67, split = 34; // Both units exercise multiple HLS row tiles.
    for (size_t columns : {32U, 64U, 96U, 1056U}) {
        const size_t row_bytes = columns / 32 * 34;
        auto w = weights(rows, columns, random);
        std::vector<float> x(columns);
        for (auto& value : x) value = static_cast<float>(static_cast<int>(random() % 2001) - 1000) / 71.0f;
        Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
        fabric.set_peer_access(0, 1, true);
        fabric.set_peer_access(1, 0, true);
        auto a = fabric.register_buffer(0, 8 * GIB + PAGE_BYTES - 3, split * row_bytes + 7);
        auto b = fabric.register_buffer(1, 500 * GIB, (rows - split) * row_bytes + 11);
        auto x0 = fabric.register_buffer(0, 20 * GIB, columns * 4 + 3);
        auto x1 = fabric.register_buffer(1, 20 * GIB, columns * 4 + 5);
        auto y0 = fabric.register_buffer(0, 40 * GIB, split * 4 + 1);
        auto y1 = fabric.register_buffer(1, 40 * GIB, (rows - split) * 4 + 3);
        auto gathered = fabric.register_buffer(0, 60 * GIB, rows * 4);
        fabric.write({a, 7}, std::span<const std::byte>(w).first(split * row_bytes));
        fabric.write({b, 11}, std::span<const std::byte>(w).subspan(split * row_bytes));
        fabric.write({x0, 3}, pack_f32_le(x));
        // Splitting a scale between local spans is valid; crossing units within a row is not.
        fabric.publish_model(1, {{7, w.size(), {{0, {a, 7}, 1},
            {1, {a, 8}, split * row_bytes - 1},
            {split * row_bytes, {b, 11}, (rows - split) * row_bytes}}, "q8_0_row_major", rows, columns}});
        fabric.submit({1, 1, {x0, 3}, {x1, 5}, columns * 4});
        fabric.execute(1);
        complete(fabric, 1);
        fabric.submit_q8_0_gemv({2, 1, 0, 7, 0, split, {x0, 3}, {y0, 1}});
        fabric.submit_q8_0_gemv({3, 1, 1, 7, split, rows - split, {x1, 5}, {y1, 3}});
        fabric.execute(3);
        fabric.execute(2);
        auto c1 = complete(fabric, 3);
        auto c0 = complete(fabric, 2);
        CHECK(c1.output_bytes == (rows - split) * 4 && c0.output_bytes == split * 4);
        CHECK(c1.macs == (rows - split) * columns && c0.macs == split * columns);
        CHECK(c1.bytes_transferred == 0 && c0.bytes_transferred == 0);
        CHECK(fabric.counters().peer_bytes == columns * 4); // No weight traffic during compute.
        fabric.submit({4, 1, {y0, 1}, {gathered}, split * 4});
        fabric.submit({5, 1, {y1, 3}, {gathered, split * 4}, (rows - split) * 4});
        fabric.execute(5);
        fabric.execute(4);
        complete(fabric, 5);
        complete(fabric, 4);
        CHECK(read_bytes(fabric, {gathered}, rows * 4) == expected_bytes(w, x, rows));
        CHECK(fabric.counters().macs == rows * columns);
        CHECK(fabric.counters().local_weight_bytes == w.size());
        CHECK(fabric.counters().gemv_commands == 2 && fabric.counters().q8_0_commands == 2);
        CHECK(fabric.counters().compute_output_bytes == rows * 4);
        CHECK(fabric.counters().bytes_copied == columns * 4 + rows * 4);
        CHECK(fabric.counters().peer_bytes == columns * 4 + (rows - split) * 4);
    }
}

static void test_validation() {
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    auto a = fabric.register_buffer(0, 0, 2 * GEMV_MAX_WEIGHT_BYTES);
    auto b = fabric.register_buffer(1, 0, 1024);
    auto x0 = fabric.register_buffer(0, 3 * GIB, 128);
    auto x1 = fabric.register_buffer(1, 3 * GIB, 128);
    auto y0 = fabric.register_buffer(0, 4 * GIB, 16);
    auto y1 = fabric.register_buffer(1, 4 * GIB, 16);
    fabric.publish_model(1, {{1, 1, {{0, {a}, 1}}}});
    const uint64_t shapes[][3] = {{34, 1, 31}, {34, 1, 0}, {33, 1, 32}, {34, 0, 32},
        {34, 1, std::numeric_limits<uint64_t>::max() - 31}};
    for (const auto& shape : shapes) {
        expect(Status::InvalidArgument, [&] {
            fabric.publish_model(2, {{7, shape[0], {{0, {a}, shape[0]}}, "q8_0_row_major", shape[1], shape[2]}});
        });
        CHECK(fabric.model_epoch() == 1);
    }
    fabric.publish_model(2, {
        {7, 136, {{0, {a}, 68}, {68, {b}, 68}}, "q8_0_row_major", 4, 32},
        {8, 64, {{0, {a}, 64}}, "int8_row_major", 2, 32},
        {9, 1048594, {{0, {a}, 1048594}}, "q8_0_row_major", 30841, 32},
        {10, 139264, {{0, {a}, 139264}}, "q8_0_row_major", 1, 131072}
    });
    GemvCommand base{1, 2, 0, 7, 0, 2, {x0}, {y0}};
    auto command = base;
    command.model_epoch = 1;
    expect(Status::StaleEpoch, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.weights = 999;
    expect(Status::UnknownTensor, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.weights = 8;
    expect(Status::InvalidArgument, [&] { fabric.submit_q8_0_gemv(command); });
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(base); });
    command = base; command.weights = 9; command.row_count = 30841;
    expect(Status::InvalidArgument, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.weights = 10; command.row_count = 1;
    expect(Status::InvalidArgument, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.first_row = std::numeric_limits<uint64_t>::max();
    expect(Status::OutOfBounds, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.input.offset = 1;
    expect(Status::OutOfBounds, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.output.offset = 12;
    expect(Status::OutOfBounds, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.output.buffer = a;
    expect(Status::ReadOnly, [&] { fabric.submit_q8_0_gemv(command); });
    fabric.set_peer_access(0, 1, true);
    fabric.set_peer_access(1, 0, true);
    command = base; command.input.buffer = x1;
    expect(Status::NonLocal, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.output.buffer = y1;
    expect(Status::NonLocal, [&] { fabric.submit_q8_0_gemv(command); });
    command = base; command.first_row = 1;
    expect(Status::NonLocal, [&] { fabric.submit_q8_0_gemv(command); });
    CHECK(fabric.pending_count() == 0 && fabric.counters().submitted == 0);
    fabric.submit_q8_0_gemv(base);
    fabric.execute(1);
    complete(fabric, 1); // Zero scale and zero activations are valid sparse operands.
}

static void test_numeric_failures() {
    // Failure in the second row must not publish the valid first row.
    for (int fault = 0; fault < 4; ++fault) {
        Fabric fabric(std::vector<UnitConfig>{{4 * PAGE_BYTES}});
        auto w = fabric.register_buffer(0, 0, 68);
        auto x = fabric.register_buffer(0, 128, 128);
        auto y = fabric.register_buffer(0, 256, 8);
        std::vector<std::byte> data(68, std::byte{0});
        data[1] = std::byte{0x3c};
        data[35] = fault == 0 ? std::byte{0x7c} : std::byte{0x3c};
        std::fill(data.begin() + 36, data.end(), std::byte{127});
        std::vector<float> activation(32, 1);
        if (fault == 1) activation[31] = std::numeric_limits<float>::quiet_NaN();
        if (fault == 2) activation[31] = std::numeric_limits<float>::infinity();
        if (fault == 3) activation.assign(32, std::numeric_limits<float>::max());
        const std::vector<std::byte> sentinel(8, std::byte{0x5a});
        fabric.write({w}, data);
        fabric.write({x}, pack_f32_le(activation));
        fabric.write({y}, sentinel);
        fabric.publish_model(1, {{7, 68, {{0, {w}, 68}}, "q8_0_row_major", 2, 32}});
        fabric.submit_q8_0_gemv({1, 1, 0, 7, 0, 2, {x}, {y}});
        fabric.execute(1);
        auto failed = complete(fabric, 1, Status::NumericError);
        CHECK(failed.macs == 0 && failed.output_bytes == 0 && failed.bytes_transferred == 0);
        CHECK(read_bytes(fabric, {y}, 8) == sentinel);
        CHECK(fabric.counters().q8_0_commands == 0 && fabric.counters().local_weight_bytes == 0);
        CHECK(fabric.counters().failed == 1);
        fabric.unregister_buffer(x);
        fabric.unregister_buffer(y);
    }
}

static void test_lifecycle() {
    Fabric fabric(std::vector<UnitConfig>{{4 * PAGE_BYTES}}, 1);
    auto w = fabric.register_buffer(0, 0, 34);
    auto x = fabric.register_buffer(0, 128, 128);
    auto y = fabric.register_buffer(0, 256, 4);
    std::vector<std::byte> data(34, std::byte{1});
    data[0] = std::byte{0}; data[1] = std::byte{0x3c};
    fabric.write({w}, data);
    fabric.write({x}, pack_f32_le(std::vector<float>(32, 127)));
    fabric.publish_model(1, {{7, 34, {{0, {w}, 34}}, "q8_0_row_major", 1, 32}});
    GemvCommand command{1, 1, 0, 7, 0, 1, {x}, {y}};
    fabric.submit_q8_0_gemv(command);
    expect(Status::Busy, [&] { read_bytes(fabric, {x}, 1); });
    expect(Status::Busy, [&] { fabric.write({x}, pack_f32_le(std::vector<float>(32))); });
    expect(Status::Busy, [&] { fabric.publish_model(2, {}); });
    expect(Status::DuplicateSequence, [&] { fabric.submit_q8_0_gemv(command); });
    command.sequence = 2;
    expect(Status::QueueFull, [&] { fabric.submit_q8_0_gemv(command); });
    fabric.set_online(0, false);
    fabric.execute(1);
    complete(fabric, 1, Status::DeviceOffline);
    fabric.set_online(0, true);
    CHECK(read_bytes(fabric, {y}, 4) == std::vector<std::byte>(4));
    fabric.submit_q8_0_gemv(command);
    fabric.cancel(2);
    complete(fabric, 2, Status::Cancelled);
    CHECK(read_bytes(fabric, {y}, 4) == std::vector<std::byte>(4));
    command.sequence = 3;
    fabric.submit_q8_0_gemv(command);
    fabric.execute(3);
    expect(Status::Busy, [&] { fabric.publish_model(2, {}); });
    CHECK(complete(fabric, 3).macs == 32);
    CHECK(unpack_f32_le(read_bytes(fabric, {y}, 4))[0] == 4064.0f);
    CHECK(fabric.counters().failed == 1 && fabric.counters().cancelled == 1);
    CHECK(fabric.counters().gemv_commands == 1 && fabric.counters().q8_0_commands == 1);
}

static void test_budget_and_overlap() {
    Fabric fabric(std::vector<UnitConfig>{{4 * PAGE_BYTES, 1}});
    auto w = fabric.register_buffer(0, 0, 68);
    auto xy = fabric.register_buffer(0, 128, 128);
    auto unavailable = fabric.register_buffer(0, PAGE_BYTES, 8);
    std::mt19937 random(8);
    auto data = weights(2, 32, random);
    std::vector<float> activation(32, 1.25f);
    fabric.write({w}, data);
    fabric.write({xy}, pack_f32_le(activation));
    fabric.publish_model(1, {{7, 68, {{0, {w}, 68}}, "q8_0_row_major", 2, 32}});
    fabric.submit_q8_0_gemv({1, 1, 0, 7, 0, 2, {xy}, {unavailable}});
    fabric.execute(1);
    CHECK(complete(fabric, 1, Status::ResourceLimit).output_bytes == 0);
    CHECK(fabric.counters().macs == 0 && fabric.counters().local_weight_bytes == 0);
    CHECK(read_bytes(fabric, {unavailable}, 8) == std::vector<std::byte>(8));
    fabric.submit_q8_0_gemv({2, 1, 0, 7, 0, 2, {xy}, {xy, 1}});
    fabric.execute(2);
    complete(fabric, 2);
    CHECK(read_bytes(fabric, {xy, 1}, 8) == expected_bytes(data, activation, 2));
    CHECK(fabric.resident_bytes() == PAGE_BYTES);
}

int main(int argc, char** argv) {
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {{"encoding", test_encoding}, {"distributed", test_distributed},
        {"validation", test_validation}, {"numeric_failures", test_numeric_failures},
        {"lifecycle", test_lifecycle}, {"budget_and_overlap", test_budget_and_overlap}};
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
