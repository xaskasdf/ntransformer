#include "ntrpu/gemv.h"
#include "ntrpu/simulator.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace nt::ntrpu;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Function>
static void expect(Status status, Function function) {
    try { function(); }
    catch (const Error& error) { CHECK(error.status() == status); return; }
    throw std::runtime_error("expected simulator error");
}

template<class Function>
static void invalid(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}

static std::vector<int32_t> reference(const std::vector<int8_t>& weights,
                                    const std::vector<int8_t>& input) {
    std::vector<int32_t> output(weights.size() / input.size());
    for (size_t row = 0; row < output.size(); ++row) {
        int64_t sum = 0;
        for (size_t column = 0; column < input.size(); ++column) {
            sum += int64_t{weights[row * input.size() + column]} * input[column];
        }
        CHECK(sum >= std::numeric_limits<int32_t>::min());
        CHECK(sum <= std::numeric_limits<int32_t>::max());
        output[row] = static_cast<int32_t>(sum);
    }
    return output;
}

static std::vector<int8_t> random_values(size_t size, std::mt19937& random) {
    std::vector<int8_t> result(size);
    for (auto& value : result) value = static_cast<int8_t>(static_cast<int>(random() % 256) - 128);
    return result;
}

static std::vector<std::byte> read_bytes(Fabric& fabric, BufferSlice source, size_t size) {
    std::vector<std::byte> result(size);
    fabric.read(source, result);
    return result;
}

static Completion completed(Fabric& fabric, uint64_t sequence, Status status = Status::Ok) {
    auto result = fabric.poll();
    CHECK(result && result->sequence == sequence && result->status == status);
    return *result;
}

static void test_numeric() {
    const std::vector<int8_t> weights{1, -2, 3, -128, 127, 0};
    const std::vector<int8_t> input{-4, 5, -6};
    CHECK(gemv_int8(weights, input, 2) == std::vector<int32_t>({-32, 1147}));
    std::mt19937 random(20260930);
    for (size_t columns : {1U, 7U, 31U, 32U, 33U, 63U, 64U, 65U, 1023U, 1024U, 1025U}) {
        for (size_t rows : {1U, 3U, 17U}) {
            auto w = random_values(rows * columns, random);
            auto x = random_values(columns, random);
            CHECK(gemv_int8(w, x, rows) == reference(w, x));
        }
    }
}

static void test_extremes() {
    std::vector<int8_t> w(GEMV_MAX_COLUMNS, -128), x(GEMV_MAX_COLUMNS, -128);
    CHECK(gemv_int8(w, x, 1) == reference(w, x));
    CHECK(gemv_int8(w, x, 1)[0] == 2147467264);
    std::fill(x.begin(), x.end(), 127);
    CHECK(gemv_int8(w, x, 1) == reference(w, x));
    std::fill(x.begin(), x.end(), 0);
    CHECK(gemv_int8(w, x, 1)[0] == 0);
    w.push_back(-128);
    x.push_back(0);
    invalid([&] { gemv_int8(w, x, 1); }); // K=131072 is rejected even for benign data.
    invalid([&] { gemv_int8({}, {}, 1); });
    invalid([&] { gemv_int8(w, x, 0); });
    invalid([&] { gemv_int8(std::span<const int8_t>(w).first(3),
                           std::span<const int8_t>(x).first(2), 2); });
    invalid([&] { gemv_int8(std::span<const int8_t>(w).first(3),
                           std::span<const int8_t>(x).first(2), std::numeric_limits<size_t>::max()); });
}

static void test_encoding() {
    const std::vector<int32_t> values{0, -1, 0x12345678,
        std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    const unsigned expected[] = {0, 0, 0, 0, 255, 255, 255, 255, 0x78, 0x56, 0x34, 0x12,
                                0, 0, 0, 0x80, 255, 255, 255, 0x7f};
    const auto bytes = pack_int32_le(values);
    CHECK(bytes.size() == 20);
    for (size_t i = 0; i < bytes.size(); ++i) CHECK(std::to_integer<unsigned>(bytes[i]) == expected[i]);
    CHECK(unpack_int32_le(bytes) == values);
    invalid([&] { unpack_int32_le(std::span<const std::byte>(bytes).first(19)); });
}

static void test_distributed() {
    std::mt19937 random(719);
    for (size_t columns : {1U, 31U, 32U, 33U, 65U, 1025U}) {
        constexpr size_t rows = 7, split = 3;
        Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
        fabric.set_peer_access(0, 1, true);
        fabric.set_peer_access(1, 0, true);
        auto a = fabric.register_buffer(0, 8 * GIB + PAGE_BYTES - 3, split * columns + 7);
        auto b = fabric.register_buffer(1, 500 * GIB, (rows - split) * columns + 11);
        auto x0 = fabric.register_buffer(0, 20 * GIB, columns + 3);
        auto x1 = fabric.register_buffer(1, 20 * GIB, columns + 5);
        auto y0 = fabric.register_buffer(0, 40 * GIB, split * 4 + 1);
        auto y1 = fabric.register_buffer(1, 40 * GIB, (rows - split) * 4 + 3);
        auto output = fabric.register_buffer(0, 100 * GIB, rows * 4);
        auto w = random_values(rows * columns, random);
        auto x = random_values(columns, random);
        const auto bytes = std::as_bytes(std::span<const int8_t>(w));
        fabric.write({a, 7}, bytes.first(split * columns));
        fabric.write({b, 11}, bytes.subspan(split * columns));
        fabric.write({x0, 3}, std::as_bytes(std::span<const int8_t>(x)));
        // An arbitrary byte boundary within a local row is legal; the row stays local.
        fabric.publish_model(1, {{9, rows * columns,
            {{0, {a, 7}, 1}, {1, {a, 8}, split * columns - 1},
             {split * columns, {b, 11}, (rows - split) * columns}},
            "int8_row_major", rows, columns}});
        fabric.submit({1, 1, {x0, 3}, {x1, 5}, columns});
        fabric.execute(1);
        CHECK(completed(fabric, 1).bytes_transferred == columns);
        fabric.submit_gemv({2, 1, 0, 9, 0, split, {x0, 3}, {y0, 1}});
        fabric.submit_gemv({3, 1, 1, 9, split, rows - split, {x1, 5}, {y1, 3}});
        fabric.execute(3);
        fabric.execute(2);
        auto second = completed(fabric, 3);
        CHECK(second.bytes_transferred == 0 && second.output_bytes == (rows - split) * 4);
        CHECK(second.macs == (rows - split) * columns);
        auto first = completed(fabric, 2);
        CHECK(first.macs == split * columns && first.output_bytes == split * 4);
        CHECK(fabric.counters().peer_bytes == columns); // GEMV moved no remote weights.
        fabric.submit({4, 1, {y0, 1}, {output}, split * 4});
        fabric.submit({5, 1, {y1, 3}, {output, split * 4}, (rows - split) * 4});
        fabric.execute(5);
        fabric.execute(4);
        completed(fabric, 5);
        completed(fabric, 4);
        CHECK(unpack_int32_le(read_bytes(fabric, {output}, rows * 4)) == reference(w, x));
        CHECK(fabric.counters().macs == rows * columns);
        CHECK(fabric.counters().local_weight_bytes == rows * columns);
        CHECK(fabric.counters().compute_output_bytes == rows * 4);
        CHECK(fabric.counters().gemv_commands == 2);
        CHECK(fabric.counters().bytes_copied == columns + rows * 4);
        CHECK(fabric.counters().peer_bytes == columns + (rows - split) * 4);
    }
}

static void test_validation() {
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    auto w0 = fabric.register_buffer(0, 0, 2 * GEMV_MAX_WEIGHT_BYTES);
    auto w1 = fabric.register_buffer(1, 0, 32);
    auto x0 = fabric.register_buffer(0, 4 * GIB, 32);
    auto x1 = fabric.register_buffer(1, 4 * GIB, 32);
    auto y0 = fabric.register_buffer(0, 5 * GIB, 32);
    auto y1 = fabric.register_buffer(1, 5 * GIB, 32);
    expect(Status::InvalidArgument, [&] {
        fabric.publish_model(1, {{9, 32, {{0, {w0}, 32}}, "int8_row_major", 3, 16}});
    });
    expect(Status::InvalidArgument, [&] {
        fabric.publish_model(1, {{9, 32, {{0, {w0}, 32}}, "int8_row_major", 2, 0}});
    });
    CHECK(fabric.model_epoch() == 0);
    fabric.publish_model(1, {
        {9, 32, {{0, {w0}, 16}, {16, {w1}, 16}}, "int8_row_major", 4, 8},
        {10, 32, {{0, {w0}, 32}}, "Q8_0"},
        {11, 2 * GEMV_MAX_WEIGHT_BYTES, {{0, {w0}, 2 * GEMV_MAX_WEIGHT_BYTES}},
         "int8_row_major", 2 * GEMV_MAX_WEIGHT_BYTES / 32, 32},
        {12, GEMV_MAX_COLUMNS + 1, {{0, {w0}, GEMV_MAX_COLUMNS + 1}},
         "int8_row_major", 1, GEMV_MAX_COLUMNS + 1}
    });
    auto base = GemvCommand{1, 1, 0, 9, 0, 2, {x0}, {y0}};
    auto changed = base;
    changed.model_epoch = 2;
    expect(Status::StaleEpoch, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.weights = 999;
    expect(Status::UnknownTensor, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.sequence = 0;
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.weights = 10;
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.row_count = 0;
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.first_row = std::numeric_limits<uint64_t>::max();
    expect(Status::OutOfBounds, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.row_count = std::numeric_limits<uint64_t>::max();
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.weights = 11; changed.row_count = 2 * GEMV_MAX_WEIGHT_BYTES / 32;
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.weights = 12; changed.row_count = 1;
    expect(Status::InvalidArgument, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.input.offset = 31;
    expect(Status::OutOfBounds, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.output.offset = 31;
    expect(Status::OutOfBounds, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.output.buffer = w0;
    expect(Status::ReadOnly, [&] { fabric.submit_gemv(changed); });
    fabric.set_peer_access(0, 1, true);
    fabric.set_peer_access(1, 0, true);
    changed = base; changed.row_count = 4;
    expect(Status::NonLocal, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.input.buffer = x1;
    expect(Status::NonLocal, [&] { fabric.submit_gemv(changed); });
    changed = base; changed.output.buffer = y1;
    expect(Status::NonLocal, [&] { fabric.submit_gemv(changed); });
    CHECK(fabric.pending_count() == 0 && fabric.counters().submitted == 0);
    fabric.submit_gemv(base); // Failed validations leaked no pins or sequence reservations.
    fabric.execute(1);
    completed(fabric, 1);
}

static void test_lifecycle() {
    Fabric fabric(std::vector<UnitConfig>{{4 * PAGE_BYTES}}, 1);
    auto weights = fabric.register_buffer(0, 0, 8);
    auto input = fabric.register_buffer(0, 32, 4);
    auto output = fabric.register_buffer(0, 64, 8);
    const std::vector<int8_t> w{1, 2, 3, 4, -1, -2, -3, -4}, x{5, 6, 7, 8};
    const std::vector<std::byte> sentinel(8, std::byte{0x5a});
    fabric.write({weights}, std::as_bytes(std::span<const int8_t>(w)));
    fabric.write({input}, std::as_bytes(std::span<const int8_t>(x)));
    fabric.write({output}, sentinel);
    fabric.publish_model(1, {{9, 8, {{0, {weights}, 8}}, "int8_row_major", 2, 4}});
    GemvCommand command{1, 1, 0, 9, 0, 2, {input}, {output}};
    fabric.submit_gemv(command);
    expect(Status::Busy, [&] { fabric.write({input}, std::as_bytes(std::span<const int8_t>(x))); });
    expect(Status::Busy, [&] { read_bytes(fabric, {weights}, 8); });
    expect(Status::Busy, [&] { fabric.unregister_buffer(output); });
    expect(Status::Busy, [&] { fabric.publish_model(2, {}); });
    expect(Status::DuplicateSequence, [&] { fabric.submit_gemv(command); });
    command.sequence = 2;
    expect(Status::QueueFull, [&] { fabric.submit_gemv(command); });
    fabric.set_online(0, false);
    fabric.execute(1);
    auto failed = completed(fabric, 1, Status::DeviceOffline);
    CHECK(failed.macs == 0 && failed.output_bytes == 0 && failed.bytes_transferred == 0);
    fabric.set_online(0, true);
    CHECK(read_bytes(fabric, {output}, 8) == sentinel);
    fabric.submit_gemv(command);
    fabric.cancel(2);
    CHECK(completed(fabric, 2, Status::Cancelled).macs == 0);
    CHECK(read_bytes(fabric, {output}, 8) == sentinel);
    command.sequence = 3;
    fabric.submit_gemv(command);
    fabric.execute(3);
    expect(Status::Busy, [&] { fabric.publish_model(2, {}); });
    CHECK(completed(fabric, 3).macs == 8);
    CHECK(unpack_int32_le(read_bytes(fabric, {output}, 8)) == reference(w, x));
    CHECK(fabric.counters().failed == 1 && fabric.counters().cancelled == 1);
    CHECK(fabric.counters().gemv_commands == 1 && fabric.counters().macs == 8);
    CHECK(fabric.counters().bytes_copied == 0 && fabric.counters().peer_bytes == 0);
    fabric.publish_model(2, {});
    fabric.unregister_buffer(weights);
    fabric.unregister_buffer(input);
    fabric.unregister_buffer(output);
}

static void test_memory_and_overlap() {
    Fabric fabric(std::vector<UnitConfig>{{4 * PAGE_BYTES, 1}});
    auto weights = fabric.register_buffer(0, 0, 8);
    auto input_output = fabric.register_buffer(0, 32, 16);
    auto unavailable = fabric.register_buffer(0, PAGE_BYTES, 8);
    const std::vector<int8_t> w{1, 2, 3, 4, -1, -2, -3, -4}, x{-128, 127, -3, 4};
    fabric.write({weights}, std::as_bytes(std::span<const int8_t>(w)));
    fabric.write({input_output}, std::as_bytes(std::span<const int8_t>(x)));
    fabric.publish_model(1, {{9, 8, {{0, {weights}, 8}}, "int8_row_major", 2, 4}});
    fabric.submit_gemv({1, 1, 0, 9, 0, 2, {input_output}, {unavailable}});
    fabric.execute(1);
    CHECK(completed(fabric, 1, Status::ResourceLimit).output_bytes == 0);
    CHECK(read_bytes(fabric, {unavailable}, 8) == std::vector<std::byte>(8));
    CHECK(fabric.counters().macs == 0);
    fabric.submit_gemv({2, 1, 0, 9, 0, 2, {input_output}, {input_output, 1}});
    fabric.execute(2);
    completed(fabric, 2);
    CHECK(unpack_int32_le(read_bytes(fabric, {input_output, 1}, 8)) == reference(w, x));
    CHECK(fabric.resident_bytes() == PAGE_BYTES);
}

int main(int argc, char** argv) {
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {{"numeric", test_numeric}, {"extremes", test_extremes},
        {"encoding", test_encoding}, {"distributed", test_distributed},
        {"validation", test_validation}, {"lifecycle", test_lifecycle},
        {"memory_and_overlap", test_memory_and_overlap}};
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
    if (ran == 0) return 2;
    return failed == 0 ? 0 : 1;
}
