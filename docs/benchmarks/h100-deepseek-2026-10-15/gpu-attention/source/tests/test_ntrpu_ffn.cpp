#include "ntrpu/activation.h"
#include "ntrpu/ffn.h"
#include "ntrpu/numeric_io.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace nt::ntrpu;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Function>
static void expect(Status status, Function function) {
    try { function(); }
    catch (const Error& error) { CHECK(error.status() == status); return; }
    throw std::runtime_error("expected simulator error");
}

static Completion complete(Fabric& f, uint64_t sequence, Status status = Status::Ok) {
    auto c = f.poll();
    CHECK(c && c->sequence == sequence && c->status == status);
    return *c;
}

static std::vector<std::byte> read(Fabric& f, BufferId b, size_t bytes) {
    std::vector<std::byte> result(bytes);
    f.read({b}, result);
    return result;
}

static void numeric() {
    const float max = std::numeric_limits<float>::max();
    const std::vector<float> gate{0, 1, -1, 100, -100, max, -max};
    const std::vector<float> up{2, 2, 2, 1, 1, 0, max};
    const auto result = swiglu(gate, up);
    CHECK(result[0] == 0 && result[3] == 100 && result[5] == 0 && result[6] == 0);
    CHECK(std::abs(result[1] - 1.46211715726f) < 2e-7f);
    CHECK(std::abs(result[2] + 0.53788284274f) < 2e-7f);
    CHECK(result[4] <= 0 && result[4] > -1e-40f);
    for (const auto& pair : std::vector<std::pair<std::vector<float>, std::vector<float>>>{
            {{}, {}}, {{1}, {}}, {{max}, {2}},
            {{std::numeric_limits<float>::infinity()}, {0}},
            {{0}, {std::numeric_limits<float>::quiet_NaN()}}}) {
        bool rejected = false;
        try { swiglu(pair.first, pair.second); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }
}

static void commands() {
    Fabric f(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}}, 1);
    auto gate = f.register_buffer(0, 8 * GIB, 16);
    auto up = f.register_buffer(0, 12 * GIB, 8);
    auto remote = f.register_buffer(1, 8 * GIB, 16);
    f.write({gate}, pack_f32_le(std::vector<float>{1, -1, 99, 99}));
    f.write({up}, pack_f32_le(std::vector<float>{2, 2}));
    f.publish_model(1, {});
    expect(Status::NonLocal, [&] { f.submit_swiglu({1, 1, 0, 2, {gate}, {remote}, {up}}); });
    expect(Status::InvalidArgument, [&] { f.submit_swiglu({1, 1, 0, 0, {gate}, {up}, {gate}}); });
    expect(Status::InvalidArgument, [&] { f.submit_swiglu({1, 1, 0, UINT64_MAX, {gate}, {up}, {gate}}); });
    expect(Status::OutOfBounds, [&] { f.submit_swiglu({1, 1, 0, 3, {gate}, {up}, {gate}}); });
    expect(Status::StaleEpoch, [&] { f.submit_swiglu({1, 2, 0, 2, {gate}, {up}, {gate}}); });
    // Partial overlap: the second gate is overwritten only after both inputs are read.
    f.submit_swiglu({1, 1, 0, 2, {gate}, {up}, {gate, 4}});
    expect(Status::Busy, [&] { read(f, up, 8); });
    expect(Status::Busy, [&] { f.unregister_buffer(gate); });
    expect(Status::DuplicateSequence, [&] { f.submit_swiglu({1, 1, 0, 2, {gate}, {up}, {gate}}); });
    expect(Status::QueueFull, [&] { f.submit_swiglu({2, 1, 0, 2, {gate}, {up}, {gate}}); });
    f.execute(1);
    const auto c = complete(f, 1);
    CHECK(c.output_bytes == 8 && c.macs == 0 && c.bytes_transferred == 0);
    const auto output = unpack_f32_le(read(f, gate, 16));
    CHECK(output[0] == 1 && output[3] == 99);
    CHECK(std::abs(output[1] - 1.46211715726f) < 2e-7f);
    CHECK(std::abs(output[2] + 0.53788284274f) < 2e-7f);
    CHECK(f.counters().swiglu_commands == 1 && f.counters().swiglu_elements == 2);
    CHECK(f.counters().compute_output_bytes == 8 && f.counters().bytes_copied == 0);
    CHECK(f.counters().gemv_commands == 0 && f.counters().macs == 0);
    f.publish_model(2, {{1, 8, {{0, {up}, 8}}}});
    expect(Status::ReadOnly, [&] { f.submit_swiglu({2, 2, 0, 2, {gate}, {up}, {up}}); });
}

static void failures() {
    Fabric f(std::vector<UnitConfig>{{512 * GIB}});
    auto gate = f.register_buffer(0, 0, 8);
    auto up = f.register_buffer(0, PAGE_BYTES, 8);
    auto output = f.register_buffer(0, 2 * PAGE_BYTES, 8);
    f.write({gate}, pack_f32_le(std::vector<float>{1, std::numeric_limits<float>::max()}));
    f.write({up}, pack_f32_le(std::vector<float>{2, 2}));
    const auto sentinel = pack_f32_le(std::vector<float>{77, 88});
    f.write({output}, sentinel);
    f.publish_model(1, {});
    f.submit_swiglu({1, 1, 0, 2, {gate}, {up}, {output}});
    f.execute(1);
    const auto c = complete(f, 1, Status::NumericError);
    CHECK(c.output_bytes == 0 && read(f, output, 8) == sentinel);
    f.submit_swiglu({2, 1, 0, 2, {gate}, {up}, {output}});
    f.cancel(2);
    complete(f, 2, Status::Cancelled);
    f.submit_swiglu({3, 1, 0, 2, {gate}, {up}, {output}});
    f.set_online(0, false);
    f.execute(3);
    complete(f, 3, Status::DeviceOffline);
    f.set_online(0, true);
    CHECK(read(f, output, 8) == sentinel && f.counters().swiglu_commands == 0);
    CHECK(f.counters().failed == 2 && f.counters().cancelled == 1);
    f.unregister_buffer(output); // All pins released after each failure.

    Fabric limited(std::vector<UnitConfig>{{512 * GIB, 1}});
    auto source = limited.register_buffer(0, 0, 8);
    auto dest = limited.register_buffer(0, PAGE_BYTES, 8);
    limited.write({source}, pack_f32_le(std::vector<float>{1, 1}));
    limited.publish_model(1, {});
    limited.submit_swiglu({1, 1, 0, 2, {source}, {source}, {dest}});
    limited.execute(1);
    complete(limited, 1, Status::ResourceLimit);
    CHECK(read(limited, dest, 8) == std::vector<std::byte>(8));
}

static std::vector<std::byte> constant_weights(size_t rows, size_t columns, size_t real_columns,
                                             uint16_t half_scale) {
    std::vector<std::byte> result;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < columns / 32; ++block) {
            result.push_back(static_cast<std::byte>(half_scale & 255));
            result.push_back(static_cast<std::byte>(half_scale >> 8));
            for (size_t j = 0; j < 32; ++j) {
                result.push_back(block * 32 + j < real_columns ? std::byte{1} : std::byte{0});
            }
        }
    }
    return result;
}

static void distributed() {
    const auto gate = constant_weights(35, 32, 32, 0x2800); // 1/32
    const auto up = constant_weights(35, 32, 32, 0x2400); // 1/64
    const auto down = constant_weights(32, 64, 35, 0x2800); // Irregular H and split.
    std::vector<float> input(32, 1);
    const auto result = q8_0_ffn_distributed(gate, up, down, input, 35);
    const double expected = 35.0 / 32 * 0.5 / (1 + std::exp(-1.0));
    for (float value : result.output) CHECK(std::abs(value - expected) < 1e-7);
    const auto& c = result.counters;
    CHECK(c.macs == 4288 && c.local_weight_bytes == 4556);
    CHECK(c.peer_bytes == 324 && c.bytes_copied == 396);
    CHECK(c.q8_0_commands == 5 && c.swiglu_commands == 2 && c.swiglu_elements == 35);
    CHECK(c.submitted == 11 && c.completed == 11 && c.failed == 0);
    bool rejected = false;
    try { q8_0_ffn_distributed(gate, up, down, input, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        const std::string test = argv[1];
        if (test == "numeric") numeric();
        else if (test == "commands") commands();
        else if (test == "failures") failures();
        else if (test == "distributed") distributed();
        else throw std::runtime_error("unknown test");
        std::cout << "PASS " << test << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
