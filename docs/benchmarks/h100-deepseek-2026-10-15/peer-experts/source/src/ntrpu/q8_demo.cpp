#include "numeric_io.h"
#include "q8_0.h"
#include "simulator.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace nt::ntrpu;

static std::vector<std::byte> make_weights(size_t rows, size_t columns, uint32_t seed) {
    std::vector<std::byte> result(rows * (columns / 32) * 34);
    const uint16_t scales[] = {0x2000, 0x2400, 0x2800}; // 1/128, 1/64, 1/32.
    for (size_t offset = 0; offset < result.size(); offset += 34) {
        const auto scale = scales[(offset / 34) % 3];
        result[offset] = static_cast<std::byte>(scale & 255U);
        result[offset + 1] = static_cast<std::byte>(scale >> 8);
        for (size_t j = 0; j < 32; ++j) {
            seed = seed * 1664525U + 1013904223U;
            const int8_t value = static_cast<int8_t>(static_cast<int>((seed >> 16) % 31) - 15);
            result[offset + 2 + j] = static_cast<std::byte>(static_cast<uint8_t>(value));
        }
    }
    return result;
}

static std::vector<float> to_f32(const std::vector<double>& values) {
    std::vector<float> output(values.size());
    for (size_t i = 0; i < values.size(); ++i) output[i] = static_cast<float>(values[i]);
    return output;
}

static void complete(Fabric& fabric, uint64_t sequence) {
    auto result = fabric.poll();
    if (!result || result->sequence != sequence || result->status != Status::Ok) {
        throw std::runtime_error("command failed; dependent stage must not run");
    }
}

static void execute(Fabric& fabric, uint64_t sequence) {
    fabric.execute(sequence);
    complete(fabric, sequence);
}

int main() {
    try {
        constexpr size_t rows = 64, columns = 96, split = 33, next_rows = 32;
        constexpr size_t row_bytes = columns / 32 * 34;
        auto first_weights = make_weights(rows, columns, 719);
        auto next_weights = make_weights(next_rows, rows, 1201);
        std::vector<float> input(columns);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i % 41) - 20) / 8.0f;

        Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
        fabric.set_peer_access(0, 1, true);
        fabric.set_peer_access(1, 0, true);
        auto w0 = fabric.register_buffer(0, 8 * GIB, split * row_bytes);
        auto w1 = fabric.register_buffer(1, 500 * GIB, (rows - split) * row_bytes);
        auto w2 = fabric.register_buffer(1, 480 * GIB, next_weights.size());
        auto x0 = fabric.register_buffer(0, 20 * GIB, columns * 4);
        auto x1 = fabric.register_buffer(1, 20 * GIB, columns * 4);
        auto y0 = fabric.register_buffer(0, 40 * GIB, split * 4);
        auto y1 = fabric.register_buffer(1, 40 * GIB, (rows - split) * 4);
        auto gathered = fabric.register_buffer(0, 100 * GIB, rows * 4);
        auto next_input = fabric.register_buffer(1, 100 * GIB, rows * 4);
        auto next_output = fabric.register_buffer(1, 120 * GIB, next_rows * 4);
        auto result_buffer = fabric.register_buffer(0, 120 * GIB, next_rows * 4);
        fabric.write({w0}, std::span<const std::byte>(first_weights).first(split * row_bytes));
        fabric.write({w1}, std::span<const std::byte>(first_weights).subspan(split * row_bytes));
        fabric.write({w2}, next_weights);
        fabric.write({x0}, pack_f32_le(input));
        fabric.publish_model(1, {
            {7, first_weights.size(), {{0, {w0}, split * row_bytes},
                {split * row_bytes, {w1}, (rows - split) * row_bytes}}, "q8_0_row_major", rows, columns},
            {8, next_weights.size(), {{0, {w2}, next_weights.size()}}, "q8_0_row_major", next_rows, rows}
        });

        fabric.submit({1, 1, {x0}, {x1}, columns * 4});
        execute(fabric, 1);
        fabric.submit_q8_0_gemv({2, 1, 0, 7, 0, split, {x0}, {y0}});
        fabric.submit_q8_0_gemv({3, 1, 1, 7, split, rows - split, {x1}, {y1}});
        fabric.execute(3);
        fabric.execute(2);
        complete(fabric, 3);
        complete(fabric, 2); // Verify both results before issuing dependent DMA.
        fabric.submit({4, 1, {y0}, {gathered}, split * 4});
        fabric.submit({5, 1, {y1}, {gathered, split * 4}, (rows - split) * 4});
        fabric.execute(5);
        fabric.execute(4);
        complete(fabric, 5);
        complete(fabric, 4);

        // The next operation consumes the FP32 bytes directly, without host rewriting.
        fabric.submit({6, 1, {gathered}, {next_input}, rows * 4});
        execute(fabric, 6);
        fabric.submit_q8_0_gemv({7, 1, 1, 8, 0, next_rows, {next_input}, {next_output}});
        execute(fabric, 7);
        fabric.submit({8, 1, {next_output}, {result_buffer}, next_rows * 4});
        execute(fabric, 8);
        std::vector<std::byte> result_bytes(next_rows * 4);
        fabric.read({result_buffer}, result_bytes);

        const auto model_first = to_f32(q8_0_gemv_hls_model(first_weights, input, rows));
        const auto model_next = to_f32(q8_0_gemv_hls_model(next_weights, model_first, next_rows));
        if (result_bytes != pack_f32_le(model_next)) throw std::runtime_error("distributed/monolithic mismatch");
        const auto reference_first = to_f32(q8_0_gemv_reference(first_weights, input, rows));
        const auto reference_next = q8_0_gemv_reference(next_weights, reference_first, next_rows);
        double squared_error = 0, squared_reference = 0;
        for (size_t row = 0; row < next_rows; ++row) {
            const double error = double(model_next[row]) - reference_next[row];
            squared_error += error * error;
            squared_reference += reference_next[row] * reference_next[row];
        }
        const auto& counters = fabric.counters();
        if (counters.local_weight_bytes != 8704 || counters.peer_bytes != 892 ||
            counters.bytes_copied != 1024 || counters.macs != 8192 || counters.q8_0_commands != 3) {
            throw std::runtime_error("unexpected transfer or compute accounting");
        }
        std::cout << "Q8_0 chain: 64x96 (33+31 rows on two units), then 32x64 on unit 1\n"
                  << "Distributed output matches the monolithic quantized model byte-for-byte\n"
                  << "Successful MACs: " << counters.macs << '\n'
                  << "Local weight bytes, including scales: " << counters.local_weight_bytes << '\n'
                  << "Peer bytes: " << counters.peer_bytes << " (384 input + 124 gather + 256 next input + 128 result)\n"
                  << "Logical capacity: " << fabric.capacity_bytes() / GIB << " GiB\n"
                  << "Materialized page backing: " << fabric.resident_bytes() << " bytes\n"
                  << "Synthetic chain relative L2 error: " << std::sqrt(squared_error / squared_reference) << '\n'
                  << "CPU functional simulation. Two linear operations, not a full FFN or transformer.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
