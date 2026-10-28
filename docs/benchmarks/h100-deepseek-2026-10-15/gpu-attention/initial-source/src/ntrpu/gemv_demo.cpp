#include "gemv.h"
#include "simulator.h"

#include <iostream>
#include <vector>

using namespace nt::ntrpu;

static Completion complete(Fabric& fabric, uint64_t sequence) {
    auto result = fabric.poll();
    if (!result || result->sequence != sequence || result->status != Status::Ok) {
        throw std::runtime_error("unexpected completion");
    }
    return *result;
}

int main() {
    try {
        constexpr size_t rows = 7, columns = 65, split = 3;
        Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
        fabric.set_peer_access(0, 1, true);
        fabric.set_peer_access(1, 0, true);
        const auto w0 = fabric.register_buffer(0, 8 * GIB, split * columns);
        const auto w1 = fabric.register_buffer(1, 500 * GIB, (rows - split) * columns);
        const auto x0 = fabric.register_buffer(0, 20 * GIB, columns);
        const auto x1 = fabric.register_buffer(1, 20 * GIB, columns);
        const auto y0 = fabric.register_buffer(0, 40 * GIB, split * 4);
        const auto y1 = fabric.register_buffer(1, 40 * GIB, (rows - split) * 4);
        const auto gathered = fabric.register_buffer(0, 100 * GIB, rows * 4);

        std::vector<int8_t> weights(rows * columns), input(columns);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<int8_t>(static_cast<int>((i * 37 + 11) % 256) - 128);
        }
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<int8_t>(static_cast<int>((i * 19 + 3) % 256) - 128);
        }
        const auto bytes = std::as_bytes(std::span<const int8_t>(weights));
        fabric.write({w0}, bytes.first(split * columns));
        fabric.write({w1}, bytes.subspan(split * columns));
        fabric.write({x0}, std::as_bytes(std::span<const int8_t>(input)));
        fabric.publish_model(1, {{42, rows * columns,
            {{0, {w0}, split * columns}, {split * columns, {w1}, (rows - split) * columns}},
            "int8_row_major", rows, columns}});

        fabric.submit({1, 1, {x0}, {x1}, columns});
        fabric.execute(1);
        complete(fabric, 1);
        fabric.submit_gemv({2, 1, 0, 42, 0, split, {x0}, {y0}});
        fabric.submit_gemv({3, 1, 1, 42, split, rows - split, {x1}, {y1}});
        fabric.execute(3);
        fabric.execute(2);
        for (auto sequence : {3ULL, 2ULL}) {
            const auto done = complete(fabric, sequence);
            std::cout << "GEMV completion " << sequence << ": " << done.macs
                      << " MACs, " << done.output_bytes << " output bytes\n";
        }
        fabric.submit({4, 1, {y0}, {gathered}, split * 4});
        fabric.submit({5, 1, {y1}, {gathered, split * 4}, (rows - split) * 4});
        fabric.execute(5);
        fabric.execute(4);
        complete(fabric, 5);
        complete(fabric, 4);

        std::vector<std::byte> result_bytes(rows * 4);
        fabric.read({gathered}, result_bytes);
        const auto result = unpack_int32_le(result_bytes);
        for (size_t row = 0; row < rows; ++row) {
            int64_t expected = 0;
            for (size_t column = 0; column < columns; ++column) {
                expected += int64_t{weights[row * columns + column]} * input[column];
            }
            if (result[row] != expected) throw std::runtime_error("INT64 reference mismatch");
        }
        const auto& counters = fabric.counters();
        if (counters.peer_bytes != columns + (rows - split) * 4 ||
            counters.macs != rows * columns || counters.local_weight_bytes != rows * columns) {
            throw std::runtime_error("unexpected compute or transport accounting");
        }
        std::cout << "INT8 GEMV 7x65, row split 3+4: exact match against INT64 reference\n"
                  << "Output:";
        for (const auto value : result) std::cout << ' ' << value;
        std::cout << "\nLogical capacity: " << fabric.capacity_bytes() / GIB << " GiB\n"
                  << "Local weight bytes: " << counters.local_weight_bytes << '\n'
                  << "Peer bytes: " << counters.peer_bytes << " (65 input + 16 output)\n"
                  << "Materialized page backing: " << fabric.resident_bytes() << " bytes\n"
                  << "CPU functional simulation; no FPGA execution or performance estimate.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
