#include "ffn.h"
#include "gemv.h"
#include "numeric_io.h"

#include <array>

namespace nt::ntrpu {

FfnResult q8_0_ffn_distributed(std::span<const std::byte> gate,
                             std::span<const std::byte> up,
                             std::span<const std::byte> down,
                             std::span<const float> input, size_t hidden) {
    const size_t dim = input.size();
    if (dim == 0 || dim % 32 != 0 || dim > 1024 || hidden < 2 || hidden > 1024) {
        throw std::invalid_argument("FFN fixture supports D=32..1024, H=2..1024");
    }
    const size_t padded = (hidden + 31) / 32 * 32;
    const size_t row_bytes = dim / 32 * 34;
    const size_t down_bytes = dim * (padded / 32) * 34;
    const size_t split = hidden / 2;
    if (gate.size() != hidden * row_bytes || up.size() != gate.size() ||
        down.size() != down_bytes || down_bytes > GEMV_MAX_WEIGHT_BYTES ||
        (hidden - split) * row_bytes > GEMV_MAX_WEIGHT_BYTES) {
        throw std::invalid_argument("FFN weight shape or command budget");
    }
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    fabric.set_peer_access(0, 1, true);
    fabric.set_peer_access(1, 0, true);
    std::array<BufferId, 2> wg{}, wu{}, x{}, g{}, u{}, a{};
    std::array<size_t, 2> count{split, hidden - split};
    for (UnitId id = 0; id < 2; ++id) {
        wg[id] = fabric.register_buffer(id, 8 * GIB, count[id] * row_bytes);
        wu[id] = fabric.register_buffer(id, 12 * GIB, count[id] * row_bytes);
        x[id] = fabric.register_buffer(id, 16 * GIB, dim * 4);
        g[id] = fabric.register_buffer(id, 20 * GIB, count[id] * 4);
        u[id] = fabric.register_buffer(id, 24 * GIB, count[id] * 4);
        a[id] = fabric.register_buffer(id, 28 * GIB, count[id] * 4);
        const size_t offset = id == 0 ? 0 : split * row_bytes;
        fabric.write({wg[id]}, gate.subspan(offset, count[id] * row_bytes));
        fabric.write({wu[id]}, up.subspan(offset, count[id] * row_bytes));
    }
    auto wd = fabric.register_buffer(1, 32 * GIB, down_bytes);
    auto gathered = fabric.register_buffer(1, 36 * GIB, padded * 4);
    auto y = fabric.register_buffer(1, 40 * GIB, dim * 4);
    auto result_buffer = fabric.register_buffer(0, 40 * GIB, dim * 4);
    fabric.write({wd}, down);
    fabric.write({x[0]}, pack_f32_le(input));
    // Untouched sparse memory is zero: gathered[hidden:padded] stays zero.
    auto sharded = [&](TensorId id, const std::array<BufferId, 2>& w) {
        return TensorSpec{id, hidden * row_bytes, {{0, {w[0]}, split * row_bytes},
            {split * row_bytes, {w[1]}, (hidden - split) * row_bytes}},
            "q8_0_row_major", hidden, dim};
    };
    fabric.publish_model(1, {sharded(1, wg), sharded(2, wu),
        {3, down_bytes, {{0, {wd}, down_bytes}}, "q8_0_row_major", dim, padded}});
    uint64_t sequence = 1;
    auto execute = [&] {
        fabric.execute(sequence);
        auto completion = fabric.poll();
        if (!completion || completion->sequence != sequence || completion->status != Status::Ok) {
            throw std::runtime_error("FFN stage failed; no dependent stage submitted");
        }
        ++sequence;
    };
    fabric.submit({sequence, 1, {x[0]}, {x[1]}, dim * 4});
    execute();
    for (UnitId id = 0; id < 2; ++id) {
        const size_t first = id == 0 ? 0 : split;
        fabric.submit_q8_0_gemv({sequence, 1, id, 1, first, count[id], {x[id]}, {g[id]}});
        execute();
        fabric.submit_q8_0_gemv({sequence, 1, id, 2, first, count[id], {x[id]}, {u[id]}});
        execute();
        fabric.submit_swiglu({sequence, 1, id, count[id], {g[id]}, {u[id]}, {a[id]}});
        execute();
        fabric.submit({sequence, 1, {a[id]}, {gathered, first * 4}, count[id] * 4});
        execute();
    }
    fabric.submit_q8_0_gemv({sequence, 1, 1, 3, 0, dim, {gathered}, {y}});
    execute();
    fabric.submit({sequence, 1, {y}, {result_buffer}, dim * 4});
    execute();
    std::vector<std::byte> result(dim * 4);
    fabric.read({result_buffer}, result);
    return {unpack_f32_le(result), fabric.counters(), fabric.resident_bytes()};
}

} // namespace nt::ntrpu
