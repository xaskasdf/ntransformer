#include "simulator.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace nt::ntrpu;

int main() {
    try {
        Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
        fabric.set_peer_access(0, 1, true);
        fabric.set_peer_access(1, 0, true);
        auto weights_a = fabric.register_buffer(0, 8 * GIB + PAGE_BYTES - 3, 32);
        auto weights_b = fabric.register_buffer(1, 500 * GIB, 32);
        auto result = fabric.register_buffer(0, 400 * GIB, 64);
        auto peer_result = fabric.register_buffer(1, 64 * GIB, 32);
        std::vector<std::byte> weights(64);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<std::byte>(i + 1);
        fabric.write({weights_a}, std::span<const std::byte>(weights).first(32));
        fabric.write({weights_b}, std::span<const std::byte>(weights).subspan(32));
        fabric.publish_model(1, {{70, 64, {{0, {weights_a}, 32}, {32, {weights_b}, 32}}}});

        fabric.submit_tensor_copy(1, 1, 70, 0, {result}, 64);
        fabric.submit({2, 1, {weights_a}, {peer_result}, 32});
        fabric.execute(2);
        fabric.execute(1);
        for (uint64_t expected : {2ULL, 1ULL}) {
            auto completion = fabric.poll();
            if (!completion || completion->sequence != expected || completion->status != Status::Ok) {
                throw std::runtime_error("unexpected DMA completion");
            }
            std::cout << "Completion " << completion->sequence << ": "
                      << completion->bytes_transferred << " bytes, "
                      << status_name(completion->status) << '\n';
        }
        std::vector<std::byte> gathered(64), peer_copy(32);
        fabric.read({result}, gathered);
        fabric.read({peer_result}, peer_copy);
        if (gathered != weights || !std::equal(peer_copy.begin(), peer_copy.end(), weights.begin())) {
            throw std::runtime_error("distributed weights did not match reference");
        }
        std::cout << "Units: 2 x 512 GiB; logical capacity: " << fabric.capacity_bytes() / GIB << " GiB\n"
                  << "Materialized page backing: " << fabric.resident_bytes() << " bytes\n"
                  << "Copied: " << fabric.counters().bytes_copied << " bytes; peer traffic: "
                  << fabric.counters().peer_bytes << " bytes\n"
                  << "Distributed tensor verified. CPU-only functional simulation; no hardware or timing emulation.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
