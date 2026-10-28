#include "ntrpu/simulator.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace nt::ntrpu;

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + ": " + #condition); \
} while (false)

static std::vector<std::byte> bytes(size_t count, unsigned value = 0) {
    return std::vector<std::byte>(count, static_cast<std::byte>(value));
}

static std::vector<std::byte> read_bytes(const Fabric& fabric, BufferId buffer,
                                         uint64_t offset, size_t count) {
    auto data = bytes(count);
    fabric.read({buffer, offset}, data);
    return data;
}

template <class F>
static void expect(Status status, F operation) {
    try {
        operation();
    } catch (const Error& error) {
        CHECK(error.status() == status);
        return;
    }
    throw std::runtime_error(std::string("expected error: ") + status_name(status));
}

static Completion completion(Fabric& fabric, uint64_t sequence, Status status, uint64_t count) {
    auto result = fabric.poll();
    CHECK(result.has_value());
    CHECK(result->sequence == sequence);
    CHECK(result->status == status);
    CHECK(result->bytes_transferred == count);
    return *result;
}

static void test_sparse_memory() {
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    CHECK(fabric.capacity_bytes() == 1024 * GIB);
    CHECK(fabric.resident_bytes() == 0);
    auto a = fabric.register_buffer(0, 8 * GIB + PAGE_BYTES - 2, 16);
    auto b = fabric.register_buffer(1, 512 * GIB - 16, 16);
    CHECK(read_bytes(fabric, a, 0, 16) == bytes(16));
    CHECK(fabric.resident_bytes() == 0);
    fabric.write({a}, bytes(16, 0x5a));
    fabric.write({b}, bytes(16, 0xa5));
    CHECK(read_bytes(fabric, a, 0, 16) == bytes(16, 0x5a));
    CHECK(read_bytes(fabric, b, 0, 16) == bytes(16, 0xa5));
    CHECK(fabric.resident_bytes() == 3 * PAGE_BYTES);
    fabric.write({a, 16}, {});
    CHECK(read_bytes(fabric, a, 16, 0).empty());
    expect(Status::OutOfBounds, [&] { fabric.write({b, 15}, bytes(2)); });
}

static void test_registration() {
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{}); });
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{{0}}); });
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{{1, 0}}); });
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{{max}, {1}}); });
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{{1}}, 0); });
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{{1}}, 1, 0); });
    expect(Status::InvalidArgument, [] { Fabric f(std::vector<UnitConfig>{{1}}, 1, max); });
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    auto a = fabric.register_buffer(0, 100, 100);
    auto b = fabric.register_buffer(0, 200, 100); // Adjacent is permitted.
    auto c = fabric.register_buffer(1, 100, 100); // A distinct physical unit.
    CHECK(a != b && b != c);
    expect(Status::InvalidArgument, [&] { fabric.register_buffer(2, 0, 1); });
    expect(Status::InvalidArgument, [&] { fabric.register_buffer(0, 0, 0); });
    expect(Status::InvalidArgument, [&] { fabric.register_buffer(0, 199, 2); });
    expect(Status::OutOfBounds, [&] { fabric.register_buffer(0, max - 1, 4); });
    expect(Status::OutOfBounds, [&] { fabric.write({a, max}, bytes(2)); });
    fabric.unregister_buffer(a);
    expect(Status::UnknownBuffer, [&] { read_bytes(fabric, a, 0, 1); });
    CHECK(fabric.register_buffer(0, 100, 100) != a);
}

static void test_publication() {
    Fabric fabric(std::vector<UnitConfig>{{4096}});
    auto weights = fabric.register_buffer(0, 0, 128);
    auto scratch = fabric.register_buffer(0, 128, 128);
    fabric.write({weights}, bytes(128, 23));
    fabric.publish_model(4, {{11, 64, {{0, {weights, 32}, 64}}, "Q8_0"}});
    auto mapping = fabric.resolve_tensor(4, 11, 3, 7);
    CHECK(mapping.size() == 1);
    CHECK(mapping[0].source.buffer == weights && mapping[0].source.offset == 35);
    CHECK(mapping[0].size_bytes == 7);
    CHECK(fabric.resolve_tensor(4, 11, 64, 0).empty());
    expect(Status::ReadOnly, [&] { fabric.write({weights}, bytes(1)); });
    expect(Status::Busy, [&] { fabric.unregister_buffer(weights); });
    expect(Status::InvalidArgument, [&] { fabric.register_buffer(0, 32, 64); });
    expect(Status::ReadOnly, [&] { fabric.submit({1, 4, {scratch}, {weights}, 1}); });
    expect(Status::OutOfBounds, [&] { fabric.resolve_tensor(4, 11, 63, 2); });
    expect(Status::UnknownTensor, [&] { fabric.resolve_tensor(4, 99, 0, 1); });
    CHECK(read_bytes(fabric, weights, 0, 128) == bytes(128, 23));
}

static void test_invalid_directory() {
    Fabric fabric(std::vector<UnitConfig>{{4096}});
    auto a = fabric.register_buffer(0, 0, 128);
    auto b = fabric.register_buffer(0, 128, 128);
    fabric.publish_model(1, {{7, 64, {{0, {a}, 64}}}});
    // Gaps, overlaps, duplicate IDs, missing bytes and invalid source ranges.
    expect(Status::InvalidArgument, [&] {
        fabric.publish_model(2, {{8, 64, {{1, {b}, 63}}}});
    });
    expect(Status::InvalidArgument, [&] {
        fabric.publish_model(2, {{8, 64, {{0, {b}, 40}, {32, {b, 40}, 24}}}});
    });
    expect(Status::InvalidArgument, [&] {
        fabric.publish_model(2, {{8, 64, {{0, {b}, 64}}}, {8, 64, {{0, {a}, 64}}}});
    });
    expect(Status::InvalidArgument, [&] {
        fabric.publish_model(2, {{8, 65, {{0, {b}, 64}}}});
    });
    expect(Status::OutOfBounds, [&] {
        fabric.publish_model(2, {{8, 64, {{0, {b, 100}, 64}}}});
    });
    expect(Status::UnknownBuffer, [&] {
        fabric.publish_model(2, {{8, 64, {{0, {999}, 64}}}});
    });
    CHECK(fabric.model_epoch() == 1);
    CHECK(fabric.resolve_tensor(1, 7, 0, 1)[0].source.buffer == a);
    expect(Status::ReadOnly, [&] { fabric.write({a}, bytes(1)); });
    fabric.write({b}, bytes(1)); // Failed publication did not freeze new buffers.

    constexpr auto max = std::numeric_limits<uint64_t>::max();
    Fabric huge(std::vector<UnitConfig>{{max}});
    auto backing = huge.register_buffer(0, 0, max - 32);
    huge.publish_model(1, {{1, max, {{0, {backing}, max - 32}, {max - 32, {backing, 16}, 32}}}});
    auto tail = huge.resolve_tensor(1, 1, max - 1, 1);
    CHECK(tail.size() == 1 && tail[0].source.offset == 47);
    expect(Status::OutOfBounds, [&] { huge.resolve_tensor(1, 1, max - 1, 2); });
    CHECK(huge.resident_bytes() == 0);
}

static void test_epochs() {
    Fabric fabric(std::vector<UnitConfig>{{4096}});
    auto a = fabric.register_buffer(0, 0, 64);
    auto b = fabric.register_buffer(0, 64, 64);
    expect(Status::StaleEpoch, [&] { fabric.submit({1, 0, {a}, {b}, 8}); });
    fabric.publish_model(2, {{7, 64, {{0, {a}, 64}}}});
    expect(Status::StaleEpoch, [&] { fabric.submit({1, 1, {a}, {b}, 8}); });
    expect(Status::StaleEpoch, [&] { fabric.publish_model(2, {}); });
    fabric.submit({1, 2, {a}, {b}, 8});
    expect(Status::Busy, [&] { fabric.publish_model(3, {}); });
    fabric.execute(1);
    expect(Status::Busy, [&] { fabric.publish_model(3, {}); });
    CHECK(completion(fabric, 1, Status::Ok, 8).model_epoch == 2);
    fabric.publish_model(3, {});
    expect(Status::StaleEpoch, [&] { fabric.submit({2, 2, {a}, {b}, 8}); });
    expect(Status::StaleEpoch, [&] { fabric.resolve_tensor(2, 7, 0, 1); });
    expect(Status::UnknownTensor, [&] { fabric.resolve_tensor(3, 7, 0, 1); });
    fabric.write({a}, bytes(64, 5)); // Retired buffers are mutable again.
    fabric.unregister_buffer(a);
}

static void test_sharded_dma() {
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    fabric.set_peer_access(0, 1, true);
    auto a = fabric.register_buffer(0, 0, 512 * GIB);
    auto b = fabric.register_buffer(1, 0, 288 * GIB);
    auto output = fabric.register_buffer(1, 400 * GIB, 64);
    fabric.write({a, 512 * GIB - 8}, bytes(8, 0x11));
    fabric.write({b}, bytes(8, 0x22));
    // An 800 GiB logical tensor; its first shard exceeds 32-bit addressing.
    fabric.publish_model(1, {{42, 800 * GIB,
        {{512 * GIB, {b}, 288 * GIB}, {0, {a}, 512 * GIB}}}});
    auto spans = fabric.resolve_tensor(1, 42, 512 * GIB - 8, 16);
    CHECK(spans.size() == 2 && spans[0].size_bytes == 8 && spans[1].size_bytes == 8);
    fabric.submit_tensor_copy(1, 1, 42, 512 * GIB - 8, {output, 7}, 16);
    fabric.execute(1);
    completion(fabric, 1, Status::Ok, 16);
    auto expected = bytes(64);
    std::fill_n(expected.begin() + 7, 8, std::byte{0x11});
    std::fill_n(expected.begin() + 15, 8, std::byte{0x22});
    CHECK(read_bytes(fabric, output, 0, 64) == expected);
    CHECK(fabric.counters().peer_bytes == 8);
    CHECK(fabric.resident_bytes() == 3 * PAGE_BYTES);
}

static void test_completions() {
    Fabric fabric(std::vector<UnitConfig>{{8192}, {8192}});
    fabric.set_peer_access(0, 1, true);
    auto source = fabric.register_buffer(0, 0, 32);
    auto out_a = fabric.register_buffer(1, 0, 32);
    auto out_b = fabric.register_buffer(1, 32, 32);
    fabric.write({source}, bytes(32, 0x73));
    fabric.publish_model(1, {});
    fabric.submit({91, 1, {source}, {out_a}, 32});
    fabric.submit({12, 1, {source}, {out_b}, 16});
    expect(Status::Busy, [&] { fabric.write({source}, bytes(1)); });
    expect(Status::Busy, [&] { read_bytes(fabric, out_a, 0, 1); });
    expect(Status::Busy, [&] { fabric.unregister_buffer(source); });
    fabric.execute(12);
    completion(fabric, 12, Status::Ok, 16);
    expect(Status::Busy, [&] { fabric.unregister_buffer(source); });
    fabric.execute(91);
    completion(fabric, 91, Status::Ok, 32);
    CHECK(!fabric.poll());
    CHECK(read_bytes(fabric, out_a, 0, 32) == bytes(32, 0x73));
    CHECK(read_bytes(fabric, out_b, 16, 16) == bytes(16));
    CHECK(fabric.counters().submitted == 2 && fabric.counters().completed == 2);
    CHECK(fabric.counters().bytes_copied == 48 && fabric.counters().peer_bytes == 48);
    fabric.unregister_buffer(source);
}

static void test_queue_limits() {
    Fabric fabric(std::vector<UnitConfig>{{4096}}, 2);
    auto a = fabric.register_buffer(0, 0, 32);
    auto b = fabric.register_buffer(0, 32, 32);
    fabric.publish_model(1, {});
    fabric.submit({1, 1, {a}, {b}, 8});
    fabric.submit({2, 1, {a}, {b, 8}, 8});
    expect(Status::DuplicateSequence, [&] { fabric.submit({1, 1, {a}, {b}, 8}); });
    expect(Status::QueueFull, [&] { fabric.submit({3, 1, {a}, {b}, 8}); });
    fabric.execute(2);
    expect(Status::DuplicateSequence, [&] { fabric.submit({2, 1, {a}, {b}, 8}); });
    expect(Status::QueueFull, [&] { fabric.submit({3, 1, {a}, {b}, 8}); });
    completion(fabric, 2, Status::Ok, 8);
    fabric.submit({3, 1, {a}, {b, 16}, 8});
    fabric.execute(3);
    fabric.execute(1);
    completion(fabric, 3, Status::Ok, 8);
    completion(fabric, 1, Status::Ok, 8);
    CHECK(fabric.pending_count() == 0 && fabric.completion_count() == 0);
    expect(Status::UnknownSequence, [&] { fabric.execute(1); });
}

static void test_invalid_commands() {
    Fabric fabric(std::vector<UnitConfig>{{4096}, {4096}}, 2, 16);
    auto a = fabric.register_buffer(0, 0, 32);
    auto b = fabric.register_buffer(1, 0, 32);
    fabric.publish_model(1, {});
    expect(Status::InvalidArgument, [&] { fabric.submit({0, 1, {a}, {b}, 8}); });
    expect(Status::InvalidArgument, [&] { fabric.submit({1, 1, {a}, {b}, 0}); });
    expect(Status::InvalidArgument, [&] { fabric.submit({1, 1, {a}, {b}, 17}); });
    expect(Status::UnknownBuffer, [&] { fabric.submit({1, 1, {999}, {b}, 8}); });
    expect(Status::OutOfBounds, [&] { fabric.submit({1, 1, {a, 30}, {b}, 8}); });
    expect(Status::OutOfBounds, [&] { fabric.submit({1, 1, {a}, {b, 30}, 8}); });
    expect(Status::PeerUnavailable, [&] { fabric.submit({1, 1, {a}, {b}, 8}); });
    fabric.set_peer_access(0, 1, true);
    expect(Status::PeerUnavailable, [&] { fabric.submit({1, 1, {b}, {a}, 8}); });
    CHECK(fabric.pending_count() == 0 && fabric.counters().submitted == 0);
    CHECK(fabric.resident_bytes() == 0);
    fabric.write({a}, bytes(8, 1)); // No failed command left a pin behind.
    fabric.submit({1, 1, {a}, {b}, 8});
    fabric.execute(1);
    completion(fabric, 1, Status::Ok, 8);
}

static void test_device_loss() {
    Fabric fabric(std::vector<UnitConfig>{{4096}, {4096}});
    fabric.set_peer_access(0, 1, true);
    auto a = fabric.register_buffer(0, 0, 32);
    auto b = fabric.register_buffer(1, 0, 32);
    fabric.write({a}, bytes(32, 7));
    fabric.write({b}, bytes(32, 9));
    fabric.publish_model(1, {});
    for (UnitId lost : {0U, 1U}) {
        fabric.submit({1, 1, {a}, {b}, 32});
        fabric.set_online(lost, false);
        fabric.execute(1);
        completion(fabric, 1, Status::DeviceOffline, 0);
        fabric.set_online(lost, true);
        CHECK(read_bytes(fabric, b, 0, 32) == bytes(32, 9));
    }
    CHECK(fabric.counters().failed == 2 && fabric.counters().bytes_copied == 0);
    fabric.unregister_buffer(a);
}

static void test_link_loss() {
    Fabric fabric(std::vector<UnitConfig>{{4096}, {4096}});
    fabric.set_peer_access(0, 1, true);
    auto a = fabric.register_buffer(0, 0, 32);
    auto b = fabric.register_buffer(1, 0, 32);
    fabric.write({a}, bytes(32, 7));
    fabric.publish_model(1, {});
    fabric.submit({1, 1, {a}, {b}, 32});
    fabric.set_peer_access(0, 1, false);
    fabric.execute(1);
    completion(fabric, 1, Status::PeerUnavailable, 0);
    CHECK(read_bytes(fabric, b, 0, 32) == bytes(32));
    CHECK(fabric.counters().peer_bytes == 0);
}

static void test_memory_budget() {
    Fabric fabric(std::vector<UnitConfig>{{8 * PAGE_BYTES, 1}, {8 * PAGE_BYTES, 2}});
    fabric.set_peer_access(1, 0, true);
    auto a = fabric.register_buffer(1, 0, PAGE_BYTES + 4);
    auto b = fabric.register_buffer(0, 0, PAGE_BYTES + 4);
    fabric.write({a}, bytes(PAGE_BYTES + 4, 7));
    fabric.write({b}, bytes(1, 9));
    CHECK(fabric.resident_bytes() == 3 * PAGE_BYTES);
    fabric.publish_model(1, {});
    fabric.submit({1, 1, {a}, {b}, PAGE_BYTES + 4});
    fabric.execute(1);
    completion(fabric, 1, Status::ResourceLimit, 0);
    auto expected = bytes(PAGE_BYTES + 4);
    expected[0] = std::byte{9};
    CHECK(read_bytes(fabric, b, 0, expected.size()) == expected);
    CHECK(fabric.resident_bytes() == 3 * PAGE_BYTES);
    expect(Status::ResourceLimit, [&] { fabric.write({b}, bytes(PAGE_BYTES + 4, 8)); });
    CHECK(read_bytes(fabric, b, 0, expected.size()) == expected);
}

static void test_cancellation() {
    Fabric fabric(std::vector<UnitConfig>{{4096}});
    auto a = fabric.register_buffer(0, 0, 32);
    auto b = fabric.register_buffer(0, 32, 32);
    fabric.write({a}, bytes(32, 7));
    fabric.publish_model(1, {});
    fabric.submit({8, 1, {a}, {b}, 32});
    fabric.cancel(8);
    CHECK(read_bytes(fabric, b, 0, 32) == bytes(32));
    expect(Status::DuplicateSequence, [&] { fabric.submit({8, 1, {a}, {b}, 32}); });
    completion(fabric, 8, Status::Cancelled, 0);
    expect(Status::UnknownSequence, [&] { fabric.cancel(8); });
    CHECK(fabric.counters().cancelled == 1 && fabric.counters().failed == 0);
    CHECK(fabric.counters().completed == 1 && fabric.counters().bytes_copied == 0);
    fabric.unregister_buffer(a);
    fabric.unregister_buffer(b);
}

static void test_overlap() {
    Fabric fabric(std::vector<UnitConfig>{{8192}});
    auto a = fabric.register_buffer(0, PAGE_BYTES - 4, 32);
    std::vector<std::byte> original(32);
    for (size_t i = 0; i < original.size(); ++i) original[i] = static_cast<std::byte>(i);
    fabric.write({a}, original);
    fabric.publish_model(1, {});
    fabric.submit({1, 1, {a}, {a, 3}, 24});
    fabric.execute(1);
    completion(fabric, 1, Status::Ok, 24);
    auto expected = original;
    std::copy_n(original.begin(), 24, expected.begin() + 3);
    CHECK(read_bytes(fabric, a, 0, 32) == expected);
    fabric.write({a}, original);
    fabric.submit({2, 1, {a, 3}, {a}, 24});
    fabric.execute(2);
    completion(fabric, 2, Status::Ok, 24);
    expected = original;
    std::copy_n(original.begin() + 3, 24, expected.begin());
    CHECK(read_bytes(fabric, a, 0, 32) == expected);
}

static void test_randomized() {
    Fabric fabric(std::vector<UnitConfig>{{512 * GIB}, {512 * GIB}});
    fabric.set_peer_access(0, 1, true);
    fabric.set_peer_access(1, 0, true);
    auto a = fabric.register_buffer(0, 7 * GIB + PAGE_BYTES - 13, 128);
    auto b = fabric.register_buffer(1, 9 * GIB + PAGE_BYTES - 7, 128);
    const BufferId outputs[] = {
        fabric.register_buffer(0, 400 * GIB, 384),
        fabric.register_buffer(1, 400 * GIB, 384)
    };
    std::mt19937 random(20260930);
    auto reference = bytes(170);
    for (auto& value : reference) value = static_cast<std::byte>(random() % 256);
    fabric.write({a, 5}, std::span<const std::byte>(reference).first(97));
    fabric.write({b, 11}, std::span<const std::byte>(reference).subspan(97));
    fabric.publish_model(1, {{71, 170, {{97, {b, 11}, 73}, {0, {a, 5}, 97}}}});
    uint64_t copied = 0;
    for (uint64_t sequence = 1; sequence <= 200; ++sequence) {
        const size_t start = random() % reference.size();
        const size_t length = 1 + random() % (reference.size() - start);
        const size_t destination = random() % (385 - length);
        const auto output = outputs[sequence % 2];
        auto expected = bytes(384, 0xdd);
        fabric.write({output}, expected);
        fabric.submit_tensor_copy(sequence, 1, 71, start, {output, destination}, length);
        fabric.execute(sequence);
        completion(fabric, sequence, Status::Ok, length);
        std::copy_n(reference.begin() + start, length, expected.begin() + destination);
        CHECK(read_bytes(fabric, output, 0, 384) == expected);
        copied += length;
    }
    CHECK(fabric.counters().bytes_copied == copied);
    CHECK(fabric.counters().failed == 0 && fabric.counters().completed == 200);
    CHECK(fabric.resident_bytes() <= 6 * PAGE_BYTES);
}

int main(int argc, char** argv) {
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {
        {"sparse_memory", test_sparse_memory}, {"registration", test_registration},
        {"publication", test_publication}, {"invalid_directory", test_invalid_directory},
        {"epochs", test_epochs}, {"sharded_dma", test_sharded_dma},
        {"completions", test_completions}, {"queue_limits", test_queue_limits},
        {"invalid_commands", test_invalid_commands}, {"device_loss", test_device_loss},
        {"link_loss", test_link_loss}, {"memory_budget", test_memory_budget},
        {"cancellation", test_cancellation}, {"overlap", test_overlap},
        {"randomized", test_randomized}
    };
    size_t ran = 0;
    size_t failed = 0;
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
    if (ran == 0) {
        std::cerr << "Unknown test case\n";
        return 2;
    }
    return failed == 0 ? 0 : 1;
}
