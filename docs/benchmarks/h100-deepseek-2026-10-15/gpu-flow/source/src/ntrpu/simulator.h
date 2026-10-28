#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nt::ntrpu {

using UnitId = uint32_t;
using BufferId = uint64_t;
using TensorId = uint64_t;
constexpr uint64_t GIB = uint64_t{1} << 30;
constexpr size_t PAGE_BYTES = 4096;

enum class Status {
    Ok, InvalidArgument, OutOfBounds, UnknownBuffer, UnknownTensor,
    StaleEpoch, ReadOnly, Busy, QueueFull, DuplicateSequence,
    UnknownSequence, PeerUnavailable, DeviceOffline, ResourceLimit, Cancelled, NonLocal,
    NumericError
};

const char* status_name(Status status);

class Error : public std::runtime_error {
public:
    explicit Error(Status status);
    Status status() const { return status_; }
private:
    Status status_;
};

struct UnitConfig {
    uint64_t capacity_bytes;
    size_t resident_page_limit = 4096; // 16 MiB of backing per unit by default.
};

struct BufferSlice {
    BufferId buffer;
    uint64_t offset = 0;
};

struct TensorShard {
    uint64_t tensor_offset;
    BufferSlice source;
    uint64_t size_bytes;
};

struct TensorSpec {
    TensorId id;
    uint64_t size_bytes;
    std::vector<TensorShard> shards;
    std::string format = "raw"; // Typed formats: int8_row_major, q8_0_row_major.
    uint64_t rows = 0;
    uint64_t columns = 0;
};

struct DmaCommand {
    uint64_t sequence;
    uint64_t model_epoch;
    BufferSlice source;
    BufferSlice destination;
    uint64_t size_bytes;
};

struct GemvCommand {
    uint64_t sequence;
    uint64_t model_epoch;
    UnitId unit;
    TensorId weights;
    uint64_t first_row;
    uint64_t row_count;
    BufferSlice input; // INT8 for submit_gemv; FP32 little endian for submit_q8_0_gemv.
    BufferSlice output; // INT32 or FP32 little endian, respectively; always local.
};

struct SwigluCommand {
    uint64_t sequence;
    uint64_t model_epoch;
    UnitId unit;
    uint64_t elements;
    BufferSlice gate; // All operands FP32 little endian and local to unit.
    BufferSlice up;
    BufferSlice output;
};

struct Completion {
    uint64_t sequence;
    uint64_t model_epoch;
    Status status;
    uint64_t bytes_transferred;
    uint64_t output_bytes = 0; // Compute only; not DMA traffic.
    uint64_t macs = 0;
};

struct Counters {
    uint64_t submitted = 0;
    uint64_t completed = 0; // Includes errors and cancellations.
    uint64_t failed = 0;
    uint64_t cancelled = 0;
    uint64_t bytes_copied = 0;
    uint64_t peer_bytes = 0;
    uint64_t gemv_commands = 0; // Successful commands only.
    uint64_t q8_0_commands = 0; // Subset of gemv_commands.
    uint64_t macs = 0;
    uint64_t local_weight_bytes = 0;
    uint64_t compute_output_bytes = 0;
    uint64_t swiglu_commands = 0;
    uint64_t swiglu_elements = 0;
};

// Functional, single-threaded simulator. No real BAR mapping, DMA or timing.
// Only touched pages consume backing memory. Buffer IDs are never reused.
class Fabric {
public:
    explicit Fabric(std::vector<UnitConfig> units, size_t queue_depth = 64,
                    uint64_t max_dma_bytes = 1024 * 1024);

    uint64_t capacity_bytes() const { return capacity_bytes_; }
    uint64_t resident_bytes() const;
    uint64_t model_epoch() const { return model_epoch_; }
    size_t pending_count() const { return pending_.size(); }
    size_t completion_count() const { return completions_.size(); }
    const Counters& counters() const { return counters_; }

    // Cross-unit routes default to disabled and are directional.
    void set_peer_access(UnitId source, UnitId destination, bool enabled);
    void set_online(UnitId unit, bool online);

    // Non-overlapping physical reservations prevent writable aliases of weights.
    BufferId register_buffer(UnitId unit, uint64_t offset, uint64_t size_bytes);
    void unregister_buffer(BufferId buffer);
    void write(BufferSlice destination, std::span<const std::byte> data);
    void read(BufferSlice source, std::span<std::byte> data) const;

    // Atomic directory replacement, only when all completions are consumed.
    // Epochs must increase. An empty directory retires the current model.
    // All buffers referenced by published tensors become immutable.
    void publish_model(uint64_t epoch, std::vector<TensorSpec> tensors);
    std::vector<TensorShard> resolve_tensor(uint64_t epoch, TensorId tensor,
                                          uint64_t offset, uint64_t size_bytes) const;

    // Invalid submissions throw Error and do not acquire queue slots or buffers.
    // Accepted work pins every buffer until execute() or cancel().
    void submit(DmaCommand command);
    // Whole rows only. Weights, input and output must belong to command.unit.
    void submit_gemv(GemvCommand command);
    // Q8_0 weights, FP32 activations/output; INT8 HLS model plus host scaling.
    // Non-finite operands or unrepresentable FP32 output produce NumericError.
    void submit_q8_0_gemv(GemvCommand command);
    // CPU functional model, not yet an HLS activation kernel. At most 1 MiB output.
    void submit_swiglu(SwigluCommand command);
    void submit_tensor_copy(uint64_t sequence, uint64_t epoch, TensorId tensor,
                            uint64_t tensor_offset, BufferSlice destination,
                            uint64_t size_bytes);
    // Explicit execution order allows testing out-of-order completions.
    // Runtime faults produce a completion with zero transferred bytes.
    void execute(uint64_t sequence);
    void cancel(uint64_t sequence);
    std::optional<Completion> poll();

private:
    using Page = std::array<std::byte, PAGE_BYTES>;
    struct Unit {
        UnitConfig config;
        bool online = true;
        std::unordered_map<uint64_t, Page> pages;
    };
    struct Buffer {
        UnitId unit;
        uint64_t offset;
        uint64_t size_bytes;
        bool frozen = false;
        size_t in_flight = 0;
    };
    struct Segment {
        BufferSlice source;
        uint64_t size_bytes;
    };
    struct Job {
        uint64_t sequence;
        uint64_t epoch;
        std::vector<Segment> sources;
        BufferSlice destination;
        uint64_t size_bytes;
        std::vector<BufferId> pins;
        struct GemvWork {
            UnitId unit;
            uint64_t rows;
            uint64_t columns;
            uint64_t weight_bytes;
            uint64_t input_bytes;
            bool q8_0;
        };
        std::optional<GemvWork> gemv = std::nullopt;
        std::optional<UnitId> swiglu_unit = std::nullopt;
    };

    Unit& unit(UnitId id);
    const Unit& unit(UnitId id) const;
    Buffer& buffer(BufferId id);
    const Buffer& buffer(BufferId id) const;
    static void check_range(uint64_t offset, uint64_t size, uint64_t capacity);
    const Buffer& validate_slice(BufferSlice slice, uint64_t size, bool writable) const;
    void check_epoch(uint64_t epoch) const;
    void submit_matrix_gemv(GemvCommand command, bool q8_0);
    void validate_job(const Job& job) const;
    void submit_job(Job job);
    void finish(std::map<uint64_t, Job>::iterator it, Status status, uint64_t peer_bytes);
    void read_memory(UnitId id, uint64_t offset, std::span<std::byte> output) const;
    void write_memory(UnitId id, uint64_t offset, std::span<const std::byte> input);

    std::vector<Unit> units_;
    std::vector<std::vector<bool>> routes_;
    std::unordered_map<BufferId, Buffer> buffers_;
    std::unordered_map<TensorId, TensorSpec> tensors_;
    std::map<uint64_t, Job> pending_;
    std::vector<Completion> completions_;
    uint64_t capacity_bytes_ = 0;
    uint64_t next_buffer_ = 1;
    uint64_t model_epoch_ = 0;
    size_t queue_depth_;
    uint64_t max_dma_bytes_;
    Counters counters_;
};

} // namespace nt::ntrpu
