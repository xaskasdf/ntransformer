#include "simulator.h"
#include "gemv.h"
#include "numeric_io.h"
#include "q8_0.h"
#include "activation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace nt::ntrpu {

const char* status_name(Status status) {
    switch (status) {
        case Status::Ok: return "ok";
        case Status::InvalidArgument: return "invalid argument";
        case Status::OutOfBounds: return "out of bounds";
        case Status::UnknownBuffer: return "unknown buffer";
        case Status::UnknownTensor: return "unknown tensor";
        case Status::StaleEpoch: return "stale model epoch";
        case Status::ReadOnly: return "published weights are read-only";
        case Status::Busy: return "buffer or model busy";
        case Status::QueueFull: return "queue full";
        case Status::DuplicateSequence: return "duplicate outstanding sequence";
        case Status::UnknownSequence: return "unknown sequence";
        case Status::PeerUnavailable: return "peer route unavailable";
        case Status::DeviceOffline: return "device offline";
        case Status::ResourceLimit: return "resident memory budget exceeded";
        case Status::Cancelled: return "cancelled";
        case Status::NonLocal: return "compute operands must be local to the executing unit";
        case Status::NumericError: return "non-finite operand or unrepresentable numeric output";
    }
    return "unknown status";
}

Error::Error(Status status) : std::runtime_error(status_name(status)), status_(status) {}

Fabric::Fabric(std::vector<UnitConfig> configs, size_t queue_depth, uint64_t max_dma_bytes)
    : queue_depth_(queue_depth), max_dma_bytes_(max_dma_bytes) {
    if (configs.empty() || configs.size() > std::numeric_limits<UnitId>::max() ||
        queue_depth == 0 || max_dma_bytes == 0 ||
        max_dma_bytes > std::vector<std::byte>{}.max_size()) {
        throw Error(Status::InvalidArgument);
    }
    for (auto config : configs) {
        if (config.capacity_bytes == 0 || config.resident_page_limit == 0 ||
            config.capacity_bytes > std::numeric_limits<uint64_t>::max() - capacity_bytes_) {
            throw Error(Status::InvalidArgument);
        }
        capacity_bytes_ += config.capacity_bytes;
        units_.push_back(Unit{config, true, {}});
    }
    routes_.assign(units_.size(), std::vector<bool>(units_.size(), false));
    for (size_t i = 0; i < units_.size(); ++i) routes_[i][i] = true;
    completions_.reserve(queue_depth_);
}

Fabric::Unit& Fabric::unit(UnitId id) {
    if (id >= units_.size()) throw Error(Status::InvalidArgument);
    return units_[id];
}

const Fabric::Unit& Fabric::unit(UnitId id) const {
    if (id >= units_.size()) throw Error(Status::InvalidArgument);
    return units_[id];
}

Fabric::Buffer& Fabric::buffer(BufferId id) {
    auto it = buffers_.find(id);
    if (it == buffers_.end()) throw Error(Status::UnknownBuffer);
    return it->second;
}

const Fabric::Buffer& Fabric::buffer(BufferId id) const {
    auto it = buffers_.find(id);
    if (it == buffers_.end()) throw Error(Status::UnknownBuffer);
    return it->second;
}

void Fabric::check_range(uint64_t offset, uint64_t size, uint64_t capacity) {
    // Subtraction avoids overflow for hostile 64-bit offsets and lengths.
    if (offset > capacity || size > capacity - offset) throw Error(Status::OutOfBounds);
}

const Fabric::Buffer& Fabric::validate_slice(BufferSlice slice, uint64_t size, bool writable) const {
    const auto& region = buffer(slice.buffer);
    check_range(slice.offset, size, region.size_bytes);
    if (!unit(region.unit).online) throw Error(Status::DeviceOffline);
    if (writable && region.frozen) throw Error(Status::ReadOnly);
    return region;
}

uint64_t Fabric::resident_bytes() const {
    uint64_t bytes = 0;
    for (const auto& device : units_) bytes += device.pages.size() * PAGE_BYTES;
    return bytes;
}

void Fabric::set_peer_access(UnitId source, UnitId destination, bool enabled) {
    unit(source);
    unit(destination);
    if (source == destination && !enabled) throw Error(Status::InvalidArgument);
    routes_[source][destination] = enabled;
}

void Fabric::set_online(UnitId id, bool online) {
    unit(id).online = online;
}

BufferId Fabric::register_buffer(UnitId id, uint64_t offset, uint64_t size_bytes) {
    const auto& device = unit(id);
    if (!device.online) throw Error(Status::DeviceOffline);
    if (size_bytes == 0 || next_buffer_ == std::numeric_limits<BufferId>::max()) {
        throw Error(Status::InvalidArgument);
    }
    check_range(offset, size_bytes, device.config.capacity_bytes);
    for (const auto& [handle, region] : buffers_) {
        (void)handle;
        if (region.unit == id && offset < region.offset + region.size_bytes &&
            region.offset < offset + size_bytes) {
            throw Error(Status::InvalidArgument);
        }
    }
    const auto handle = next_buffer_;
    buffers_.emplace(handle, Buffer{id, offset, size_bytes});
    ++next_buffer_;
    return handle;
}

void Fabric::unregister_buffer(BufferId handle) {
    const auto& region = buffer(handle);
    if (region.frozen || region.in_flight != 0) throw Error(Status::Busy);
    buffers_.erase(handle);
}

void Fabric::read_memory(UnitId id, uint64_t offset, std::span<std::byte> output) const {
    const auto& device = unit(id);
    size_t done = 0;
    while (done < output.size()) {
        const uint64_t address = offset + done;
        const auto in_page = static_cast<size_t>(address % PAGE_BYTES);
        const auto count = std::min(PAGE_BYTES - in_page, output.size() - done);
        const auto page = device.pages.find(address / PAGE_BYTES);
        if (page == device.pages.end()) {
            std::fill_n(output.data() + done, count, std::byte{0});
        } else {
            std::copy_n(page->second.data() + in_page, count, output.data() + done);
        }
        done += count;
    }
}

void Fabric::write_memory(UnitId id, uint64_t offset, std::span<const std::byte> input) {
    auto& device = unit(id);
    if (input.empty()) return;
    const uint64_t first = offset / PAGE_BYTES;
    const uint64_t last = (offset + input.size() - 1) / PAGE_BYTES;
    size_t missing = 0;
    for (uint64_t page = first; page <= last; ++page) {
        if (!device.pages.contains(page)) ++missing;
    }
    if (missing > device.config.resident_page_limit - device.pages.size()) {
        throw Error(Status::ResourceLimit);
    }
    // Finish allocation before changing any payload bytes. Unwritten pages read as zero.
    for (uint64_t page = first; page <= last; ++page) {
        device.pages.try_emplace(page, Page{});
    }
    size_t done = 0;
    while (done < input.size()) {
        const uint64_t address = offset + done;
        const auto in_page = static_cast<size_t>(address % PAGE_BYTES);
        const auto count = std::min(PAGE_BYTES - in_page, input.size() - done);
        auto& page = device.pages.at(address / PAGE_BYTES);
        std::copy_n(input.data() + done, count, page.data() + in_page);
        done += count;
    }
}

void Fabric::write(BufferSlice destination, std::span<const std::byte> data) {
    const auto& region = validate_slice(destination, data.size(), true);
    if (region.in_flight != 0) throw Error(Status::Busy);
    write_memory(region.unit, region.offset + destination.offset, data);
}

void Fabric::read(BufferSlice source, std::span<std::byte> data) const {
    const auto& region = validate_slice(source, data.size(), false);
    if (region.in_flight != 0) throw Error(Status::Busy);
    read_memory(region.unit, region.offset + source.offset, data);
}

void Fabric::check_epoch(uint64_t epoch) const {
    if (epoch == 0 || epoch != model_epoch_) throw Error(Status::StaleEpoch);
}

void Fabric::publish_model(uint64_t epoch, std::vector<TensorSpec> tensors) {
    if (epoch == 0 || epoch <= model_epoch_) throw Error(Status::StaleEpoch);
    if (!pending_.empty() || !completions_.empty()) throw Error(Status::Busy);
    std::unordered_map<TensorId, TensorSpec> directory;
    for (auto& tensor : tensors) {
        if (tensor.size_bytes == 0 || tensor.shards.empty() || directory.contains(tensor.id)) {
            throw Error(Status::InvalidArgument);
        }
        if (tensor.format == "int8_row_major" || tensor.format == "q8_0_row_major") {
            uint64_t row_bytes = tensor.columns;
            if (tensor.format == "q8_0_row_major") {
                const uint64_t blocks = tensor.columns / Q8_0_BLOCK_ELEMENTS;
                if (tensor.columns % Q8_0_BLOCK_ELEMENTS != 0 ||
                    blocks > std::numeric_limits<uint64_t>::max() / Q8_0_BLOCK_BYTES) {
                    throw Error(Status::InvalidArgument);
                }
                row_bytes = blocks * Q8_0_BLOCK_BYTES;
            }
            if (tensor.rows == 0 || row_bytes == 0 ||
                tensor.rows != tensor.size_bytes / row_bytes || tensor.size_bytes % row_bytes != 0) {
                throw Error(Status::InvalidArgument);
            }
        }
        std::sort(tensor.shards.begin(), tensor.shards.end(),
                  [](const auto& a, const auto& b) { return a.tensor_offset < b.tensor_offset; });
        uint64_t covered = 0;
        for (const auto& shard : tensor.shards) {
            if (shard.size_bytes == 0 || shard.tensor_offset != covered) {
                throw Error(Status::InvalidArgument);
            }
            check_range(covered, shard.size_bytes, tensor.size_bytes);
            validate_slice(shard.source, shard.size_bytes, false);
            covered += shard.size_bytes;
        }
        if (covered != tensor.size_bytes) throw Error(Status::InvalidArgument);
        directory.emplace(tensor.id, std::move(tensor));
    }
    // No mutation of the old epoch/directory/permissions until all validation succeeds.
    tensors_.swap(directory);
    for (auto& [handle, region] : buffers_) {
        (void)handle;
        region.frozen = false;
    }
    for (const auto& [id, tensor] : tensors_) {
        (void)id;
        for (const auto& shard : tensor.shards) buffer(shard.source.buffer).frozen = true;
    }
    model_epoch_ = epoch;
}

std::vector<TensorShard> Fabric::resolve_tensor(uint64_t epoch, TensorId id,
                                               uint64_t offset, uint64_t size_bytes) const {
    check_epoch(epoch);
    auto it = tensors_.find(id);
    if (it == tensors_.end()) throw Error(Status::UnknownTensor);
    check_range(offset, size_bytes, it->second.size_bytes);
    std::vector<TensorShard> result;
    if (size_bytes == 0) return result;
    const uint64_t end = offset + size_bytes;
    for (const auto& shard : it->second.shards) {
        const uint64_t begin = std::max(offset, shard.tensor_offset);
        const uint64_t stop = std::min(end, shard.tensor_offset + shard.size_bytes);
        if (begin >= stop) continue;
        BufferSlice source{shard.source.buffer, shard.source.offset + (begin - shard.tensor_offset)};
        validate_slice(source, stop - begin, false);
        result.push_back({begin, source, stop - begin});
    }
    return result;
}

void Fabric::validate_job(const Job& job) const {
    check_epoch(job.epoch);
    if (job.sequence == 0 || job.size_bytes == 0 ||
        (!job.gemv && !job.swiglu_unit && job.size_bytes > max_dma_bytes_)) {
        throw Error(Status::InvalidArgument);
    }
    const auto& destination = validate_slice(job.destination, job.size_bytes, true);
    if (job.gemv || job.swiglu_unit) {
        const auto compute_unit = job.gemv ? job.gemv->unit : *job.swiglu_unit;
        if (destination.unit != compute_unit) throw Error(Status::NonLocal);
        for (const auto& segment : job.sources) {
            const auto& source = validate_slice(segment.source, segment.size_bytes, false);
            if (source.unit != compute_unit) throw Error(Status::NonLocal);
        }
        return;
    }
    uint64_t covered = 0;
    for (const auto& segment : job.sources) {
        check_range(covered, segment.size_bytes, job.size_bytes);
        const auto& source = validate_slice(segment.source, segment.size_bytes, false);
        if (!routes_[source.unit][destination.unit]) throw Error(Status::PeerUnavailable);
        covered += segment.size_bytes;
    }
    if (covered != job.size_bytes) throw Error(Status::InvalidArgument);
}

void Fabric::submit_job(Job job) {
    if (pending_.contains(job.sequence) ||
        std::any_of(completions_.begin(), completions_.end(),
                    [&](const auto& c) { return c.sequence == job.sequence; })) {
        throw Error(Status::DuplicateSequence);
    }
    if (pending_.size() + completions_.size() >= queue_depth_) throw Error(Status::QueueFull);
    validate_job(job);
    job.pins.push_back(job.destination.buffer);
    for (const auto& source : job.sources) job.pins.push_back(source.source.buffer);
    std::sort(job.pins.begin(), job.pins.end());
    job.pins.erase(std::unique(job.pins.begin(), job.pins.end()), job.pins.end());
    auto [it, inserted] = pending_.emplace(job.sequence, std::move(job));
    (void)inserted;
    for (auto handle : it->second.pins) ++buffer(handle).in_flight;
    ++counters_.submitted;
}

void Fabric::submit(DmaCommand command) {
    submit_job({command.sequence, command.model_epoch, {{command.source, command.size_bytes}},
                command.destination, command.size_bytes, {}});
}

void Fabric::submit_gemv(GemvCommand command) {
    submit_matrix_gemv(command, false);
}

void Fabric::submit_q8_0_gemv(GemvCommand command) {
    submit_matrix_gemv(command, true);
}

void Fabric::submit_swiglu(SwigluCommand command) {
    check_epoch(command.model_epoch);
    unit(command.unit);
    if (command.elements == 0 || command.elements > (1024 * 1024) / 4) {
        throw Error(Status::InvalidArgument);
    }
    const uint64_t bytes = command.elements * 4;
    Job job{command.sequence, command.model_epoch, {{command.gate, bytes}, {command.up, bytes}},
            command.output, bytes, {}};
    job.swiglu_unit = command.unit;
    submit_job(std::move(job));
}

void Fabric::submit_matrix_gemv(GemvCommand command, bool q8_0) {
    check_epoch(command.model_epoch);
    unit(command.unit);
    auto it = tensors_.find(command.weights);
    if (it == tensors_.end()) throw Error(Status::UnknownTensor);
    const auto& tensor = it->second;
    if (tensor.format != (q8_0 ? "q8_0_row_major" : "int8_row_major") || command.row_count == 0 ||
        tensor.columns == 0 || tensor.columns > GEMV_MAX_COLUMNS) {
        throw Error(Status::InvalidArgument);
    }
    const uint64_t row_bytes = q8_0 ? tensor.columns / Q8_0_BLOCK_ELEMENTS * Q8_0_BLOCK_BYTES : tensor.columns;
    if (command.row_count > GEMV_MAX_WEIGHT_BYTES / row_bytes) throw Error(Status::InvalidArgument);
    check_range(command.first_row, command.row_count, tensor.rows);
    Job job{command.sequence, command.model_epoch, {}, command.output, command.row_count * 4, {}};
    job.gemv = Job::GemvWork{command.unit, command.row_count, tensor.columns,
                            command.row_count * row_bytes, tensor.columns * (q8_0 ? 4 : 1), q8_0};
    for (const auto& shard : resolve_tensor(command.model_epoch, command.weights,
            command.first_row * row_bytes, command.row_count * row_bytes)) {
        job.sources.push_back({shard.source, shard.size_bytes});
    }
    job.sources.push_back({command.input, job.gemv->input_bytes});
    submit_job(std::move(job));
}

void Fabric::submit_tensor_copy(uint64_t sequence, uint64_t epoch, TensorId tensor,
                                uint64_t tensor_offset, BufferSlice destination,
                                uint64_t size_bytes) {
    Job job{sequence, epoch, {}, destination, size_bytes, {}};
    for (const auto& shard : resolve_tensor(epoch, tensor, tensor_offset, size_bytes)) {
        job.sources.push_back({shard.source, shard.size_bytes});
    }
    submit_job(std::move(job));
}

void Fabric::finish(std::map<uint64_t, Job>::iterator it, Status status, uint64_t peer_bytes) {
    const auto& job = it->second;
    const bool success = status == Status::Ok;
    const bool compute = job.gemv.has_value() || job.swiglu_unit.has_value();
    const auto transferred = success && !compute ? job.size_bytes : 0;
    const auto output_bytes = success && compute ? job.size_bytes : 0;
    const auto macs = success && job.gemv ? job.gemv->rows * job.gemv->columns : 0;
    completions_.push_back({job.sequence, job.epoch, status, transferred, output_bytes, macs});
    for (auto handle : job.pins) --buffer(handle).in_flight;
    ++counters_.completed;
    if (status == Status::Cancelled) ++counters_.cancelled;
    else if (status != Status::Ok) ++counters_.failed;
    counters_.bytes_copied += transferred;
    if (success) counters_.peer_bytes += peer_bytes;
    if (success && job.gemv) {
        ++counters_.gemv_commands;
        if (job.gemv->q8_0) ++counters_.q8_0_commands;
        counters_.macs += macs;
        counters_.local_weight_bytes += job.gemv->weight_bytes; // Includes Q8_0 scales.
    }
    counters_.compute_output_bytes += output_bytes;
    if (success && job.swiglu_unit) {
        ++counters_.swiglu_commands;
        counters_.swiglu_elements += job.size_bytes / 4;
    }
    pending_.erase(it);
}

void Fabric::execute(uint64_t sequence) {
    auto it = pending_.find(sequence);
    if (it == pending_.end()) throw Error(Status::UnknownSequence);
    const auto& job = it->second;
    Status status = Status::Ok;
    uint64_t peer_bytes = 0;
    try {
        validate_job(job); // Devices and routes may have changed since submission.
        const auto& destination = buffer(job.destination.buffer);
        const uint64_t source_bytes = job.gemv ?
            job.gemv->weight_bytes + job.gemv->input_bytes :
            (job.swiglu_unit ? job.size_bytes * 2 : job.size_bytes);
        std::vector<std::byte> snapshot(static_cast<size_t>(source_bytes));
        size_t done = 0;
        for (const auto& segment : job.sources) {
            const auto& source = buffer(segment.source.buffer);
            const auto count = static_cast<size_t>(segment.size_bytes);
            read_memory(source.unit, source.offset + segment.source.offset,
                        std::span<std::byte>(snapshot).subspan(done, count));
            if (source.unit != destination.unit) peer_bytes += segment.size_bytes;
            done += count;
        }
        if (job.gemv && job.gemv->q8_0) {
            const size_t count = static_cast<size_t>(job.gemv->weight_bytes);
            const auto values = std::span<const std::byte>(snapshot);
            const auto input = unpack_f32_le(values.subspan(count));
            const auto computed = q8_0_gemv_hls_model(values.first(count), input,
                                                     static_cast<size_t>(job.gemv->rows));
            std::vector<float> output(computed.size());
            for (size_t i = 0; i < output.size(); ++i) {
                if (!std::isfinite(computed[i]) ||
                    std::abs(computed[i]) > double(std::numeric_limits<float>::max())) {
                    throw Error(Status::NumericError);
                }
                output[i] = static_cast<float>(computed[i]);
            }
            snapshot = pack_f32_le(output);
        } else if (job.swiglu_unit) {
            const auto bytes = std::span<const std::byte>(snapshot);
            const auto half = static_cast<size_t>(job.size_bytes);
            snapshot = pack_f32_le(swiglu(unpack_f32_le(bytes.first(half)),
                                          unpack_f32_le(bytes.subspan(half))));
        } else if (job.gemv) {
            std::vector<int8_t> operands(snapshot.size());
            for (size_t i = 0; i < snapshot.size(); ++i) {
                const int value = std::to_integer<int>(snapshot[i]);
                operands[i] = static_cast<int8_t>(value < 128 ? value : value - 256);
            }
            const size_t count = static_cast<size_t>(job.gemv->rows * job.gemv->columns);
            const auto values = std::span<const int8_t>(operands);
            snapshot = pack_int32_le(gemv_int8(values.first(count), values.subspan(count),
                                              static_cast<size_t>(job.gemv->rows)));
        }
        // Read every operand before writing, including overlapping input/output.
        write_memory(destination.unit, destination.offset + job.destination.offset, snapshot);
    } catch (const Error& error) {
        status = error.status();
    } catch (const std::bad_alloc&) {
        status = Status::ResourceLimit;
    } catch (const std::invalid_argument&) {
        status = Status::NumericError;
    }
    finish(it, status, peer_bytes);
}

void Fabric::cancel(uint64_t sequence) {
    auto it = pending_.find(sequence);
    if (it == pending_.end()) throw Error(Status::UnknownSequence);
    finish(it, Status::Cancelled, 0);
}

std::optional<Completion> Fabric::poll() {
    if (completions_.empty()) return std::nullopt;
    Completion completion = completions_.front();
    completions_.erase(completions_.begin());
    return completion;
}

} // namespace nt::ntrpu
