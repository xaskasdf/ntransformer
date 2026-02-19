#include "streamer.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace nt {

// ============================================================
// Helper: compute byte size from GGUF tensor info
// ============================================================
size_t LayerStreamer::tensor_bytes(const GGUFTensorInfo& info) {
    return info.nbytes;
}

// ============================================================
// Destructor
// ============================================================
LayerStreamer::~LayerStreamer() {
    shutdown();
}

// ============================================================
// Init: build per-layer layout, allocate GPU double-buffers
// ============================================================
void LayerStreamer::init(const GGUFLoader& loader, const ModelConfig& config) {
    int n_layers = config.n_layers;
    layers_.resize(n_layers);

    const char* tensor_suffixes[] = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };

    size_t max_layer_bytes = 0;

    for (int i = 0; i < n_layers; i++) {
        std::string pfx = "blk." + std::to_string(i) + ".";
        LayerLayout& lay = layers_[i];

        TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };

        size_t offset = 0;

        for (int t = 0; t < 7; t++) {
            std::string name = pfx + tensor_suffixes[t];
            const GGUFTensorInfo* info = loader.tensor_info(name);
            NT_CHECK(info != nullptr, ("Missing tensor: " + name).c_str());

            // Align offset to 256 bytes for efficient GPU access
            offset = (offset + 255) & ~(size_t)255;

            slots[t]->gpu_offset = offset;
            slots[t]->cpu_ptr    = loader.tensor_data(name);
            slots[t]->nbytes     = info->nbytes;
            slots[t]->dtype      = ggml_to_dtype(info->ggml_type);

            NT_CHECK(slots[t]->cpu_ptr != nullptr, ("Null data for tensor: " + name).c_str());

            offset += info->nbytes;
        }

        size_t layer_bytes = (offset + 255) & ~(size_t)255;
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
    }

    buf_size_ = max_layer_bytes;

    fprintf(stderr, "LayerStreamer: %d layers, buffer size: %.1f MB each (%.1f MB total for 2 buffers)\n",
        n_layers, buf_size_ / (1024.0 * 1024.0), 2.0 * buf_size_ / (1024.0 * 1024.0));

    // Allocate two GPU buffers
    for (int s = 0; s < 2; s++) {
        cudaError_t err = cudaMalloc(&gpu_buf_[s], buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate GPU layer buffer");
        current_layer_[s] = -1;
    }

    // Create CUDA events (disable timing for lower overhead)
    for (int s = 0; s < 2; s++) {
        cudaEvent_t ev;
        NT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        transfer_done_[s] = static_cast<void*>(ev);

        NT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        compute_done_[s] = static_cast<void*>(ev);
    }

    // Try to pin the mmap'd region for true async DMA
    const void* data_ptr = loader.mmap_data_ptr();
    size_t data_size = loader.tensor_data_size();

    cudaError_t pin_err = cudaHostRegister(
        const_cast<void*>(data_ptr), data_size, cudaHostRegisterReadOnly);

    if (pin_err == cudaSuccess) {
        mmap_pinned_ = true;
        fprintf(stderr, "LayerStreamer: mmap region pinned (%.1f GB) — true async DMA enabled\n",
            data_size / (1024.0 * 1024.0 * 1024.0));
    } else {
        mmap_pinned_ = false;
        fprintf(stderr, "LayerStreamer: cudaHostRegister failed (%s), using pinned staging buffer\n",
            cudaGetErrorString(pin_err));

        // Allocate a pinned staging buffer for the largest layer
        pinned_size_ = buf_size_;
        cudaError_t err = cudaMallocHost(&pinned_staging_, pinned_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate pinned staging buffer");
        fprintf(stderr, "LayerStreamer: pinned staging buffer: %.1f MB\n",
            pinned_size_ / (1024.0 * 1024.0));
    }

    // Record initial compute_done events so the first begin_transfer doesn't deadlock
    auto& dev = CUDADevice::instance();
    for (int s = 0; s < 2; s++) {
        dev.record_event(compute_done_[s], STREAM_COMPUTE);
    }

    fprintf(stderr, "LayerStreamer: initialized\n");
}

// ============================================================
// Shutdown: free everything
// ============================================================
void LayerStreamer::shutdown() {
    for (int s = 0; s < 2; s++) {
        if (gpu_buf_[s]) { cudaFree(gpu_buf_[s]); gpu_buf_[s] = nullptr; }
        if (transfer_done_[s]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(transfer_done_[s]));
            transfer_done_[s] = nullptr;
        }
        if (compute_done_[s]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(compute_done_[s]));
            compute_done_[s] = nullptr;
        }
    }

    if (mmap_pinned_) {
        mmap_pinned_ = false;
    }

    if (pinned_staging_) {
        cudaFreeHost(pinned_staging_);
        pinned_staging_ = nullptr;
    }

    layers_.clear();
    buf_size_ = 0;
}

// ============================================================
// Begin async transfer of layer_idx into GPU buffer slot
// ============================================================
void LayerStreamer::begin_transfer(int layer_idx, int slot) {
    NT_CHECK(layer_idx >= 0 && layer_idx < (int)layers_.size(), "Layer index out of range");
    NT_CHECK(slot == 0 || slot == 1, "Slot must be 0 or 1");

    current_layer_[slot] = layer_idx;

    auto& dev = CUDADevice::instance();
    StreamType xfer_stream = (slot == 0) ? STREAM_TRANSFER0 : STREAM_TRANSFER1;

    // Wait until compute on this slot is done (safe to overwrite)
    dev.wait_event(xfer_stream, compute_done_[slot]);

    const LayerLayout& lay = layers_[layer_idx];
    uint8_t* gpu_base = static_cast<uint8_t*>(gpu_buf_[slot]);

    const TensorSlot* slots[] = {
        &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
        &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
    };

    if (mmap_pinned_) {
        // Direct async copy from pinned mmap to GPU
        for (int t = 0; t < 7; t++) {
            dev.memcpy_h2d_async(
                gpu_base + slots[t]->gpu_offset,
                slots[t]->cpu_ptr,
                slots[t]->nbytes,
                xfer_stream
            );
        }
    } else {
        // CPU memcpy from mmap to pinned staging, then async to GPU
        uint8_t* staging = static_cast<uint8_t*>(pinned_staging_);

        for (int t = 0; t < 7; t++) {
            memcpy(staging + slots[t]->gpu_offset,
                   slots[t]->cpu_ptr,
                   slots[t]->nbytes);
        }

        // Single large async transfer from pinned staging to GPU
        size_t total = 0;
        for (int t = 0; t < 7; t++) {
            size_t end = slots[t]->gpu_offset + slots[t]->nbytes;
            if (end > total) total = end;
        }

        dev.memcpy_h2d_async(gpu_base, staging, total, xfer_stream);
    }

    // Record that transfer is done
    dev.record_event(transfer_done_[slot], xfer_stream);
}

// ============================================================
// Wait for transfer into slot to complete (called from compute stream)
// ============================================================
void LayerStreamer::wait_transfer(int slot) {
    auto& dev = CUDADevice::instance();
    dev.wait_event(STREAM_COMPUTE, transfer_done_[slot]);
}

// ============================================================
// Signal that compute on slot is done (called from compute stream)
// ============================================================
void LayerStreamer::signal_compute_done(int slot) {
    auto& dev = CUDADevice::instance();
    dev.record_event(compute_done_[slot], STREAM_COMPUTE);
}

// ============================================================
// Get weight pointers for a given slot
// Uses the layer that was most recently transferred to this slot.
// ============================================================
LayerWeightPtrs LayerStreamer::get_weights(int slot) const {
    NT_CHECK(slot == 0 || slot == 1, "Slot must be 0 or 1");
    NT_CHECK(current_layer_[slot] >= 0, "No layer transferred to this slot yet");

    const LayerLayout& lay = layers_[current_layer_[slot]];
    const uint8_t* gpu_base = static_cast<const uint8_t*>(gpu_buf_[slot]);

    LayerWeightPtrs wp;
    wp.attn_q       = gpu_base + lay.attn_q.gpu_offset;
    wp.attn_q_dtype = lay.attn_q.dtype;
    wp.attn_k       = gpu_base + lay.attn_k.gpu_offset;
    wp.attn_k_dtype = lay.attn_k.dtype;
    wp.attn_v       = gpu_base + lay.attn_v.gpu_offset;
    wp.attn_v_dtype = lay.attn_v.dtype;
    wp.attn_output  = gpu_base + lay.attn_output.gpu_offset;
    wp.attn_o_dtype = lay.attn_output.dtype;
    wp.ffn_gate       = gpu_base + lay.ffn_gate.gpu_offset;
    wp.ffn_gate_dtype = lay.ffn_gate.dtype;
    wp.ffn_up         = gpu_base + lay.ffn_up.gpu_offset;
    wp.ffn_up_dtype   = lay.ffn_up.dtype;
    wp.ffn_down       = gpu_base + lay.ffn_down.gpu_offset;
    wp.ffn_down_dtype = lay.ffn_down.dtype;

    return wp;
}

} // namespace nt
