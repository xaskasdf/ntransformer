/*
 * gpu-nvme-direct: Layer Loader API
 *
 * High-level API for GPU-initiated NVMe reads of large contiguous regions
 * (e.g., model layers). Encapsulates BAR0 mapping, controller init, PRP
 * building, and pipelined GPU kernel launch into a 3-call interface:
 *   init → load_layer (repeated) → destroy
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_LAYER_LOADER_H
#define GPUNVME_LAYER_LOADER_H

#include <stdint.h>
#include <stddef.h>
#include "error.h"
#include "controller.h"
#include "queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpunvme_layer_loader {
    gpunvme_ctrl_t ctrl;
    gpunvme_io_queue_t ioq;
    int bar0_fd;
    volatile void *bar0;
    void *bar0_gpu;
    size_t bar0_size;

    /* PRP pool: single page-aligned alloc, one 4KB page per command */
    void *prp_pool;
    size_t prp_pool_bytes;
    uint64_t *prp1_array;     /* pinned, max_commands entries */
    uint64_t *prp2_array;     /* pinned, max_commands entries */

    uint32_t max_commands;     /* ceil(max_layer_bytes / MDTS) */
    uint32_t blocks_per_cmd;   /* MDTS / block_size */
    uint32_t pipeline_depth;

    void *kernel_params;       /* layer_read_params* (pinned) */
    void *kernel_result;       /* layer_read_result* (pinned) */

    int pagemap_fd;            /* cached /proc/self/pagemap fd */
    int gpu_clock_khz;

    /* BAR1 direct VRAM mode (Tier 2) */
    int bar1_fd;               /* fd for GPU resource1_wc */
    uint64_t gpu_bar1_phys;    /* GPU BAR1 PCIe physical base (e.g., 0x7000000000) */
    uint64_t bar1_vram_offset; /* static BAR1 offset where VRAM starts (e.g., 0x20000000) */
    int bar1_enabled;          /* nonzero if BAR1 mode initialized */
} gpunvme_layer_loader_t;

/*
 * Initialize the layer loader.
 *
 * Opens BAR0, registers with CUDA, initializes the NVMe controller,
 * creates an I/O queue, and pre-allocates all PRP/kernel buffers.
 *
 * pci_bdf:         PCI BDF string (e.g., "0000:0b:00.0")
 * max_layer_bytes: Maximum layer size to support (determines PRP pool size)
 * pipeline_depth:  Number of NVMe commands in flight (recommended: 32)
 */
gpunvme_err_t gpunvme_layer_loader_init(gpunvme_layer_loader_t *loader,
                                         const char *pci_bdf,
                                         size_t max_layer_bytes,
                                         uint32_t pipeline_depth);

/*
 * Load a contiguous region from NVMe into host pinned memory.
 *
 * Rebuilds PRP entries for dest_pinned, launches the GPU kernel,
 * and waits for completion. Queue state rolls naturally between calls.
 *
 * start_lba:   First LBA to read
 * size_bytes:  Number of bytes to read (will be rounded up to block size)
 * dest_pinned: Destination buffer (must be page-aligned, cudaMallocHost or
 *              posix_memalign + mlock + cudaHostRegister)
 */
gpunvme_err_t gpunvme_load_layer(gpunvme_layer_loader_t *loader,
                                  uint64_t start_lba,
                                  size_t size_bytes,
                                  void *dest_pinned);

/* Query helpers */
uint32_t gpunvme_layer_loader_block_size(const gpunvme_layer_loader_t *loader);
uint32_t gpunvme_layer_loader_max_transfer(const gpunvme_layer_loader_t *loader);
uint64_t gpunvme_layer_loader_ns_blocks(const gpunvme_layer_loader_t *loader);

/*
 * Initialize BAR1 direct VRAM mode (Tier 2).
 *
 * Reads the GPU's BAR1 physical base from PCI config,
 * opens resource1_wc for VRAM pattern scanning.
 * Requires nvidia module loaded with NVreg_RegistryDwords="RMForceStaticBar1=1"
 *
 * gpu_bdf:            GPU PCI BDF string (e.g., "0000:0a:00.0")
 * static_bar1_offset: BAR1 offset where VRAM mapping starts (typically 0x20000000)
 */
gpunvme_err_t gpunvme_bar1_init(gpunvme_layer_loader_t *loader,
                                 const char *gpu_bdf,
                                 uint64_t static_bar1_offset);

/*
 * Discover BAR1 physical address for a VRAM allocation.
 *
 * Writes a unique pattern to vram_ptr via GPU kernel, then scans BAR1
 * resource1_wc to find it. Returns the BAR1 physical address of the
 * start of the allocation. Verifies physical contiguity.
 *
 * vram_ptr:      cudaMalloc'd pointer
 * vram_size:     total allocation size in bytes
 * bar1_phys_out: receives the BAR1 physical address
 */
gpunvme_err_t gpunvme_bar1_resolve(gpunvme_layer_loader_t *loader,
                                    void *vram_ptr,
                                    size_t vram_size,
                                    uint64_t *bar1_phys_out);

/*
 * Load a contiguous region from NVMe directly to VRAM via BAR1.
 *
 * Like gpunvme_load_layer, but PRP entries use BAR1 physical addresses
 * instead of host pinned addresses. NVMe DMA goes through BAR1 into VRAM.
 *
 * start_lba:      First LBA to read
 * size_bytes:     Number of bytes to read
 * dest_bar1_phys: BAR1 physical address of destination (from gpunvme_bar1_resolve)
 */
gpunvme_err_t gpunvme_load_layer_vram(gpunvme_layer_loader_t *loader,
                                       uint64_t start_lba,
                                       size_t size_bytes,
                                       uint64_t dest_bar1_phys);

/*
 * Destroy the layer loader.
 * Frees all resources, deletes the I/O queue, shuts down the controller.
 */
void gpunvme_layer_loader_destroy(gpunvme_layer_loader_t *loader);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_LAYER_LOADER_H */
