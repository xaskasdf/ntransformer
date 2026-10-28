/*
 * gpu-nvme-direct: Layer Loader Implementation
 *
 * Encapsulates the proven large-read pattern (from test_large_read.cu) into
 * a reusable API. The GPU kernel issues pipelined NVMe READ commands at MDTS
 * granularity with pre-built PRP lists.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

#include <gpunvme/layer_loader.h>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/controller.h>
#include <gpunvme/queue.h>
#include <gpunvme/dma.h>
#include <gpunvme/error.h>

#include "device/queue_state.cuh"
#include "device/mmio_ops.cuh"
#include "device/sq_submit.cuh"
#include "device/cq_poll.cuh"

/* ---- Kernel parameter structs (pinned, passed to GPU) ---- */

struct layer_read_params {
    uint64_t start_lba;
    uint32_t total_blocks;
    uint32_t blocks_per_cmd;
    uint32_t n_commands;
    uint32_t pipeline_depth;
    uint64_t *prp1_array;
    uint64_t *prp2_array;
};

struct layer_read_result {
    uint32_t status;            /* 0 = success */
    uint32_t commands_completed;
    uint32_t error_code;        /* 0=ok, 2=timeout, 3=nvme_error */
    uint16_t cqe_status;
    uint64_t gpu_cycles;
};

/* ---- GPU kernel: pipelined sequential read ---- */

__global__
void gpu_layer_read(gpu_nvme_queue *q,
                    layer_read_params *params,
                    layer_read_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->commands_completed = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint32_t n_cmds = params->n_commands;
    uint32_t pipe_depth = params->pipeline_depth;
    uint32_t blocks_per = params->blocks_per_cmd;
    uint64_t base_lba = params->start_lba;

    uint64_t t_start = clock64();

    uint32_t submitted = 0;
    uint32_t completed = 0;

    while (completed < n_cmds) {
        /* Submit commands up to pipeline depth */
        while (submitted < n_cmds && (submitted - completed) < pipe_depth) {
            uint32_t cmd_idx = submitted;
            uint64_t lba = base_lba + (uint64_t)cmd_idx * blocks_per;

            /* Last command may be smaller */
            uint32_t remaining = params->total_blocks - cmd_idx * blocks_per;
            uint32_t nlb = (remaining < blocks_per) ? remaining : blocks_per;

            sq_submit_read(
                q, lba, nlb - 1,
                params->prp1_array[cmd_idx],
                params->prp2_array[cmd_idx]
            );
            submitted++;
        }

        /* Poll for ANY next completion (NVMe may complete out of order) */
        cq_poll_result cqr = cq_poll_completion(q, 3400000000ULL);

        if (cqr.timed_out) {
            result->status = 1;
            result->error_code = 2;
            result->commands_completed = completed;
            result->gpu_cycles = clock64() - t_start;
            return;
        }
        if (!cqr.success) {
            result->status = 1;
            result->error_code = 3;
            result->cqe_status = cqr.status;
            result->commands_completed = completed;
            result->gpu_cycles = clock64() - t_start;
            return;
        }

        completed++;
    }

    result->commands_completed = completed;
    result->gpu_cycles = clock64() - t_start;
}

/* ---- Helper: resolve physical address via pagemap ---- */

static uint64_t virt_to_phys_pagemap(int pm_fd, void *vaddr) {
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t va = (uint64_t)(uintptr_t)vaddr;
    uint64_t page_idx = va / page_size;
    uint64_t entry;
    if (pread(pm_fd, &entry, 8, page_idx * 8) != 8) return 0;
    if (!(entry & (1ULL << 63))) return 0;  /* page not present */
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return pfn * page_size + (va % page_size);
}

/* ---- API implementation ---- */

gpunvme_err_t gpunvme_layer_loader_init(gpunvme_layer_loader_t *loader,
                                         const char *pci_bdf,
                                         size_t max_layer_bytes,
                                         uint32_t pipeline_depth) {
    if (!loader || !pci_bdf || max_layer_bytes == 0 || pipeline_depth == 0)
        return GPUNVME_ERR_INVALID_PARAM;

    memset(loader, 0, sizeof(*loader));
    loader->bar0_fd = -1;
    loader->pagemap_fd = -1;
    loader->bar1_fd = -1;

    /* Map BAR0 */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", pci_bdf);

    loader->bar0_fd = open(path, O_RDWR | O_SYNC);
    if (loader->bar0_fd < 0) {
        fprintf(stderr, "layer_loader: failed to open %s\n", path);
        return GPUNVME_ERR_BAR_MAP;
    }

    loader->bar0_size = lseek(loader->bar0_fd, 0, SEEK_END);
    loader->bar0 = (volatile void *)mmap(NULL, loader->bar0_size,
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED, loader->bar0_fd, 0);
    if (loader->bar0 == MAP_FAILED) {
        loader->bar0 = NULL;
        fprintf(stderr, "layer_loader: mmap BAR0 failed\n");
        close(loader->bar0_fd);
        loader->bar0_fd = -1;
        return GPUNVME_ERR_BAR_MAP;
    }

    /* Register BAR0 with CUDA for GPU MMIO access */
    cudaError_t cerr = cudaHostRegister(
        (void *)loader->bar0, loader->bar0_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "layer_loader: cudaHostRegisterIoMemory failed: %s\n",
                cudaGetErrorString(cerr));
        munmap((void *)loader->bar0, loader->bar0_size);
        close(loader->bar0_fd);
        loader->bar0 = NULL;
        loader->bar0_fd = -1;
        return GPUNVME_ERR_CUDA;
    }

    cudaHostGetDevicePointer(&loader->bar0_gpu, (void *)loader->bar0, 0);

    /* Initialize NVMe controller */
    gpunvme_err_t err = gpunvme_ctrl_init(&loader->ctrl, loader->bar0, loader->bar0_size);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "layer_loader: controller init failed: %s\n", gpunvme_err_str(err));
        goto fail_bar0;
    }
    loader->ctrl.bar0_gpu = loader->bar0_gpu;

    loader->blocks_per_cmd = loader->ctrl.max_transfer_bytes / loader->ctrl.block_size;
    loader->max_commands = (uint32_t)((max_layer_bytes + loader->ctrl.max_transfer_bytes - 1)
                                      / loader->ctrl.max_transfer_bytes);

    /* Clamp pipeline depth */
    loader->pipeline_depth = pipeline_depth;
    if (loader->pipeline_depth > 60) loader->pipeline_depth = 60;

    /* Create I/O queue */
    err = gpunvme_create_io_queue(&loader->ctrl, 1, 64, 4096, GPUNVME_TIER1, &loader->ioq);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "layer_loader: I/O queue creation failed: %s\n", gpunvme_err_str(err));
        gpunvme_ctrl_shutdown(&loader->ctrl);
        goto fail_bar0;
    }

    /* Allocate PRP pool: one 4KB page per command */
    loader->prp_pool_bytes = (size_t)loader->max_commands * 4096;
    if (posix_memalign(&loader->prp_pool, 4096, loader->prp_pool_bytes) != 0) {
        fprintf(stderr, "layer_loader: PRP pool alloc failed (%zu bytes)\n",
                loader->prp_pool_bytes);
        err = GPUNVME_ERR_NOMEM;
        goto fail_ioq;
    }
    mlock(loader->prp_pool, loader->prp_pool_bytes);
    cudaHostRegister(loader->prp_pool, loader->prp_pool_bytes, cudaHostRegisterDefault);
    memset(loader->prp_pool, 0, loader->prp_pool_bytes);

    /* Allocate prp1/prp2 arrays */
    if (cudaMallocHost(&loader->prp1_array, loader->max_commands * sizeof(uint64_t)) != cudaSuccess ||
        cudaMallocHost(&loader->prp2_array, loader->max_commands * sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "layer_loader: PRP array alloc failed\n");
        err = GPUNVME_ERR_NOMEM;
        goto fail_prp;
    }

    /* Allocate kernel params and result */
    layer_read_params *params;
    layer_read_result *result;
    if (cudaMallocHost(&params, sizeof(layer_read_params)) != cudaSuccess ||
        cudaMallocHost(&result, sizeof(layer_read_result)) != cudaSuccess) {
        fprintf(stderr, "layer_loader: kernel param alloc failed\n");
        err = GPUNVME_ERR_NOMEM;
        goto fail_prp;
    }
    loader->kernel_params = params;
    loader->kernel_result = result;

    /* Open pagemap (cached for all load_layer calls) */
    loader->pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    if (loader->pagemap_fd < 0) {
        fprintf(stderr, "layer_loader: failed to open /proc/self/pagemap\n");
        err = GPUNVME_ERR_IO;
        goto fail_prp;
    }

    /* Get GPU clock for throughput reporting */
    cudaDeviceGetAttribute(&loader->gpu_clock_khz, cudaDevAttrClockRate, 0);

    fprintf(stderr, "layer_loader: init OK — MDTS=%uK, block_size=%u, max_commands=%u, pipeline=%u\n",
            loader->ctrl.max_transfer_bytes / 1024, loader->ctrl.block_size,
            loader->max_commands, loader->pipeline_depth);

    return GPUNVME_OK;

fail_prp:
    if (loader->kernel_result) cudaFreeHost(loader->kernel_result);
    if (loader->kernel_params) cudaFreeHost(loader->kernel_params);
    if (loader->prp2_array) cudaFreeHost(loader->prp2_array);
    if (loader->prp1_array) cudaFreeHost(loader->prp1_array);
    if (loader->prp_pool) {
        cudaHostUnregister(loader->prp_pool);
        munlock(loader->prp_pool, loader->prp_pool_bytes);
        free(loader->prp_pool);
    }
fail_ioq:
    gpunvme_delete_io_queue(&loader->ctrl, &loader->ioq);
    gpunvme_ctrl_shutdown(&loader->ctrl);
fail_bar0:
    cudaHostUnregister((void *)loader->bar0);
    munmap((void *)loader->bar0, loader->bar0_size);
    close(loader->bar0_fd);
    loader->bar0 = NULL;
    loader->bar0_fd = -1;
    return err;
}

gpunvme_err_t gpunvme_load_layer(gpunvme_layer_loader_t *loader,
                                  uint64_t start_lba,
                                  size_t size_bytes,
                                  void *dest_pinned) {
    if (!loader || !dest_pinned || size_bytes == 0)
        return GPUNVME_ERR_INVALID_PARAM;

    uint32_t block_size = loader->ctrl.block_size;
    uint32_t total_blocks = (uint32_t)((size_bytes + block_size - 1) / block_size);
    uint32_t n_commands = (total_blocks + loader->blocks_per_cmd - 1) / loader->blocks_per_cmd;

    if (n_commands > loader->max_commands) {
        fprintf(stderr, "layer_loader: size %zu requires %u commands, max is %u\n",
                size_bytes, n_commands, loader->max_commands);
        return GPUNVME_ERR_INVALID_PARAM;
    }

    /* Rebuild PRP entries for this dest_pinned buffer */
    int pm_fd = loader->pagemap_fd;

    for (uint32_t i = 0; i < n_commands; i++) {
        uint32_t remaining_blocks = total_blocks - i * loader->blocks_per_cmd;
        uint32_t cmd_blocks = (remaining_blocks < loader->blocks_per_cmd)
                              ? remaining_blocks : loader->blocks_per_cmd;
        uint32_t cmd_bytes = cmd_blocks * block_size;
        uint32_t cmd_pages = (cmd_bytes + loader->ctrl.page_size - 1) / loader->ctrl.page_size;

        /* This command's PRP list page */
        uint64_t *list_virt = (uint64_t *)((uint8_t *)loader->prp_pool + (size_t)i * 4096);

        /* Resolve PRP list physical address */
        uint64_t list_phys = virt_to_phys_pagemap(pm_fd, list_virt);

        /* Build PRP entries for each data page */
        uint8_t *chunk = (uint8_t *)dest_pinned + (uint64_t)i * loader->ctrl.max_transfer_bytes;
        uint64_t prp1 = 0, prp2 = 0;

        for (uint32_t p = 0; p < cmd_pages; p++) {
            uint64_t phys = virt_to_phys_pagemap(
                pm_fd, chunk + (uint64_t)p * loader->ctrl.page_size);
            if (phys == 0) {
                fprintf(stderr, "layer_loader: failed to resolve phys addr for cmd %u page %u\n", i, p);
                return GPUNVME_ERR_DMA;
            }

            if (p == 0) {
                prp1 = phys;
            } else {
                list_virt[p - 1] = phys;
            }
        }

        if (cmd_pages <= 1) {
            prp2 = 0;
        } else if (cmd_pages == 2) {
            prp2 = list_virt[0];  /* Direct second page phys */
        } else {
            prp2 = list_phys;    /* PRP list phys addr */
        }

        loader->prp1_array[i] = prp1;
        loader->prp2_array[i] = prp2;
    }

    /* Fill kernel params */
    layer_read_params *params = (layer_read_params *)loader->kernel_params;
    layer_read_result *result = (layer_read_result *)loader->kernel_result;

    params->start_lba = start_lba;
    params->total_blocks = total_blocks;
    params->blocks_per_cmd = loader->blocks_per_cmd;
    params->n_commands = n_commands;
    params->pipeline_depth = loader->pipeline_depth;
    params->prp1_array = loader->prp1_array;
    params->prp2_array = loader->prp2_array;
    memset(result, 0, sizeof(layer_read_result));

    /* Launch GPU kernel */
    gpu_layer_read<<<1, 1>>>(loader->ioq.gpu_queue, params, result);
    cudaError_t cerr = cudaDeviceSynchronize();

    if (cerr != cudaSuccess) {
        fprintf(stderr, "layer_loader: GPU kernel error: %s\n", cudaGetErrorString(cerr));
        return GPUNVME_ERR_CUDA;
    }

    if (result->status != 0) {
        fprintf(stderr, "layer_loader: read failed at command %u/%u: error=%u, cqe_status=0x%04x\n",
                result->commands_completed, n_commands,
                result->error_code, result->cqe_status);
        if (result->error_code == 2)
            return GPUNVME_ERR_TIMEOUT;
        return GPUNVME_ERR_NVME_STATUS;
    }

    /* Report throughput */
    if (loader->gpu_clock_khz > 0 && result->gpu_cycles > 0) {
        double seconds = (double)result->gpu_cycles / ((double)loader->gpu_clock_khz * 1000.0);
        double mb_per_sec = (double)size_bytes / (1024.0 * 1024.0) / seconds;
        fprintf(stderr, "layer_loader: read %zu bytes (%u cmds) in %.1f ms — %.1f MB/s\n",
                size_bytes, n_commands, seconds * 1000.0, mb_per_sec);
    }

    return GPUNVME_OK;
}

/* ---- BAR1 direct VRAM support (Tier 2) ---- */

/* GPU kernel: write a unique pattern to VRAM for BAR1 offset discovery */
__global__
void bar1_fill_pattern(uint64_t *addr, uint64_t pattern, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count) addr[i] = pattern;
}

/* Parse GPU BAR1 physical base from /sys/bus/pci/devices/.../resource */
static uint64_t parse_bar1_phys(const char *gpu_bdf) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", gpu_bdf);

    FILE *f = fopen(path, "r");
    if (!f) return 0;

    /* resource file: one line per BAR, format "start end flags"
     * Line 0 = BAR0, Line 1 = BAR1 (64-bit spans lines 1-2) */
    char line[256];
    int lineno = 0;
    uint64_t bar1_start = 0;
    while (fgets(line, sizeof(line), f)) {
        if (lineno == 1) {
            /* BAR1 line: "0x7000000000 0x77ffffffff 0x000000000014220c" */
            unsigned long long start, end, flags;
            if (sscanf(line, "0x%llx 0x%llx 0x%llx", &start, &end, &flags) >= 2) {
                bar1_start = (uint64_t)start;
            }
            break;
        }
        lineno++;
    }
    fclose(f);
    return bar1_start;
}

gpunvme_err_t gpunvme_bar1_init(gpunvme_layer_loader_t *loader,
                                 const char *gpu_bdf,
                                 uint64_t static_bar1_offset) {
    if (!loader || !gpu_bdf)
        return GPUNVME_ERR_INVALID_PARAM;

    /* Read GPU BAR1 physical base from PCI config */
    uint64_t bar1_phys = parse_bar1_phys(gpu_bdf);
    if (bar1_phys == 0) {
        fprintf(stderr, "bar1_init: failed to read BAR1 from %s\n", gpu_bdf);
        return GPUNVME_ERR_BAR_MAP;
    }

    /* Open GPU resource1_wc for pattern scanning */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource1_wc", gpu_bdf);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        /* Fallback to non-WC */
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource1", gpu_bdf);
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "bar1_init: failed to open %s\n", path);
            return GPUNVME_ERR_BAR_MAP;
        }
    }

    loader->bar1_fd = fd;
    loader->gpu_bar1_phys = bar1_phys;
    loader->bar1_vram_offset = static_bar1_offset;
    loader->bar1_enabled = 1;

    fprintf(stderr, "bar1_init: GPU BAR1 phys=0x%llx, VRAM starts at BAR1+0x%llx\n",
            (unsigned long long)bar1_phys,
            (unsigned long long)static_bar1_offset);

    return GPUNVME_OK;
}

gpunvme_err_t gpunvme_bar1_resolve(gpunvme_layer_loader_t *loader,
                                    void *vram_ptr,
                                    size_t vram_size,
                                    uint64_t *bar1_phys_out) {
    if (!loader || !loader->bar1_enabled || !vram_ptr || !bar1_phys_out)
        return GPUNVME_ERR_INVALID_PARAM;

    /* Step 1: Write unique pattern to first 64 bytes of VRAM buffer */
    uint64_t pattern = 0xBA01CAFE00000000ULL | (uint64_t)((uintptr_t)vram_ptr & 0xFFFFFFFF);
    int nwords = 8;  /* 64 bytes = 8 x uint64_t */
    bar1_fill_pattern<<<1, nwords>>>((uint64_t *)vram_ptr, pattern, nwords);
    cudaDeviceSynchronize();

    /* Step 2: Fast stride scan — probe first 8 bytes of each 2MB GPU page.
     * cudaMalloc returns 2MB-aligned addresses on Ampere, so the pattern
     * lands at the start of a 2MB BAR1 page. This reduces the scan from
     * reading every byte (minutes) to 12288 probes (~12ms for 24GB VRAM). */
    uint64_t scan_start = loader->bar1_vram_offset;
    uint64_t scan_end = scan_start + 24ULL * 1024 * 1024 * 1024;  /* 24GB VRAM */
    size_t stride = 2 * 1024 * 1024;  /* 2MB GPU page size */
    size_t page_size = sysconf(_SC_PAGESIZE);  /* 4KB for mmap alignment */
    uint64_t found_offset = 0;
    int found = 0;

    /* Mmap a sliding window (256MB) for efficient scanning */
    size_t window_size = 256ULL * 1024 * 1024;

    for (uint64_t win_start = scan_start; win_start < scan_end && !found; win_start += window_size) {
        size_t map_size = window_size;
        if (win_start + map_size > scan_end) map_size = scan_end - win_start;

        void *map = mmap(NULL, map_size, PROT_READ, MAP_SHARED,
                         loader->bar1_fd, win_start);
        if (map == MAP_FAILED) continue;

        /* Probe first 8 bytes at each 2MB stride within this window */
        for (size_t off = 0; off < map_size && !found; off += stride) {
            volatile uint64_t *p = (volatile uint64_t *)((uint8_t *)map + off);
            if (*p == pattern) {
                /* Verify consecutive words */
                int consecutive = 0;
                for (int j = 0; j < nwords; j++) {
                    if (p[j] == pattern) consecutive++;
                }
                if (consecutive >= 4) {
                    found_offset = win_start + off;
                    found = 1;
                }
            }
        }
        munmap(map, map_size);
    }

    if (!found) {
        fprintf(stderr, "bar1_resolve: pattern not found in BAR1 — static BAR1 not enabled?\n");
        return GPUNVME_ERR_DMA;
    }

    uint64_t bar1_phys = loader->gpu_bar1_phys + found_offset;
    fprintf(stderr, "bar1_resolve: VRAM ptr=%p → BAR1 offset=0x%llx → phys=0x%llx\n",
            vram_ptr,
            (unsigned long long)found_offset,
            (unsigned long long)bar1_phys);

    /* Step 3: Verify contiguity at midpoint of allocation */
    if (vram_size >= 2 * stride) {
        size_t check_offset = (vram_size / 2) & ~(size_t)(stride - 1);  /* 2MB-aligned */
        uint64_t pattern2 = pattern ^ 0xFFFFFFFFULL;
        uint64_t *check_ptr = (uint64_t *)((uint8_t *)vram_ptr + check_offset);
        bar1_fill_pattern<<<1, nwords>>>(check_ptr, pattern2, nwords);
        cudaDeviceSynchronize();

        uint64_t expected_bar1_off = found_offset + check_offset;
        void *map = mmap(NULL, page_size, PROT_READ, MAP_SHARED,
                         loader->bar1_fd, expected_bar1_off);
        if (map != MAP_FAILED) {
            volatile uint64_t *p = (volatile uint64_t *)map;
            if (p[0] == pattern2) {
                fprintf(stderr, "bar1_resolve: contiguity verified at +0x%zx\n", check_offset);
            } else {
                fprintf(stderr, "bar1_resolve: WARNING — VRAM not contiguous at +0x%zx "
                        "(expected 0x%llx, got 0x%llx)\n",
                        check_offset,
                        (unsigned long long)pattern2,
                        (unsigned long long)p[0]);
                munmap(map, page_size);
                return GPUNVME_ERR_DMA;
            }
            munmap(map, page_size);
        }
    }

    *bar1_phys_out = bar1_phys;
    return GPUNVME_OK;
}

gpunvme_err_t gpunvme_load_layer_vram(gpunvme_layer_loader_t *loader,
                                       uint64_t start_lba,
                                       size_t size_bytes,
                                       uint64_t dest_bar1_phys) {
    if (!loader || !loader->bar1_enabled || size_bytes == 0)
        return GPUNVME_ERR_INVALID_PARAM;

    uint32_t block_size = loader->ctrl.block_size;
    uint32_t page_size = loader->ctrl.page_size;
    uint32_t total_blocks = (uint32_t)((size_bytes + block_size - 1) / block_size);
    uint32_t n_commands = (total_blocks + loader->blocks_per_cmd - 1) / loader->blocks_per_cmd;

    if (n_commands > loader->max_commands) {
        fprintf(stderr, "load_layer_vram: size %zu requires %u commands, max is %u\n",
                size_bytes, n_commands, loader->max_commands);
        return GPUNVME_ERR_INVALID_PARAM;
    }

    /* Build PRP entries using BAR1 physical addresses.
     * Unlike pagemap, we compute addresses directly:
     * dest_bar1_phys + offset = physical address visible on PCIe bus */
    int pm_fd = loader->pagemap_fd;

    for (uint32_t i = 0; i < n_commands; i++) {
        uint32_t remaining_blocks = total_blocks - i * loader->blocks_per_cmd;
        uint32_t cmd_blocks = (remaining_blocks < loader->blocks_per_cmd)
                              ? remaining_blocks : loader->blocks_per_cmd;
        uint32_t cmd_bytes = cmd_blocks * block_size;
        uint32_t cmd_pages = (cmd_bytes + page_size - 1) / page_size;

        /* This command's PRP list page (still in host pinned memory) */
        uint64_t *list_virt = (uint64_t *)((uint8_t *)loader->prp_pool + (size_t)i * 4096);
        uint64_t list_phys = virt_to_phys_pagemap(pm_fd, list_virt);

        /* Compute BAR1 physical address for each data page */
        uint64_t chunk_phys = dest_bar1_phys + (uint64_t)i * loader->ctrl.max_transfer_bytes;
        uint64_t prp1 = 0, prp2 = 0;

        for (uint32_t p = 0; p < cmd_pages; p++) {
            uint64_t phys = chunk_phys + (uint64_t)p * page_size;

            if (p == 0) {
                prp1 = phys;
            } else {
                list_virt[p - 1] = phys;
            }
        }

        if (cmd_pages <= 1) {
            prp2 = 0;
        } else if (cmd_pages == 2) {
            prp2 = list_virt[0];
        } else {
            prp2 = list_phys;
        }

        loader->prp1_array[i] = prp1;
        loader->prp2_array[i] = prp2;
    }

    /* Fill kernel params and launch (same GPU kernel as Tier 1) */
    layer_read_params *params = (layer_read_params *)loader->kernel_params;
    layer_read_result *result = (layer_read_result *)loader->kernel_result;

    params->start_lba = start_lba;
    params->total_blocks = total_blocks;
    params->blocks_per_cmd = loader->blocks_per_cmd;
    params->n_commands = n_commands;
    params->pipeline_depth = loader->pipeline_depth;
    params->prp1_array = loader->prp1_array;
    params->prp2_array = loader->prp2_array;
    memset(result, 0, sizeof(layer_read_result));

    gpu_layer_read<<<1, 1>>>(loader->ioq.gpu_queue, params, result);
    cudaError_t cerr = cudaDeviceSynchronize();

    if (cerr != cudaSuccess) {
        fprintf(stderr, "load_layer_vram: GPU kernel error: %s\n", cudaGetErrorString(cerr));
        return GPUNVME_ERR_CUDA;
    }

    if (result->status != 0) {
        fprintf(stderr, "load_layer_vram: read failed at command %u/%u: error=%u, cqe_status=0x%04x\n",
                result->commands_completed, n_commands,
                result->error_code, result->cqe_status);
        if (result->error_code == 2)
            return GPUNVME_ERR_TIMEOUT;
        return GPUNVME_ERR_NVME_STATUS;
    }

    if (loader->gpu_clock_khz > 0 && result->gpu_cycles > 0) {
        double seconds = (double)result->gpu_cycles / ((double)loader->gpu_clock_khz * 1000.0);
        double mb_per_sec = (double)size_bytes / (1024.0 * 1024.0) / seconds;
        fprintf(stderr, "load_layer_vram: %zu bytes (%u cmds) in %.1f ms — %.1f MB/s [BAR1→VRAM]\n",
                size_bytes, n_commands, seconds * 1000.0, mb_per_sec);
    }

    return GPUNVME_OK;
}

uint32_t gpunvme_layer_loader_block_size(const gpunvme_layer_loader_t *loader) {
    return loader ? loader->ctrl.block_size : 0;
}

uint32_t gpunvme_layer_loader_max_transfer(const gpunvme_layer_loader_t *loader) {
    return loader ? loader->ctrl.max_transfer_bytes : 0;
}

uint64_t gpunvme_layer_loader_ns_blocks(const gpunvme_layer_loader_t *loader) {
    return loader ? loader->ctrl.ns_size_blocks : 0;
}

void gpunvme_layer_loader_destroy(gpunvme_layer_loader_t *loader) {
    if (!loader) return;

    if (loader->pagemap_fd >= 0) {
        close(loader->pagemap_fd);
        loader->pagemap_fd = -1;
    }

    if (loader->kernel_result) cudaFreeHost(loader->kernel_result);
    if (loader->kernel_params) cudaFreeHost(loader->kernel_params);
    if (loader->prp2_array) cudaFreeHost(loader->prp2_array);
    if (loader->prp1_array) cudaFreeHost(loader->prp1_array);

    if (loader->prp_pool) {
        cudaHostUnregister(loader->prp_pool);
        munlock(loader->prp_pool, loader->prp_pool_bytes);
        free(loader->prp_pool);
        loader->prp_pool = NULL;
    }

    gpunvme_delete_io_queue(&loader->ctrl, &loader->ioq);
    gpunvme_ctrl_shutdown(&loader->ctrl);

    if (loader->bar1_fd >= 0) {
        close(loader->bar1_fd);
        loader->bar1_fd = -1;
    }
    loader->bar1_enabled = 0;

    if (loader->bar0) {
        cudaHostUnregister((void *)loader->bar0);
        munmap((void *)loader->bar0, loader->bar0_size);
        loader->bar0 = NULL;
    }
    if (loader->bar0_fd >= 0) {
        close(loader->bar0_fd);
        loader->bar0_fd = -1;
    }

    memset(loader, 0, sizeof(*loader));
    loader->bar0_fd = -1;
    loader->pagemap_fd = -1;
    loader->bar1_fd = -1;
}
