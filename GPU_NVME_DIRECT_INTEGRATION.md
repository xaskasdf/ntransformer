# gpu-nvme-direct Integration for ntransformer

## Resumen

Este documento describe los cambios necesarios para integrar **gpu-nvme-direct**
como backend de I/O en ntransformer, eliminando el CPU del data path de streaming
de layers.

**Proyecto gpu-nvme-direct**: `../gpu-nvme-direct`
**Estado**: Milestones 1-8 completos. GPU lee 669MB (1 layer Q6_K) @ 2.1 GB/s
desde NVMe via MMIO doorbells, sin intervención del CPU.

### Pipeline actual (CPU bottleneck)

```
NVMe → page cache → mmap → CPU memcpy → staging → H2D DMA → GPU compute
                            (worker thread)   (pinned)    (PCIe)
```

Resultado: ~0.02 tok/s en 70B Q6_K. El memcpy del worker thread es el cuello de botella.

### Pipeline objetivo (GPU-autónomo)

```
GPU doorbell write → NVMe DMA → host pinned buffer → GPU compute
  (MMIO a BAR0)      (autónomo)   (sin CPU memcpy)    (lee directo)
```

**Nota**: En Tier 1, los datos llegan a host pinned memory (no directamente a VRAM).
El GPU lee desde pinned memory, lo cual es eficiente vía PCIe UVA.

---

## Hardware requerido

| Componente | Requerimiento |
|---|---|
| GPU | NVIDIA con soporte cudaHostRegisterIoMemory (RTX 3090 probado) |
| NVMe | Cualquier NVMe en VFIO (WD SN530 PCIe 3.0 x4 probado) |
| CPU | AMD Zen 3 probado (Intel debería funcionar, P2P reads también) |
| OS | Linux (kernel 6.12+, necesita patch nvidia DKMS para follow_pfn) |
| IOMMU | OFF (`amd_iommu=off` en GRUB) |

---

## Arquitectura de la integración

### Lo que NO cambia

- `forward_streaming()` en `transformer.cpp` — el pipeline loop se mantiene idéntico
- `LayerWeightPtrs`, `get_weights()` — la GPU sigue leyendo pesos desde `gpu_buf_[slot]`
- CUDA events, streams, double-buffering — toda la sincronización se mantiene
- `GGUFLoader` — parsing de GGUF, metadata, vocab

### Lo que cambia

| Componente | Antes | Después |
|---|---|---|
| **Datos source** | mmap del GGUF file | NVMe DMA directo (via gpu-nvme-direct) |
| **CPU worker thread** | memcpy mmap→staging | **Eliminado** (GPU inicia reads) |
| **staging_buf_[]** | 2 pinned buffers para memcpy | **Reutilizados** como destino DMA del NVMe |
| **begin_h2d()** | cudaMemcpyAsync staging→GPU | **Se reemplaza** por GPU kernel que: (1) inicia NVMe reads a staging, (2) espera completions, (3) copia staging→GPU |
| **prefetch_staging()** | Queue work al worker thread | **Se reemplaza** por submit de NVMe reads desde GPU |
| **Dependencia nueva** | Solo CUDA | CUDA + libgpunvme + VFIO setup |

---

## Cambios detallados por archivo

### 1. `CMakeLists.txt` — Build system

```cmake
# Agregar al inicio:
option(USE_GPUNVME "Enable gpu-nvme-direct backend for NVMe streaming" OFF)

# Agregar las fuentes de gpunvme si está habilitado:
if(USE_GPUNVME)
    # Ruta al proyecto gpu-nvme-direct (ajustar según instalación)
    set(GPUNVME_DIR "${CMAKE_SOURCE_DIR}/../gpu-nvme-direct")

    # Include paths
    target_include_directories(ntransformer_lib PRIVATE
        ${GPUNVME_DIR}/include
    )

    # Compilar las fuentes de gpu-nvme-direct como parte del proyecto.
    # Archivos host (C):
    target_sources(ntransformer_lib PRIVATE
        ${GPUNVME_DIR}/src/host/controller.c
        ${GPUNVME_DIR}/src/host/admin.c
        ${GPUNVME_DIR}/src/host/io_queue.c
        ${GPUNVME_DIR}/src/host/bar_map.c
        ${GPUNVME_DIR}/src/host/dma_alloc.c
    )

    # Para las fuentes C (.c files) dentro del target CUDA, necesitan
    # compilarse con gcc-14, no nvcc:
    set_source_files_properties(
        ${GPUNVME_DIR}/src/host/controller.c
        ${GPUNVME_DIR}/src/host/admin.c
        ${GPUNVME_DIR}/src/host/io_queue.c
        ${GPUNVME_DIR}/src/host/bar_map.c
        ${GPUNVME_DIR}/src/host/dma_alloc.c
        PROPERTIES LANGUAGE C
    )

    target_compile_definitions(ntransformer_lib PRIVATE USE_GPUNVME=1)
endif()
```

### 2. `src/memory/streamer.h` — Nuevos miembros

```cpp
// Agregar al inicio del archivo:
#ifdef USE_GPUNVME
#include <gpunvme/controller.h>
#include <gpunvme/queue.h>
#include <gpunvme/dma.h>
#endif

// Agregar a la clase LayerStreamer (sección private):

#ifdef USE_GPUNVME
    // gpu-nvme-direct state
    gpunvme_ctrl_t nvme_ctrl_ = {};
    gpunvme_io_queue_t nvme_ioq_ = {};
    bool nvme_initialized_ = false;

    // GGUF file location on NVMe
    uint64_t gguf_start_lba_ = 0;       // LBA where the GGUF file starts
    uint64_t gguf_data_offset_ = 0;     // byte offset of tensor data within GGUF
    uint32_t nvme_block_size_ = 512;     // NVMe logical block size

    // Per-layer NVMe offsets (byte offset from start of file)
    struct NvmeLayerInfo {
        uint64_t file_offset;   // offset of first tensor in GGUF file
        uint64_t total_bytes;   // total bytes for all 7 tensors (with alignment)
    };
    std::vector<NvmeLayerInfo> nvme_layers_;

    // Init/shutdown helpers
    void init_nvme(const char* pci_bdf, uint64_t gguf_start_lba);
    void shutdown_nvme();

    // GPU-initiated NVMe read of a full layer to staging[slot]
    void nvme_read_layer_to_staging(int layer_idx, int slot);
#endif
```

### 3. `src/memory/streamer.cu` — Implementación NVMe backend

#### 3a. Nuevo método `init_nvme()`

```cpp
#ifdef USE_GPUNVME
#include <gpunvme/controller.h>
#include <gpunvme/queue.h>
#include <gpunvme/dma.h>
#include <gpunvme/error.h>
#include <sys/mman.h>
#include <fcntl.h>

// GPU kernel que lee una layer completa desde NVMe
// Incluir los headers de device:
#include "../../gpu-nvme-direct/src/device/sq_submit.cuh"
#include "../../gpu-nvme-direct/src/device/cq_poll.cuh"

__global__
void gpu_nvme_read_layer(gpu_nvme_queue* q,
                         uint64_t start_lba,
                         uint32_t total_blocks,
                         uint32_t blocks_per_cmd,
                         uint64_t* prp1_array,
                         uint64_t* prp2_array,
                         uint32_t n_commands,
                         uint32_t* out_status) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t submitted = 0, completed = 0;
    uint32_t pipe_depth = 32;  // commands in flight

    while (completed < n_commands) {
        while (submitted < n_commands && (submitted - completed) < pipe_depth) {
            uint64_t lba = start_lba + (uint64_t)submitted * blocks_per_cmd;
            uint32_t remaining = total_blocks - submitted * blocks_per_cmd;
            uint32_t nlb = (remaining < blocks_per_cmd) ? remaining : blocks_per_cmd;

            sq_submit_read(q, lba, nlb - 1,
                           prp1_array[submitted], prp2_array[submitted]);
            submitted++;
        }

        cq_poll_result cqr = cq_poll_completion(q, 3400000000ULL);
        if (cqr.timed_out || !cqr.success) {
            *out_status = 1;
            return;
        }
        completed++;
    }
    *out_status = 0;
}

void LayerStreamer::init_nvme(const char* pci_bdf, uint64_t gguf_start_lba) {
    // 1. Map NVMe BAR0
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", pci_bdf);

    int fd = open(path, O_RDWR | O_SYNC);
    NT_CHECK(fd >= 0, "Failed to open NVMe BAR0 resource");

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void* bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    close(fd);
    NT_CHECK(bar0 != MAP_FAILED, "Failed to mmap NVMe BAR0");

    // Register BAR0 for GPU MMIO access
    cudaError_t cerr = cudaHostRegister(
        (void*)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped);
    NT_CHECK(cerr == cudaSuccess, "cudaHostRegisterIoMemory failed");

    void* gpu_bar0;
    cudaHostGetDevicePointer(&gpu_bar0, (void*)bar0, 0);

    // 2. Initialize NVMe controller
    gpunvme_err_t err = gpunvme_ctrl_init(&nvme_ctrl_, bar0, bar_size);
    NT_CHECK(err == GPUNVME_OK, "NVMe controller init failed");
    nvme_ctrl_.bar0_gpu = gpu_bar0;

    // 3. Create I/O queue
    err = gpunvme_create_io_queue(&nvme_ctrl_, 1, 64, 4096,
                                   GPUNVME_TIER1, &nvme_ioq_);
    NT_CHECK(err == GPUNVME_OK, "NVMe I/O queue creation failed");

    // 4. Store NVMe parameters
    gguf_start_lba_ = gguf_start_lba;
    nvme_block_size_ = nvme_ctrl_.block_size;

    // 5. Compute per-layer NVMe offsets
    // Los tensors están en el GGUF file a offsets conocidos.
    // El NVMe lee bloques físicos. Necesitamos mapear:
    //   tensor_file_offset = gguf_data_offset + tensor.offset
    //   tensor_lba = gguf_start_lba + (tensor_file_offset / block_size)
    //
    // IMPORTANTE: Cada layer se lee como un bloque contiguo desde
    // el primer tensor hasta el último (incluyendo padding de alignment).
    // Esto funciona porque GGUF almacena tensors en orden secuencial.

    nvme_layers_.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); i++) {
        const LayerLayout& lay = layers_[i];
        const TensorSlot* first = &lay.attn_q;
        const TensorSlot* last = &lay.ffn_down;

        // El file_offset del primer tensor de esta layer
        // cpu_ptr apunta a mmap_base + data_offset + tensor.offset
        // Necesitamos el offset desde el inicio del archivo GGUF
        // Esto se calcula durante init() cuando se mapean los tensors.
        // Por ahora, almacenamos el offset relativo al data section.
        nvme_layers_[i].file_offset = first->gpu_offset;  // placeholder
        nvme_layers_[i].total_bytes = layer_transfer_size(i);
    }

    nvme_initialized_ = true;
    fprintf(stderr, "LayerStreamer: gpu-nvme-direct initialized (NVMe at %s)\n", pci_bdf);
    fprintf(stderr, "LayerStreamer: MDTS=%u KB, block_size=%u\n",
            nvme_ctrl_.max_transfer_bytes / 1024, nvme_block_size_);
}

void LayerStreamer::shutdown_nvme() {
    if (!nvme_initialized_) return;
    gpunvme_delete_io_queue(&nvme_ctrl_, &nvme_ioq_);
    gpunvme_ctrl_shutdown(&nvme_ctrl_);
    nvme_initialized_ = false;
}
#endif
```

#### 3b. Nuevo método `nvme_read_layer_to_staging()`

```cpp
#ifdef USE_GPUNVME
void LayerStreamer::nvme_read_layer_to_staging(int layer_idx, int slot) {
    // Lee una layer completa desde NVMe directamente a staging_buf_[slot]
    // usando el GPU como iniciador de I/O.

    const NvmeLayerInfo& nlay = nvme_layers_[layer_idx];
    const LayerLayout& lay = layers_[layer_idx];

    // Calcular LBA de inicio para esta layer
    // file_byte_offset = data_offset (dentro del GGUF) + offset del primer tensor
    uint64_t file_byte_offset = gguf_data_offset_ + /* offset del primer tensor */;
    // En la práctica, necesitamos que el GGUFLoader nos dé el offset absoluto
    // del primer tensor de cada layer. Ver sección de cambios a loader.h.

    uint64_t start_lba = gguf_start_lba_ + (file_byte_offset / nvme_block_size_);
    uint32_t total_bytes_aligned = (nlay.total_bytes + nvme_block_size_ - 1)
                                   & ~(nvme_block_size_ - 1);
    uint32_t total_blocks = total_bytes_aligned / nvme_block_size_;
    uint32_t blocks_per_cmd = nvme_ctrl_.max_transfer_bytes / nvme_block_size_;
    uint32_t n_commands = (total_blocks + blocks_per_cmd - 1) / blocks_per_cmd;

    // Build PRP lists (CPU side, one-time per layer read)
    // Usar pool allocation para evitar múltiples cudaHostRegister
    size_t prp_pool_bytes = n_commands * 4096;
    void* prp_pool = NULL;
    posix_memalign(&prp_pool, 4096, prp_pool_bytes);
    mlock(prp_pool, prp_pool_bytes);
    cudaHostRegister(prp_pool, prp_pool_bytes, cudaHostRegisterDefault);
    memset(prp_pool, 0, prp_pool_bytes);

    uint64_t *prp1_arr, *prp2_arr;
    cudaMallocHost(&prp1_arr, n_commands * sizeof(uint64_t));
    cudaMallocHost(&prp2_arr, n_commands * sizeof(uint64_t));

    // Resolver PRP entries para cada comando
    int pm_fd = open("/proc/self/pagemap", O_RDONLY);
    long sys_page_size = sysconf(_SC_PAGESIZE);
    uint8_t* dest = (uint8_t*)staging_buf_[slot];

    for (uint32_t i = 0; i < n_commands; i++) {
        uint32_t cmd_bytes = (i == n_commands - 1)
            ? total_bytes_aligned - i * nvme_ctrl_.max_transfer_bytes
            : nvme_ctrl_.max_transfer_bytes;
        uint32_t cmd_pages = (cmd_bytes + 4095) / 4096;
        uint64_t* list_virt = (uint64_t*)((uint8_t*)prp_pool + i * 4096);

        // Resolver phys del PRP list
        uint64_t list_vaddr = (uint64_t)(uintptr_t)list_virt;
        uint64_t pm_entry;
        pread(pm_fd, &pm_entry, 8, (list_vaddr / sys_page_size) * 8);
        uint64_t list_phys = (pm_entry & ((1ULL << 55) - 1)) * sys_page_size;

        uint8_t* chunk = dest + (uint64_t)i * nvme_ctrl_.max_transfer_bytes;
        uint64_t prp1 = 0;

        for (uint32_t p = 0; p < cmd_pages; p++) {
            uint64_t va = (uint64_t)(uintptr_t)(chunk + (uint64_t)p * 4096);
            uint64_t entry;
            pread(pm_fd, &entry, 8, (va / sys_page_size) * 8);
            uint64_t phys = (entry & ((1ULL << 55) - 1)) * sys_page_size
                          + (va % sys_page_size);
            if (p == 0) prp1 = phys;
            else list_virt[p - 1] = phys;
        }

        prp1_arr[i] = prp1;
        prp2_arr[i] = (cmd_pages <= 1) ? 0
                     : (cmd_pages == 2) ? list_virt[0]
                     : list_phys;
    }
    close(pm_fd);

    // Lanzar GPU kernel para hacer los NVMe reads
    uint32_t* d_status;
    cudaMallocHost(&d_status, sizeof(uint32_t));
    *d_status = 99;

    gpu_nvme_read_layer<<<1, 1>>>(
        nvme_ioq_.gpu_queue,
        start_lba, total_blocks, blocks_per_cmd,
        prp1_arr, prp2_arr, n_commands, d_status);
    cudaDeviceSynchronize();

    NT_CHECK(*d_status == 0, "gpu-nvme-direct: NVMe read failed for layer");

    // Cleanup
    cudaFreeHost(d_status);
    cudaFreeHost(prp1_arr);
    cudaFreeHost(prp2_arr);
    cudaHostUnregister(prp_pool);
    munlock(prp_pool, prp_pool_bytes);
    free(prp_pool);
}
#endif
```

#### 3c. Modificar `prefetch_staging()` y `begin_h2d()`

```cpp
void LayerStreamer::prefetch_staging(int layer_idx, int slot) {
    if (mmap_pinned_) return;

#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        // GPU-initiated NVMe read directamente a staging
        nvme_read_layer_to_staging(layer_idx, slot);
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            staging_ready_[slot] = true;
        }
        staging_ready_cv_.notify_all();
        return;
    }
#endif

    // Fallback: CPU worker thread memcpy
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        staging_ready_[slot] = false;
        worker_request_ = {layer_idx, slot};
        worker_has_work_ = true;
    }
    worker_cv_.notify_one();
}
```

### 4. `src/model/loader.h` — Exponer offsets para NVMe

Agregar métodos para obtener el offset absoluto de un tensor en el archivo GGUF:

```cpp
// Agregar a la clase GGUFLoader:

    // Get absolute byte offset of a tensor within the GGUF file
    // (for NVMe direct reads: LBA = file_start_lba + byte_offset / block_size)
    uint64_t tensor_file_offset(const std::string& name) const {
        auto it = tensor_map_.find(name);
        if (it == tensor_map_.end()) return 0;
        return data_offset_ + tensors_[it->second].offset;
    }

    // Get the absolute file offset of the data section start
    uint64_t file_data_offset() const { return data_offset_; }
```

### 5. `src/model/transformer.cpp` — Inicialización NVMe

Agregar inicialización del backend NVMe al startup del Transformer:

```cpp
// En Transformer::init() o Transformer::init_streaming(), después de streamer_.init():

#ifdef USE_GPUNVME
    // Leer parámetros NVMe de environment variables o config
    const char* nvme_bdf = getenv("GPUNVME_PCI_BDF");    // e.g. "0000:0b:00.0"
    const char* nvme_lba = getenv("GPUNVME_GGUF_LBA");   // LBA donde empieza el GGUF

    if (nvme_bdf && nvme_lba) {
        uint64_t start_lba = strtoull(nvme_lba, NULL, 0);
        streamer_.init_nvme(nvme_bdf, start_lba);
    }
#endif
```

### 6. `src/memory/streamer.cu` — Modificar `init()` para NVMe layer offsets

En `init()`, después de construir `layers_[]`, calcular los offsets NVMe:

```cpp
#ifdef USE_GPUNVME
    // Store GGUF data offset for NVMe LBA calculation
    gguf_data_offset_ = loader.file_data_offset();

    // Pre-compute per-layer file offsets
    // El primer tensor de cada layer tiene el menor offset en el archivo.
    // Los tensors de una layer son contiguos en GGUF (orden blk.0.*, blk.1.*, ...).
    nvme_layers_.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        std::string first_tensor = "blk." + std::to_string(i) + ".attn_q.weight";
        nvme_layers_[i].file_offset = loader.tensor_file_offset(first_tensor);
        nvme_layers_[i].total_bytes = layer_transfer_size(i);
    }
#endif
```

---

## Requisitos de setup NVMe

### Antes de ejecutar (después de cada reboot)

```bash
# 1. Bind NVMe a VFIO
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo modprobe vfio-pci
sudo bash ../gpu-nvme-direct/scripts/setup_vfio.sh 0000:0b:00.0

# 2. Activar NVMe
sudo sh -c 'echo on > /sys/bus/pci/devices/0000:0b:00.0/power/control'
sudo setpci -s 0000:0b:00.0 0x84.W=0x0008
sudo setpci -s 0000:0b:00.0 COMMAND=0x0006
```

### Copiar el GGUF al NVMe (una sola vez)

El archivo GGUF debe estar en el NVMe crudo (sin filesystem), empezando en un
LBA conocido. Esto se hace una vez:

```bash
# Escribir GGUF file al NVMe desde el LBA 0
# CUIDADO: esto destruye cualquier filesystem en el NVMe
sudo dd if=/path/to/model.gguf of=/dev/nvme0n1 bs=1M oflag=direct

# Verificar el tamaño (para calcular LBAs):
ls -la /path/to/model.gguf
# 57,398,476,800 bytes = 112,106,400 bloques de 512B
```

**NOTA**: El NVMe usado para gpu-nvme-direct NO debe tener filesystem ni estar
montado. Se accede como raw block device via VFIO.

### Ejecutar ntransformer con gpu-nvme-direct

```bash
# Build con NVMe backend
cmake .. -DUSE_GPUNVME=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build . -j$(nproc)

# Ejecutar (requiere root para VFIO y pagemap)
sudo GPUNVME_PCI_BDF=0000:0b:00.0 GPUNVME_GGUF_LBA=0 \
     ./ntransformer --model /dev/null --streaming
# (--model /dev/null porque los datos vienen del NVMe, no de mmap)
# (esto requiere refactoring del loader para modo NVMe, ver TODO abajo)
```

---

## Limitaciones conocidas

1. **Linux only**: gpu-nvme-direct requiere VFIO, /proc/self/pagemap, etc.
2. **Root requerido**: para VFIO bind y pagemap reads.
3. **NVMe dedicado**: el NVMe no puede tener filesystem mientras se usa con VFIO.
4. **AMD: solo writes**: GPU reads de NVMe BAR0 fallan en AMD (CmpltTO). Tier 1
   solo necesita writes (doorbells), los datos llegan vía NVMe DMA a host memory.
5. **Throughput**: 2.1-2.7 GB/s en SN530 PCIe 3.0 x4. Un NVMe Gen4 x4 daría ~4-6 GB/s.
6. **gcc-14 requerido**: gcc-15 es incompatible con CUDA 13.1.

---

## TODO (orden de implementación)

### Fase 1: Integración básica
- [ ] Agregar `USE_GPUNVME` option a CMakeLists.txt
- [ ] Agregar campos NVMe a `LayerStreamer`
- [ ] Implementar `init_nvme()` y `shutdown_nvme()`
- [ ] Implementar `nvme_read_layer_to_staging()` con GPU kernel
- [ ] Modificar `prefetch_staging()` para usar NVMe cuando disponible
- [ ] Agregar `tensor_file_offset()` a `GGUFLoader`
- [ ] Computar per-layer NVMe offsets en `init()`
- [ ] Agregar env vars para configuración NVMe en transformer init

### Fase 2: Optimización
- [ ] Pre-build PRP lists para todas las layers (una vez en init)
- [ ] Eliminar staging buffer: NVMe DMA → gpu_buf_ directamente (los staging
      buffers son host pinned, que es exactamente lo que necesita NVMe DMA)
- [ ] Async NVMe reads: lanzar GPU kernel en transfer stream, no en default
- [ ] Overlap NVMe read con compute (el pipeline actual ya soporta esto)

### Fase 3: Eliminar dependencia de mmap
- [ ] Modo NVMe puro: no necesitar abrir el GGUF file con mmap
- [ ] Parsear GGUF header leyendo primeros bloques del NVMe
- [ ] Solo necesitar BDF + start_lba + model config

### Fase 4: Port completo a Linux
- [ ] Verificar que todo el codebase de ntransformer compila en Linux con gcc-14
- [ ] Paths de modelo configurable (no hardcoded Windows paths)
- [ ] Testear con CUDA 13.1 (no solo 12.4)
- [ ] Resolver incompatibilidades C++20 / nvcc en Linux

---

## Números de rendimiento esperados

| Métrica | Actual (mmap+memcpy) | gpu-nvme-direct (SN530) | NVMe Gen4 (futuro) |
|---|---|---|---|
| I/O throughput | ~1.5-2 GB/s | 2.1-2.7 GB/s | 4-6 GB/s |
| 1 layer (669MB Q6_K) | ~400ms | ~250-315ms | ~130ms |
| 80 layers | ~32s | ~20-25s | ~10s |
| tok/s (70B Q6_K) | 0.03 | 0.04-0.05 | 0.08-0.10 |
| CPU utilization | 100% (1 core memcpy) | ~0% (GPU autónomo) | ~0% |

**La ganancia principal NO es solo throughput** — es eliminar el CPU del data path.
Esto libera cores del CPU para otros procesos y elimina el cuello de botella
del worker thread sincrónico.

---

## Diagrama de flujo: Layer streaming con gpu-nvme-direct

```
Token N forward pass:
                                                            time →
CPU:      [idle]──────────────────────────────────────────────────
GPU SM:   [compute L0][compute L1][compute L2]... [compute L79][norm+head]
GPU MMIO: [doorbell L1 ][doorbell L2 ][doorbell L3 ]...
NVMe DMA: [      DMA L1→staging1    ][DMA L2→staging0   ]...
PCIe H2D: [stg0→gpu0  ][stg1→gpu1  ][stg0→gpu0  ]...
           └──prefill──┘

Nota: En Fase 2+, staging se elimina — NVMe DMA va directo a gpu_buf_[slot]
      (que ya está en host pinned memory en Tier 1).
```
