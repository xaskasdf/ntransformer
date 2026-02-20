# gpu-nvme-direct Integration for ntransformer

## Resumen

Integrar **gpu-nvme-direct** como backend de I/O en ntransformer, eliminando
el CPU del data path de streaming de layers.

**Proyecto gpu-nvme-direct**: `../gpu-nvme-direct`
**Estado**: Layer Loader API lista (`gpunvme_layer_loader_init` / `gpunvme_load_layer` / `gpunvme_layer_loader_destroy`).
GPU lee 669MB (1 layer Q6_K) @ 2.1 GB/s desde NVMe via MMIO doorbells, sin intervención del CPU.

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

## Layer Loader API (nuevo)

El Layer Loader API encapsula todo el boilerplate de BAR0 mapping, controller init,
PRP building y kernel launch en 3 llamadas:

```c
#include <gpunvme/layer_loader.h>

gpunvme_layer_loader_t loader;

// Init: abre BAR0, registra CUDA, init controller, crea I/O queue, pre-alloc PRP pool
gpunvme_layer_loader_init(&loader, "0000:0b:00.0", max_layer_bytes, /*pipeline_depth=*/32);

// Load: rebuild PRPs para dest, lanza GPU kernel, sincroniza
gpunvme_load_layer(&loader, start_lba, size_bytes, dest_pinned);

// Destroy: cleanup completo
gpunvme_layer_loader_destroy(&loader);
```

**Helpers**:
- `gpunvme_layer_loader_block_size(&loader)` — block size del NVMe (512)
- `gpunvme_layer_loader_max_transfer(&loader)` — MDTS en bytes (512K)
- `gpunvme_layer_loader_ns_blocks(&loader)` — capacidad total del namespace

**Queue state rueda naturalmente** entre llamadas a `gpunvme_load_layer()` — no hay
reset, los CIDs, sq_tail, cq_head, phase bit continúan desde donde quedaron.

**Código fuente**:
- Header: `gpu-nvme-direct/include/gpunvme/layer_loader.h`
- Impl: `gpu-nvme-direct/src/host/layer_loader.cu`
- Test: `gpu-nvme-direct/tests/test_layer_loader.cu`

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
| **Datos source** | mmap del GGUF file | NVMe DMA directo via Layer Loader |
| **CPU worker thread** | memcpy mmap→staging | **Eliminado** (GPU inicia reads) |
| **staging_buf_[]** | 2 pinned buffers para memcpy | **Reutilizados** como destino DMA del NVMe |
| **prefetch_staging()** | Queue work al worker thread | `gpunvme_load_layer()` directamente a staging |
| **Dependencia nueva** | Solo CUDA | CUDA + libgpunvme_layer_loader + VFIO setup |

---

## Cambios detallados por archivo

### 1. `CMakeLists.txt` — Build system

```cmake
# Agregar al inicio:
option(USE_GPUNVME "Enable gpu-nvme-direct backend for NVMe streaming" OFF)

if(USE_GPUNVME)
    set(GPUNVME_DIR "${CMAKE_SOURCE_DIR}/../gpu-nvme-direct")

    # Incluir la librería pre-compilada (build-hw debe existir)
    # Opción A: Link contra las librerías estáticas pre-built
    add_library(gpunvme_layer_loader STATIC IMPORTED)
    set_target_properties(gpunvme_layer_loader PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_layer_loader.a)

    add_library(gpunvme_host STATIC IMPORTED)
    set_target_properties(gpunvme_host PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_host.a)

    add_library(gpunvme_device STATIC IMPORTED)
    set_target_properties(gpunvme_device PROPERTIES
        IMPORTED_LOCATION ${GPUNVME_DIR}/build-hw/libgpunvme_device.a)

    target_include_directories(ntransformer_lib PRIVATE ${GPUNVME_DIR}/include)
    target_link_libraries(ntransformer_lib PRIVATE
        gpunvme_layer_loader gpunvme_host gpunvme_device)
    target_compile_definitions(ntransformer_lib PRIVATE USE_GPUNVME=1)
endif()
```

### 2. `src/memory/streamer.h` — Nuevos miembros

```cpp
// Agregar al inicio del archivo:
#ifdef USE_GPUNVME
#include <gpunvme/layer_loader.h>
#endif

// Agregar a la clase LayerStreamer (sección private):
#ifdef USE_GPUNVME
    gpunvme_layer_loader_t nvme_loader_ = {};
    bool nvme_initialized_ = false;

    // Per-layer: start LBA and byte size for NVMe reads
    struct NvmeLayerInfo {
        uint64_t start_lba;     // LBA of first byte of this layer's tensor data
        size_t   total_bytes;   // total bytes for all 7 tensors
    };
    std::vector<NvmeLayerInfo> nvme_layers_;
    uint64_t gguf_start_lba_ = 0;
    uint32_t nvme_block_size_ = 512;
#endif
```

### 3. `src/memory/streamer.cu` — init / prefetch / shutdown

#### 3a. `init()` — después de construir `layers_[]`

```cpp
#ifdef USE_GPUNVME
    // Leer parámetros NVMe de environment variables
    const char* nvme_bdf = getenv("GPUNVME_PCI_BDF");
    const char* nvme_lba_str = getenv("GPUNVME_GGUF_LBA");

    if (nvme_bdf && nvme_lba_str) {
        uint64_t gguf_start_lba = strtoull(nvme_lba_str, NULL, 0);

        gpunvme_err_t err = gpunvme_layer_loader_init(
            &nvme_loader_, nvme_bdf, buf_size_, /*pipeline_depth=*/32);

        if (err == GPUNVME_OK) {
            nvme_initialized_ = true;
            gguf_start_lba_ = gguf_start_lba;
            nvme_block_size_ = gpunvme_layer_loader_block_size(&nvme_loader_);

            // Pre-compute per-layer LBAs
            uint64_t data_offset = loader.file_data_offset();
            nvme_layers_.resize(n_layers);
            for (int i = 0; i < n_layers; i++) {
                std::string first = "blk." + std::to_string(i) + ".attn_q.weight";
                uint64_t byte_offset = loader.tensor_file_offset(first);
                nvme_layers_[i].start_lba = gguf_start_lba + (byte_offset / nvme_block_size_);
                nvme_layers_[i].total_bytes = layer_transfer_size(i);
            }

            fprintf(stderr, "LayerStreamer: NVMe backend OK (MDTS=%uK, block=%u)\n",
                    gpunvme_layer_loader_max_transfer(&nvme_loader_) / 1024,
                    nvme_block_size_);
        } else {
            fprintf(stderr, "LayerStreamer: NVMe init failed (%s), fallback to mmap\n",
                    gpunvme_err_str(err));
        }
    }
#endif
```

#### 3b. `prefetch_staging()` — reemplazar body

```cpp
void LayerStreamer::prefetch_staging(int layer_idx, int slot) {
    if (mmap_pinned_) return;

#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        const auto& nlay = nvme_layers_[layer_idx];
        gpunvme_err_t err = gpunvme_load_layer(
            &nvme_loader_, nlay.start_lba, nlay.total_bytes, staging_buf_[slot]);

        if (err == GPUNVME_OK) {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            staging_ready_[slot] = true;
            staging_ready_cv_.notify_all();
            return;
        }
        fprintf(stderr, "LayerStreamer: NVMe read failed for layer %d, fallback\n", layer_idx);
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

#### 3c. `shutdown()` — agregar cleanup

```cpp
void LayerStreamer::shutdown() {
#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        gpunvme_layer_loader_destroy(&nvme_loader_);
        nvme_initialized_ = false;
    }
#endif
    // ... resto del shutdown existente ...
}
```

### 4. `src/model/loader.h` — Ya tiene los métodos necesarios

Los métodos `tensor_file_offset()` y `file_data_offset()` ya están implementados
(líneas 75-80). No se necesitan cambios adicionales.

---

## Setup NVMe (después de cada reboot)

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

El archivo GGUF debe estar en el NVMe crudo (sin filesystem), empezando en LBA 0:

```bash
# ANTES de bind a VFIO (necesita driver NVMe nativo)
# CUIDADO: esto destruye cualquier dato en el NVMe
sudo dd if=/path/to/model.gguf of=/dev/nvme0n1 bs=1M oflag=direct status=progress

# Verificar:
ls -la /path/to/model.gguf
# 57,398,476,800 bytes → empieza en LBA 0
```

**NOTA**: El NVMe usado para gpu-nvme-direct NO debe tener filesystem ni estar
montado. Se accede como raw block device via VFIO.

---

## Guía de desarrollo y testing

### Prerequisitos

1. **gpu-nvme-direct compilado** con hardware build:
   ```bash
   cd ~/gpu-nvme-direct/build-hw
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF \
     -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
     -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
     -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build . -j$(nproc)
   ```

2. **NVMe setup** (ver sección anterior)

3. **Layer Loader test** pasando:
   ```bash
   cd ~/gpu-nvme-direct/build-hw
   sudo ./test_layer_loader 0000:0b:00.0       # 4MB, 3/3 tests
   sudo ./test_layer_loader 0000:0b:00.0 669   # full layer, 3/3 tests
   ```

### Flujo de desarrollo incremental

#### Paso 1: Verificar Layer Loader aislado

Antes de tocar ntransformer, verificar que el Layer Loader funciona con el
tamaño exacto de las layers del modelo objetivo:

```bash
# Tamaños típicos de layer (70B Llama):
#   Q6_K: ~669 MB por layer (80 layers)
#   Q8_0: ~875 MB por layer (80 layers)
cd ~/gpu-nvme-direct/build-hw
sudo ./test_layer_loader 0000:0b:00.0 669   # Q6_K layer size
sudo ./test_layer_loader 0000:0b:00.0 875   # Q8_0 layer size
```

Esperado: 3/3 tests pass, throughput ~2.1 GB/s para 669MB.

#### Paso 2: Standalone integration test

Crear un test mínimo en ntransformer que usa el Layer Loader para leer una
layer real del GGUF en el NVMe:

```cpp
// tests/test_nvme_layer.cu

#include <gpunvme/layer_loader.h>
#include "../src/model/loader.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gguf_path> <pci_bdf>\n", argv[0]);
        return 1;
    }

    // 1. Parse GGUF header (still need mmap for metadata)
    nt::GGUFLoader loader;
    loader.load(argv[1]);
    loader.print_info();

    // 2. Find layer 0's first tensor offset and total size
    uint64_t data_offset = loader.file_data_offset();
    uint64_t layer0_offset = loader.tensor_file_offset("blk.0.attn_q.weight");
    printf("GGUF data_offset=%lu, layer0_offset=%lu\n", data_offset, layer0_offset);

    // Get all layer 0 tensor sizes
    const char* names[] = {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"
    };
    size_t total_bytes = 0;
    for (auto& n : names) {
        auto* info = loader.tensor_info(n);
        if (info) total_bytes += info->nbytes;
    }
    printf("Layer 0 total: %zu bytes (%.1f MB)\n", total_bytes, total_bytes / 1e6);

    // 3. Init Layer Loader
    gpunvme_layer_loader_t nvme;
    gpunvme_err_t err = gpunvme_layer_loader_init(&nvme, argv[2], total_bytes, 32);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Layer loader init failed: %s\n", gpunvme_err_str(err));
        return 1;
    }

    // 4. Alloc dest buffer and read layer 0
    void* buf;
    cudaMallocHost(&buf, total_bytes);
    memset(buf, 0xDE, total_bytes);

    uint32_t block_size = gpunvme_layer_loader_block_size(&nvme);
    uint64_t start_lba = layer0_offset / block_size;  // GGUF starts at LBA 0

    err = gpunvme_load_layer(&nvme, start_lba, total_bytes, buf);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Load failed: %s\n", gpunvme_err_str(err));
        gpunvme_layer_loader_destroy(&nvme);
        return 1;
    }

    // 5. Compare with mmap'd data
    const uint8_t* mmap_data = (const uint8_t*)loader.tensor_data("blk.0.attn_q.weight");
    const uint8_t* nvme_data = (const uint8_t*)buf;

    // El offset dentro del bloque NVMe puede diferir del offset dentro del GGUF
    // si layer0_offset no es múltiplo de block_size. Ajustar:
    size_t block_offset = layer0_offset % block_size;

    int mismatch = 0;
    for (size_t i = 0; i < 4096 && i < total_bytes; i++) {
        if (nvme_data[block_offset + i] != mmap_data[i]) {
            printf("MISMATCH at byte %zu: nvme=0x%02x mmap=0x%02x\n",
                   i, nvme_data[block_offset + i], mmap_data[i]);
            mismatch++;
            if (mismatch > 10) break;
        }
    }

    if (mismatch == 0) printf("PASS: NVMe data matches mmap (first 4KB)\n");
    else printf("FAIL: %d mismatches\n", mismatch);

    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&nvme);
    return mismatch > 0 ? 1 : 0;
}
```

Compilar:
```bash
cd ~/ntransformer/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86

# Compilar el test manualmente (antes de integrar al CMake):
nvcc -std=c++20 -O2 --compiler-bindir=/usr/bin/gcc-14 -arch=sm_86 \
  -I ~/gpu-nvme-direct/include -I ~/ntransformer/src \
  tests/test_nvme_layer.cu ~/ntransformer/src/model/loader.cpp \
  ~/ntransformer/src/model/config.cpp ~/ntransformer/src/core/tensor.cpp \
  ~/ntransformer/src/core/allocator.cpp \
  -L ~/gpu-nvme-direct/build-hw \
  -lgpunvme_layer_loader -lgpunvme_host -lgpunvme_device \
  -lcudart -lstdc++ -lm -o test_nvme_layer

# Correr:
sudo ./test_nvme_layer /path/to/model.gguf 0000:0b:00.0
```

#### Paso 3: Integrar en LayerStreamer

Una vez que el standalone test pasa:
1. Agregar `USE_GPUNVME` al CMakeLists.txt (ver sección anterior)
2. Agregar miembros NVMe a `streamer.h`
3. Modificar `init()`, `prefetch_staging()`, `shutdown()`
4. Compilar con `-DUSE_GPUNVME=ON`

#### Paso 4: Test end-to-end con ntransformer

```bash
# Asegurar GGUF está en el NVMe (ver sección de setup)
# Build ntransformer con NVMe backend
cd ~/ntransformer/build
cmake .. -DUSE_GPUNVME=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)

# Ejecutar benchmark con NVMe backend
sudo GPUNVME_PCI_BDF=0000:0b:00.0 GPUNVME_GGUF_LBA=0 \
     ./ntransformer -m /path/to/model.gguf --streaming --benchmark -n 8

# Comparar con mmap backend (sin env vars)
./ntransformer -m /path/to/model.gguf --streaming --benchmark -n 8
```

Verificar:
- Mismo output (bit-identical tokens)
- stderr muestra "LayerStreamer: NVMe backend OK"
- stderr muestra throughput ~2.1 GB/s por layer read

### Troubleshooting

| Síntoma | Causa probable | Fix |
|---------|---------------|-----|
| `layer_loader: failed to open resource0` | VFIO no configurado | Correr setup NVMe |
| `cudaHostRegisterIoMemory failed` | Driver nvidia no parcheado | Parchear os-mlock.c (ver gpu-nvme-direct docs) |
| `controller init failed: timeout` | NVMe en D3 / link down | `setpci` commands + power/control |
| `read failed: timeout` at cmd N | PRP list no page-aligned | Verificar que `dest_pinned` está page-aligned |
| `NVMe init failed, fallback to mmap` | Cualquier error de init | Revisar stderr, correr test_layer_loader primero |
| Data mismatch vs mmap | LBA offset mal calculado | Verificar `tensor_file_offset()` y block alignment |
| `CSTS.CFS=1` | NVMe fatal error | Power cycle del NVMe (unplug/replug o reboot) |

### Consideraciones de alignment

**GGUF tensor alignment**: Los tensors en GGUF están alineados a 32 bytes por
defecto (GGUF v3). Esto NO coincide con el block size del NVMe (512B).

Opciones:
1. **Leer bloques completos**: start_lba = floor(byte_offset / 512), leer bytes
   extra al inicio. El offset dentro del bloque se aplica al parsear los tensors.
   Los staging_buf_[] ya son suficientemente grandes.

2. **Alinear el GGUF a 512B**: Usar `gguf-split` o similar para forzar alignment
   de tensors a 512 bytes. Esto simplifica el cálculo de LBAs.

La opción 1 es más simple y no requiere modificar el GGUF. El overhead de leer
bytes extra es despreciable (<512B por layer).

### Performance profiling

Para medir dónde se gasta el tiempo:

```bash
# 1. Solo NVMe read (sin compute)
# El layer_loader imprime throughput en stderr:
# "layer_loader: read 669000000 bytes (1306 cmds) in 315.2 ms — 2023.4 MB/s"

# 2. nsight systems profile
sudo GPUNVME_PCI_BDF=0000:0b:00.0 GPUNVME_GGUF_LBA=0 \
     nsys profile -o nvme_streaming \
     ./ntransformer -m /path/to/model.gguf --streaming --benchmark -n 4

# 3. Verificar que compute overlap funciona
# En nsys, los GEMV kernels deben solaparse con NVMe reads.
# Si hay gaps entre layers → el NVMe read es el bottleneck puro.
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

## TODO (orden de implementación)

### Fase 1: Integración básica (usa Layer Loader API)
- [ ] Agregar `USE_GPUNVME` option a CMakeLists.txt
- [ ] Agregar `gpunvme_layer_loader_t` a `LayerStreamer`
- [ ] En `init()`: llamar `gpunvme_layer_loader_init()`, pre-compute per-layer LBAs
- [ ] En `prefetch_staging()`: llamar `gpunvme_load_layer()` cuando NVMe disponible
- [ ] En `shutdown()`: llamar `gpunvme_layer_loader_destroy()`
- [ ] Standalone test: leer layer 0 del NVMe, comparar con mmap
- [ ] End-to-end: `--streaming` con NVMe, verificar output idéntico

### Fase 2: Optimización
- [ ] Eliminar staging buffer: NVMe DMA → gpu_buf_[] directamente (los staging
      buffers son host pinned, que es exactamente lo que necesita NVMe DMA)
- [ ] Pre-compute PRP lists en init (evitar rebuild por layer)
- [ ] Lanzar GPU kernel en transfer stream (no default stream) para overlap

### Fase 3: Eliminar dependencia de mmap
- [ ] Modo NVMe puro: no necesitar abrir el GGUF file con mmap
- [ ] Parsear GGUF header leyendo primeros bloques del NVMe
- [ ] Solo necesitar BDF + start_lba + model config

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

Nota: En Fase 2, staging se elimina — NVMe DMA va directo a gpu_buf_[slot]
      (que ya está en host pinned memory en Tier 1).
```
