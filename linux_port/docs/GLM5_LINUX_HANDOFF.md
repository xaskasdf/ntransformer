# HANDOFF: GLM-5.3-Flash × gpu-nvme-direct — continuar en Linux

> ✅ **EJECUTADO Y COMPLETADO (Oct 22, 2026)** — paridad bit-exacta lograda en
> Linux con expertos leídos del disco crudo por el GPU. Decode 0.45–0.5 tok/s
> (techo físico del disco; supera el 0.24–0.26 de Windows). Estado completo,
> bench y hallazgos: **`GLM5_LINUX_BENCH.md`** y la sección "ESTADO LINUX" de
> `GLM5_PROJECT.md`. Binario verificado: `~/ntransformer-gpnvme`.
> Desviaciones del plan original: el índice LBA se regeneró en Linux (FIEMAP),
> Secure Boot quedó desactivado (el lockdown bloquea el acceso PCI que la lib
> necesita), `iommu=pt` en el cmdline, y Ubuntu live 25.10 en vez de Arch.

> Este archivo es una instrucción completa para una sesión nueva de opencode en Linux.
> Contexto total de lo hecho en Windows está en `docs/GLM5_PROJECT.md`. Leer AMBOS.

## La misión

GLM-5.3-Flash (320B/18B activo, Q8_0, 341GB) ya corre en ntransformer con paridad
exacta vs llama.cpp, pero el decode está limitado a 0.24-0.26 tok/s por el fill de
expertos desde NVMe (1.33 GiB/s efectivo con ReadFile CPU-mediado). El objetivo:
usar **gpu-nvme-direct** (GPU-issued NVMe reads a 3.35 GB/s medidos en este mismo
hardware) para el fill → target **0.5-0.6 tok/s**.

## Hardware y discos (verificar con `lspci` / `lsblk` al bootear)

| Disco | Windows | Linux (probable) | Rol |
|---|---|---|---|
| WD PC SN5000S 512GB NVMe | E: (el modelo) | **VFIO bind** — BDF a confirmar (`lspci -d 15b9:` o similar) | Raw store = el GGUF original en su lugar |
| Micron 2400 512GB NVMe | C: (boot Win) | montar ro (`ntfs3`) | Tiene `C:\linux_kit\` con todo el kit |
| WD SATA 1TB ×2 (span 1.86TB) | D: (`projects`) | **NO montable** (volumen dinámico Windows) — ignorar | — |
| USB FAT32 15GB | H: | — | Instalador Arch, no sirve de kit (FAT32 4GB) |

## El kit: `C:\linux_kit\` (montar C: read-only y copiar a ext4 home)

| Archivo | Qué es |
|---|---|
| `glm5_backbone.bin` (9.4GB) | 1,283 tensores no-expertos del modelo |
| `glm5_backbone_manifest.json` | name → {offset, bytes, dims, ggml_type} |
| `expert_lba_index.json` (0.8MB) | **El mapa LBA**: layers[3..44].{gate,up,down}.experts[e] = [[lba, nbytes], ...]. 36,288 slices, 97.7% contiguos, +2048 sectores de offset de partición YA incluido |
| `glm5_nvme_probe.cu` | Test de validación (3 LBAs vs hashes del primer bloque de 3 slices) |
| `map_lbas.py`, `build_lba_index.py`, `pack_backbone.py` | Herramientas Windows para regenerar todo si E: cambia |

## El insight clave (no repetir el descubrimiento)

**El GGUF en E: ES el raw store.** No hay que copiar 341GB. Los archivos quedaron
físicamente casi contiguos (109-142 runs por shard de 50GB), y cada slice de
experto de 8.9MB cae dentro de un único run (0.00 cruces promedio). El índice LBA
apunta directo a los bloques del archivo original.

**NO TOCAR E: DESDE WINDOWS** (defrag/chkdsk/escrituras) — invalida el mapa.

## Paso 1 — setup (antes de VFIO)

```bash
# montar C: y copiar el kit al home
sudo mkdir -p /mnt/win && sudo mount -t ntfs3 -o ro /dev/nvme1n1p3 /mnt/win   # C: (ver partición)
mkdir -p ~/glm5 && cp /mnt/win/linux_kit/{glm5_backbone.bin,glm5_backbone_manifest.json,expert_lba_index.json} ~/glm5/
sudo umount /mnt/win

# fuentes
cd ~ && git -C /path/a/ntransformer pull 2>/dev/null || true   # o copiar de dónde sea
```

## Paso 2 — VFIO bind del SN5000S

```bash
cd ~/gpu-nvme-direct
BDF=$(lspci | grep -i 'sn5000\|15b9' | awk '{print $1}')   # CONFIRMAR que es el disco del modelo, NO el boot
sudo scripts/setup_vfio.sh $BDF
sudo modprobe vfio enable_unsafe_noiommu_mode=1
echo on | sudo tee /sys/bus/pci/devices/0000:$BDF/power/control
sudo setpci -s $BDF 84.W=0x0008
sudo setpci -s $BDF COMMAND=0x0006
```

El driver nvidia debe ser el parcheado (os-mlock.c por cudaHostRegisterIoMemory);
`ntransformer/scripts/setup_system.sh:463-523` lo aplica (necesita rebuild DKMS si
el kernel cambió).

## Paso 3 — probe (VALIDACIÓN CRÍTICA)

```bash
nvcc -O2 -arch=sm_86 -I ~/gpu-nvme-direct/include \
    ~/linux_kit/glm5_nvme_probe.cu ~/gpu-nvme-direct/src/host/layer_loader.cu -o ~/probe
sudo ~/probe $BDF ~/glm5/expert_lba_index.json
```

Esperado: `PROBE OK` con los 3 hashes (`220df462cfd41245`, `d40d9dc76320007f`,
`e80ff6110c1dd0e7`). Si falla: revisar que el BDF sea el disco correcto y que el
offset de partición sea el mismo (el índice asume LBA absolutos de disco).

## Paso 4 — integración glm5 (el trabajo de verdad)

Archivos a tocar:
- `src/model/glm5/glm5_gpu.cu` → `ExpertEngine`:
  - nuevo init: `GLM5_GPNVME=1` + `GPUNVME_PCI_BDF` + `GLM5_LBA_INDEX=path` →
    parsear el JSON (o convertirlo a bin en el kit con un script), crear el
    `gpunvme_layer_loader_t` con `max_layer_bytes = slice_bytes`, depth 8
  - `fill_slice(wid, ring, src)`: resolver `src` (puntero base del tensor +
    expert*slice) contra el registro de capas → `gpunvme_load_layer(loader, lba,
    slice, ring_buf)`. Registrar cada tensor experto al load del modelo con su
    array de 288 LBAs.
  - Slices no contiguos (843/36,288): el JSON ya trae lista de runs — iterar
- `src/model/glm5/glm5_model.cpp` / `glm5_config.cpp`:
  - si `GLM5_BACKBONE_PACK=path` set: cargar tensores desde backbone.bin+manifest
    (mmap) en vez de shards GGUF; el modelo "no ve" los tensores de expertos del
    pack — los punteros base de expertos se registran igual (para las claves del
    cache) pero nunca se dereferencian en CPU
  - alternativa rápida: cargar el GGUF "fantasma" — solo headers de los shards
    (64MB c/u) para que GGUFLoader dé config+tensor_info, y backbone.bin para data
- CMake: los `CORE_SOURCES` de glm5 ya compilan en Linux; enlazar con las libs de
  gpu-nvme-direct (ver `USE_GPUNVME` existente en CMakeLists:150-171)

Gotcha conocido (doc de la integración original, GPU_NVME_DIRECT_INTEGRATION.md:569):
el loader lanza `gpu_layer_read<<<1,1>>>` en el legacy stream con
`cudaDeviceSynchronize()` — serializa contra TODO. Mitigación v1: como fill_slice
se llama desde los 8 workers del ExpertEngine con mutex del loader, y el loader
tiene pipeline_depth interno, dejar así y medir; si el overhead de sync domina,
mover a stream dedicado (el TODO original).

## Paso 5 — validar y medir

```bash
# paridad (greedy, mismo resultado que en Windows):
./build/ntransformer -m <backbone-pack-mode> --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512
# esperado: " Paris. In French, Paris is spelled \"Paris\", but"
# bench: GLM5_GPU=1 GLM5_GPNVME=1 → target >= 0.5 tok/s decode
```

## Números de referencia (Windows, para comparar)

| Config | decode | prefill |
|---|---|---|
| CPU puro | 0.29 tok/s | ~5.5 s/tok |
| GPU sync path (producción) | 0.24-0.26 | 1.05 s/tok (chunk 512) |
| GPU fast-path device-residente | 0.16 | — |
| **Target gpnvme** | **0.5-0.6** | ~1.6 min/chunk |

El muro era: 45 capas × 24 slices × 8.5MB = 9.2GB/token ÷ 1.33 GiB/s = 6.9s/token
de I/O puro. Con 3.35 GB/s: 2.75s/token → 0.36 tok/s worst case sin overlap;
con LRU VRAM de 13GB (miss ~65%): ~1.8s/token → 0.55 tok/s.
