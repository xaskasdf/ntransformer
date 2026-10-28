# GLM-5.3-Flash × gpu-nvme-direct — plan de integración Linux

## Estado (hecho desde Windows)

| Artefacto | Qué es | Dónde |
|---|---|---|
| `E:\Q8_0\*.gguf` | El modelo Q8_0 original — **ES el raw store** (no se copia nada) | NVMe SN5000S |
| `E:\Q8_0\expert_lba_index.json` (0.8MB) | 42 capas × 288 expertos × 3 tensores → LBA. 97.7% slices contiguos (843 con 2 runs) | copiar a boot disk |
| `E:\glm5_backbone.bin` (9.84GB) | 1,283 tensores no-expertos + manifest JSON | copiar a boot disk |
| `tools/linux/glm5_nvme_probe.cu` | Test: 3 probes LBA vs hash | compilar en Linux |

## Proyección

- Fill de expertos: 1.33 GiB/s (ReadFile CPU) → **3.35 GB/s** (GPU-issued) = 2.5×
- Decode: 0.26 → **0.5-0.6 tok/s** (floor 1.8s/token con 65% miss)
- Prefill chunk: ~2.4min → **~1.6min** por chunk de 512

## Paso a paso en Linux

### 1. Preparación (antes del VFIO bind)
```bash
# montar E: read-only (ntfs3) y copiar los artefactos al boot disk
mkdir -p /mnt/ntfs && mount -t ntfs3 -o ro /dev/nvmeXnYp2 /mnt/ntfs   # partición de E:
cp /mnt/ntfs/Q8_0/expert_lba_index.json ~/
cp /mnt/ntfs/glm5_backbone.bin ~/
cp /mnt/ntfs/glm5_backbone_manifest.json ~/
umount /mnt/ntfs
```

### 2. VFIO bind del SN5000S
```bash
cd ~/gpu-nvme-direct
sudo scripts/setup_vfio.sh 01:00.0        # BDF del SN5000S (verificar con lspci)
# post-bind (flujo ntransformer):
sudo modprobe vfio enable_unsafe_noiommu_mode=1
echo on | sudo tee /sys/bus/pci/devices/0000:01:00.0/power/control
sudo setpci -s 01:00.0 84.W=0x0008
sudo setpci -s 01:00.0 COMMAND=0x0006
```

### 3. Verificación del mapa (el paso crítico)
```bash
nvcc -O2 -arch=sm_86 -I gpu-nvme-direct/include \
    tools/linux/glm5_nvme_probe.cu gpu-nvme-direct/src/host/layer_loader.cu -o probe
sudo ./probe 01:00.0 ~/expert_lba_index.json
# espera: PROBE OK — los 3 hashes de bloques coinciden con los leidos en Windows
```

### 4. Integración glm5 (código a escribir tras el probe OK)

`glm5_gpu.cu` / ExpertEngine:
- `glm5_gpunvme_init(bdf, index_path)`: carga el LBA index, crea el loader
- `fill_slice(wid, ring, src)`: si gpunvme activo → resuelve `src` a (layer_base → experts[e] lba) y llama `gpunvme_load_layer(loader, lba, slice, ring_buf)` — GPU emite el NVMe read, sin ReadFile
- Preactivar PRPs por slice en init (TODO de la integración original)

`glm5_model.cpp`:
- Si `GLM5_BACKBONE_PACK` set: cargar tensores desde backbone.bin+manifest (mmap local) en vez de shards GGUF; config KV hardcodeado o duplicado en manifest

Modo stream (el TODO de la doc original):
- Cambiar `gpu_layer_read<<<1,1>>>` (legacy stream) a un stream dedicado para solapar con GEMV
- O más simple para v1: el loader ya es síncrono por slice pero con pipeline_depth=8 en el pool de threads (el ExpertEngine ya tiene 8 workers que llaman fill_slice en paralelo → 8 loaders o mutex alrededor de 1 loader... NO: single queue pair. v1: serializar las llamadas gpunvme con el mutex del engine, depth interno del loader 8)

### 5. Verificación final
- Paridad greedy: ` Paris. In French, Paris is spelled "Paris", but`
- Bench: GLM5_GPU=1 GLM5_GPNVME=1 → target ≥0.5 tok/s decode

## Riesgos
- El mapa LBA queda invalidado si Windows desfragmenta/mueve el archivo → **no tocar E: desde Windows** (ni defrag, ni chkdsk /f, ni escribir)
- Reversibilidad: el VFIO bind se deshace con unbind/rebind de vfio-pci → nvme (script teardown)
- ~~Offset de partición~~ **RESUELTO**: los LBAs del index ya incluyen +2048 sectores (offset de la partición E: = 1MB). Los probes del .cu usan los valores corregidos.
- Nota: en Linux el SN5000S es `/dev/nvmeXnY`, la partición E: es `nYp1` — para el paso 1 (copiar artefactos) montar `p1`; el VFIO bind es del **disco entero** (función 0000:01:00.0), sin particiones
