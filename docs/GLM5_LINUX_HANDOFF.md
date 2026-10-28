# HANDOFF: GLM-5.3-Flash × gpu-nvme-direct — continuar en Linux

> Este archivo es una instrucción completa para una sesión nueva de opencode en Linux.
> Contexto total en `docs/GLM5_PROJECT.md` y resultados del bench en `GLM5_LINUX_BENCH.md`.

## La misión

GLM-5.3-Flash (320B/18B activo, Q8_0, 341GB) ya corre en ntransformer con paridad
bit-exacta vs llama.cpp. En Linux con gpu-nvme-direct ya logrado: **0.45-0.5 tok/s
decode, 56 ms/tok prefill** (techo físico del SN5000S). El trabajo sigue desde ahí.

## Hardware y discos (verificar con `lspci` / `lsblk` al bootear)

| Disco | Windows | Linux | Rol |
|---|---|---|---|
| WD PC SN5000S 512GB NVMe | E: (el modelo) | **VFIO bind** — BDF `0000:0b:00.0` (verificar) | Raw store = el GGUF original |
| WD SATA 1TB ×2 (span 1.86TB) | D: (projects + linux_kit) | **montar via ldmtool** (`ldmtool create <guid> Volume1` → `/dev/mapper/...` → ntfs3 ro) | Tiene `D:\linux_kit\` y `D:\projects\` |
| Micron 2400 512GB NVMe | C: (boot Win) | ignorar | — |

## El kit: `D:\linux_kit\`

| Archivo | Qué es |
|---|---|
| `glm5_backbone.bin` (9.4GB) | 1,283 tensores no-expertos del modelo |
| `glm5_backbone_manifest.json` | name → {offset, bytes, dims, ggml_type} |
| `expert_lba_index.json` (0.8MB) | Mapa LBA (STALE — regenerar SIEMPRE con FIEMAP en Linux) |
| `glm5_nvme_probe.cu` | Test de validación (3 LBAs vs hashes) |
| `sync_to_repo.sh` | Sincroniza fuentes de la sesión al repo en D: |
| `map_lbas.py`, `build_lba_index.py`, `pack_backbone.py` | Herramientas Windows (regeneración) |
| `GLM5_LINUX_BENCH.md` | Resultados + hallazgos completos de la sesión Linux |
| `GLM5_GPU_UNLOCK.md` | Plan de iteración con Zagitta/gpu-unlock |

## El insight clave

**El GGUF en E: ES el raw store.** Sin copiar 341GB. El índice LBA apunta directo
a los bloques del archivo original (97.7% de slices contiguos, 0.00 cruces de
extent promedio). **NO TOCAR E: DESDE WINDOWS** (defrag/chkdsk/escrituras).

**El índice de Windows queda STALE** — NTFS mueve extents con el uso normal.
Regenerar SIEMPRE con FIEMAP en Linux (`regen_lba_index_linux.py`, ioctl read-only)
antes de correr.

## Paso 0 — sincronizar fuentes (si hay cambios sin respaldar)

```bash
sudo bash ~/sync_to_repo.sh    # copia ~/src/ntransformer → D:\projects\ntransformer via ldmtool
```

## Paso 1 — setup

```bash
# montar D: (span LDM) y copiar el kit al home ext4
sudo ldmtool scan all
VOL=$(ldmtool scan all 2>/dev/null | grep -oP 'f2d76644[0-9a-f-]+' | head -1)
sudo ldmtool create "$VOL" Volume1
sudo mkdir -p /mnt/win && sudo mount -t ntfs3 -o ro /dev/mapper/ldmvol_*Volume1 /mnt/win
mkdir -p ~/glm5
cp /mnt/win/linux_kit/{glm5_backbone.bin,glm5_backbone_manifest.json} ~/glm5/
sudo umount /mnt/win
```

## Paso 2 — VFIO bind del SN5000S

```bash
cd ~/gpu-nvme-direct
BDF=0000:0b:00.0   # verificar con lspci — CONFIRMAR que NO es el disco de boot
sudo scripts/setup_vfio.sh $BDF
sudo modprobe vfio enable_unsafe_noiommu_mode=1
echo on | sudo tee /sys/bus/pci/devices/$BDF/power/control
sudo setpci -s $BDF 84.W=0x0008        # OBLIGATORIO (registro WD DRAM-less)
sudo setpci -s $BDF COMMAND=0x0006     # + bus master, con read-back
```

**La secuencia completa o nada** — sin el 0x84.W el DMA muere silenciosamente
(NaN → salida `!`×N). Usar `~/run_glm5.sh` que ya lo automatiza todo.

## Paso 3 — regenerar índice LBA + probe

```bash
python3 ~/regen_lba_index_linux.py     # FIEMAP, read-only sobre E:
# (re)compilar el probe si hace falta (fuente: tools/linux/glm5_nvme_probe.cu)
nvcc -O2 -arch=sm_86 -I ~/src/gpu-nvme-direct/include \
    glm5_nvme_probe.cu ~/src/gpu-nvme-direct/src/host/layer_loader.cu -o ~/probe
sudo ~/probe $BDF ~/glm5/expert_lba_index.json   # esperar: PROBE OK
```

`PROBE OK` = los 3 hashes del primer bloque de 3 slices (`220df462cfd41245`,
`d40d9dc76320007f`, `e80ff6110c1dd0e7`). Si falla: BDF equivocado o índice
stale (los LBA son absolutos de disco, offset de partición incluido).

## Paso 4 — correr

```bash
sudo env GLM5_BACKBONE_PACK=~/glm5/pack GLM5_GPNVME=1 \
     GLM5_LBA_BIN=~/glm5/glm5_lba.bin GPUNVME_PCI_BDF=$BDF GLM5_GPU=1 GLM5_CHUNK=3 \
  ~/ntransformer-gpnvme -m ~/glm5/pack/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf \
  --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512
# esperado: " Paris. In French, Paris is spelled \"Paris\", but"
```

`GLM5_CHUNK=3` es el workaround de la carrera del union load con chunks grandes
(`GLM5_LINUX_BENCH.md` §10; `run_glm5.sh` ya lo fija por defecto). Antes de
cualquier escritura cruda a un disco (p. ej. `fill_second_disk.py`), verificar
el serial del destino: los nombres `nvmeXnY` cambian tras un hot-swap (§9).

## Iteración: gpu-unlock

Ver `GLM5_GPU_UNLOCK.md` — BF16 +80%/TF32 +94% en GA102, benchar contra el
baseline, para compounding con el 2° NVMe futuro.

## Números de referencia

| Config | decode | prefill |
|---|---|---|
| Windows (repo, sync path) | 0.24-0.26 tok/s | 1.05 s/tok |
| **Linux gpnvme (logrado)** | **0.45-0.5 tok/s** | **56 ms/tok** |
| Techo físico SN5000S | 0.37 tok/s (sin hits) / 0.45-0.5 (con LRU) | — |
| 2° NVMe stripe | ~0.9-1.0 tok/s | ~28 ms/tok |
| NVMe PCIe4 | ~1.5-1.9 tok/s | — |
