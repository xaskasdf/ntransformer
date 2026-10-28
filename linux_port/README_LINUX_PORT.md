# linux_port — port de GLM-5.3-Flash a Linux con expertos por GPU→NVMe

Port completo y verificado (paridad bit-exacta) de GLM-5.3-Flash (320B/18B MoE,
Q8_0) corriendo en Linux con los expertos leídos del disco NVMe crudo por el
GPU (gpu-nvme-direct, VFIO). Estado: **operativo, 0.45-0.5 tok/s decode
(techo físico del disco SN5000S) en RTX 3090**, frente a 0.24-0.26 tok/s en
Windows nativo.

El código de la integración (rama gpnvme de `fill_slice`, pack mode,
verificación FILL-HASH/SLOT-VERIFY, índice `LBA1`/`LBA2`) está **mergeado en
`src/model/glm5/`**. Esta carpeta guarda lo que es propio del host Linux.

## Contenido

| Carpeta | Qué es |
|---|---|
| `tools/` | `run_glm5.sh` (runner: bootstrap backbone → wake → vfio → setpci 84.W+COMMAND con read-back → exec; fija `GLM5_CHUNK=3`), `post_reboot.sh`, `regen_lba_index_linux.py` (regen FIEMAP, read-only), `gen_lba2.py` (índice LBA2 de dos discos), `fill_second_disk.py` (llenado crudo, exige `DST_SERIAL`), fuentes de los probes 2/3 y `bench_slice.cu`. |
| `index/` | `expert_lba_index.json` (regen FIEMAP en Linux), `disk_layout.json`, `probe3_vectors.json` (verificación on-device), `gguf_lba_map.json` y `glm5_backbone_manifest.json` (generados en Windows por `tools/linux/`). |
| `gpunvme_patch/` | gpu-nvme-direct parcheado: `layer_loader.cu` (stream dedicado non-blocking + guard de geometría cero) y `layer_loader.h` (campo stream), con `.repo.orig` = upstream para diff. |

No versionados (quedan en disco local; ver `.gitignore`): el binario verificado
`bin/ntransformer-gpnvme-linux`, los ejecutables `tools/probe2`/`probe3` y los
índices binarios `index/*.bin` (`glm5_lba.bin` 'LBA1', `glm5_lba2_two.bin`
'LBA2', `glm5_backbone_manifest.bin` 'MBN1'), regenerables con las
herramientas de arriba.

Las herramientas del lado Windows del puente (`D:\linux_kit`) están en
`tools/linux/`: `map_lbas.py`, `build_lba_index.py`, `pack_backbone.py`,
`glm5_nvme_probe.cu` y `sync_to_repo.sh` (sincroniza la sesión Linux al repo
montando D: con `ldmtool`).

## Documentación

Canónica en `docs/`: `GLM5_PROJECT.md` (master: arquitectura, flags, hallazgos,
setup Linux), `GLM5_LINUX_BENCH.md` (bench, verificación, 17 hallazgos,
incidente §9, carrera del union load §10), `GLM5_LINUX_HANDOFF.md` (cómo
retomar) y `GLM5_GPU_UNLOCK.md` (siguiente iteración). La copia divergente que
vivía aquí (`linux_port/docs`, snapshot del 22-10) se fusionó en `docs/` y
queda en el historial de git.

## Cómo correr (en el setup Linux existente)

```bash
sudo bash ~/run_glm5.sh --token-ids 785,6722,315,9621,374 -n 12 -t 0 \
  --repeat-penalty 1.0 -c 512
# →  Paris. In French, Paris is spelled "Paris", but   (golden bit-exacto)
```

## Workarounds y estado de bugs

- `GLM5_CHUNK=3` (default en run_glm5.sh): workaround del race del union load
  con chunks grandes (BENCH §10: bisección por chunk-size).
- `GLM5_PREFETCH=2` NO usar (rompe paridad, carrera documentada BENCH §7.14).
- SN520 (PCIe3×2) descartado como disco de expertos: lento y su primer fill
  cuelga en gpunvme (BENCH §10).
- Cualquier escritura cruda a un disco exige serial verificado (incidente
  BENCH §9: un fill por nombre de device sobreescribió el disco de boot).

## Para recompilar

Fuentes: host Linux `~/src/ntransformer` + `~/src/gpu-nvme-direct`
(siblings, con `gpunvme_patch/` aplicado). `cmake -DUSE_GPUNVME=ON
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13 -DNTRANSFORMER_BUILD_NTRPU_SIM=OFF`
(CUDA 12.4 nvcc; kernel 6.17 necesita gcc-13 y el parche os-mlock.c del
driver; Secure Boot OFF + iommu=pt).
