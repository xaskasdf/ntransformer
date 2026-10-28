# GLM-5.3-Flash en NTransformer — documentación del proyecto

> Cronología completa del port + optimización. Todo el trabajo se hizo en sesiones
> de opencode sobre Windows nativo (build MSVC+CUDA 13.2), Oct 2026.

## Qué corre hoy

**GLM-5.3-Flash** (zai-org, 320B total / 18B activo, MoE 288-experto, Q8_0 341GB)
en ntransformer, Windows nativo, RTX 3090 + Ryzen 5800X + 48GB RAM, con **paridad
greedy bit-exacta vs llama.cpp** (` Paris. In French, Paris is spelled "Paris", but`).

| Métrica | Valor |
|---|---|
| Decode | 0.24-0.26 tok/s (GLM5_GPU=1, sync path) |
| Prefill | 1.05 s/tok @ 138 tok; ~2.4 min fijos por chunk de 512 (GLM5_CHUNK=512) |
| Fast-path device-residente | 0.16 tok/s (correcto, listo para graphs; I/O-bound) |

## Arquitectura del modelo (`glm5-next`, referenciada de llama.cpp PR #27754)

45 capas trunk (block_count 46 incluye 1 capa MTP no usada):

- **34 capas KDA** (Kimi Delta Attention — gated delta rule lineal): conv1d causal
  (d_conv=4) sobre proyecciones Q/K/V [4096→8192], decay por dim
  `exp(sigmoid(-(g1·A))·(-5))`, beta por cabeza, estado S[i,j] 128×128 por cabeza,
  out-gate RMSNorm×sigmoid, wo [8192→4096]. Posiciones: kv_heads=0 en el array
  `head_count_kv` (capas 0-2, 4-6, ...).
- **11 capas MLA nope-only** (3,7,11,...,43): q_lora 1536 → q_b 64×256, kv_lora
  512 como cache K=V comprimido, wk_b/wv_b absorbidos (atención sobre latentes,
  escala 1/√256), indexer DSA k-pool 4 (top 2048 celdas de pools de 4 tokens).
  Sin RoPE (rope.dimension_count=0).
- **MoE** (capas 3-44; 0-2 dense lead): router F32 [4096,288] + bias, sigmoid
  gating, top-8, pesos normalizados ×2.5, experto compartido Q8, swiglu-clamp
  ±10. 288 expertos × 2048 ff.
- **mHC** (hyper-connections con Sinkhorn 4×4, hc=4): pre/post/comb por
  subcapa; reutiliza `ds41_hc.h` (el mHC de DeepSeek-V4.1 es idéntico).
- Q8_0 en TODAS las matrices grandes (ggml type 8; ojo: no confundir con Q8_1).

## Estructura del código

```
src/model/glm5/
├── glm5_config.{h,cpp}    load_config (probe arch "glm5-next"), roles por capa
├── glm5_model.{h,cpp}     TensorRef, matvec AVX2 (Q8_0/F16/BF16/F32/MXFP4), thread pool
├── glm5_forward.{h,cpp}   Session CPU (referencia exacta), forward_chunk (prefill
│                          por capas con unión de expertos), run_cli, sampling
├── glm5_gpu.{h,cu}        Motor GPU: residentes, LRU VRAM (ExpertCacheIndex de ds41),
│                          rings RAM + pool ReadFile, kernels Q8 (single/batch/grouped/
│                          tokens), stats. GLM5_GPU=1
└── glm5_gpu2.{h,cu}       Fast-path device-residente (Fase A del playbook ter):
                           forward completo en GPU, router on-device, tabla de
                           punteros de expertos. GLM5_FAST=1
```

## Flags

| Env | Default | Qué hace |
|---|---|---|
| GLM5_GPU | 0 | Motor GPU (residentes + expertos LRU + rings) |
| GLM5_FAST | 0 | Fast-path device-residente (paridad OK) |
| GLM5_CHUNK | 512 | Tamaño de chunk de prefill |
| GLM5_VRAM_GB | free-8.5 | Presupuesto del cache de expertos (VER techo WDDM abajo) |
| GLM5_RINGS | 96 | Buffers pinned del tier RAM |
| GLM5_FILL_THREADS | 8 | Workers de fill (ReadFile) |
| GLM5_PREFETCH | 1 | 1=warming PrefetchVirtualMemory, 2=+ring prefetch (mejor 0) |
| GLM5_PROFILE / GLM5_DBG_LAYERS / GLM5_DBG_CHUNK / GLM5_TRACE_EXPERTS | 0 | Diagnóstico |

## Hallazgos duros (por los que hubo que pasar)

1. **Techo WDDM del cache VRAM**: cache+residentes < ~21.5GB o Windows pagina la
   VRAM a RAM silenciosamente (19GB cache → 0.02 tok/s; 16GB → 0.05; 13.3GB OK).
2. **Router casi uniforme**: hit-rate del LRU ≈ fracción de capacidad (sin
   localidad explotable) → el prefetch layer-ahead no tiene señal; el page cache
   (48GB RAM) churnea (190GB/run) con hit 0.3-2%.
3. **El muro de I/O**: 45 capas × 24 slices × 8.5MB = 9.2GB/token; a 1.33 GiB/s
   (ReadFile efectivo) = 6.9s/token de floor. El disco hace 3.35-3.5 GB/s QD8 —
   la brecha es el camino CPU (por eso gpu-nvme-direct).
4. **Fill mmap-fault lee a 1.6 GB/s** vs ReadFile 2.9-3.5 (page-faults del archivo
   mapeado) — el motor usa ReadFile directo a rings pinned.
5. **Ley del prefill**: cada chunk lee ~331GB (la unión satura 288/capa con T≥36)
   → costo fijo ~2.4 min/chunk a 2.4 GB/s; chunks grandes amortizan.

## Bugs de port encontrados (todos con bisect de checkpoints vs CPU)

- conv weight layout: GGUF `[d_conv,1,DI]` = window-es-dim-rápida (`w[c*4+s]`), no
  `[d_conv][DI]`
- race: `attn_lat` compartido entre cabezas en parallel_for del attention MLA
- grid.y faltante en launches batcheados de expertos (NaN)
- overflow `d_x` en grouped (subía 64 grupos a buffer de 1)
- down-projection leía `x` compartido en vez de `xv` por-experto
- KDA fast-path: head-offset faltante en GDN, conv-shift guardaba post-conv en
  vez de la proyección cruda, dt-bias faltante en gates, overflow 8× de d_gate,
  scramble de índices en la tabla de expertos, shared leyendo d_x sobrescrito

## Referencias externas

- llama.cpp `src/models/glm5-next.cpp` (PR #27754) — el grafo exacto, línea por línea
- ter (`D:\projects\ter`) — playbook de optimización CUDA (graphs, dp4a, fusions)
- glm-flash-lite (sybil-solutions) — diseño de tiers (VRAM LRU + RAM + NVMe
  records + CPU lane), nuestros números de referencia de techo
- gpu-nvme-direct (`D:\projects\gpu-nvme-direct`) — GPU-issued NVMe reads
  (3.35 GB/s en este hardware), milestone 8 ya integrado para Llama streaming

## Próxima fase: `GLM5_LINUX_HANDOFF.md`

El plan gpu-nvme-direct con el mapa LBA del archivo original (cero repack de
341GB), backbone pack de 9.4GB, y probe de validación.

---

# ESTADO LINUX (Oct 22, 2026) — PARIDAD LOGRADA

## Qué corre hoy en Linux

**Paridad bit-exacta confirmada**: ` Paris. In French, Paris is spelled "Paris", but`
con `--token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512`.

| Métrica | Valor |
|---|---|
| Decode | ~0.45 tok/s (1008 fills × 2.9ms SERIALIZADOS por stream único del loader) |
| Prefill | 5.8 s/tok (chunked; el union load serializa igual) |
| Throughput del fill | 2.7–3.2 GB/s medidos (single-run 8.9MB, 18 cmds) |

El target 0.5–0.6 tok/s requiere eliminar la serialización: hoy cada
`gpunvme_load_layer` lanza UN kernel en UN stream dedicado → los fills se
ejecutan uno a la vez. Siguientes pasos de optimización: múltiples queues/streams
(o un solo kernel que atienda varios loads), y dejar que el motor solape fills
con compute (hoy `wait_ready` bloquea por ring).

## Setup del sistema (live USB Ubuntu 25.10, kernel 6.17)

1. **Discos**: C: (Micron) montado ro en `/media/ubuntu/cp1` (symlink `/media/ubuntu/cp`).
   D: = span LDM dinámico de Windows: **MONTABLE con `ldmtool create volume
   f2d76644-54af-11ee-85b3-8c882b08068c Volume1`** → ro en /mnt/d (el handoff
   lo daba por imposible: no lo es). E: (SN5000S, el raw store) → VFIO.
   BDFs: GPU `0a:00.0`, SN5000S `0b:00.0`, Micron `01:00.0`.
2. **Secure Boot**: desactivado en BIOS (OS Type: Other OS). Alternativa MOK:
   se enroló el cert (`/var/lib/shim-signed/mok/MOK.der`) y
   el driver 580.159.03 carga firmado, PERO el lockdown `integrity` de SB bloquea
   mmap de PCI sysfs y setpci → gpu-nvme-direct **necesita SB off sí o sí**.
3. **IOMMU**: `iommu=pt` en el cmdline del USB (`/cdrom/boot/grub/grub.cfg`,
   entry `persistent`). Sin eso el DMA del NVMe muere (dominio Translated).
4. **Driver**: nvidia-dkms-580-open + parche `os-mlock.c` (kernel 6.17 no tiene
   `follow_pfn`) — el parche lo aplica `scripts/setup_system.sh` (correrlo con
   `sed -i 's/\r$//'` antes: CRLF). CUDA toolkit 12.4 de apt (nvcc), host
   compiler gcc-13 (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13`).
5. **CRÍTICO tras cada rebind a vfio-pci**: `setpci -s 0b:00.0 COMMAND=0x0006`
   (Memory + Bus Master). Sin bus master el DMA no llega: el controller "arranca"
   (CAP/enable OK) pero el identify/cmds cuelgan o devuelven datos vacíos →
   logits NaN → salida basura ('!'×N). Leer COMMAND de vuelta para verificar.
   Tras remove/rescan o FLR el COMMAND se resetea: re-aplicar.
6. Estado del controller tras muchos rebinding: si el init cuelga en identify,
   re-secuencia: unbind vfio → bind nvme (3s) → unbind → bind vfio → setpci.

## Descubrimientos del port

- **El índice LBA de Windows quedó stale** (el fs movió extents). Regenerado en
  Linux con FIEMAP (`~/glm5/regen_lba_index_linux.py`) y validado 1003/1003
  contra lecturas raw. Cifras idénticas: 36,288 slices, 843 multi-run.
- **Los slices NO están alineados a 512B** en el archivo: cada tensor experto
  tiene `in_off = tensor_offset % 512` (0..511, constante por tensor). El fill
  lee desde el sector alineado (`lba0`) y el slice vive en `buf[in_off..]`.
  Los PRPs del loader exigen destino page-aligned: single-run → directo al ring;
  multi-run (2.3%) → scratch por worker + memcpy (ya normalizado, in_off=0).
- **El GGUF sí declara los expertos** (bug de substring en el análisis previo:
  `ffn_gate_exps` no contiene `.experts.`). Sus data están en el archivo, y el
  índice LBA apunta a ellos dentro del rango del archivo en el disco crudo.
- **Backbone verificado**: manifest contiguo 0 solapes/0 huecos, scales f16
  sanos; capas 0–2 (dense) producen estados normales → contenido correcto.
- **vram cache**: 13.71 GiB = 1652 slots de 8.5MiB.
- El `GGUFLoader` de ntransformer tolera los headers truncados (data beyond EOF
  = warning no fatal): el "GGUF fantasma" funciona sin tocar el loader.
- **GLM5_PREFETCH=2 ROMPE LA PARIDAD** ("Paris is pronounced "pa-REE"" vs
  "spelled"): carrera en el ring prefetch layer-ahead (sospechado: ring robado
  con in_off stale del fill previo). Usar GLM5_PREFETCH=1 (default): la config
  verificada con paridad exacta. El warm fadvise (1) es no-op en Linux.
- **GLM5_VRAM_GB no cambia el cache en este port** (13.71 GiB fijo: deriva del
  free VRAM al momento del init). Revisar la fórmula si se quiere más cache.
- **Pipeline depth del loader no es palanca**: pipe 8/16/32/48 = mismo GB/s
  (el SN5000S está al techo secuencial ~3.4GB/s, DRAM-less).
- **Decode 0.45–0.5 tok/s = techo físico del disco**: 9.2GB/token ÷ 3.4GB/s
  ≈ 2.7s puro, + ~20-25% hits reales del LRU entre tokens → 2.0-2.2s. El
  target 0.55 del handoff asumía 65% hits (imposible: router uniforme, ver
  hallazgos de Windows). Para superarlo: segundo NVMe (ancho de banda) o
  requant de expertos (rompe la paridad Q8_0).
- `bash scripts/setup_system.sh --phase 3` NO se detiene en la fase 3: sigue
  hasta VFIO/Build y puede bindear el disco equivocado (autodetect). Verificar
  BDF a mano (`lspci | grep SN5000` = `0b:00.0`; el Micron/C: es `01:00.0`).

## Layout en la máquina

```
~/src/ntransformer        fuentes (con integración gpnvme + pack mode)
~/src/gpu-nvme-direct     lib (build-hw/*.a con stream dedicado)
~/glm5/pack/              headers renombrados + symlinks backbone/manifest.bin
~/glm5/glm5_lba.bin       índice binario para el runtime ('LBA1')
~/glm5/glm5_backbone_manifest.bin ('MBN1')
~/glm5/expert_lba_index.json   índice regenerado (fuente de glm5_lba.bin)
~/glm5/probe3_vectors.json     vectores de verificación on-device (GLM5_VERIFY)
~/ntransformer-gpnvme     binario Release (verificado; rebuild: /dev/shm/ntr-build)
~/post_reboot.sh          restaura mounts + driver + VFIO tras reboot
~/probe ~/probe2 ~/probe3 ~/bench_slice   validadores standalone
```

## Cómo correr (paridad)

```bash
sudo env GLM5_BACKBONE_PACK=/home/ubuntu/glm5/pack \
         GLM5_GPNVME=1 \
         GLM5_LBA_BIN=/home/ubuntu/glm5/glm5_lba.bin \
         GPUNVME_PCI_BDF=0000:0b:00.0 \
         GLM5_GPU=1 \
  ~/ntransformer-gpnvme -m /home/ubuntu/glm5/pack/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf \
  --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512
# →  Paris. In French, Paris is spelled "Paris", but
```

Integración: `Model::tensor()` sintetiza bases fantasma deterministas para
`ffn_{gate,up,down}_exps.weight` (`glm5_gpnvme_fake_base(il,kind)`), y
`ExpertEngine::fill_slice` recupera (tensor, expert) por aritmética de
punteros y lee los runs LBA con `gpunvme_load_layer`. El H2D copia desde
`ring.buf + ring.in_off`. GLM5_VERIFY permite re-verificar fills on-device.
