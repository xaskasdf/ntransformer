# GLM-5.3-Flash en NTransformer — documentación del proyecto

> Cronología completa del port + optimización. Trabajo en Windows (Oct 2026,
> build MSVC+CUDA 13.2) e integración gpu-nvme-direct en Linux (Ubuntu 25.10 live,
> 21–22 Oct). Bench y verificación Linux: `GLM5_LINUX_BENCH.md`. El puente entre
> ambos sistemas fue `D:\linux_kit` (Linux no puede vivir en el volumen del modelo:
> el disco de expertos queda en VFIO para gpu-nvme-direct, y el kernel del live
> USB es distinto).

## Qué corre hoy

**GLM-5.3-Flash** (zai-org, 320B total / 18B activo, MoE 288-experto, Q8_0 341GB)
en ntransformer con **paridad greedy bit-exacta vs llama.cpp**
(` Paris. In French, Paris is spelled "Paris", but`).

Hardware: RTX 3090 24GB + Ryzen 5800X + 48GB RAM; modelo en WD SN5000S 512GB
(DRAM-less, PCIe3×4, ~3.4 GB/s secuencial).

| Métrica | Windows nativo | Linux × gpu-nvme-direct |
|---|---|---|
| Decode | 0.24-0.26 tok/s (`GLM5_GPU=1`, sync path) | **0.45-0.5 tok/s** (techo del disco) |
| Prefill | 1.05 s/tok @ 138 tok; ~2.4 min fijos por chunk de 512 | **~29 s/chunk** (≈56 ms/tok amortizado en chunk 512)¹ |
| Fill de expertos | 1.33 GiB/s (ReadFile) | **2.7-3.2 GB/s** (GPU→NVMe VFIO) |
| Fast-path device-residente (`GLM5_FAST=1`) | 0.16 tok/s (correcto, I/O-bound) | — |
| Paridad greedy | ✓ | ✓ bit-exacta (con `GLM5_CHUNK=3`, ver abajo) |
| Residentes VRAM | 6.57 GiB | 6.57 GiB |
| Cache expertos VRAM | 13.3 GiB | 13.71 GiB (1652 slots de 8.5 MiB) |

¹ Medido sobre un único chunk (prompt de 5 tokens): el costo del chunk lo domina
la carga del union de expertos, no el número de tokens. Con chunks grandes hay
una carrera en el union load (`GLM5_LINUX_BENCH.md` §10): la config operativa
verificada es `GLM5_CHUNK=3` (default de `run_glm5.sh`), que encarece el
prefill de prompts largos (no medido).

> **Estado del código**: la integración gpnvme (rama gpnvme de `fill_slice`,
> pack mode `GLM5_BACKBONE_PACK`, índice `GLM5_LBA_BIN`) está **mergeada** en
> `src/model/glm5/`. Runbook, índices LBA, scripts de la sesión Linux
> (`run_glm5.sh`, `post_reboot.sh`, `regen_lba_index_linux.py`, ...) y el patch
> de gpu-nvme-direct: `linux_port/`. Herramientas del kit (lado Windows):
> `tools/linux/`. Ver hallazgo #8 abajo: el índice LBA de Windows queda stale
> con el uso normal de E: — regenerar SIEMPRE con FIEMAP en Linux antes de correr.

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
| GLM5_PREFETCH | 1 | 1=warming PrefetchVirtualMemory (no-op en Linux), 2=+ring prefetch — **2 rompe la paridad** |
| GLM5_GPNVME | 0 | (Linux, `USE_GPUNVME`) expertos leídos del NVMe crudo por el GPU; requiere `GLM5_LBA_BIN` + `GPUNVME_PCI_BDF` |
| GLM5_BACKBONE_PACK | — | Directorio del pack: tensores no-expertos desde `backbone.bin`+manifest; exige `GLM5_GPNVME=1` |
| GLM5_LBA_BIN | — | Índice binario de slices de expertos (`LBA1` un disco / `LBA2` stripe de dos discos) |
| GPUNVME_PCI_BDF / GPUNVME_PCI_BDF1 | — | BDF del disco de expertos / segundo disco para `LBA2` |
| GLM5_VERIFY / GLM5_VERIFY_SLOTS / GLM5_FILL_HASH | — | Verificación on-device de fills (vectores probe3), muestreo de slots, hash de fills |
| GLM5_PROFILE / GLM5_DBG_LAYERS / GLM5_DBG_CHUNK / GLM5_TRACE_EXPERTS | 0 | Diagnóstico |
| GLM5_DBG_IDS / GLM5_DBG_FAST / GLM5_DUMP_TOKENS / GLM5_MAX_LAYERS / GLM5_GPU_MOE_OFF | — | Diagnóstico (ids del router, fast-path, tokens, truncar capas, MoE en CPU) |

## Hallazgos duros (por los que hubo que pasar)

**Linux/gpnvme** (detalle completo en `GLM5_LINUX_BENCH.md`):
0. **Techo físico alcanzado**: 9.2 GB de expertos/token ÷ 3.4 GB/s del SN5000S
   = 0.37 tok/s sin hits; con ~20-25% hits del LRU → 0.45-0.5 tok/s medidos.
   Vías reales para más: 2° NVMe en stripe (~0.9-1.0), NVMe PCIe4 (~1.5-1.9).
1. El registro WD `0x84.W=0x0008` + `COMMAND=0x0006` con read-back es
   OBLIGATORIO tras CADA rebind vfio (sin él: DMA silenciosamente muerto →
   NaN → salida `!`×N).
2. Slices de expertos NO alineados a 512B: `in_off = tensor_offset % 512`
   (352 típico); el fill lee desde sector alineado y el H2D copia con offset.
3. `iommu=pt` + Secure Boot off + bus master: sin la combinación exacta el
   DMA no funciona (CQE timeout, status 0x4013 = PRP desalineado).
4. El loader original serializaba en el legacy stream — parcheado con stream
   dedicado en layer_loader.cu.
5. GLM5_PREFETCH=2 rompe paridad (carrera del ring prefetch) — usar 1.
6. `ffn_down_` tiene 9 caracteres (off-by-one del compare → SIGSEGV capa 3).
7. `ldmtool` monta el span dinámico de Windows (D:) que dábamos por perdido.
8. **El índice LBA generado en Windows queda stale** por movimiento natural de
   extents del NTFS — regenerar con FIEMAP en Linux (read-only) antes de cada
   sesión; verificar con los 30 probes on-device.
8b. **Carrera del union load con chunks grandes** (bisección por chunk-size,
   BENCH §10): `GLM5_CHUNK=512` → 1/6 corridas exactas; 3 → 3/3; 2 y 1 → 6/6.
   El fill en sí es determinista (FILL-HASH 0 mismatches). Workaround
   `GLM5_CHUNK=3`; fix de raíz pendiente en el eviction path del cache ds41
   (proteger slots con H2D en vuelo).
8c. **Escrituras crudas a disco solo con serial verificado**: un fill por
   nombre de device (`/dev/nvme0n1`) tras un hot-swap sobreescribió ~150GB del
   disco de boot de Windows (BENCH §9). `fill_second_disk.py` exige `DST_SERIAL`.
8d. SN520 (PCIe3×2) descartado como segundo disco de expertos: lento (~1.7 GB/s)
   y gpu-nvme-direct cuelga en su primer fill.

**Windows**:
9. Techo WDDM del cache VRAM: cache+residentes < ~21.5GB o Windows pagina la
   VRAM a RAM silenciosamente (19GB cache → 0.02 tok/s; 16GB → 0.05; 13.3GB OK).
10. Router casi uniforme: hit-rate del LRU ≈ fracción de capacidad, sin
   localidad → el prefetch layer-ahead no tiene señal; el page cache (48GB RAM)
   churnea (190GB/run) con hit 0.3-2%.
11. Fill mmap-fault 1.6 GB/s vs ReadFile 2.9-3.5 (page-faults del mmap) — el
   motor usa ReadFile directo a rings pinned.
12. Ley del prefill: cada chunk lee ~331GB (unión satura a 288/capa con T≥36).
13. Muro de I/O en Windows: 45 capas × 24 slices × 8.5MB = 9.2GB/token; a
   1.33 GiB/s efectivo (ReadFile) = 6.9 s/token de piso. El disco da 3.35-3.5
   GB/s QD8: la brecha es el camino CPU — por eso gpu-nvme-direct en Linux.

## Bugs de port (todos con bisect de checkpoints vs CPU)

- conv weight layout: GGUF `[d_conv,1,DI]` = window-es-dim-rápida (`w[c*4+s]`)
- race: `attn_lat` compartido entre cabezas en parallel_for del attention MLA
- grid.y faltante en launches batcheados de expertos (NaN)
- overflow `d_x` en grouped (subía 64 grupos a buffer de 1)
- down-projection leía `x` compartido en vez de `xv` por-experto
- KDA fast-path: head-offset faltante en GDN, conv-shift guardaba post-conv,
  dt-bias faltante, overflow 8× de d_gate, scramble de índices de la tabla de
  expertos, shared leyendo d_x sobrescrito

## Referencias externas

- llama.cpp `src/models/glm5-next.cpp` (PR #27754) — el grafo exacto, línea por línea
- ter (`D:\projects\ter`) — playbook de optimización CUDA (graphs, dp4a, fusions)
- glm-flash-lite (sybil-solutions) — diseño de tiers (VRAM LRU + RAM + NVMe
  records + CPU lane), nuestros números de referencia de techo
- gpu-nvme-direct (`D:\projects\gpu-nvme-direct`) — GPU-issued NVMe reads
  (3.35 GB/s en este hardware), milestone 8 ya integrado para Llama streaming

## Estado Linux — setup y operación (live USB Ubuntu 25.10, kernel 6.17)

1. **Discos**: E: (SN5000S, el raw store = el GGUF original en su lugar) → VFIO,
   BDF `0000:0b:00.0`. D: = span LDM dinámico de Windows (projects + linux_kit),
   montable read-only con `ldmtool create volume <guid> Volume1` + `ntfs3`.
   GPU en `0a:00.0`. Verificar SIEMPRE el BDF a mano (`lspci`): el disco de
   boot de Windows también es NVMe.
2. **Secure Boot off**: con SB on, el lockdown `integrity` bloquea mmap de PCI
   sysfs (BAR0) y setpci. El driver firmado con MOK carga, pero no alcanza.
3. **IOMMU**: `iommu=pt` en el cmdline del USB (entry `persistent`). Sin eso el
   dominio Translated mata el DMA del NVMe.
4. **Driver/toolchain**: nvidia-dkms-580-open (580.159.03) + parche `os-mlock.c`
   (kernel 6.17 no tiene `follow_pfn`; lo aplica `scripts/setup_system.sh`,
   correrlo tras `sed -i 's/\r$//'` por CRLF). CUDA 12.4 de apt (nvcc), host
   compiler gcc-13: `cmake -DUSE_GPUNVME=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13
   -DNTRANSFORMER_BUILD_NTRPU_SIM=OFF`.
5. **Tras cada rebind a vfio-pci**: `setpci 84.W=0x0008` + `COMMAND=0x0006` con
   read-back de ambos. FLR o remove/rescan resetean COMMAND. Si el init cuelga
   en identify: unbind vfio → bind nvme (3s) → unbind → bind vfio → setpci.
   `linux_port/tools/run_glm5.sh` automatiza wake + rebind + registros + exec.

Layout en el host Linux:

```
~/src/ntransformer        fuentes (integración gpnvme + pack mode, ya en el repo)
~/src/gpu-nvme-direct     lib (build-hw/*.a con stream dedicado: linux_port/gpunvme_patch)
~/glm5/pack/              headers renombrados + symlinks backbone/manifest.bin
~/glm5/glm5_lba.bin       índice binario para el runtime ('LBA1')
~/glm5/glm5_backbone_manifest.bin ('MBN1')
~/glm5/expert_lba_index.json   índice regenerado con FIEMAP (fuente de glm5_lba.bin)
~/glm5/probe3_vectors.json     vectores de verificación on-device (GLM5_VERIFY)
~/ntransformer-gpnvme     binario Release verificado
~/post_reboot.sh          restaura mounts + driver + VFIO tras reboot
```

Cómo correr (paridad):

```bash
sudo bash ~/run_glm5.sh --token-ids 785,6722,315,9621,374 -n 12 -t 0 \
  --repeat-penalty 1.0 -c 512
# equivalente explícito (run_glm5.sh además fija GLM5_CHUNK=3):
sudo env GLM5_BACKBONE_PACK=/home/ubuntu/glm5/pack GLM5_GPNVME=1 \
         GLM5_LBA_BIN=/home/ubuntu/glm5/glm5_lba.bin GPUNVME_PCI_BDF=0000:0b:00.0 \
         GLM5_GPU=1 GLM5_CHUNK=3 \
  ~/ntransformer-gpnvme -m /home/ubuntu/glm5/pack/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf \
  --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512
# →  Paris. In French, Paris is spelled "Paris", but
```

Mecanismo: `Model::tensor()` sintetiza bases fantasma deterministas para
`ffn_{gate,up,down}_exps.weight` (`glm5_gpnvme_fake_base(il,kind)`, nunca
dereferenciadas en host), y `ExpertEngine::fill_slice` recupera
(tensor, experto) por aritmética de punteros y lee los runs LBA con
`gpunvme_load_layer`. El H2D copia desde `ring.buf + ring.in_off`. El
`GGUFLoader` tolera los headers truncados del "GGUF fantasma" sin cambios.

## Próxima fase

- `GLM5_LINUX_HANDOFF.md` — cómo retomar en Linux desde `D:\linux_kit`.
- `GLM5_GPU_UNLOCK.md` — medir gpu-unlock (BF16/TF32 en GA102) contra este baseline.
- Palanca real de decode: segundo NVMe rápido en stripe (`LBA2`, ~0.9-1.0 tok/s
  proyectado) o NVMe PCIe4 (~1.5-1.9). Requant de expertos rompe la paridad Q8_0.
- Fix de raíz de la carrera del union load (§8b) para volver a chunks grandes.
