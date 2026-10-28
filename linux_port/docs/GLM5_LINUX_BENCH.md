# BENCH & VERIFICACIÓN — GLM-5.3-Flash × gpu-nvme-direct en Linux

> Ejecutado: 21–22 Oct 2026. Ubuntu 25.10 live USB (kernel 6.17.0-5-generic),
> RTX 3090 + Ryzen 5800X + 48GB RAM. Driver NVIDIA 580.159.03 (DKMS parcheado,
> firmado MOK) + CUDA 12.4 (nvcc). SN5000S 512GB en vfio-pci (BDF 0000:0b:00.0).

---

## 1. Resultado principal: PARIDAD BIT-EXACTA

```
$ ntransformer -m <pack>/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf \
    --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512

 Paris. In French, Paris is spelled "Paris", but
```

Idéntico al golden de Windows (`GLM5_LINUX_HANDOFF.md`). Logrado con expertos
leídos del disco NVMe crudo (VFIO, sin filesystem) por el GPU, y los tensores
no-expertos desde el backbone pack de 9.8GB sobre NTFS3 read-only.

## 2. Números end-to-end

| Métrica | Windows (referencia) | Linux (este bench) |
|---|---|---|
| Paridad greedy | ✓ golden | ✓ **idéntico** |
| Throughput del fill (8.9MB slice, single-run) | 1.33 GiB/s (ReadFile CPU) | **2.7–3.2 GB/s** (GPU→NVMe) |
| Decode | 0.24–0.26 tok/s | **0.45–0.5 tok/s** (2.0–2.2 s/token) |
| Prefill chunked | ~1.6 min/chunk-512 (target) | **~29 s/chunk** (cualquier tamaño ≤512) |
| Prefill amortizado | 187 ms/tok | **~56 ms/tok** |
| Residentes en VRAM | 6.57 GiB backbone | 6.57 GiB ✓ |
| Cache de expertos VRAM | 13.3–13.7 GiB | 13.71 GiB = 1652 slots de 8.5MiB |

Comando de reproducibilidad:

```bash
sudo env GLM5_BACKBONE_PACK=/home/ubuntu/glm5/pack \
         GLM5_GPNVME=1 \
         GLM5_LBA_BIN=/home/ubuntu/glm5/glm5_lba.bin \
         GPUNVME_PCI_BDF=0000:0b:00.0 \
         GLM5_GPU=1 \
  ~/ntransformer-gpnvme -m /home/ubuntu/glm5/pack/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf \
  --token-ids 785,6722,315,9621,374 -n 12 -t 0 --repeat-penalty 1.0 -c 512
```

## 3. Throughput del loader (micro-bench)

`bench_slice` (gpunvme_load_layer, slice real 8,913,408B = 18 comandos de 512KB,
pipeline depth 8, 12–20 iteraciones):

| Config | Tiempo por slice | Throughput |
|---|---|---|
| single-run, pipe 8 | 2.6–2.9 ms | **2934–3216 MB/s** |
| pipe 16 / 32 / 48 | idéntico | 1.70–1.81 GB/s (avg bench c/ memset) |
| 4KB (1 cmd, probe2) | ~0.15 ms | latencia ~150µs |

**El pipeline depth NO es palanca** (8=16=32=48): el SN5000S (DRAM-less,
PCIe3×4) está en su techo secuencial ~3.4GB/s. QD adicional no escala.

El promedio del bench (1.7–1.8 GB/s) incluye el memset de verificación de 8.9MB
por iteración; el throughput real del loader es el del log interno (~3 GB/s).

## 4. Verificación del índice LBA (regenerado en Linux)

El índice de Windows quedó **stale** (el NTFS movió extents). Regenerado con
FIEMAP (`regen_lba_index_linux.py`, ioctl read-only, cero escrituras en E:):

| Verificación | Resultado |
|---|---|
| Estructura | 36,288 slices, 37,134 runs, 843 multi-run (97.7% contiguos) — idéntico al de Windows |
| Contenido vs disco raw (kernel) | **1003/1003 checks OK** (slices random × 16 bytes en 5 offsets) |
| On-device (GPU→NVMe, probe2, 4KB + in_off) | **30/30 OK** |
| On-device, slice completo 8.9MB (GLM5_VERIFY en-run, 5 offsets: 0 / 4KB / 512KB / 4MB / 8MB) | **0/5 bad por vector** |
| Modelo en disco vs hashes del golden | ✓ (los 3 probes del kit matchean el contenido del archivo) |

Formato binario para el runtime (`glm5_lba.bin`, 'LBA1'): por
(layer,kind,expert): `in_off` (offset del slice dentro del primer sector) +
runs `[lba_abs, nbytes]` contiguos. Los LBAs son absolutos del disco
(FIEMAP physical relativo a partición + 2048 sectores de offset).

## 5. Correctitud del modelo en pack mode

- **Backbone**: 1,283 tensores, contiguo sin solapes ni huecos (end = tamaño del
  archivo exacto), scales f16 de bloques Q8_0 con distribución sana → contenido
  correcto. Capas 0–2 (dense, solo backbone) producen estados normales.
- **Expertos**: sintetizados como bases fantasma deterministas
  (`0x500000000000 + (il*3+kind)*3GiB`); nunca dereferenciados en host.
  `ffn_down_` son **9 caracteres** (bug off-by-one rompía la clasificación y
  dejaba `data=nullptr` → `nullptr + e*slice` → SIGSEGV).
- **NaN check**: con fills corruptos los logits van a NaN completo (top-8 todos
  `logit=nan`, argmax degenerado a tokens contiguos `!`×N). Útil como síntoma:
  logits NaN ⇒ DMA/copiado de expertos roto.
- Estado intermedio por capa (`GLM5_DBG_LAYERS`): capas 0–2 ss normales
  (0.59 / 0.88 / 19.4), capa 3 +expertos → si los fills están mal, ss explota y
  NaN a partir de la capa 5–6.

## 6. Contadores del run (12 tokens, chunked)

```
[glm5-gpu] residents=6.57 GiB cache=13.71 GiB (1176/1652 slots) calls=714
           misses=1176 hits=17094 evict=0
[glm5-gpu] fills: disk=1176 (9.8 GiB)   ← prefill union + decode
[prefill] tokens=5 seconds=28.9 (chunked)
generated 12 tokens in 55.4s (0.22 tok/s total; decode puro ≈ 2.0–2.2 s/tok)
```

El decode mueve **9.2 GB de expertos por token** (42 capas × 8 expertos × 3
tensores × 8.9MB menos hits del LRU). Techo físico del disco:
9.2GB ÷ 3.4GB/s ≈ 2.7s/token (0.37 tok/s) sin hits; con los ~20–25% de hits
reales del LRU → **2.0–2.2s/token = 0.45–0.5 tok/s = techo de este hardware**.
El target 0.55 del handoff asumía 65% hits del LRU, pero el router es uniforme
(hallazgo de Windows): no hay localidad que explotar.

## 9. INCIDENTE (22-10-2026): llenado al disco equivocado

Durante el llenado del SSD de 256GB tras un hot-swap físico, el script tomó el
disco destino **por nombre de device** (`/dev/nvme0n1`) sin verificar identidad.
Por la re-enumeración del hot-swap, ese nombre era el **Micron (C: de Windows)**
→ los primeros ~150GB del Micron quedaron sobreescritos con slices de expertos.

- El SN5000S (raw store) quedó **intacto** (golden verificado por 3 métodos).
- El SN520 no recibió escrituras.
- Daño: contenido de 0–150GB del Micron (C:\Windows + archivos al inicio del
  volumen NTFS). La MFT del NTFS (~centro del volumen) está fuera de la zona →
  con el Micron reconectado: testdisk para reconstruir la partición y listar
  lo recuperable (todo lo que viva por encima de ~150GB).
- **Precaución aplicada**: `fill_second_disk.py` ahora exige `DST_SERIAL` del
  destino (verificación por serial, nunca por nombre) y aborta si no coincide.
- Regla definitiva: **cualquier escritura cruda a un disco exige serial
  verificado**; los nombres nvmeXnY no sobreviven a un hot-swap.

## 10. ESTADO FINAL (cierre del día 2) — race del chunked cazado por bisección

**Hallazgo principal del día 2 (cazado por bisección de chunk-size)**:

| GLM5_CHUNK | Fills del union/capa | Paridad (12 tokens, -t 0) |
|---|---|---|
| 512 | 102 | **1/6 exacto** (5 divergen, 3 salidas distintas) |
| 3 | ~24 | **3/3 exactos** |
| 2 | ~16 | **6/6 exactos** |
| 1 (token-a-token) | 24 | **6/6 exactos** |

⇒ **La divergencia intermitente escala con los fills concurrentes del union
load del prefill chunked** (race: evict-vs-H2D-in-flight o ring con fills
múltiples — los slots del cache reciben el slice equivocado bajo presión).
El fill per-se es determinista y correcto (FILL-HASH 0 mismatches entre
re-fills post-evict; SLOT-VERIFY muestreado y 300 checks on-device OK).

**Workaround verificado**: `GLM5_CHUNK=3` (o 2): paridad 12/12 en todas las
corridas. Costo: el union se recarga por chunk (para prompts largos el prefill
encarece; para el decode y prompts cortos: sin impacto). El fix de raíz es en
el eviction path del ds41 cache (proteger slots con H2D in-flight) — próximo
turno.

- **SN520 descartado como disco de expertos**: PCIe3×2 (~1.7GB/s) — como disco 1
  daría 0.26 tok/s (peor que el disco solo). Además el gpunvme cuelga en su
  primer fill (requeriría debugging específico del controlador SN520).
- **Config operativa**: SN5000S solo, `GLM5_CHUNK=3`, paridad exacta verificada.
  Decode 0.45–0.5 tok/s = techo del disco.
- **Micron (C:) dañado** (ver incidente §9): fuera del equipo. Recuperación:
  testdisk con el disco reconectado; MFT fuera de la zona dañada. La copia de
  linux_kit vive en `~/kit-docs`.

## 7. Hallazgos de ingeniería (port y entorno)

### Transporte (gpu-nvme-direct + driver)

1. **Bus Master tras cada rebind a vfio-pci**: `setpci -s 0b:00.0 COMMAND=0x0006`
   y **leer de vuelta**. Sin bus master el doorbell MMIO funciona (CAP/enable OK)
   pero ningún DMA llega: identify cuelga, o fills "exitosos" con buffer vacío →
   logits NaN → salida `!`×N. FLR y remove/rescan resetean COMMAND → re-aplicar.
2. **Secure Boot off obligatorio**: con SB on, el kernel en lockdown
   `[integrity]` bloquea mmap de PCI sysfs (BAR0) y setpci → la lib no puede
   operar. El driver firmado con MOK sí carga con SB on, pero no alcanza.
3. **iommu=pt obligatorio**: sin eso el dominio Translated mata el DMA
   (timeout del CQE, `error=2` en el kernel del loader). El cmdline va en el
   grub.cfg del USB (`persistent iommu=pt`).
4. **CQE status 0x4013** (Data Transfer Error, DNRR) = PRP desalineado o buffer
   inválido; **0x0000 + timeout** en cascada = el kernel abandona dejando el SQ
   con comandos huérfanos (desincroniza la cola). Prevención: no enviar dst
   no page-aligned al loader.
5. **Stream dedicado** para `gpu_layer_read` (no legacy default): el kernel del
   loader se serializaba con los streams del motor. Parcheado en
   `layer_loader.cu` (cudaStreamCreateWithFlags non-blocking + stream sync).

### Índice y datos

6. **Los slices de expertos NO están alineados a 512B**: `in_off =
   tensor_offset % 512` (0..511, constante por tensor; 352 es el más común).
   El fill lee desde `lba0` (sector alineado) y el slice vive en `buf[in_off..]`.
   El H2D del engine copia desde `ring.buf + ring.in_off`.
7. **PRP page-alignment**: el loader arma PRPs por página 4KB desde el destino →
   destino debe ser page-aligned. Single-run (97.7%) → ring directo (alineado);
   multi-run (843 slices, 2.3%) → scratch por worker + memcpy con in_off, y
   `ring.in_off = 0` al final (el copy ya normaliza).
8. **El índice de Windows quedó stale** por movimiento de extents del NTFS
   (verificado: 2/3 probes del kit fallaban por LBA viejo; los extents actuales
   vía FIEMAP = LBA viejo − 2048 para los NO movidos). El regen de Linux produce
   cifras idénticas de estructura.
9. **El GGUF sí declara los expertos** (129 = 43 capas × 3, incluye la capa MTP
   blk.45). Análisis previo que decía "0 expertos" era un bug de substring
   (`.experts.` vs `_exps.`). Sus dims/types sirven para los TensorRef del
   pack mode sin hardcodear shapes.
10. **El backbone pack está íntegro**: 1,283 tensores contiguos, sin solapes,
    9,840,532,856 bytes = tamaño del archivo exacto.

### Código (ntransformer)

11. **`ffn_down_` = 9 caracteres** — el `compare(len=8)` dejaba el down_exps con
    `data=nullptr` y el submit armaba `nullptr + e*slice` → SIGSEGV garantizado
    en la capa 3.
12. `run_cli` hace un `GGUFLoader peek.load(path)` previo (detección de
    arquitectura): explica la doble carga del modelo en los logs. Inofensivo.
13. `scripts/setup_system.sh --phase N` no se detiene en la fase N: ejecuta
    hasta VFIO/Build con autodetección de BDF (puede bindear el disco del
    sistema). Además tiene CRLF: `sed -i 's/\r$//'` antes de correrlo.
14. **GLM5_PREFETCH=2 rompe la paridad** (carrera en el ring prefetch
    layer-ahead; texto diverge a "pronounced "pa-REE""). GLM5_PREFETCH=1
    (default) es la config verificada. Pendiente: cazar la carrera si se quiere
    el overlap.
15. **GLM5_VRAM_GB no afecta el cache en este port** (13.71 GiB derivan del
    free VRAM al init). Revisar fórmula si se quiere más cache.
16. El `GGUFLoader` tolera headers truncados (data beyond EOF = warning no
    fatal, punteros lazy): el "GGUF fantasma" no requirió tocar el loader.
17. `ldmtool` monta el span dinámico de Windows (D:) que el handoff daba por
    irrecuperable: `ldmtool create volume f2d76644-...-8c882b08068c Volume1`.

## 8. Techo físico y salidas para superarlo

```
techo I/O puro:   9.2 GB/token ÷ 3.4 GB/s = 2.7 s/token = 0.37 tok/s
con hits LRU:     ~20–25% → 2.0–2.2 s/token  = 0.45–0.5 tok/s  ← ESTAMOS AQUÍ
```

Vías reales para más tok/s (en orden de impacto):

1. **Segundo NVMe** para los expertos (stripe por slice): 2× ancho de banda →
   ~0.9–1.0 tok/s. Los WD SATA (sda/sdb) no sirven (500MB/s).
2. **NVMe PCIe4** (7GB/s) → ~1.5–1.9 tok/s con el mismo software.
3. **Requant de expertos** (Q8_0→Q5_K/Q4_K): −35–50% de bytes → 0.7–0.9 tok/s
   desde el mismo disco, pero **rompe la paridad** con el modelo Q8_0 original.
4. Eliminar los bubbles compute↔I/O (el compute GPU no-I/O es <15% del token;
   ganancia máx ~10% y ya la cubren los hits).

El código de la integración está en `~/src/ntransformer`
(`glm5_gpu.cu: fill_slice rama gpnvme + GLM5_VERIFY`,
`glm5_model.cpp: pack mode`), fuentes persistentes; binario verificado en
`~/ntransformer-gpnvme`; restore del entorno en `~/post_reboot.sh`.
