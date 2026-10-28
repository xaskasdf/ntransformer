# Camino KV: resultados XSim y síntesis OOC

Estado al 2026-10-13, 18:19 UTC: cargador, append, caché, RAM privada y
camino KV tienen PASS revisado y respaldo D:/NFS. Son pruebas contra peers
de memoria/protocolo; el motor con aritmética de atención se valida después.

## Recursos obtenidos

Vivado 2026.1, `xcvu5p-flva2104-2-i`, reloj solicitado de 5 ns.
Los valores corresponden a cada top completo: **no sumarlos**, porque caché
incluye cargador/append y el camino KV incluye caché/RAM.
BRAM está expresado en tiles equivalentes de 36 Kib.

| Top | LUT | FF | BRAM | DSP | WNS, ns | WHS, ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cargador KV | 1.300 | 2.013 | 0 | 2 | +2,572 | +0,066 |
| Append | 998 | 1.131 | 7,5 | 1 | +2,667 | +0,065 |
| Caché | 2.796 | 3.923 | 7,5 | 3 | +2,435 | +0,062 |
| RAM privada | 912 | 398 | 16 | 0 | +1,125 | +0,066 |
| Camino KV | 3.971 | 4.474 | 23,5 | 3 | +0,394 | +0,046 |

Los cinco tops tienen cero cajas negras y latches; los checks de clocks,
loops y endpoints internos pasan. Las E/S externas quedan sin delays en
estos proyectos OOC. El margen del camino KV es menor que el de sus piezas:
requiere revisión al integrar contexto/HLS y después colocación/ruteo.
Estos resultados no califican DDR física ni timing ruteado.

## Cobertura del camino

Cuatro suites: semillas 1/7/42 con watchdog 32 y semilla 1 con watchdog
65.536. Cada una verifica 70 consultas, cinco resets, 78 appends y
67.364 palabras K/V leídas de la RAM privada. Los bancos registran errores,
stalls, cancelación, epoch y solapamiento de append/lectura.
Los ciclos totales fueron 70.209, 70.375, 70.000 y 135.651.
No son una medida de tokens/s ni latencia de atención.

## Archivos verificables

Cada revisión independiente está en `build/cloud-setup/20261002-board/`;
los certificados y archivos están en `artifacts/ntrpu-board-vivado/`, con
almacenamiento local en D: y copia NFS.

| Etapa / run | SHA-256 del archivo |
| --- | --- |
| Append `20261013T175629Z-V7-llama32-kv-append-vivado-r1` | `5cf5c4430449194b013fb0540a03f5cdb6d64a8068a9a7d1dd89a47f701fda2a` |
| Caché `20261013T180115Z-V7-llama32-kv-cache-vivado-r1` | `aeb79304428643e8553ab025e5a51ebd89df6a1cd27d86970f68649f40bb1a34` |
| RAM `20261013T180601Z-V7-llama32-attention-kv-ram-vivado-r1` | `559f3622fc7d538084870a30c9a17a6bd37b951394ab6000a7dee64a9bc8b032` |
| Camino `20261013T181003Z-V7-llama32-attention-kv-path-vivado-r1` | `1098194f4f986821528e32e31928d4ff7288c09a1a316a8c3fd4cd1b5a2854ed` |

Para el cargador y la adopción de su revisión, consultar
[su resultado específico](LLAMA32_KV_LOADER_XSIM_SYNTH_RESULTS_R1.md).
