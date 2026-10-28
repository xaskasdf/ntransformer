# Llama 3.2 1B Q8: primer banco RTL con pesos reales

Revisado el 2026-10-12 UTC. La tanda
`20261012T002948Z-V7-llama32-q8-gemv-slices-r1` pasó **84 trabajos y
2.688 salidas FP64 exactas**. Se comparan las primeras y últimas 32 filas
de Q/K/V/O/gate/up/down en las capas 0 y 15, con semillas 1/7/42.
Las activaciones INT8 y sus escalas son sintéticas; no son estados ocultos
obtenidos de inferencia del modelo.

## Pesos y alcance

Se usa el [GGUF Q8_0 auditado](LLAMA32_Q8_REAL_WEIGHTS_R1.md), revisión
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`, conservando los bloques originales.
SHA-256 del modelo:
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
Cada semilla procesa 81.920 bloques y 2.785.280 bytes de pesos.

| Forma | Slices por semilla | Ciclos observados, tres semillas |
| --- | ---: | ---: |
| 32 × 2048 | 24 | 25.078–25.277 |
| 32 × 8192 | 4 | 100.386–100.833 |

La revisión independiente comprueba valores, orden, geometría, ráfagas,
límites de página y pausas. La [auditoría de contadores](LLAMA32_Q8_GEMV_METRICS_RESULTS_R2.md)
reconcilia las seis fases por trabajo, bytes AXI, bloques y ocupación;
rechaza catorce alteraciones y verifica los doce archivos HLS copiados.

## Decisión de diseño

Al menos uno de los dos MAC está ocupado durante 93,59–93,61%; la entrada
Q8 espera 85,03–85,04% de los ciclos. El caudal es 2,7670–2,7675 bytes
útiles por ciclo de trabajo. Los contadores se solapan y no deben sumarse.
La condición RTL de aceptación depende de que el MAC seleccionado esté
libre; estos resultados señalan al despacho/cálculo como siguiente objetivo
de optimización de este banco. No justifican dimensionar más canales DDR.

Se conserva el núcleo dual mientras se valida la FFN completa. Un futuro
experimento con mayor concurrencia deberá preservar el orden K de cada
acumulador, cubrir filas impares, errores/reset y comparar coste de BRAM,
DSP y timing. La mejora se decidirá con mediciones; aquí no se atribuye
aceleración a una variante aún no ejecutada.

## Evidencia y límites

Revisión numérica: `68d1ee1b28fb2850c1e6c173f1b85670a4aab11aae6b3bf705e3814c52570b80`.
Revisión de contadores R2: `ea3b4abc4fb61a4772d4907e195b97ba0f8f717f289c1067a054acf69f0b4499`.
Archivo de tanda: `8f2d768200f13a181bf3c7cd50c05548bf99ae58b5eef40e335fa702d42914c2`.
Fuente: `148b8a2e52328137975c661e12af66d3c6dc71ec32ea3e1929e9be51af0be661`.
El archivo y todos sus miembros se verificaron en D:; la revisión suplementaria
también tiene cápsula local/NFS. R1 de ese auditor falló al resolver un
nombre interno del ZIP como ruta del repositorio; R2 corrige la resolución
sin cambiar el cálculo, los pesos ni la simulación.

Faltan proyecciones completas, FFN real, atención/KV, logits e inferencia
en placa. Los ciclos anteriores no acreditan frecuencia física, ancho de
banda DDR ni tokens/s.
