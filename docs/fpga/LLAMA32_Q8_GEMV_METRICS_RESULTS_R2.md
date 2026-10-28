# Llama Q8: ciclos y ocupación del GEMV RTL

Los 84 trabajos y 2.688 salidas FP64 pasaron la revisión numérica y de contadores.
Usan slices de pesos Q8 originales de las capas 0/15 y activaciones INT8 sintéticas.
Son métricas de simulación; no hay frecuencia física validada ni tokens/s medidos.

| Semilla | Ciclos dentro de trabajos | Algún MAC ocupado | Espera Q8 | Bytes útiles/ciclo |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1006433 | 93.61% | 85.04% | 2.7675 |
| 7 | 1006599 | 93.59% | 85.03% | 2.7670 |
| 42 | 1006578 | 93.59% | 85.03% | 2.7671 |

Los seis estados particionan cada trabajo. Ocupación y esperas pueden solaparse:
sus porcentajes no deben sumarse. Bytes/ciclo excluye huecos entre trabajos y
no representa ancho de banda DDR físico. Los 28 jobs por semilla conservan
orden, geometría, bloques, beats AXI y escrituras de activaciones exactos.

Catorce corrupciones de registros/contadores fueron rechazadas. Se verificaron
fixture, testbench compilado y doce archivos HLS copiados contra el snapshot.
El JSON contiene min/mediana/max de ciclos por slice sobre las tres semillas;
esa muestra no representa una distribución de rendimiento general.

Tanda: `20261012T002948Z-V7-llama32-q8-gemv-slices-r1`. Revisión: `20261012T002948Z-V7-llama32-q8-gemv-slices-r1-independent-llama32-gemv-metrics-review-r2.json` en `build/cloud-setup/20261002-board/`.
SHA-256 de revisión: `ea3b4abc4fb61a4772d4907e195b97ba0f8f717f289c1067a054acf69f0b4499`.

Archivo validado: `8f2d768200f13a181bf3c7cd50c05548bf99ae58b5eef40e335fa702d42914c2`.

Esto no cierra proyecciones completas, atención, KV, logits ni inferencia en placa.
