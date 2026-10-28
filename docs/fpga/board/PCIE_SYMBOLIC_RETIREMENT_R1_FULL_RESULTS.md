# R1: regresión completa de retiro aprobada

Resultado del 2026-10-09: **182529 filas originales en catorce bancos**,
más **277 filas dirigidas**. Se reutilizan 23243 filas de la prueba corta
con las mismas diecinueve fuentes SV y hashes de vectores; esta tanda
ejecutó las 159563 positivas restantes y nueve mutantes reales.

## Estado y origen

Se mantiene la comparación independiente con R7 posterior al flanco,
con cinco posiciones de historia y cinco flancos de vaciado. Incluye
410 bits públicos, control de 391 bits, guardas, validez, secuencia y
obligación del origen. Las comprobaciones R7/modelo/S1/C/E/D/A/B y del
validador tardío siguen activas. Los campos solapados no se suman.

| Banco original | Filas | Retiradas | Canceladas por frío |
| --- | ---: | ---: | ---: |
| `directed` | 217 | 114 | 71 |
| `controls` | 352 | 150 | 142 |
| `drained_main` | 22674 | 22203 | 335 |
| `main` | 22339 | 21868 | 335 |
| `ownership` | 46356 | 19257 | 19355 |
| `rq` | 8850 | 7177 | 1193 |
| `prefix` | 544 | 129 | 295 |
| `lanes` | 984 | 345 | 455 |
| `context` | 1047 | 926 | 85 |
| `drained_ownership` | 65711 | 38612 | 19355 |
| `drained_rq` | 10045 | 8370 | 1195 |
| `drained_prefix` | 839 | 424 | 295 |
| `drained_lanes` | 1439 | 800 | 455 |
| `drained_context` | 1132 | 1011 | 85 |

Los seis bancos con vaciado previo al reset cancelan **cero registros
de tráfico original**. Las cancelaciones de registros de vaciado y de
los bancos sin vaciado se contabilizan expresamente.

## Persistencia del fallo de bytes

256 lecturas distintas llevan el estado a **1056760 bytes** en el origen
259. Una terminación retira 8188: vuelve a **1048572 bytes** en el origen
260, bajo el máximo 1052672. El bit 13 persiste durante tres observaciones
en rango, con el bit 13 temprano siempre a cero. R publica los mismos
orígenes cinco flancos después. El reset frío de la fila 272 borra bytes
y fallos de ambos diseños.

El mutante que elimina exclusivamente la persistencia del bit 13 falla
en fila 265/origen 260: XOR de estados igual a 8192. El mutante que conserva
fallos durante frío falla en fila 272. Otros siete detectan errores de
origen, habilitación, doble suma, reset y representación/realimentación.

| Mutante | Detección | Fila | Origen |
| --- | --- | ---: | ---: |
| `drop_byte_sticky` | `RETIRE_STATE_MISMATCH` | 265 | 260 |
| `keep_fault_cold` | `RETIRE_COLD_MISMATCH` | 272 | — |
| `wrong_origin` | `RETIRE_ORIGIN_MISMATCH` | 7 | 2 |
| `ignore_enable` | `RETIRE_GUARD_MISMATCH` | 7 | 2 |
| `double_add` | `RETIRE_STATE_MISMATCH` | 9 | 4 |
| `omit_cold` | `RETIRE_COLD_MISMATCH` | 13 | — |
| `omit_sticky` | `RETIRE_STATE_MISMATCH` | 265 | 260 |
| `omit_high` | `RETIRE_GUARD_MISMATCH` | 22399 | 22394 |
| `wide_feedback` | `RETIRE_GUARD_MISMATCH` | 22400 | 22395 |

## Evidencia y límites

La revisión enlaza logs, scripts compilados, vectores y archivos sellados;
rechaza nueve corrupciones de los nuevos testigos. Conserva los trece
controles de contabilidad de la revisión corta. Solo aparecieron las
21 advertencias XSIM, las 21 de declaración de `rq_held` ya conocidas y
una de declaración anticipada de `byte_range_high` en el mutante de
realimentación ancha. La revisión R2 exige esa línea y ese caso exactos;
el revisor R1 y su diagnóstico se conservan, sin repetir simulaciones.
Fuentes y resultados están verificados en D: y NFS.

Siguiente: [síntesis propia y reproducción binaria](PCIE_SYMBOLIC_RETIREMENT_SYNTHESIS_PLAN_R1.md).
Siguen pendientes quiet, integración del monitor, mediciones conjuntas,
cierre físico y las validaciones nativas y de placa.

- Tanda: `20261009T175903Z-V4-ffn-supervised-pcie-symbolic-retirement-full-r1`.
- Archivo D:/NFS: SHA-256 `c54e2e97abe5cb4eeadf586736715d9e94530cb4817059012879113fc072df0e`.
- Revisión: `20261009T175903Z-V4-ffn-supervised-pcie-symbolic-retirement-full-r1-independent-symbolic-retirement-full-review.json`, SHA-256 `84dcac60bc8a50d064e3ea61237ba0dd1876f03ae207572141b57a23b4c19e33`.
- Fuente: SHA-256 `6b097880bcdd3111f2406b1531aea2073921a934f98bc24c88f155b5e3cfc9a2`.
