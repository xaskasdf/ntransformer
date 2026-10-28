# Validación tardía desde B: equivalencia RTL aprobada

Tanda del 2026-10-09: **178.185 filas en catorce bancos**, con los errores,
tráfico posterior al fallo y valores X/Z originales. Se mantienen todas
las comprobaciones R7, modelo, S1, C/E/D y A/B. El nuevo consumidor recibe
exclusivamente el registro B capturado; no consulta la tabla actual.

## Comparaciones y cobertura

Por origen se compararon los 16 bits de fallos tardíos y los cuatro
temporales de cabecera de 15 bits contra observaciones añadidas a R7.
El probe conserva los predicados, escrituras y estado de R7. Los temporales
usan el [guard previamente probado](PCIE_SYMBOLIC_HEADER_GUARD_R1_RESULTS.md).

La historia independiente tiene cuatro posiciones. Reset frío cancela
los orígenes pendientes; reset de usuario y pérdida de link no los borran.
Cada banco añade cuatro flancos finales. La contabilidad distingue filas
originales de vaciado y cancela según los resets realmente presentes.

| Banco | Filas | Orígenes retirados | Orígenes cancelados |
| --- | ---: | ---: | ---: |
| `main` | 22339 | 21935 | 268 |
| `ownership` | 46356 | 23128 | 15484 |
| `rq` | 8850 | 7414 | 956 |
| `prefix` | 544 | 188 | 236 |
| `lanes` | 984 | 436 | 364 |
| `context` | 1047 | 943 | 68 |
| `drained_main` | 22607 | 22203 | 268 |
| `drained_ownership` | 61840 | 38612 | 15484 |
| `drained_rq` | 9806 | 8370 | 956 |
| `drained_prefix` | 780 | 424 | 236 |
| `drained_lanes` | 1348 | 800 | 364 |
| `drained_context` | 1115 | 1011 | 68 |
| `directed` | 217 | 126 | 59 |
| `controls` | 352 | 177 | 115 |

Los seis bancos con vaciado previo a reset no cancelaron tráfico original.
Los bancos sin ese vaciado conservan sus cancelaciones; no se reclasifican.
Los cinco bits tardíos tienen cobertura positiva. Para cada mutante se usó
el primer banco con cobertura, según el orden congelado de ejecución.

## Controles negativos reales

Cada módulo alterado suprime todas las asignaciones que activan un único
bit. Los cinco fueron rechazados con exactamente ese bit de diferencia,
en el origen correcto y cuatro flancos después de capturarlo.

| Bit suprimido | Banco seleccionado | Origen | Fila de detección |
| --- | --- | ---: | ---: |
| 4 | `drained_main` | 4292 | 4296 |
| 7 | `drained_main` | 4201 | 4205 |
| 8 | `ownership` | 45994 | 45998 |
| 9 | `ownership` | 165 | 169 |
| 10 | `ownership` | 4 | 8 |

La revisión independiente verificó fuentes, archivos compilados y parámetros
de los diecinueve casos; rechazó trece corrupciones de evidencia. Las únicas
advertencias fueron las diecinueve de entorno XSIM y las diecinueve del
orden de declaración de `rq_held` ya presente en C R3. No hubo errores ni
advertencias críticas. El respaldo y su contenido se verificaron en D: y NFS.

## Evidencia y siguiente dependencia

- Tanda: `20261009T170317Z-V4-ffn-supervised-pcie-symbolic-late-validation-r1`.
- Archivo D:/NFS: SHA-256 `c6091dce8ab29011d5d0eb5a636a7a84fa68c357df491365a71a2529bd512215`.
- Revisión: `20261009T170317Z-V4-ffn-supervised-pcie-symbolic-late-validation-r1-independent-symbolic-late-validation-review.json`, SHA-256 `aae6a7f93970bb889dc5090ff9d63402c1ce15cfb1aa2002af7f7121fdac72ac`.
- Fuente: SHA-256 `4f4d2a817dbf9e0d23865594b69941ec5193f9beb1c38ce60a6c9c5eee579b18`.

Este resultado habilita implementar R: candidato de bytes de 23 bits,
realimentación de los 21 bajos, fallo persistente y estado público del
mismo origen. R necesita cinco flancos de vaciado y nueva comparación
contra el estado de R7 posterior al flanco. Siguen pendientes quiet,
síntesis/netlist propias del conjunto, monitor completo y cierre físico.
Las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md) permanecen abiertas.
