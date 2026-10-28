# C/E/D con control R3: replay propio

La tanda `20261009T151232Z-V4-ffn-supervised-pcie-symbolic-ced-control-r3-binary` pasó la revisión independiente el 2026-10-09.
Se compararon **2.499 bits externos** del circuito combinado sintetizado,
sin modificarlo, sobre **205.493 filas positivas**. La simulación funcional
no incorpora retardos SDF: **no demuestra cierre físico ni 250 MHz**.

## Cobertura

| Banco | Filas | Orígenes retirados |
| --- | ---: | ---: |
| smoke | 64 | 26 |
| raw_main | 18,538 | 18,280 |
| raw_ownership | 46,356 | 30,870 |
| raw_rq | 8,850 | 7,892 |
| raw_prefix | 544 | 306 |
| raw_lanes | 632 | 394 |
| raw_context | 997 | 943 |
| drained_main | 18,666 | 18,408 |
| drained_ownership | 54,098 | 38,612 |
| drained_rq | 9,328 | 8,370 |
| drained_prefix | 662 | 424 |
| drained_lanes | 752 | 514 |
| drained_context | 1,023 | 969 |
| directed | 191 | 137 |
| established_main | 22,330 | 22,064 |
| established_main_drained | 22,462 | 22,196 |

Los bancos con vaciado previo al reset no cancelaron tráfico original.
Se comprobaron 200 ns de inicialización GSR, muestreo a 1 ns tras cada
flanco y dos ciclos finales de vaciado, también comparados. Los módulos
RTL de referencia solo recibieron otros nombres; la revisión coteja esa
transformación, los 88 puertos físicos, el adaptador y las fuentes realmente
compiladas en cada caso. Las referencias S1/R7/modelo y la contabilidad
independiente de captura, retiro y cancelación permanecen en el banco.

## Controles negativos

Cada alteración de entrada del circuito mapeado produjo la discrepancia
esperada antes del final del banco dirigido:

| Alteración | Primera fila discrepante |
| --- | ---: |
| omit_rq | 4 |
| omit_rc | 73 |
| omit_cold | 18 |
| wrong_sequence | 4 |

El revisor también rechazó nueve corrupciones de evidencia. Las
advertencias del compilador y simulador se cotejaron con las permitidas,
incluida la igualdad del cuerpo de las dos definiciones de `glbl`.

## Alcance y siguiente diseño

La netlist se ejercitó con entradas binarias. Se excluyeron escenarios
completos con X/Z; los bancos RTL previos de cuatro estados se conservan
como requisito independiente. El corpus principal antiguo se añadió como
prueba suplementaria y no se trató como subconjunto exacto del corpus R7.

Queda habilitada la prueba de las [etapas A/B](PCIE_SYMBOLIC_NUMERIC_AB_R1_PLAN.md).
El WNS OOC del conjunto sigue siendo **−2,546 ns** con periodo de 4 ns.
La aplicación numérica, validación tardía, retiro de bytes y monitor completo
siguen pendientes. R12 conserva su condición de referencia; tampoco se
cierran CDC/reset, DDR de 512 GiB, pesos reales o placa.

## Evidencia recuperable

- Archivo local en D: y remoto en NFS: SHA-256 `4933bfa9032c693b7639e7857a5301650a4b0f24034e17b410116e1de55f701c`.
- Fuentes: SHA-256 `c2443665fa90c55c243f067f54970d962e76e32effe2ddf9570388c6093cb520`.
- Revisión: `20261009T151232Z-V4-ffn-supervised-pcie-symbolic-ced-control-r3-binary-independent-symbolic-ced-binary-review.json`, SHA-256 `7c8391c4d595b45341f1846e8bc887186eecd93645d1987872e825d30bb537a9`.
- Todos los miembros del archivo verificados; respaldo sin extracción adicional.

Este resultado corresponde exactamente a la
[sintesis R3](PCIE_SYMBOLIC_CED_CONTROL_R3_SYNTHESIS.md).
