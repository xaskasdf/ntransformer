# Revisión de primitivas eliminadas durante placement

El 13 de octubre de 2026 se revisó la detención de la cola física: el comprobador R2 rechazó primitivas eliminadas por `place_design`. Se conservaron el fallo, los dos diseños ruteados y sus archivos de readback. No se volvió a sintetizar ni a rutear.

## Evidencia estructural

Se releyeron los EDIF `reference.edf` y `reloaded.edf` directamente de cada archivo terminal verificado. El extractor conserva jerarquía, nombres originales, parámetros, dirección de pines y conexiones, y rechaza controladores múltiples o entradas conectadas sin controlador. Los controles DAG y de reloj previos siguen siendo requisitos separados.

| Variante | Primitivas de referencia | Implementadas | Eliminadas | Réplicas añadidas | Clases equivalentes |
| --- | ---: | ---: | ---: | ---: | ---: |
| RQ bitmap | 161284 | 161301 | 7 | 24 | 157611 |
| Row enable | 166288 | 166271 | 25 | 8 | 162208 |

El algoritmo comienza con identidades distintas y une únicamente primitivas del mismo tipo, parámetros y señales de entrada ya equivalentes. Para registros se comparan INIT, reloj, habilitación, reset, datos y polaridades. Después verifica **cada** primitiva implementada, la representación de todas las clases de referencia y las 447 salidas. Los nombres de réplica sólo proponen correspondencias; no las certifican.

Las eliminaciones corresponden a 31 FDRE y una LUT6. Cada una conserva una contraparte equivalente. No hubo cambios de tipo. Se excluyen de la definición lógica sólo `SOFT_HLUTNM`, `OPT_MODIFIED`, `PHYS_OPT_MODIFIED` y `ORIG_CELL_NAME`; se conservan en los grafos. AMD describe las propiedades de [agrupación para placement](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Placement-Constraints), [registro de optimizaciones](https://docs.amd.com/r/2025.2-English/ug912-vivado-properties/PHYS_OPT_MODIFIED) y [origen de réplicas](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Adjusting-Constraints-for-Synthesis-Logic-Replication).

## Controles y alcance

`python tests/test_ntrpu_primitive_congruence_r1.py` rechaza 21 alteraciones, incluidas conexiones, INIT, polaridad, pines, salidas, propiedades desconocidas y realimentación no demostrada. El comprobador físico R3 conserva y supera los 22 controles negativos del R2. R3 exige comparar grafos completos cuando hay eliminaciones; una lista de nombres permitidos no basta.

El resultado describe una relación lógica sin retardos. **No califica timing, CDC, reset físico, DDR ni la tarjeta.** Se mantienen las violaciones de setup/hold, las limitaciones OOC y la obligación de simular ambos netlists ruteados con sus vectores originales.

La evidencia está en `build/cloud-setup/20261002-board/pcie-symbolic-monitor-primitive-congruence-r1-review.json` (SHA-256 `7aad2958c6586f8ebf19cd1337f7fc8bdad1c4944103896c46cf4b2042e3f5d2`). Los cuatro grafos se guardan en `D:/AI/ntransformer/artifacts/ntrpu-primitive-congruence-r1/`. La recuperación crea coordinadores nuevos para las 13 etapas pendientes y conserva las fuentes, bancos, límites y dependencias de hardware anteriores.
