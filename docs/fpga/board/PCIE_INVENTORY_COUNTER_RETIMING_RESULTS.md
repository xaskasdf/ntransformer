# Monitor PCIe: deltas de bytes registrados

El 2026-10-08 el candidato pasó el banco RTL completo y el replay de su
propia netlist. **No mejoró el timing:** WNS −9,959 ns frente a −9,877 ns de
la referencia, con 2.348 LUT y 66 FF adicionales. Se conserva el productor
original como preferido entre las medidas OOC; ambos incumplen 4 ns.

## Cambio y validación

`ntrpu_pcie_inventory_scheduler_r3.sv` instancia un acumulador que registra
los deltas RC/RQ y una base de 21 bits. El candidato de 23 bits se calcula
entre registros, manteniendo la latencia pública y los controles de rango.
No cambia tablas, capacidad, RQ2/RC4 ni pausas del observador.
El [contrato congelado](PCIE_INVENTORY_COUNTER_RETIMING_CONTRACT.md) describe
las obligaciones previas; este documento registra su ejecución posterior.

El banco completo aprobó **22.339 filas/68 escenarios** y **5.718.784
comparaciones de palabras internas**, incluidas las filas posteriores a fallos.
Retiró 22.136 registros, canceló 67 por cold reset y comprobó 71 revocaciones
inmediatas de vacío, 3.609 filas vacías y una vuelta de secuencia. Detectó
siete mutantes reales y rechazó 17 corrupciones de evidencia.

El banco suplementario instancia el acumulador real: **20.414 muestras
secuenciales**, con 136 inyecciones X/Z, reset, pausa y recuperación. La
reconstrucción independiente confirmó 211 resultados negativos y 21 superiores
al límite. Dos controles rechazaron evidencia incompleta. Esto no prueba
equivalencia formal universal de cuatro estados.

Los bancos binarios RTL y netlist aprobaron **22.330 filas/67 escenarios**,
con 12 controles de corrupción cada uno. Todas las filas públicas de estado,
fallos, secuencia, cancelación y vacío coinciden. El RTL comparó 5.716.480
palabras internas; la netlist no compila productor, wrapper ni validador RTL
como DUT y no inspecciona sus tablas internas. Startup de 200 ns, muestreo
de 1 ns y sin SDF. Las dos definiciones archivadas de `glbl` son idénticas.

## Síntesis del monitor completo

Comparación sobre `xcvu5p-flva2104-2-i`, Vivado 2026.1, `RuntimeOptimized`,
reloj de 4 ns, retardos I/O cero y sin excepciones temporales:

| Métrica | Referencia | Deltas registrados |
| --- | ---: | ---: |
| LUT | 48189 | 50537 |
| FF | 10729 | 10795 |
| CARRY8 | 260 | 260 |
| WNS, ns | -9.877 | -9.959 |
| TNS, ns | -109536.984 | -143085.438 |
| Endpoints setup incumplidos | 15851 | 15910 |
| WHS, ns | -0.075 | -0.075 |
| THS, ns | -139.02 | -139.154 |
| Endpoints hold incumplidos | 1936 | 1973 |

La peor ruta pasa de la tabla de direcciones al contador global en la referencia
a `producer/allocated_tags_reg[51]` → `producer/remaining_reg[50][10]`:
**13,941 ns y 55 niveles**, con 3,071 ns lógicos y 10,870 ns de ruta estimada.
El contador deja de ser el destino de la peor ruta reportada; esto no acredita
una mejora física. Hay 29.933 endpoints setup/hold evaluados, cero latches,
BRAM, URAM y DSP.

La auditoría del DCP comprobó **71.819 primitivas y 85.740 nets sin ciclos**,
con ocho controles. Las 12 categorías de `check_timing` quedaron en cero;
el DRC enumeró cero comprobaciones encontradas. Persisten `TIMING-16` × 1.000,
`XDCH-2` × 1.829 y la advertencia OOC de reloj sin `HD.CLK_SRC`.
Son estimaciones sin colocación/ruteo; setup y hold permanecen abiertos.

## Evidencia reproducible

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T113713Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-rtl` | 142.744 | `3c0ee674154f9be6d23bae27ca500939dccf12bc21d2deb29a5652c89e8ceb99` |
| `20261008T114354Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-synth` | 326.297 | `c1596c221754bd84dede35e6f8b8df68f59dc83c085d8470e8ecf538652f671f` |
| `20261008T115028Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-binary-rtl` | 46.248 | `7182b9cee03d8aacbcfe4137ba72aa3013cbe1569eed2ccdbe5fb33552620e8d` |
| `20261008T115824Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-binary-netlist` | 120.242 | `1a3803c4ce2a36785b0b1c43f099f205de4f4717a3db33fe8af0b3d3389ebcbf` |

Las cuatro tandas usan `source-20261008T113646Z.zip`, 2.129 archivos,
SHA-256 `c38edda066755a6f4091c860027f2ac151deb031e5a3afb8728c5b2eda1e1427`.
Cada respaldo y sus miembros están verificados en
`artifacts/ntrpu-board-vivado/<tanda>/`, con almacenamiento físico en D: y
copia en NFS. La netlist reproducida tiene SHA-256
`7f3b06206146c9e648a118f1157e6d6b4f0f0a9b70addd27da5d19fee96d58e7`.

Revisiones independientes en `build/cloud-setup/20261002-board/`:

- `20261008T113713Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-rtl-independent-inventory-join-review.json`: `ac3d9b340368c17d7ae209af507d4289a175ed716cac8c1281f39cf19a3bade7`.
- `20261008T114354Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-synth-independent-synthesis-review.json`: `7c3e480f49b5b50b9bb6708d553bf3441565145bf89dd14d90429a70b67fab2a`.
- `20261008T115028Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `4faf279900e86a334c38964b3775f0ed5084f17f863d53c7be15d139e7455be7`.
- `20261008T115824Z-V4-ffn-supervised-pcie-inventory-join-counter-retime-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `b8b568a804ac42c008a6569931aeb55e20ccc1d241d8035a9d6763fb8a3b8a54`.

El revisor RTL R2 reconoce los registros archivados exactos de compilación,
snapshot y finalización: Vivado truncó un nombre largo en xelab y reemplazó
el `.prj` del primer top. El revisor de síntesis R1 corrige únicamente una
expectativa heredada de `Netlist 29-101`, ausente en esta ejecución. Se
conservan los revisores previos y sus diagnósticos; no se modificó la evidencia
ni se repitieron las herramientas para resolver esas diferencias del revisor.

## Trabajo siguiente

Se encola la [separación de eventos y actualizaciones RC](PCIE_INVENTORY_RC_FORWARDING_PLAN.md)
para atacar la cadena de tabla. La propuesta aritmética necesita validación
antes de incorporarse al RTL. Siguen abiertos conexión nativa, identidad/BDF,
CDC/reset, DIMM exacto, cuatro DDR/512 GiB, pesos reales e implementación
física. La captura nativa reutilizada cubre sólo tres lecturas de 32/64/32
bytes; la cobertura amplia procede de estímulos sintéticos.
