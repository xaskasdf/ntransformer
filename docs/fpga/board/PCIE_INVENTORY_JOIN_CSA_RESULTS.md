# Monitor PCIe: experimento de acumulación carry-save

El 2026-10-08 la revisión CSA pasó el banco completo RTL y el replay de su
propia netlist. **Se conserva la referencia original como candidata preferida
de timing**: CSA reduce 510 LUT, pero empeora WNS en
1.006 ns. Ninguna de las dos cumple 4 ns.

## Cambio y cobertura

`ntrpu_pcie_inventory_scheduler_r2.sv` reemplaza solamente la suma/resta final
del contador global por compresores de suma y acarreo. Se mantienen el productor
canónico, el wrapper, el validador y los bancos principales. El
[contrato](PCIE_INVENTORY_JOIN_CSA_CONTRACT.md) explica la identidad modular
de 23 bits y conserva el truncamiento y los guardas de rango originales.

El banco completo aprobó **22.339 filas, 68 escenarios y 5.718.784 comparaciones
de palabras de tabla**, incluso después de fallos. Retiró 22.136 registros y
canceló 67 por cold reset; observó 71 revocaciones inmediatas de vacío, 3.609
filas vacías y una vuelta de la secuencia diagnóstica. Rechazó los siete mutantes
reales y 17 controles de corrupción de evidencia.

Un banco suplementario llamó a la función real del DUT: **20.242 comparaciones**,
incluidas 178 con X/Z en cada bit de entrada. Dos controles adicionales
rechazaron evidencia incompleta. No se afirma equivalencia formal universal.

Los bancos binarios RTL y netlist aprobaron **22.330 filas y 67 escenarios**,
con 12 controles de evidencia cada uno. Todas las filas públicas de estado,
fallo, secuencia, reset, pendiente y vacío fueron idénticas. El RTL comparó
5.716.480 palabras internas; la netlist no usa jerarquía interna ni compila
el productor, wrapper o validador RTL como DUT. Sus dos definiciones `glbl`
son idénticas. Startup 200 ns, muestreo 1 ns, sin SDF.

## Medición del monitor completo

Ambas tandas usan `xcvu5p-flva2104-2-i`, Vivado 2026.1, `RuntimeOptimized`,
reloj de 4 ns, retardos I/O cero y ninguna excepción temporal.

| Métrica | Referencia | CSA |
| --- | ---: | ---: |
| LUT | 48189 | 47679 |
| FF | 10729 | 10729 |
| CARRY8 | 260 | 258 |
| WNS, ns | -9.877 | -10.883 |
| TNS, ns | -109536.984 | -144776.094 |
| Endpoints setup incumplidos | 15851 | 15867 |
| WHS, ns | -0.075 | -0.075 |
| THS, ns | -139.02 | -139.887 |
| Endpoints hold incumplidos | 1936 | 1974 |

CSA conserva cero latches, BRAM, URAM y DSP. La peor ruta va desde
`rc_user[78]` a `producer/byte_range_high_reg[0]`: **14,894 ns y 58 niveles**,
con 2,999 ns lógicos y 11,895 ns de ruta estimada. La referencia medía
13,860 ns y 56 niveles desde `address_next[51][0]` al contador global.
Son estimaciones OOC sin colocación/ruteo; no son frecuencias físicas medidas.

La auditoría del DCP verificó 69.155 primitivas y 75.592 nets sin ciclos,
con ocho controles. Las 12 categorías de `check_timing` quedaron en cero;
el reporte DRC enumera cero comprobaciones encontradas. Persisten
`TIMING-16` × 1.000, `XDCH-2` × 1.829 y la advertencia OOC de reloj sin
`HD.CLK_SRC`. La ausencia de errores de exportación no cierra esos límites.

## Ejecuciones y respaldo

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T110153Z-V4-ffn-supervised-pcie-inventory-join-csa-rtl` | 138.792 | `3095a78cb0251d3f4c9d8e64320cb2917e69f5f943fb45c5e217bb06155deda1` |
| `20261008T110648Z-V4-ffn-supervised-pcie-inventory-join-csa-synth` | 304.450 | `cab98c739f4ae085f911ff1c6120b213c73e664b7b4c26764034b270bf86f85f` |
| `20261008T111306Z-V4-ffn-supervised-pcie-inventory-join-csa-binary-rtl` | 45.298 | `f961bfe073adc8c625e7a58bcd502e1628744e78849094996438b3f6de15ad94` |
| `20261008T111722Z-V4-ffn-supervised-pcie-inventory-join-csa-binary-netlist` | 104.842 | `1379597bb8b9761803ebdd3d5dcbd8dd5034e28264a8d6394379b4eba1b13e95` |

Las cuatro tandas comparten `source-20261008T110108Z.zip`, 2.113 archivos,
SHA-256 `2765bd13ee9bfd2682076ddba084efdead7ec7f44cbae8729665d9cfb5e54335`. Cada archivo y sus fuentes están verificados
localmente en `artifacts/ntrpu-board-vivado/<tanda>/` y conservados en NFS.
La netlist exacta reproducida tiene SHA-256
`c21cbc15d6b45d09335d50c2f613da2008f8190d01c1fdee274740a2d07d230c`.

Las revisiones independientes quedan en `build/cloud-setup/20261002-board/`:

- `20261008T110153Z-V4-ffn-supervised-pcie-inventory-join-csa-rtl-independent-inventory-join-review.json`: `f71279ad527bdf2387752257443ee6d0dd2f5d5f84ace6ea0621b803f638fd84`.
- `20261008T110648Z-V4-ffn-supervised-pcie-inventory-join-csa-synth-independent-synthesis-review.json`: `14766dba0004c71e4cf3348537279f3eabf405f4aa1f3b2ad033301d278c81c4`.
- `20261008T111306Z-V4-ffn-supervised-pcie-inventory-join-csa-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `f34e22b5da534efb3ee4825dc83bded297791cdef0273b09accd780acd8b9142`.
- `20261008T111722Z-V4-ffn-supervised-pcie-inventory-join-csa-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `c20ac4eb899e28ef7d79c78493c5d08d251b511168a8c3a4c5b816585a945d0d`.

El revisor inicial suponía que sobrevivía el `.prj` del primer top; Vivado lo
reemplaza al cambiar al banco principal. La primera corrección encontró una
cadena dentro del Tcl impreso. `review_pcie_inventory_join_csa_rtl_r2.py`
vincula las líneas exactas archivadas de compilación, snapshot y finalización
con las fuentes verificadas. Se conservan ambos diagnósticos; no se alteró
ni repitió la ejecución para resolverlos.

## Siguiente paso y límites

Se encola el [registro de deltas de bytes](PCIE_INVENTORY_COUNTER_RETIMING_PLAN.md)
para separar el cálculo de eventos de la acumulación global, conservando la
latencia pública. Es una hipótesis pendiente de ejecución. Siguen abiertos
timing, conexión nativa, identidad/BDF, CDC/reset, DIMM exacto, cuatro DDR/512 GiB,
pesos reales e implementación física. La captura nativa reutilizada continúa
limitada a tres lecturas de 32/64/32 bytes; el banco amplio es sintético.
