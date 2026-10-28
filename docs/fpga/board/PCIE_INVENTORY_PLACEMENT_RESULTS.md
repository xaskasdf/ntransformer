# Implementación diagnóstica R12: resultados

Revisada el 2026-10-08. La implementación y el replay de su netlist propia
quedaron respaldados. **El diseño falla el objetivo de 4 ns**; no es una
FPGA ni una frecuencia de placa calificadas.

## Procedencia y correcciones

Se reutilizó el DCP R12 de síntesis
`293ee2f048b1f8c6624d871e1cdb5a1d8ded88f1c58b717a39f89903e57b437a`,
sin resíntesis. Mismos puertos, parte `xcvu5p-flva2104-2-i`, reloj de 4 ns
y E/S cero; único contexto añadido: `HD.CLK_SRC BUFGCE_X0Y60`.
Flujo predeterminado opt/place/route, cuatro hilos; 1608.016 s.

El intento inicial falló antes de abrir el DCP por `redirect`. R1 completó
la implementación, pero descartó el retorno de `help`. La auditoría
posterior capturó las once ayudas: no estaban disponibles en el log R1.
Los intentos y sus ledgers se conservan.

El EDIF embebido usa una codificación Xilinx que el lector Zstd anterior
no interpreta. El EDIF de texto pasó una revisión preliminar; después,
Vivado reabrió el **DCP exacto** y lo exportó otra vez. Ambas exportaciones
coinciden salvo la fecha; inventarios de primitivas idénticos. La auditoría
DAG de la exportación recargada pasó sus ocho controles y un control
adicional de identidad. 73.355 primitivas lógicas, 77.808 declaraciones
de nets; sin ciclos combinacionales. No se repitió place/route.

## Timing y recursos

| Fase | WNS ns | TNS ns | WHS ns | Fallos setup/total | Fallos hold/total |
| --- | ---: | ---: | ---: | ---: | ---: |
| before_context | -7.068 | -70828.984 | -0.075 | 15867/29685 | 2035/29685 |
| with_context | -7.068 | -70828.984 | -0.075 | 15867/29685 | 2035/29685 |
| optimized | -7.015 | -68051.914 | -0.047 | 15855/29676 | 1863/29676 |
| placed | -7.156 | -92444.570 | -1.773 | 16000/29684 | 26570/29684 |
| final | -18.945 | -269784.312 | -1.773 | 17626/29684 | 11240/29684 |

Uso físico final: **51.867 CLB LUT**, **10.727 FF**, **1.895 CARRY8**;
cero latches, BRAM, URAM y DSP. Los conteos lógicos de LUT del EDIF no
se suman directamente como CLB LUT: Vivado combina LUTs.

Peor setup interno: `producer/allocated_tags_reg[242]/C` →
`producer/address_next_reg[60][11]/D`, **22,926 ns**, 37 niveles.
Lógica 2,911 ns; ruteo 20,015 ns (87,303%). El subconjunto registro→registro
reproduce WNS −18,945 ns; su peor hold es **+0,010 ns**. Esto no sustituye
el WHS global −1,773 ns ni sus 11.240 endpoints con fallos.

El router informó 51.664 pines críticos tanto para setup como hold y
correcciones de hold a costa de setup. Metodología emitió 1.000 avisos
TIMING-15, 1.000 TIMING-16 y 1.829 XDCH-2; son los conteos del reporte,
no un inventario ilimitado de todas las rutas. Las 30 redes de mayor
fanout sólo tienen cargas en SLR0: reset llega a 9.340, ocho bits RC a
1.993–2.025 y una red interna de direcciones a 1.429. No basta para
atribuir causalidad ni justificar otra colocación por sí solo.

## Frontera y limitaciones

39.590 nets reportadas como completamente ruteadas, cero errores de
ruteo y **un gap explícito del reloj padre** (`clk`, alias `producer/clk`).
Los indicadores son PLACED_FULLY=1, PARTIALLY_ROUTED=0, ROUTED_FULLY=0:
el reporte distingue gaps de otras rutas parciales. Retardos del reloj
padre todavía estimados. Los 1830 puertos no tienen
HD.PARTPIN_LOCS; el límite de 100 avisos Route 35-198 no cuenta puertos.

DRC informó cero violaciones, advirtiendo que OOC omite comprobaciones.
No hay excepciones temporales y las restricciones se conservaron durante
la auditoría. No interpretar slack infinito/retardo cero de puertos en
el reporte de fanout como cierre temporal.

## Replay funcional y respaldo

Netlist enrutada SHA-256 `a02d17a36c1afcf076782e26f804d97f4f7776cd82bcaa318690e66d2acf7cc1`.
Los cinco bancos compararon todas las filas públicas, cancelaciones y
estados posteriores a fallos contra R12: 22.330/67, 46.356/3.872,
8.850/240, 544/60 y 632/60 filas/escenarios. **78.712 filas de entrada**
y 30 controles de evidencia; sin RTL del DUT, sin SDF, arranque 200 ns
y muestreo 1 ns. Duración 631.652 s. No es una
medición de rendimiento físico ni una prueba de inferencia Llama.

| Tanda | ID | SHA-256 del archivo |
| --- | --- | --- |
| initialization_failure | `20261008T193901Z-V4-ffn-supervised-pcie-inventory-join-placement` | `6b1cc7480d0138155459df83e705776df77215b9659c7bbd0345bdcfea81fe50` |
| implementation | `20261008T194246Z-V4-ffn-supervised-pcie-inventory-join-placement-r1` | `ca0f3e1dbf3c39c940a187ce12c3bd3fd9d245eed056de165ce8561aacc3fb43` |
| boundary_audit | `20261008T202208Z-V4-ffn-supervised-pcie-inventory-boundary-audit` | `4e20679a9052e82abd917840720a19ac77563f3627e86f05966644a7ef6ed883` |
| binary_netlist | `20261008T202831Z-V4-ffn-supervised-pcie-inventory-placement-binary-netlist` | `842650defacb77626016be7752b18068f2583a76c3a6383b24b8fde9b553e60b` |

Archivos y fuentes verificados en NFS y en `artifacts/ntrpu-board-vivado/`
(respaldo físico en D:). Revisiones bajo `build/cloud-setup/20261002-board/`.
Se conservan reset nativo, CDC, endpoint, DIMM exacto, DDR/512 GiB y V5–V8
como criterios abiertos. Siguiente ensayo:
[preflight del reloj interno](PCIE_INVENTORY_PLACEMENT_FOLLOWUP.md).
