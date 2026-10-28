# Monitor PCIe R1: optimización con reloj global

La tanda `20261011T053520Z-V4-ffn-supervised-pcie-symbolic-monitor-clock-r1` terminó en 183.381 s con dos hilos.
Se abrió dos veces el mismo DCP de la síntesis revisada: `opt_design`
produjo la referencia y una segunda ejecución insertó un BUFGCE.
La netlist optimizada todavía debe pasar su propia simulación funcional.

## Comprobaciones estructurales

Los 1831 bits públicos conservan dirección y nombre:
1384 entradas y 447 salidas. Ambas variantes
optimizadas mantienen exactamente los mismos 26521 registros
y la misma huella de conexiones y parámetros lógicos tras contraer el
buffer comprobado. Frente a la referencia solo se añadieron un BUFGCE
y un VCC. `clk` alimenta exclusivamente su entrada; CE está unido a VCC
y la salida alimenta los 26521 pines C, sin derivaciones.

Los dos grafos EDIF archivados son acíclicos. Cada variante superó ocho
controles de jerarquía/ciclo; se rechazaron trece alteraciones del reloj
y doce corrupciones adicionales de puertos, registros, conexiones y XDC.
Las restricciones siguen en 4 ns, I/O cero, sin excepciones ni `HD.CLK_SRC`.

## Recursos y temporización provisional

| Medida | Resultado optimizado |
| --- | ---: |
| LUT CLB | 103238 |
| FF | 26521 |
| CARRY8 | 2058 |
| BUFGCE | 1 |
| WNS / TNS | -3.723 / -2267.675 ns |
| WHS / THS | -2.702 / -77724.258 ns |
| Extremos con fallo setup / hold | 1502 / 45618 |

No hay BRAM, URAM, DSP ni latches. Las doce categorías de `check_timing`
reportan cero incidencias y el reporte DRC cero violaciones, con tres
avisos explícitos de cobertura OOC incompleta. No se ejecutó colocación,
ruteo ni `report_methodology`; los retardos son estimados y **no cierran
temporización**. La inserción del reloj no acredita mejora física.

## Siguiente dependencia y respaldo

Ejecutar los dieciséis bancos binarios completos y seis controles sobre
`optimized_clock_funcsim.v`, conservando los 447 bits comparados, ambas
fases, sondas de quiet y todos los oráculos originales. La biblioteca
instalada define `SIM_DEVICE="ULTRASCALE"`; conservar el artefacto sin
parametrizar y registrar la biblioteca usada. Su fuente no demuestra
por sí sola la identidad interna de la biblioteca precompilada.
Después de revisar esa simulación, seguir el
[plan físico](PCIE_SYMBOLIC_MONITOR_PHYSICAL_PLAN_R1.md).
PCIe nativo, CDC, reset, DDR de 512 GiB y placa siguen abiertos.

- Archivo D:/NFS: SHA-256 `d5edfc980753ee89af2d69de2044a026aea1f453eca51d512b569081c769cb81`.
- Fuente: SHA-256 `91da9657889fed72ac0c14f53f76624ce40ff00b86e1052d402f0b987053a339`.
- DCP con buffer: SHA-256 `08c46e5503d1ddb88c1d25b7b823f0938ae876b885dfcef9b1523e8d493a3a28`.
- Netlist: SHA-256 `9520e678925e2f8026568c9c6949084d022dce2e6b5884e8e42a451136b1ec98`.
- Revisión: `20261011T053520Z-V4-ffn-supervised-pcie-symbolic-monitor-clock-r1-independent-clock-review.json`, SHA-256 `7d736e5ed83f54d9d9f935e8be67f439419db0b6c72d32d142196fd493840997`.
