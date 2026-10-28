# Síntesis propia E/D: recursos y timing estimado

Fecha: 2026-10-11 UTC. Tanda
`20261011T233631Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-synth-r1`.
Terminó a las 23:43:44 UTC en 428,709 s. La revisión independiente y la
comparación quedaron respaldadas a las 23:46:47 UTC. No cierra timing.

## Comparación controlada con RQ984

Se comprobaron iguales Vivado, dispositivo `xcvu5p-flva2104-2-i`, reloj 4 ns,
E/S cero, dos hilos y estrategia OOC `RuntimeOptimized`, con jerarquía rebuilt.
Los números son mediciones de síntesis y estimaciones anteriores al ruteo.

| Métrica | RQ984 | E/D | Diferencia E/D−RQ |
| --- | ---: | ---: | ---: |
| LUT | 92.913 | 97.050 | +4.137 |
| FF | 27.560 | 28.148 | +588 |
| CARRY8 | 568 | 568 | 0 |
| WNS, ns | −2,768 | −2,773 | −0,005 |
| TNS, ns | −1.472,519 | −1.540,027 | −67,508 |
| Endpoints setup incumplidos | 1.439 | 1.569 | +130 |
| WHS, ns | −0,075 | −0,075 | 0 |
| Endpoints hold incumplidos | 2.068 | 2.086 | +18 |

Hay 71.537 endpoints y doce verificaciones de cobertura temporal sin omisiones
reportadas. No usa latches, BRAM, URAM ni DSP. La peor ruta va de
`c_stage/allocated_tags_reg[51]` a `c_stage/read_range_high_reg[0]`: 28 niveles,
6,756 ns de datos, con 1,679 ns de lógica y 5,077 ns de interconexión estimada.

## Validaciones y límites

La prevalidación repitió los bancos `monitor` y `range_return`, conservando
529 filas, sondas de quiet y comparación de 21.951 bits internos. La auditoría
EDIF revisó primitivas y grafo combinacional de las netlists archivadas.
Los avisos se revisaron individualmente; los de igualdad de caso y puertos
sin uso alcanzan el límite de cien mensajes del fabricante, por lo que no
se atribuye cobertura a instancias suprimidas.

La integración RTL anterior sigue aprobada. Falta el replay binario de esta
netlist concreta y el diagnóstico por etapas. El aumento de recursos y la
regresión global no justifican promover E/D. La comparación física, si llega
a ejecutarse, seguirá siendo diagnóstica y deberá usar RQ984 bajo condiciones
idénticas; no el ruteo de otra variante.

## Identidad de los artefactos

- Revisión: `479a10c5060fa3043ffa522d16f2951cfbbfb0473c258960854a1d5c674229b5`.
- Archivo: `5d2674818306bfbd584a7aaf8b8a1debab8296185bcd1b1c8d2a1e860cc72c48`.
- Fuentes: `3d584c1ca793a567fece2804f00925d0bbfdab5903ab750d063c29b33e73f53b`.
- DCP: `706cec142a9b51a62efac046d1479c471f5545dcff48e75c1fbce68520c876ca`.

Fuentes, evidencia, DCP y diagnósticos permanecen en D: y NFS. Ninguno de estos
resultados acredita throughput, timing físico, PCIe/DDR o placa.
