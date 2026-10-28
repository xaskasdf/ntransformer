# Predecode R1: reloj interno revisado

## Resultado estructural

La tanda `20261011T085600Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-clock-r1`
terminó con código 0 en 221,546 segundos. Partió del DCP sintetizado cuya
[netlist propia pasó](PCIE_SYMBOLIC_MONITOR_PREDECODE_MAPPED_RESULTS_R1.md).
Optimizó dos copias del mismo DCP, una de referencia y otra con BUFGCE.

La revisión independiente comprobó que los grafos lógicos coinciden al
contraer el buffer. La variante añade un BUFGCE y su VCC, conserva puertos
y registros, y conecta exclusivamente los **27.086 pines de reloj FDRE** a
su salida. La entrada recibe `clk` y CE permanece constante. Los valores
por defecto de la biblioteca instalada se conservaron, incluido
`SIM_DEVICE="ULTRASCALE"`; no se cambió la biblioteca.

Ambos grafos son acíclicos. Se ejecutaron ocho controles de grafo por
variante, trece controles del auditor de reloj y doce corrupciones de
evidencia estructural. El archivo y sus fuentes quedaron verificados en D: y NFS.

## Medidas estimadas a 4 ns

| Medida tras optimización con BUFGCE | Referencia R1 | Predecode R1 |
|---|---:|---:|
| LUT | 103.238 | 103.954 |
| FF | 26.521 | 27.086 |
| CARRY8 | 2.058 | 2.058 |
| WNS | −3,723 ns | −3,723 ns |
| WHS | −2,702 ns | −2,702 ns |

El candidato aumenta 716 LUT y 565 FF, sin mejorar el peor slack estimado.
Por jerarquía, añade netos 553 FF en E y 12 en C; D, A, B y R conservan
sus conteos. El inventario incluye los 512 registros explícitos de selección.
Esta comparación de conteos no afirma equivalencia individual entre registros
de variantes distintas.

La ruta crítica sigue de `delivery_stage/out_b_reg[…][rc_user][64]/C`
a `retirement_i/fault_latched_reg[10]/D`: 31 niveles, 7,568 ns estimados,
con 6,030 ns de interconexión. Hay 1.514 endpoints con fallo de setup y
53.240 con fallo de hold. Los tres avisos DRC recuerdan la cobertura OOC
limitada; no se añadieron excepciones ni una fuente artificial `HD.CLK_SRC`.

## Evidencia y siguiente paso

Archivo: SHA-256
`2bc46c4a2fe68e1b67c3c4c2f5a3a21113c2600bb4371f4ab6c1ffc965801c74`.
Fuentes: `source-20261011T085408Z.zip`, SHA-256
`bdb5fc44d4258b5395fa6affde5b5cbdea580a39d94180687568f5ca0dae53b1`.
Revisión: SHA-256
`f39b25a5d1eb3e6781ba9cea4b6195f570533ca6b0211ab4e488c98bd6789de1`.

La [cola](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_QUEUE_R1.md) exige ahora
reproducir esta netlist optimizada. Solo después permitirá colocación/ruteo,
reapertura y replay ruteado propios. No hay todavía medida física del candidato
ni aprobación de timing, PCIe nativo, DDR o placa.
