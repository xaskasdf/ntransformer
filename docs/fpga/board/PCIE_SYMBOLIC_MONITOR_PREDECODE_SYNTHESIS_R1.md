# Monitor predecode R1: síntesis propia

## Resultado y comparación

La tanda `20261011T081251Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-synth-r1`
terminó con código 0 en 476,070 s. La revisión independiente aprobó la
coherencia de artefactos, restricciones y grafo; habilita su replay funcional,
sin aprobar timing. Se usaron dos threads, `xcvu5p-flva2104-2-i`, período de
4 ns, retardos E/S cero y síntesis OOC `RuntimeOptimized` con jerarquía reconstruida.

| Medida en síntesis | Monitor R1 | Predecode R1 | Diferencia |
|---|---:|---:|---:|
| LUT | 104.259 | 104.980 | +721 |
| FF | 26.976 | 27.542 | +566 |
| CARRY8 | 2.066 | 2.066 | 0 |
| WNS | −3,584 ns | −3,584 ns | 0 ns |
| WHS | −0,075 ns | −0,075 ns | 0 ns |

Ambos son resultados de síntesis sin colocación. No se mezclan con los
recursos optimizados o ruteados de la referencia. El candidato no contiene
RAM, DSP ni latches. El aumento de registros debe evaluarse junto con el
efecto físico de predecodificar las admisiones; todavía no hay mejora medida.

La peor ruta conserva origen `delivery_stage/out_b_reg[…][rc_user][64]/C`
y destino `retirement_i/fault_latched_reg[10]/D`: 31 niveles, 7,566 ns,
de los cuales 6,028 ns son interconexión estimada. Corresponde a validación
tardía, diferente de la ruta E→D que motivó este experimento.

## Evidencia y límites

Los dos bancos previos a síntesis aprobaron 529 filas. La
[regresión RTL completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_FULL_R1_RESULTS.md)
ya cubrió 183.058 filas. La auditoría del EDIF contenido en el DCP comprobó
176.049 instancias, 249.943 nets, ausencia de ciclos combinacionales y ocho
controles negativos. Las restricciones guardadas no añaden excepciones de rutas.

`check_timing` reporta cero problemas en sus doce categorías, pero permanecen
1.415 endpoints con fallo de setup y 2.177 con fallo de hold. Metodología
reporta 1.000 `TIMING-16` y 1.830 `XDCH-2`. La cobertura DRC es OOC;
esto no valida reloj físico, restricciones de placa ni PCIe nativo.

Fuente: `source-20261011T081125Z.zip`, SHA-256
`9e852fc08da5c50eb16189cde846d3d2c0bd30584dcbf111bb3b7fd11095925b`.
Archivo de ejecución verificado en D: y NFS:
`1e338bc22864cb420350ca3d7ca3684b6f6ac42c19b457823b7de5f6effa6515`.
Revisión independiente:
`edbea9715eec15ff445a06d26611beca866fe7b970c46dbe14d1a2b53be68981`.

## Dependencia siguiente

La netlist propia debe pasar los dieciséis bancos binarios y seis mutantes
antes de optimizar con BUFGCE. Esa netlist optimizada requiere su propio
replay antes de colocar/rutear y comparar E→D. Se conservan cinco clocks,
256 tags, dos RQ y cuatro RC por ciclo; no se reduce capacidad ni tráfico.
