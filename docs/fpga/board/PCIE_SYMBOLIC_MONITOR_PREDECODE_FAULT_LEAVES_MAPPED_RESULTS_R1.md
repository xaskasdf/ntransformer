# Replay de la netlist propia de fallos por lane

## Resultado observado

La tanda 963, `20261011T122746Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-binary-r1`,
terminó con salida **0**, en **1.503,802 s**. La revisión independiente aprobó
`PASS_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_OWN_MAPPED_BINARY`.
Resultados y fuentes están verificados en D: y NFS.

| Comprobación | Resultado |
| --- | ---: |
| Filas binarias nuevas, en 16 bancos | 174.436 |
| Bits públicos comparados | 447 |
| Observaciones anteriores / posteriores al flanco | 174.516 / 174.516 |
| Sondas de quiet | 10, con 20 muestras de prueba/restauración |
| Bancos drenados sin cancelar tráfico original | 6 |
| Controles negativos rechazados | 6 |
| Corrupciones de evidencia rechazadas | 34 |

El candidato utiliza exclusivamente la netlist de su síntesis, tanda 962;
no hereda filas aprobadas de otra variante. El revisor comprobó compilación,
referencias, primeros puntos de divergencia y los contadores de cancelación
de tráfico original en las seis familias drenadas. Se revisaron 24 avisos:
22 de entorno XSIM, uno de declaración previa y uno de `glbl` duplicado,
cuyo cuerpo se verificó idéntico. No hubo avisos críticos.

## Evidencia

Archivo terminal: **93.928.197 bytes**, 305 archivos; SHA-256
`45d55df045c6ead281692d7c4c267801329e9e71cc5c64f79e3ea2e41bfc45c0`.
Fuentes: `source-20261011T121419Z.zip`, SHA-256
`6c0831e40c3b112500387f4a566bbd034dca2b63483fd78f40a57c24674413fb`.
Netlist propia: SHA-256
`1887f09cb9477b298a55374d6031ee02e4b51e1543f2a0e89a9dd6327ccae64d`.
Revisión independiente: SHA-256
`0fe0bb52c0bb93a974d233d20c454c30dad79b85011f53ea5849f4b8c1af7ad6`.

## Decisión y límites

La simulación funcional binaria no incorpora SDF, equivalencia de cuatro
estados ni calificación física. El [WNS estimado de −3,746 ns](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_SYNTHESIS_RESULTS_R1.md)
sigue incumplido; esta aprobación no cambia la decisión de evitar otro ruteo
de la variante por hojas. Continúa el experimento de contexto por prefijo,
con pruebas propias aisladas, integradas y de síntesis antes de decidir
nuevos trabajos físicos.
