# Resultado RTL integrado de fallos por lane

## Resultado observado

La tanda 961, `20261011T115750Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-r1`,
terminó con salida **0** y duración registrada de **777,337 s**. La revisión
independiente aprobó `PASS_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_R1`.
El resultado y sus fuentes están verificados en D: y NFS.

| Comprobación | Resultado |
| --- | ---: |
| Filas RTL nuevas, incluidos X/Z | 183.058 |
| Filas reutilizadas de otra variante | 0 |
| Bits públicos comparados | 447 |
| Observaciones anteriores / posteriores al flanco | 183.138 / 183.138 |
| Probes de quiet | 31, con 62 muestras entre prueba y restauración |
| Bancos drenados sin cancelar tráfico original | 6 |
| Controles de entrada alterada rechazados | 6 |
| Corrupciones de evidencia rechazadas por el revisor | 34 |

Se ejecutaron los 16 bancos completos y seis controles negativos en
22 proyectos. La revisión verificó las 24 fuentes SystemVerilog compiladas
por caso, las referencias originales y los primeros puntos de divergencia.
Se mantienen la capacidad, la latencia de cinco clocks y la condición de quiet.

## Cambio validado

La variante mantiene la predecodificación E/D y sustituye el cálculo tardío
de fallos por 32 vectores de lane y su reducción OR. La instancia de retiro
conserva registros, rangos y reset frío. La referencia R7, el modelo y los
estímulos no cambian. El [documento de preparación](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_R1.md)
describe la transformación exacta.

Este banco valida la integración por hojas. El nuevo experimento de
[contexto por prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_R1.md)
es otra variante y todavía debe ejecutar su propia simulación.

## Evidencia y siguiente paso

Archivo terminal SHA-256:
`cbf7256d33ff009ceb86687cd7d4a62d457bc020125d753f043698d9e3b5aaa3`.
Fuentes SHA-256:
`784ce10e05a35fc7d8847168a166e1deee7967da277cb744d753244cc7a50c5e`.
Revisión independiente SHA-256:
`9ec9b66e790aeaf0eccc1e865e4fa75b1e4e79ec4ffdd80b88aa5ec9d9977037`.

La aprobación habilita la síntesis propia ya encolada. Se compararán LUT,
FF, profundidad y estimaciones de timing con la referencia bajo los mismos
parámetros. Después debe aprobar el replay de su propia netlist.
Todavía no hay mejora de timing demostrada, cierre físico ni validación
de la placa; no se atribuye rendimiento de inferencia a estas pruebas RTL.
