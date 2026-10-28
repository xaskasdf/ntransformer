# Predecode: replay de la netlist ruteada

## Resultado al 2026-10-11 UTC

La tanda `20261011T101255Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-route-binary-r1`
terminó con código 0 en 1.353,671 s. La revisión independiente aprobó la
comparación binaria contra las referencias originales y el resultado está
respaldado en D: y NFS. No se usó la netlist de la referencia ni la del
candidato antes de colocar: se compiló el `routed_funcsim.v` propio de
`20261011T093541Z`, SHA-256
`c70b29f56fc816820bf6e63e370f1f2cbc056d42d44285574a97fbe188b9e201`.

| Comprobación | Resultado |
| --- | --- |
| Bancos positivos | 16 |
| Filas binarias | 174.436 |
| Bits públicos comparados | 447 |
| Muestras antes/después del flanco | 174.516 por fase |
| Probes binarios de quiet y restauración | 10 + 10 |
| Controles de entrada alterada | 6 detectados |
| Corrupciones de evidencia | 34 rechazadas |
| Bancos drenados sin cancelación de tráfico original | 6 |

La revisión enlaza fuentes congeladas, archivos compilados, invocación,
informes y logs al archivo certificado. Conserva 24 advertencias conocidas
y ninguna crítica: una por `rq_held`, una redefinición idéntica de `glbl`
y 22 avisos del entorno de XSIM. Verificó antes y después la fuente instalada
de BUFGCE; no demuestra la identidad de la biblioteca precompilada.

## Evidencia

Archivo de 92.031.952 bytes y 300 miembros verificados:
SHA-256 `2938f508d76678e3a01285fee90ac7068528fb22753107c148c10487609580c5`.
Revisión `...-independent-symbolic-monitor-predecode-route-binary-review-r1.json`:
SHA-256 `4ff3e2ed2ae33c4d827763bc546a6119546382d6a209d65224eeed567b1e8bd8`.
La fuente sigue siendo `source-20261011T093408Z.zip`,
SHA-256 `3049daa98ab5edb3c5f7b7b0066938de673320dc3e3c0729ef66ecbcb079e716`.

## Alcance y siguiente paso

La simulación funcional no usa SDF ni demuestra funcionamiento a 250 MHz.
El resultado físico Quick mantiene WNS −13,498 ns y WHS −2,972 ns.
Las 8.622 filas de escenarios X/Z excluidas del replay mapeado permanecen
cubiertas por la regresión RTL de 183.058 filas; no se atribuyen a esta prueba.

Continúan el [ensayo aislado de fallos](PCIE_SYMBOLIC_LATE_FAULT_LEAVES_R1.md)
y la [comparación Quick/Default](PCIE_SYMBOLIC_MONITOR_PREDECODE_TIMING_ROUTE_R1.md).
Siguen abiertos timing, PCIe nativo, CDC/reset, DDR y validación de placa.
