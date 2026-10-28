# Monitor PCIe: primera medida ruteada

La primera tanda física usará colocación por defecto y
`route_design -directive Quick`, con **dos hilos y 1800 segundos** de
presupuesto. Abrirá el DCP exacto con BUFGCE cuya netlist optimizada ya
pasó los dieciséis bancos y seis controles. Mantiene 4 ns, I/O cero y
las restricciones heredadas; no añade excepciones, pines ni `HD.CLK_SRC`.

## Motivo y límites

En la tanda anterior del inventario sin la nueva tubería, el ruteo
normal agotó 3600 segundos durante la corrección intensiva de hold OOC.
Conservó el DCP colocado pero no un DCP ruteado final. Su revisión y
archivo fueron cotejados de nuevo antes de elegir esta estrategia.
Ese antecedente no demuestra que el monitor actual también agotaría
el tiempo: justifica obtener primero datos físicos con coste acotado.

`Quick` no busca cierre de temporización. Sus retardos ruteados y sus
fallos de setup/hold se registrarán íntegros. Aprobar esta tanda como
diagnóstico **no cierra temporización** ni acredita la FPGA completa.
El cierre posterior deberá cumplir las restricciones reales del reloj
y endpoint y repetir implementación cuando corresponda.

## Evidencia y dependencias

Exportar DCP de entrada, colocado y final o parcial; netlist, EDIF, XDC,
inventario de rutas, DRC, metodología, clocks y excepciones. Separar
setup/hold entre registros, entrada–registro, registro–salida y
entrada–salida. Leer de nuevo el DCP archivado y simular su propia
netlist antes de aceptar equivalencia funcional. Conservar todos los
resultados en D: y NFS, incluso si la tanda termina por timeout.

El plan R1 no llegó a congelarse ni ejecutarse. Solo se detuvo su
coordinador local mientras esperaba; la simulación y su cierre
continuaron activos. R2 mantiene las dependencias del
[plan físico](PCIE_SYMBOLIC_MONITOR_PHYSICAL_PLAN_R1.md), el caudal,
los 256 tags, contexto por valor y cancelación exclusivamente por frío.
PCIe nativo, CDC, reset, DDR, ECAD y placa siguen abiertos.

- Revisión antecedente: `20261008T213940Z-V4-ffn-supervised-pcie-inventory-clock-placement-r1-independent-default-route-timeout-review-r1.json`.
- SHA-256 de revisión: `460dd43b5a4b7a1d4c08a9b14a83c60a82689ebd714a9d993328b9a98eacab93`.
- SHA-256 del archivo: `980131e4dc07c74b998949ba3c3627bf5618fcf883b5bc1f7741b5684b56d169`.
- Script: `tools/ntrpu_board_pcie_symbolic_monitor_placement_r2.tcl`.
