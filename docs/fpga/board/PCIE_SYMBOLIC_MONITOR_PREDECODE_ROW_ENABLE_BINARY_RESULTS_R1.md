# Replay propio de la netlist E/D: resultado R1

Revisado el 2026-10-12 a las 00:14 UTC. **PASS funcional binario** de la
netlist sintetizada E/D, contra las referencias RTL independientes.
Resultado y fuentes tienen respaldo verificado localmente y en NFS.

## Evidencia

Tanda: `20261011T234828Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-binary-r1`.
Usó la netlist propia de la [síntesis E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_SYNTH_RESULTS_R1.md),
no la netlist RQ984. Vivado terminó con código 0 a las 00:10:42 UTC,
después de 1.329,182 s de ejecución del banco; este tiempo no mide
rendimiento del hardware ni inferencia.

| Comprobación | Resultado |
| --- | --- |
| Bancos positivos / filas nuevas | 16 / 174.436 |
| Bits públicos comparados | 447 |
| Sondas quietas / muestras con restauración | 10 / 20 |
| Mutantes reales de entrada detectados | 6 de 6 |
| Corrupciones de evidencia rechazadas | 34 de 34 |
| Advertencias / advertencias críticas | 24 esperadas / 0 |

Incluye los seis bancos drenados, controles y retorno de rangos.
Se revisaron procedencia del `glbl`, unidades compiladas, contabilidad
y hashes contra el archivo recuperado. El Tcl limita `general.maxThreads`
a 2; `invocation.json` conserva el valor inicial 4 del supervisor.
No se modifica esa evidencia histórica ni se infiere utilización de CPU.

Revisión independiente, en `build/cloud-setup/20261002-board/`:
`20261011T234828Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-binary-r1-independent-symbolic-monitor-predecode-row-enable-binary-review-r1.json`.

SHA-256 de revisión: `73bfbda84236bc6f3a5ea5a6fdada1c90a1883ae788297d889f44683b2f25395`.

SHA-256 de archivo: `b455d75d9e3350091cc1543dc86e9b4b088b7a984bb281b148a6a4721b5b2faf`.

SHA-256 de netlist: `e9bac74c44f6738b411c70c54936e6582e460cf06ccc3985153819150f5e5f7d`.

## Decisión

Habilita el diagnóstico por etapas de su DCP frente a RQ984.
No modifica el WNS sintetizado de −2,773 ns ni aprueba timing físico.
Es un replay binario sin SDF; tampoco valida PCIe nativo, DDR, la placa
ni inferencia completa. La comparación física y los replays ruteados
permanecen detrás de sus dependencias en la cola.
