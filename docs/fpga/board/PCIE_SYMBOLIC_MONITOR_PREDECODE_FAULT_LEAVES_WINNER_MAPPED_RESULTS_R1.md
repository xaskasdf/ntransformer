# Replay de la netlist propia de la bandera

## Resultado revisado

La tanda 971, `20261011T144252Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-winner-binary-r1`,
terminó con salida **0** en **1.515,690 s**. La revisión independiente dio
`PASS_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_OWN_MAPPED_BINARY`.
El paquete terminal y las fuentes quedaron verificados en D: y NFS.

Se compararon **174.436 filas binarias** de dieciséis bancos y los **447
bits públicos** contra las referencias independientes. Pasaron los seis
controles negativos, diez sondas de quiet más sus restauraciones y 34
corrupciones de evidencia. Se compiló la netlist de la síntesis 970;
no se sustituyó por el RTL ni por la netlist del padre.

## Evidencia

- Netlist usada: SHA-256 `87c98a455183295bcc9a7a8f58364c6261f72e50bd481fab42049d31184df19b`.
- Paquete: **94.663.789 bytes**, 305 archivos; SHA-256
  `3af3819232df10578e8c233e734be9a0ad0b9c2ad854609eb045685c231dbfbb`.
- Revisión: SHA-256 `082f5412d2c5abadc338e334816d28e737087aa96c77c22a46280113154bf01b`.
- Fuentes: `source-20261011T142914Z.zip`, SHA-256
  `846ea982aeabcf541a4b1986c6556219079c028a86678de4d03e156f358392c9`.

Los registros y revisores están en `build/cloud-setup/20261002-board/`.
La cola continúa con la [lectura comparativa por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_STAGE_TIMING_PLAN_R1.md)
de los DCP 966 y 970.

## Alcance

Es replay funcional binario sin SDF. La [regresión RTL de 183.058 filas](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_RESULTS_R1.md)
cubre X/Z por separado. No aprueba timing físico, PCIe nativo ni placa.
El [WNS estimado −2,768 ns](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_SYNTHESIS_RESULTS_R1.md)
conserva su fallo; este PASS funcional no lo sustituye.
