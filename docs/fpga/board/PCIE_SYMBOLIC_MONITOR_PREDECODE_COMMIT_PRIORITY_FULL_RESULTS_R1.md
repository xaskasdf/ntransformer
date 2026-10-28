# Prioridad RC: integración completa revisada

La tanda 978 `20261011T163442Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-full-r1`
terminó con código 0 en **770,539 s**. Su revisión independiente aprobó
183.058 filas RTL nuevas, sin reutilizar filas de otros candidatos.

## Cobertura y resultado

Se compararon 447 bits públicos, con 183.138 muestras antes y después
del flanco y 31 probes de quietud más restauración. Los seis bancos
drenados conservaron su tráfico original sin cancelación RAW. Los seis
controles negativos de entradas divergieron como se esperaba; el revisor
rechazó 34 corrupciones de evidencia.

Los 26 archivos realmente compilados incluyen el selector y C originales
para las referencias y los módulos con nombres propios para el candidato.
Los oráculos originales permanecen intactos. El resultado habilita síntesis
propia; no acredita timing, netlist, PCIe nativo ni placa física.

## Respaldo y recuperación del coordinador

El archivo de resultados está verificado en D: y NFS:
SHA-256 `1d3b46e664f4479d621926cb22ecd947f5c5b73eed2e607a742635922aa4c53e`,
49.950.588 bytes y 1.575 archivos. El snapshot es
`source-20261011T163329Z.zip`, SHA-256
`229cdcc11cb8786da5c2e09cf8251ec1e139f1d1ce69887cbd2a8670e7f69818`.
La revisión independiente tiene SHA-256
`5da47179d1ce728b748f010546d53c4cda9094bb702a8799a0c40e2f7b7c8431`.

Tras respaldar, el coordinador Full R2 buscó por error un owner Full R2;
el ejecutor conservaba correctamente su identidad Full R1. Se preservaron
el error y las colas dependientes detenidas. La revisión se ejecutó sobre
el archivo ya sellado, sin repetir Vivado. Full R3 respalda esta revisión;
las colas de síntesis y replay R2 enlazan ese cierre corregido. RTL, Tcl,
ejecutores, revisores e identidades de configuración siguen siendo R1.

Manifiesto `pcie-symbolic-monitor-predecode-commit-priority-queue-recovery-r3.json`
en `build/cloud-setup/20261002-board/`. Las siguientes etapas son la
[síntesis comparable con 970](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_SYNTH_R1.md)
y el [replay de la netlist propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_MAPPED_R1.md).
