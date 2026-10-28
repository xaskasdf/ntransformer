# Integración del contador C: tanda 998

## Resultado

`20261011T213737Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-admission-count-full-r1`
terminó con código 0 en **778,667 s**. La revisión independiente aprobó
**183.058 filas nuevas en 16 bancos**, con 447 bits públicos y 183.138
muestras antes y después del flanco. Las 31 sondas quiet aportaron 62
muestras de perturbación/restauración. Los seis bancos drenados conservaron
su tráfico original sin cancelarlo mediante reset.

También se compararon los **13 bits internos del próximo conteo** en todos
los bancos y sondas. Los seis controles negativos de entrada divergieron
donde se esperaba; el revisor rechazó 34 corrupciones de evidencia y ocho
alteraciones de la cobertura del contador. Se comprobaron las 26 unidades
compiladas en cada uno de los 22 casos.

Solo cambia la asignación del conteo de admisiones en C respecto al monitor
RQ calificado: suma equilibrada de los cuatro bits de admisión. E, B,
referencias, oráculos y latencia permanecen como en esa referencia. La
[prueba aislada 997](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FOCUS_RESULTS_R1.md)
es una dependencia revisada de esta integración.

## Evidencia y respaldo

Resultados: **50.149.851 bytes y 1.575 archivos**, verificados en D: y NFS,
SHA-256 `34a6e1867be82524d4ecef0d4642454ae86b9e2f889a7b60db15a2112064519c`.
El snapshot `source-20261011T213614Z.zip` tiene SHA-256
`71ca9231a0c7778d228e149e94b804edaa3e1a345ea054a0d60a7e3a7923af9a`.

La revisión `-independent-symbolic-monitor-predecode-admission-count-full-review-r1.json`
tiene SHA-256 `7afa8221058d9bbf61019a0fec1a0843dda9dbdc4d7865fb19fd8a1c99a6e990`.
El coordinador cerró revisión y respaldo el 2026-10-11 a las 21:54:08 UTC.
Registros en `build/cloud-setup/20261002-board/` y certificados bajo
`artifacts/ntrpu-board-vivado/<tanda>/`.

## Decisión

La [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_SYNTH_R1.md)
comenzó como tanda `20261011T215539Z` después del cierre. Comparará recursos
y tiempos con RQ 984 bajo iguales restricciones; su replay espera la
revisión y el respaldo de esa síntesis. Este PASS RTL no acredita timing
físico, throughput, PCIe nativo ni placa.
