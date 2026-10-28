# Consumidor de límites: resultado aislado, tanda 989

## Resultado

`20261011T190901Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-late-focus-r1`
terminó con código 0 en **166,648 s**. La revisión independiente aprobó
**309.884 muestras nuevas**, comparando los 16 bits agregados y las
32 contribuciones de 16 bits por lane antes del OR.

Se ejecutaron 262.144 combinaciones enteras de límite, 10.240 casos
SOP/EOP, 17.352 casos de bits X/Z y 20.000 registros pseudoaleatorios.
Hubo 217.011 muestras sin fallos y 92.873 con fallos; ambas categorías
coincidieron con las referencias. El estado pseudoaleatorio final fue
`d8abb5e0`. Se conservaron el validador original y el winner por lane.

Los seis mutantes reales divergieron en las muestras exactas 1/2/3/4/5/3,
con fallo agregado `0000`→`0400`. El revisor rechazó 21 corrupciones de
evidencia y verificó ocho fuentes compiladas en cada uno de siete proyectos.

## Evidencia y recuperación

Archivo de **3.365.311 bytes y 363 archivos**, verificado en D: y NFS,
SHA-256 `e34a1d0ce03a7b3314a5f8352d93534665da42758f0fb88c7f6b8073ef6909ed`.
Snapshot `source-20261011T190744Z.zip`, SHA-256
`611bee94cbbf81691c01adc19efc68d5e67ea5ae00be53a985d1da99c3f16ebf`.
Revisión `-independent-bound-late-focus-review-r1.json`, SHA-256
`d295f4c4d40481639ff619cb59f217ac4bc51dd776670bcf254f68b9daeea1d6`.

Tras archivar la tanda, el coordinador R2 buscó un owner R2 inexistente;
el runner había creado correctamente el owner R1. Se ejecutó el revisor
original contra ese mismo archivo, con PASS. La [recuperación R3](PCIE_SYMBOLIC_MONITOR_PREDECODE_PIPELINE_RECOVERY_R3.md)
sella esa revisión y continúa sin repetir la simulación.

## Decisión y límites

Este ensayo consume límites combinacionales calculados desde B; no
conecta todavía el productor registrado. Además, su candidato deriva
del contexto por prefijo, mientras el monitor RQ usa winner.

Se preparó un [consumidor sobre winner](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FOCUS_R1.md)
para medir después una integración atribuible únicamente a los límites.
El PASS presente no acredita conexión completa, netlist ni timing.
