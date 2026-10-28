# Integración B-bound/winner: tanda 994

## Resultado

`20261011T202003Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-winner-full-r1`
terminó con código 0 en **782,546 s**. La revisión independiente aprobó
**183.058 filas nuevas en 16 bancos**, con 447 bits públicos y 183.138
muestras antes y después del flanco. Las 31 sondas quiet aportaron 62
muestras de perturbación/restauración. Los seis bancos drenados conservaron
su tráfico original sin cancelarlo mediante reset.

Los seis controles negativos de entrada divergieron donde se esperaba;
el revisor rechazó 34 corrupciones de evidencia. También verificó los
**75 bits de límites registrados B** en todas las muestras y rechazó
ocho alteraciones de su cobertura. Las 27 unidades compiladas corresponden
al snapshot previsto, incluyendo la referencia de delivery original.

El candidato conecta el productor B calificado al consumidor winner,
manteniendo C RQ, E original y la latencia de cinco flancos. El resultado
prueba esta integración; no combina la variante de suma E ni la de
contador C.

## Evidencia y respaldo

Resultados: **50.596.966 bytes y 1.597 archivos**, verificados en D: y NFS,
SHA-256 `f700c0e28aa645bc64b011ddae13c4dba3cea18166ac48deed311d1cafd0a053`.
El snapshot `source-20261011T201841Z.zip` tiene SHA-256
`cc4551992076b7c8e6f0c94609287a7aaf66b3248d09293f954a7948d5efba62`.

La revisión `-independent-symbolic-monitor-predecode-bound-winner-full-review-r1.json`
tiene SHA-256 `c44bb283f9c9c5f3e7608fac78329e0f28db53323e86f05775dfd98be95ae8ee`.
El coordinador cerró revisión y respaldo el 2026-10-11 a las 20:36:36 UTC.
Registros en `build/cloud-setup/20261002-board/` y certificados bajo
`artifacts/ntrpu-board-vivado/<tanda>/`.

## Decisión

La [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_SYNTH_R1.md)
comenzó como tanda `20261011T203802Z`, después de ese cierre. Su comparación
es con RQ 984 bajo las mismas restricciones. El replay de la netlist propia
espera la síntesis revisada y respaldada. Este PASS RTL no acredita timing
físico, throughput, PCIe nativo ni placa.
