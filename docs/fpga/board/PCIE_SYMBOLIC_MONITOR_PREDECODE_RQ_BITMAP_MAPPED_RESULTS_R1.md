# Replay propio del bitmap RQ: tanda 985

## Resultado

`20261011T181716Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-binary-r1`
terminó con código 0 en **1.340,282 s**. La revisión independiente aprobó
**174.436 filas binarias nuevas en 16 bancos**, comparando 447 bits públicos.
Los seis controles negativos de entrada divergieron como se esperaba;
el revisor rechazó 34 corrupciones de evidencia. Diez sondas binarias
aportaron veinte muestras de perturbación/restauración.

La netlist procede de la síntesis propia 984, SHA-256
`f10f199bca923910f957b12c52bec048fbd42a38c23e4c677b8d304eb69040b7`.
Se conservaron las referencias y funciones de contabilidad calificadas.
El ensayo es funcional binario, sin SDF: no califica X/Z de la netlist,
timing físico, PCIe nativo ni placa.

## Evidencia y respaldo

Resultados: **83.180.709 bytes y 294 archivos**, verificados en D: y NFS,
SHA-256 `1ac780037714625c2eb49f47d383d4d3ddcb80b7d56b947dc66424ffce63d7db`.
El snapshot `source-20261011T181556Z.zip` tiene SHA-256
`f07eb1a0650079fbbde2ae40c5bca4f3099b19889efbb65d7c66b5632ffe9fb0`.

La revisión `-independent-symbolic-monitor-predecode-rq-bitmap-binary-review-r1.json`
tiene SHA-256 `3f43eff409952239a36ac0a3868cdb24d573644f223ceb9c9ce0037c8f278320`.
Registros en `build/cloud-setup/20261002-board/` y certificados bajo
`artifacts/ntrpu-board-vivado/<tanda>/`. El coordinador cerró revisión
y respaldo el 2026-10-11 a las 18:43 UTC, sin repetir la tanda.

## Decisión

Se conserva el resultado de [síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_SYNTHESIS_RESULTS_R1.md):
92.913 LUT y WNS estimado −2,768 ns. La equivalencia aprobada permite
continuar el diseño, pero el timing todavía incumple.

La cola pasa a la [prueba independiente de suma en E](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_RTL_R1.md),
seguida de la [comparación RQ/979 por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_STAGE_TIMING_R1.md).
Después quedan los bancos de productor B y consumidor tardío; ninguno
forma parte de la netlist validada aquí.
