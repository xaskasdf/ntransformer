# Replay propio de la suma en E: tanda 992

## Resultado

`20261011T194744Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-effect-sum-binary-r1`
terminó y quedó revisada y respaldada el 2026-10-11 a las 20:13 UTC.
La revisión independiente aprobó **174.436 filas binarias nuevas en
16 bancos**, comparando 447 bits públicos. Los seis controles negativos
de entrada divergieron como se esperaba; el revisor rechazó 34 corrupciones
de evidencia. Diez sondas aportaron veinte muestras de perturbación y
restauración.

La netlist procede de la síntesis propia 991, SHA-256
`570bd5eecaf7afe159ce276234d66cf02c7ce0c841c216bddaad137f98f85687`.
Se conservaron las referencias y funciones de contabilidad calificadas
del replay RQ 985. El ensayo funcional binario no usa SDF: no califica
X/Z de la netlist, timing físico, PCIe nativo ni placa.

## Evidencia y respaldo

Resultados: **82.867.401 bytes y 294 archivos**, verificados en D: y NFS,
SHA-256 `fb6257ba620694b416da0a75bbe3cd825c98796f47f936858d94eba285351814`.
El snapshot `source-20261011T194621Z.zip` tiene SHA-256
`acbdc5c916d8612317e4a224a711986ff9ed13665f9339fd4cf95aaf9084ab46`.

La revisión `-independent-symbolic-monitor-predecode-effect-sum-binary-review-r1.json`
tiene SHA-256 `ea4744586e39c478c995523c8a3dc025bde9b059a4921815e473332ea60f0f68`.
Registros en `build/cloud-setup/20261002-board/` y certificados bajo
`artifacts/ntrpu-board-vivado/<tanda>/`. El coordinador R3 cerró esta etapa
a las 20:13:27 UTC sin repetir la simulación.

## Decisión

Se conserva el resultado de [síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_SYNTHESIS_RESULTS_R1.md):
92.644 LUT, 27.562 FF y WNS estimado −2,768 ns. La equivalencia aprobada
no cambia el incumplimiento de timing. La
[comparación por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMPONENT_STAGE_QUEUE_R1.md)
queda encolada detrás de las validaciones B/C para medir E frente a RQ 984.

El banco del [consumidor B con límites registrados y winner](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FOCUS_R1.md)
comenzó después del cierre de E como tanda `20261011T201441Z`.
E y B siguen siendo candidatos separados; no se atribuye una mejora
a su combinación.
