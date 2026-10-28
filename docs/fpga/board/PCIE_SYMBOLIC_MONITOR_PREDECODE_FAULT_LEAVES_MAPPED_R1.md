# Replay propio de la variante con fallos por lane

## Dependencias

Preparado el 2026-10-11; aún sin resultado de simulación mapeada.
El coordinador espera la [síntesis propia revisada](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_SYNTH_R1.md),
que a su vez exige el banco RTL integrado de 183.058 filas aprobado y respaldado.
No se utiliza una netlist de otra variante para aprobar este candidato.

## Banco conservado

El testbench cambia únicamente su nombre y el módulo de la instancia
`mapped`. Las referencias RTL originales con alias, sus oráculos y los
estímulos conservan sus hashes. Los algoritmos de revisión son textualmente
idénticos a los usados en el replay previo aprobado.

La tanda ejecuta 16 bancos y **174.436 filas binarias**, con comparación de
los 447 bits públicos antes y después del flanco, diez probes de quiet con
restauración y seis mutantes reales de entrada. Los seis bancos de drenaje
deben conservar cero cancelaciones de tráfico original. El revisor rechaza
34 formas de evidencia alterada.

La exclusión de filas X/Z completas pertenece al dominio binario de esta
simulación. Su validación de cuatro estados permanece en el banco RTL
integrado obligatorio; no se declara equivalencia de síntesis para X/Z.

## Ejecución y evidencia

`continue_symbolic_monitor_predecode_fault_leaves_binary_r1.py` espera la
aprobación, congela el vínculo al `funcsim.v` del archivo de síntesis, guarda
el respaldo previo y ejecuta una sola tanda con dos hilos y límite de
3.000 segundos. Verifica puertos, fuentes compiladas, hashes y cuerpo de
`glbl`; conserva artefactos y resultados terminales en D: y NFS.

Manifiesto `pcie-symbolic-monitor-predecode-fault-leaves-binary-r1-preparation.json`,
SHA-256 `0c8ae915a2e527e7d42d57f93a715d114062424f65868a4a3e97ac8c375020c8`.
El revisor autoriza únicamente la funcionalidad observada en estos bancos.
No hay SDF: red de reloj, equivalencia tras optimización/ruteo y timing físico
requieren sus propias tandas y revisiones.
