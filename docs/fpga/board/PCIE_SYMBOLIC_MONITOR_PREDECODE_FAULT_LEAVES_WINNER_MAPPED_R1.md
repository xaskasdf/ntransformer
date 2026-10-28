# Replay condicional de la bandera de contexto

## Dependencia

Flujo preparado sin resultado mapeado. Después de la prueba aislada y la
integración completa, se exige [síntesis propia revisada y respaldada](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_SYNTH_R1.md).
El congelador enlaza el hash de la netlist realmente exportada, sus puertos,
fuentes y archivo terminal; conserva las referencias independientes originales.

## Cobertura requerida

Una tanda compara las 174.436 filas binarias nuevas de 16 bancos y los
447 bits públicos antes y después del flanco. Incluye diez sondas de quiet
con restauración, seis controles de entrada alterada y 34 corrupciones de
evidencia. Los seis bancos drenados deben conservar el tráfico original sin
cancelarlo. No se heredan filas aprobadas de otra variante.

`continue_symbolic_monitor_predecode_fault_leaves_winner_binary_r1.py`
respeta las dependencias, congela las entradas, verifica el respaldo previo
y ejecuta con límite de 3.000 s y dos hilos. Toda salida terminal queda
respaldada en D: y NFS antes de la revisión; exit 0 no basta para aprobar.

## Alcance

Se mantienen los comparadores y las funciones de revisión de la referencia.
Los cambios del banco se limitan a nombres de módulo y top.
Esta simulación funcional no incluye SDF ni acredita timing físico, una
frecuencia de hardware o funcionamiento de la placa. La cobertura RTL X/Z
de 183.058 filas permanece como requisito separado.
