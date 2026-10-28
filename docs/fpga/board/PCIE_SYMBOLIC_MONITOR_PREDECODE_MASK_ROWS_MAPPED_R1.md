# Replay propio de la netlist de máscaras fijas

Preparación 2026-10-12 UTC. Depende de la
[síntesis propia revisada y respaldada](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_SYNTH_R2.md).
Todavía no ejecuta ni aprueba la netlist del candidato.

## Entradas y referencia

El congelador recupera `monitor/pcie_symbolic_monitor_predecode_mask_rows_monitor_funcsim.v`
del archivo verificado de la síntesis propia. Comprueba hash, puertos y
ausencia de colisiones con los nombres de las referencias RTL.
Las referencias originales y los bancos binarios proceden del replay
RQ985 calificado. No se reutilizan resultados de ese diseño como
resultados de las máscaras fijas.

## Cobertura y aceptación

Se mantienen 16 bancos, 174.436 filas nuevas y 447 bits públicos;
diez sondas quietas con restauración, seis mutantes reales de entrada
y 34 controles de corrupción de evidencia. Se comprueban las 19
unidades SystemVerilog compiladas, `glbl`, vectores, contabilidad,
advertencias y reportes contra el archivo recuperado. Las funciones
de comparación numérica y de protocolo no cambian.

El banco usa dos hilos por Tcl y hasta 3.000 s. Mantiene muestras antes
y después del flanco y sondas entre flancos, con los tiempos ya
calificados para simulación funcional de netlist. No introduce SDF.

## Cola y alcance

`continue_symbolic_monitor_predecode_mask_rows_binary_r1.py` espera
el coordinador de síntesis R2. Congela entradas, respalda y ejecuta
una sola vez; conserva cualquier resultado terminal en D: y NFS.
Un error detiene la rama para revisión.

Un PASS demostrará equivalencia binaria dentro de estos bancos.
La decisión de arquitectura deberá considerar también recursos y
timing estimado; ni el replay ni la síntesis aprueban implementación
física, PCIe nativo, DDR, placa o inferencia completa.
