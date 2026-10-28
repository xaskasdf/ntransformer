# Integración condicional de fallos por lane

## Estado al 2026-10-11 UTC

Fuentes y banco completo preparados; **sin simulación integrada todavía**.
La prueba aislada de [fallos por lane](PCIE_SYMBOLIC_LATE_FAULT_LEAVES_R1.md)
aprobó 37.494 muestras y seis mutantes. Esta integración no selecciona una arquitectura ni sustituye
el monitor actual. Se mantienen las tandas Quick/Default encoladas.

## Alcance exacto

La copia `ntrpu_pcie_symbolic_retirement_fault_leaves_r1.sv` cambia únicamente
el nombre del módulo y la instancia del cálculo de fallos tardíos. Conserva
los registros de retiro, acumulación de fallos, rangos de bytes y reset frío.
La copia `ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1.sv` instancia
ese retiro y mantiene C/E/D/A/B, puertos, capacidad, latencia de cinco clocks
y expresión de quiet. El banco cambia solo el nombre de su candidato;
referencia R7, monitor original, modelo y estímulos permanecen iguales.

El generador revirtió las sustituciones y reconstruyó exactamente las tres
fuentes anteriores. Esto verifica el alcance textual, no equivalencia RTL.
Manifiesto `pcie-symbolic-monitor-predecode-fault-leaves-r1-sources.json`,
SHA-256 `a0d3045c0d8d94171ab071b96b8eafde03c59d66dc7e18e3164386638cfae2dd`.

## Prueba preparada y orden

`tools/ntrpu_board_pcie_symbolic_monitor_predecode_fault_leaves_r1.tcl`
prepara 22 casos: los 16 bancos originales completos, 183.058 filas con X/Z,
y seis controles de entrada alterada. Compila 24 fuentes SystemVerilog
por caso. Compara los 447 bits públicos antes y después del flanco y conserva
31 probes de quiet, cinco ciclos de drenaje y los seis bancos sin cancelación
de tráfico original. No reutiliza filas aprobadas por otra variante.

La ejecución requiere el resultado aislado aprobado y respaldado, fuentes
congeladas, respaldo previo y un revisor propio del resultado integrado.
Se ejecutará después de que las tandas físicas actuales liberen su turno,
sin exigir que esta comparación RTL independiente apruebe su timing.
El lanzador y revisor integrado están preparados a partir del flujo anterior.
`continue_symbolic_monitor_predecode_fault_leaves_r1.py` espera la cola física,
congela las fuentes, verifica respaldo previo y ejecuta una sola tanda de
22 casos con dos hilos y límite de 1.800 s. La revisión exige los mismos
primeros puntos de divergencia de los seis controles y 34 corrupciones
rechazadas. Toda salida terminal se respalda, incluido un fallo.

Una aprobación RTL permitiría síntesis propia y comparación de recursos,
profundidad y timing. Después seguirían replay de la netlist y validación
física. El árbol OR expresado en RTL no garantiza un árbol físico equivalente
ni resuelve por sí solo el camino E→D observado con Quick.
