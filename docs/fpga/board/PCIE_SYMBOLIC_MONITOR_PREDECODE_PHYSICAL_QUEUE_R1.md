# Predecode R1: cola de caracterización física

## Punto de partida

La [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_SYNTHESIS_R1.md) está
revisada y respaldada. Conserva WNS −3,584 ns y aumenta recursos frente a
la referencia. La [netlist sintetizada](PCIE_SYMBOLIC_MONITOR_PREDECODE_MAPPED_RESULTS_R1.md)
aprobó 174.436 filas y sus seis mutantes. Los pasos siguientes requieren
resultados aprobados, no solo código de salida 0.

## Dependencias encoladas

1. Replay sintetizado aprobado: 174.436 filas, 447 bits públicos, dieciséis
   bancos, diez sondas de quiet y sus restauraciones, seis mutantes de entrada
   y 34 controles de evidencia; respaldo D:/NFS verificado.
2. [Optimización BUFGCE aprobada](PCIE_SYMBOLIC_MONITOR_PREDECODE_CLOCK_RESULTS_R1.md):
   dos copias del mismo DCP sintetizado. Se verificaron igualdad del grafo
   tras contraer el buffer, puertos, registros, CE constante y reloj exclusivo.
3. [Replay optimizado aprobado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CLOCK_MAPPED_RESULTS_R1.md):
   174.436 filas, seis mutantes y 34 controles de evidencia contra la referencia
   independiente. La fuente de biblioteca coincide antes y después; esto no
   acredita la identidad interna de la biblioteca precompilada.
4. Colocar con `place_design` y rutear con `route_design -directive Quick`.
   Mantener parte, período de 4 ns, E/S cero y restricciones heredadas de la
   referencia. Límite: dos threads y 1.800 segundos; preservar también fallos.
5. Reabrir los DCP optimizado y ruteado sin cambiar diseño ni restricciones.
   Exportar EDIF, clocks, rutas, DRC, metodología, timing y los cuatro grupos
   de fronteras. Límite: dos threads y 900 segundos; auditar el grafo archivado.
6. Contrastar los diez grupos de timing tras reapertura y revisar
   independientemente restricciones, avisos, primitivas y límites OOC.
   La identidad de reportes no reemplaza esa revisión física.
7. Reproducir la netlist ruteada exacta: los mismos dieciséis bancos,
   174.436 filas, 447 bits, seis mutantes y 34 controles de evidencia.
   Mantener dos threads y límite de 3.000 segundos. Un PASS funcional
   no convierte los fallos de setup/hold en timing aprobado.

Los coordinadores `continue_symbolic_monitor_predecode_clock_r1.py`,
`continue_symbolic_monitor_predecode_clock_binary_r1.py` y
`continue_symbolic_monitor_predecode_physical_r1.py` viven en
`build/cloud-setup/20261002-board/`. Cada uno espera al anterior y conserva
fuentes congeladas y respaldo verificado en D: y NFS. Un fallo detiene sus
dependencias para inspección; no relanza la tanda ni altera trabajos externos.

## Criterio de comparación

Además del peor timing global, se exportan setup y hold desde los registros
de `e_stage` hasta `d_stage/symbolic_rows_reg*`, junto con el inventario
exacto de endpoints. La reapertura repite esos reportes. Se comparan recursos,
retardo lógico, interconexión y fanout con la referencia ruteada; `Quick`
no optimiza timing ni demuestra el mejor resultado posible.

El comprobador `tools/audit_ntrpu_timing_subset_r1.py` contrastó 124 rutas
entre los dos archivos certificados de la referencia y rechazó 18
alteraciones de evidencia. Verifica endpoints, tipo setup/hold, descomposición
del retardo, suma de tiempos y coincidencia tras reapertura. El coordinador
`continue_symbolic_monitor_predecode_timing_review_r1.py` aplicará esos controles
a los diez grupos de reportes del candidato después de la auditoría física.
Esta prueba del comprobador no aporta todavía medidas físicas del candidato.

La [revisión física preparada](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_REVIEW_R1.md)
usa su propio comprobador R2: reprodujo la evidencia certificada de referencia
y rechazó 22 corrupciones dirigidas. El coordinador
`continue_symbolic_monitor_predecode_physical_review_r1.py` espera la comparación
temporal respaldada y comprueba archivos, restricciones, rutas, reloj, recursos,
avisos y DRC. Sus cantidades proceden del candidato; no fija los conteos ni
la ubicación física de la referencia. Solo permite el replay diagnóstico.

El coordinador `continue_symbolic_monitor_predecode_route_binary_r1.py`
espera ambas auditorías y la revisión
`<tanda>-independent-monitor-predecode-physical-review-r1.json` antes de
congelar la netlist ruteada. Reutiliza el banco aprobado y exige la identidad
del archivo de ruteo real; no toma prestado el PASS de la netlist optimizada.

Después de la auditoría quedan pendientes la revisión independiente de los
resultados físicos y el replay de la netlist ruteada propia. No se trasladan
los PASS de variantes anteriores ni se califica PCIe nativo, DDR o placa.

Tras completar esa cadena, está encolado el
[experimento aislado de fallos por lane](PCIE_SYMBOLIC_LATE_FAULT_LEAVES_R1.md).
Comprueba una alternativa combinacional contra el RTL anterior; no cambia
la selección del monitor ni evita las pruebas de integración y timing.
