# Síntesis propia de límites B con winner — R1

## Dependencia y objetivo

Esta tanda espera el PASS revisado y respaldado de la
[integración completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FULL_R1.md).
La congelación exige sus 183.058 filas, seis controles negativos, 34
controles de evidencia y comprobación de los 75 bits de límites en los
16 bancos. También conserva los certificados del productor y consumidor
aislados. Un diagnóstico detiene los sucesores; no se repite automáticamente.

Se compara contra la síntesis RQ 984, que mantiene el mismo E original.
El cambio es registrar límites junto con B y consumirlos en la selección
winner. No se mezclan los resultados de la variante de sumas E.

## Ejecución y revisión

El preflight compila 27 fuentes y repite 529 filas de monitor/range_return,
31 perturbaciones de quietud y las comprobaciones de alineación B.
La síntesis usa 14 fuentes, `xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos
de E/S de cero, `RuntimeOptimized` y jerarquía `rebuilt`. El Tcl limita
el trabajo a dos hilos y el lanzador a 1.800 segundos.

Se archivan DCP, netlist funcional, EDIF, restricciones y reportes.
El revisor comprueba las fuentes efectivamente compiladas, recursos,
advertencias, cobertura de restricciones, ausencia de excepciones y
grafo combinacional acíclico. La comparación exige iguales dispositivo,
reloj, estrategia y versión del compilador.

## Respaldo y siguiente paso

Los helpers y fuentes se fijan por SHA-256 antes de ejecutar. R14 conserva
también snapshots pendientes; cada resultado terminal queda en D: y NFS.
La evaluación de recursos y timing habilita únicamente el
[replay de su propia netlist](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_MAPPED_R1.md).
La simulación RTL no establece mejora de timing; una estimación favorable
tampoco sustituye colocación, ruteo ni validación de hardware.
