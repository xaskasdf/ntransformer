# Síntesis comparable de las máscaras fijas

Preparación 2026-10-12 UTC. Depende del PASS y respaldo de la
[integración completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_FULL_R1.md).
No hay métricas del candidato todavía.

## Ensayo

Se conserva el preflight RTL de 529 filas y 31 sondas quietas antes de
sintetizar el monitor propio. Se verifican los 26 archivos de simulación
y 14 de síntesis, sus hashes y todas las referencias originales.

Parte `xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos de E/S de 0 ns,
`RuntimeOptimized`, jerarquía `rebuilt`, dos hilos y límite de 1.800 s.
No se añaden excepciones, multicycle ni cambios de latencia para obtener
slack favorable. La referencia es la síntesis RQ984 de `20261011T180537Z`.

## Revisión y decisión

El revisor enlaza la ejecución con el PASS completo, DCP, netlist, EDIF,
inventario, reportes y archivo recuperado. Mantiene la auditoría de
ciclos combinacionales, los doce controles de cobertura temporal y la
revisión de advertencias por fuente y línea.

El comparador exige iguales comandos de caracterización y versión
completa de Vivado. Publica recursos, WNS/TNS, hold, endpoints y cambios
del peor camino, incluyendo regresiones. No presupone que reemplazar
índices dinámicos mejore el circuito.

El coordinador `continue_symbolic_monitor_predecode_mask_rows_synth_r1.py`
espera la integración, congela fuentes y respalda antes de ejecutar.
Todo resultado terminal se conserva en D: y NFS. La síntesis revisada
solo habilita preparar el replay de su propia netlist. Faltan ese replay,
la implementación física y cualquier aprobación de PCIe/DDR/placa.
