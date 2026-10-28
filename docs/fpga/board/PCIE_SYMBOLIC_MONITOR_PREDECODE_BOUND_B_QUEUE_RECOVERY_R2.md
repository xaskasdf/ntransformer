# Recuperación de la cola tras la revisión de B

## Diagnóstico conservado

La tanda B 988 terminó y quedó archivada. El revisor R1 rechazó el
primer testigo del mutante de reset; esto detuvo de forma controlada
los coordinadores R1 del consumidor y de E completo antes de crear
sus owners o snapshots. La [revisión R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_RESULTS_R2.md)
resuelve el error de expectativa sobre la evidencia existente.

El generador `prepare_pcie_symbolic_monitor_predecode_bound_b_review_recovery_r2.py`
conserva las recetas, hashes y razón de la corrección. El finalizador
`continue_symbolic_monitor_predecode_bound_b_focus_r2.py` solo sella
la revisión y publica su cierre; no invoca Vivado.

## Cadena retomada

1. Productor B: revisión R2 y respaldo de sus 211.606 filas y seis mutantes.
2. Consumidor de límites: coordinador/congelador R2, banco R1 intacto
   de 309.884 muestras, seis mutantes y 21 controles de evidencia.
3. E completo: coordinador/congelador R2, variante y banco R1 intactos
   de 183.058 filas; B conserva su implementación original en este monitor.
4. Síntesis E: espera E completo R2, compara únicamente contra 984.
5. Replay E: espera síntesis revisada y respaldada; usa su propia netlist
   para ejecutar 174.436 filas binarias nuevas.

Los runners, revisores del consumidor/E y fuentes RTL permanecen iguales.
Los nuevos congeladores incorporan sus dependencias de recuperación.
La preparación de síntesis aún no ejecutada se amplió mediante un
manifiesto R2; la preparación R1 y su generador se conservan.

La auditoría estática de mapeo R1 confundió el literal `.py` de
`removesuffix` con un nombre de archivo. R2 corrige esa comprobación y
verifica nombres reales, recetas, fuentes y enlaces de la nueva cadena.
Es una auditoría de preparación, no evidencia de ejecución ni de timing.

## Evidencia

`pcie-symbolic-monitor-predecode-queue-recovery-after-bound-b-r2.json`
registra las derivaciones, SHA-256
`1a748930613539a0ba6611e897729e33857e505ab9ba6d58a75d4f01ec559a4c`.
La auditoría `pcie-symbolic-monitor-predecode-effect-sum-mapping-r2-queue-audit.json`
tiene SHA-256 `c8931cdfef3a8ae258b2018e44068cac22ca299348ce9a9430f70e8f7a9afbf6`.
Se preserva cada desenlace y revisión en D: y NFS mediante la cápsula R11.
