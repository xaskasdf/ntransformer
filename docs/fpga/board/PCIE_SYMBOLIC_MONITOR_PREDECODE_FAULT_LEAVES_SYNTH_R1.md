# Síntesis condicional de la variante con fallos por lane

## Estado y dependencia

Flujo preparado y encolado el 2026-10-11; todavía sin resultado de síntesis.
Solo podrá ejecutarse después de aprobar y respaldar las **183.058 filas
nuevas** del [banco integrado](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_R1.md),
incluidos X/Z, los 447 bits públicos, 31 probes de quiet, seis controles
alterados y 34 comprobaciones de evidencia corrupta.
La prueba aislada de 37.494 muestras no sustituye este requisito.
El coordinador quedó en `waiting_for_integrated_rtl_review`; las
comprobaciones locales de sintaxis, fuentes y transformaciones dieron
`PREPARED_ONLY_STATIC_CHECKS_PASS` y no ejecutaron Vivado.

## Ejecución y comparación

`continue_symbolic_monitor_predecode_fault_leaves_synth_r1.py` conserva el
orden de la cola. Tras la aprobación integrada, congela fuentes, guarda un
respaldo previo y ejecuta una tanda con dos hilos y límite de 1.800 segundos.
Repite primero 529 filas dirigidas y de rangos con la referencia original;
la síntesis usa únicamente las 14 unidades del candidato.

Se mantienen dispositivo `xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos de
entrada/salida de 0 ns, `RuntimeOptimized`, jerarquía `rebuilt` y modo OOC.
Esto permite comparar recursos, profundidad y estimaciones de timing con
la síntesis previa, sin confundirlas con resultados posteriores al ruteo.
No se añaden registros, ciclos ni excepciones de timing.

## Evidencia y límites

Cada salida terminal se archiva y verifica en D: y NFS, incluso ante error.
Se conservan DCP, EDIF, netlist funcional, constraints, primitivas, informes
y logs. El auditor comprueba el grafo del DCP archivado, inventario y ocho
controles de ciclos o referencias inválidas. El revisor exige preflight,
compilaciones, constraints completos y advertencias reconocidas; cualquier
diagnóstico nuevo detiene la aprobación para inspección.

Manifiesto `pcie-symbolic-monitor-predecode-fault-leaves-synth-r1-preparation.json`:
SHA-256 `5b13b991c58cbd44d9be8712a6eef421ff7f8145020f9b1d55a9dd8146e7bebe`.
Las transformaciones conservan los algoritmos de revisión anteriores.
Una síntesis revisada habilitaría el replay de su propia netlist. La
equivalencia binaria, el cierre físico y una mejora de timing siguen pendientes.
