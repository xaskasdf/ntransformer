# Síntesis condicional del contexto por prefijo

## Estado y condición de entrada

Flujo preparado; todavía no se ejecutó síntesis de esta variante.
`continue_symbolic_monitor_predecode_fault_leaves_prefix_synth_r1.py`
espera la aprobación y el respaldo de las **183.058 filas nuevas** de la
[integración por prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_INTEGRATION_R1.md).
La prueba aislada de 47.734 muestras y los resultados de otras variantes
no sustituyen esa integración.

## Ejecución

El congelador enlaza la revisión integrada, hashes, fuentes y artefactos
certificados. Guarda una copia previa y ejecuta una sola tanda con dos
hilos y límite de 1.800 s. Repite 529 filas dirigidas y de rangos con la
referencia original; después sintetiza las 14 unidades del candidato.

Se mantienen `xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos de E/S de 0 ns,
`RuntimeOptimized`, jerarquía `rebuilt` y modo OOC. No se añaden ciclos,
registros ni excepciones de timing. Se medirán LUT, FF, CARRY8, profundidad,
setup y hold bajo esos mismos parámetros.

## Revisión y comparación

Cada salida terminal se archiva en D: y NFS. La revisión exige preflight,
compilaciones, inventario, constraints y avisos reconocidos. La auditoría
del DCP comprueba referencias, ausencia de ciclos combinacionales y sus
controles negativos. Un aviso nuevo exige inspección; no se aprueba
automáticamente por provenir de un cambio intencional del RTL.

El comparador encolado enlaza los informes archivados y contrasta la
variante con la referencia 951, verificando también los builds de Vivado.
La variante por hojas de la tanda 962 conserva sus métricas como segundo
antecedente: 105.147 LUT, 27.542 FF, 31 niveles y WNS estimado −3,746 ns.

Manifiesto de preparación SHA-256:
`2884d82282354cc4f9650eceba5ff733c1e81fb5dd566e603d28fa1724c21d42`.
La síntesis revisada habilitará el [replay propio](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_MAPPED_R1.md).
No hay mejora demostrada, implementación física ni frecuencia de hardware
aprobada para el contexto por prefijo.
