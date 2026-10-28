# Replay propio del contexto por prefijo

## Estado y dependencias

Banco y flujo preparados, sin resultado de simulación mapeada.
El coordinador espera la [síntesis propia revisada](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_SYNTH_R1.md)
de esta variante y su respaldo certificado. Conserva la secuencia: prueba
aislada, integración completa, síntesis y replay.

## Alcance de la prueba

Se utilizará exclusivamente la netlist funcional exportada por esa síntesis,
con hash y top verificados. El banco mantiene las referencias independientes,
los estímulos originales y las funciones de revisión ya utilizadas por la
referencia. Los cambios en el banco se limitan a los nombres de top y módulo.

La ejecución debe cubrir **174.436 filas binarias nuevas** en 16 bancos,
comparar los **447 bits públicos** antes y después del flanco y preservar
las diez pruebas de quiet con restauración. Incluye seis controles de entrada
alterada y 34 comprobaciones de evidencia corrupta. Los seis bancos drenados
deben conservar tráfico original, sin cancelaciones que oculten pendientes.

`continue_symbolic_monitor_predecode_fault_leaves_prefix_binary_r1.py`
congela la identidad de la netlist y sus fuentes, verifica el respaldo previo
y ejecuta una sola tanda con límite de 3.000 s y dos hilos. Toda salida
terminal se respalda, incluso errores y timeout. El revisor comprueba las
fuentes realmente compiladas y la netlist archivada; un código 0 por sí solo
no cierra la prueba.

## Límites

Manifiesto de preparación SHA-256:
`e5df7b35908dd0620c1bb9b6f25dd9eace4ffc98eecdb30775b860c49133fda2`.
No se reutilizan filas aprobadas por otras variantes. El banco no lleva
SDF y no demuestra equivalencia de síntesis de cuatro estados: esa cobertura
permanece en el banco RTL separado de 183.058 filas.

El resultado tampoco aprobará reloj optimizado, colocación, ruteo ni la
placa. Cualquier paso físico posterior se decidirá con las métricas reales
de esta síntesis y el resultado de este replay.
