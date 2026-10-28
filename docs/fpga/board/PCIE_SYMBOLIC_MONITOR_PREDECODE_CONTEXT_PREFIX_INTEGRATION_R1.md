# Integración condicional del contexto por prefijo

## Estado

Fuentes, banco, lanzador y revisor preparados el 2026-10-11. No hay resultado
RTL integrado de esta variante. El éxito de la variante por hojas en la
tanda 961 no se hereda: el contexto por prefijo cambia la lógica combinacional
y debe aprobar de nuevo el banco completo.

## Alcance

Las nuevas copias `ntrpu_pcie_symbolic_retirement_fault_leaves_prefix_r1.sv`
y `ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_prefix_r1.sv` conectan
el módulo de contexto por prefijo. Solo cambian los nombres de módulos e
instancias; registros de retiro, puertos, E/D, capacidad, reset y latencia
de cinco clocks permanecen iguales a sus padres. La referencia R7 y sus
oráculos permanecen intactos.

El banco ejecutará **183.058 filas nuevas** en 16 bancos y seis controles
de entrada alterada. Verificará 447 bits públicos antes y después del flanco,
31 probes de quiet y los seis bancos drenados sin cancelar tráfico original.
Se mantienen las 24 fuentes compiladas por caso y las 34 comprobaciones
de evidencia corrupta del revisor que aprobó la tanda 961.

## Dependencias y ejecución

1. Terminar y respaldar la síntesis y el replay propios de la variante por hojas.
2. Aprobar el [banco aislado de contexto](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_R1.md):
   47.734 muestras, 10.240 casos de frontera, 32 vectores por lane y seis mutantes.
3. Congelar fuentes, verificar el respaldo previo y ejecutar la integración
   con dos hilos y límite de 1.800 s.
4. Respaldar cualquier salida terminal y revisar el resultado independiente.

El coordinador `continue_symbolic_monitor_predecode_fault_leaves_prefix_r1.py`
usa el congelador R2. La preparación R1 se detuvo al exigir un marcador
ausente en el coordinador; se conservaron sus archivos parciales. R2 reutilizó
el generador idéntico y creó una nueva revisión del congelador. No afectó
ninguna ejecución de Vivado.

## Evidencia de preparación y límites

Manifiesto de fuentes SHA-256:
`98229eaf8645720214b1eb37e2122e283af43c52f787ea7719068b63ee82e62e`.
Comprobaciones estáticas SHA-256:
`15726ba77cc20b33629a27b1e1ca0f8e24b567299b8d273db84f91d1ff883900`.

Las recetas reconstruyen exactamente sus padres; las funciones de revisión
de bancos y controles son idénticas a las de la integración aprobada.
Esto no equivale a simulación. Síntesis, replay de netlist y timing físicos
del contexto por prefijo siguen pendientes.
