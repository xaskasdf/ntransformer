# Comparación de optimización del prefijo y la bandera

## Decisión de diseño

El replay propio de la bandera pasó; ambas netlists sintetizadas tienen
evidencia funcional binaria. Su peor ruta estimada atraviesa dos salidas
[binariamente constantes](PCIE_SYMBOLIC_MONITOR_PREDECODE_WINNER_CONES_R1.md).
Antes de invertir en otra cadena RTL completa, se comparará el efecto de
la optimización habitual sobre los DCP 966 y 970. La reducción de LUT o
el cambio del peor camino deberán medirse, no inferirse de la prueba local.

## Tanda acotada

`tools/ntrpu_board_pcie_symbolic_monitor_predecode_paired_opt_r2.tcl` abre
copias de ambos checkpoints revisados y aplica **`opt_design`** a cada uno.
Conserva el VU5P, período de 4 ns y retardos E/S de 0 ns. Presupuesto:
**900 segundos**, dos hilos. No modifica RTL, inserta buffers de reloj,
cambia constraints ni ejecuta colocación/ruteo.

Se guardan DCP, EDIF y netlist funcional optimizados; inventarios de
primitivas, registros/INIT y puertos antes/después; constraints, timing
global, ocho informes B→R/E→D/C→C/C→E, recursos, DRC, ayuda de comandos y versión.
Las dos copias de entrada quedan vinculadas por SHA-256 a las síntesis
originales. Cada resultado terminal, incluso fallo o timeout, se respalda.

## Dependencias y revisión

El coordinador `continue_symbolic_monitor_predecode_paired_opt_r1.py`
espera la revisión y respaldo del ensayo aislado 973, ya completo. El congelador
vincula la revisión R2 de la lectura por etapas, con las cifras conservadas.
También comprueba los replay propios 967 y 971, sus fuentes y semillas.
Los helpers derivados conservan sus padres; las comprobaciones estáticas
no se presentan como una ejecución Vivado.
La corrección del lector y la ampliación de fronteras se prepararon antes
de congelar o ejecutar este flujo; el registro `paired-opt-r2-unfrozen-adjustments`
conserva las modificaciones de los helpers aún no ejecutados.

La revisión exige grafos acíclicos, inventarios concordantes, puertos y
restricciones conservados, cobertura de timing, informes coherentes y
doce corrupciones de evidencia rechazadas. Registra eliminación/adición
de registros y cambios de INIT; no deduce equivalencia funcional de esas
listas ni del auditor del grafo.

## Uso del resultado

Comparar primero recursos, WNS/WHS y B→R/E→D bajo el mismo flujo.
Seleccionar después la siguiente netlist para su propio replay, o el
[borrador de distancia](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_R1.md)
para su ensayo aislado. El coordinador no inicia esos pasos automáticamente.
La optimización sin colocación sigue siendo una estimación; el problema
físico de la [tabla D](PCIE_SYMBOLIC_MONITOR_PREDECODE_D_TABLE_CONSTRAINTS_R1.md)
y los criterios V0–V9 permanecen abiertos donde falta evidencia.
