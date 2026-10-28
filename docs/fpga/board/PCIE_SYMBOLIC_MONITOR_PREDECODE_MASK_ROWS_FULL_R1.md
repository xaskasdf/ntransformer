# Integración completa de máscaras RC con índices fijos

Preparación 2026-10-12 UTC. Hipótesis de mapeo para las rutas de entrada
descritas en [el análisis RQ/E/B/C](PCIE_SYMBOLIC_MONITOR_PREDECODE_INPUT_PATHS_R1.md).
Todavía no demuestra mejora de recursos ni de timing.

## Cambio y referencia

Se integra el [selector de máscaras fijas](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_R1.md)
en la composición completa RQ. Cambia exclusivamente la formación de
`header_prefix_mask`; conserva puertos, registros y latencia de la cadena.
Selector, control y monitor tienen nombres propios para mantener las
referencias R3, numéricas y de valores originales. No combina las
variantes experimentales E, B, C ni E/D.

## Dependencias y ejecución

Exige PASS aislado del selector: 1.186.064 comprobaciones de propiedad
y ancestros, 1.026.544 con X/Z, 89.632 vectores de selección, tres
mutantes de fuente y trece controles de evidencia. Además espera el
cierre de la cola actual hasta los replays ruteados RQ/E/D; no disputa
el turno de las pruebas Llama ni altera las validaciones activas.

El coordinador `continue_symbolic_monitor_predecode_mask_rows_full_r1.py`
congela fuentes nuevas solo tras esas dependencias. La tanda dispone
de dos hilos por Tcl y 1.800 s; el transporte SSH usa stdin. Se respaldan
entradas y cualquier resultado terminal localmente y en NFS.

## Criterios

Los 26 archivos compilados y todos los vectores se vinculan a sus hashes.
Se conservan los 16 bancos completos: 183.058 filas nuevas, 447 bits
públicos, 31 sondas quietas, seis bancos drenados y seis controles de
entrada. Se exige comparación antes y después del reloj, comprobación
entre flancos y contabilidad sin cancelación de tráfico raw en los bancos
drenados. Los comprobadores y oráculos no cambian.

Un PASS permite preparar síntesis comparable y replay de la netlist
propia. No aprueba timing, PCIe, DDR ni placa. Un fallo se conserva para
depuración; no se modifica una referencia para obtener igualdad.
