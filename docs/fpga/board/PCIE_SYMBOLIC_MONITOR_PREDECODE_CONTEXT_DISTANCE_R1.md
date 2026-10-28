# Experimento de distancia constante al último SOP

## Hipótesis y cambio preparado

La síntesis de la bandera conserva dos salidas CARRY8 binariamente
constantes en su peor ruta. El siguiente borrador selecciona la distancia
al SOP dentro del recorrido del prefijo: cuando una rama SOP se acepta,
asigna `context_sop_distance = lane - context_prior`. Ambas posiciones son
índices de los bucles de elaboración. RC y RQ usan después esa distancia
de once bits en lugar de restar el índice de SOP seleccionado.

El objetivo es permitir que Vivado seleccione constantes en lugar de
mapear aritmética posterior a la selección. Su efecto sobre recursos y
timing **aún no está medido**. Se deriva de la bandera aprobada en RTL,
conserva el orden SOP/EOP y los `if` originales, sin registros adicionales,
puertos nuevos ni ciclos de latencia. No reemplaza directamente nodos
mapeados por constantes ni añade excepciones de timing.

## Comprobaciones locales

El modelo abstracto comparó **1.114.112 combinaciones**: 65.536 máscaras
de eventos por 17 entradas de lane. Cuando existe SOP anterior, la distancia
seleccionada coincide con `unsigned11(lane - last_sop)`. Tres alteraciones
—sumar uno, elegir el primer SOP e incluir el SOP del lane actual— divergen.
Esto no demuestra por sí solo equivalencia RTL X/Z.

Se prepararon fuentes, banco de 47.734 muestras contra el padre y el
agregado original, seis mutantes, Tcl y revisores. Las recetas reversibles,
interfaces, sintaxis Python y estructura Tcl pasaron comprobaciones estáticas.
El [ensayo aislado 973](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_RESULTS_R1.md)
ya aprobó 47.734 muestras, seis mutantes y quince controles de evidencia,
con respaldo en D: y NFS. La integración y síntesis propias siguen pendientes.
La siguiente comparación prioriza [optimizar los dos DCP existentes](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAIRED_OPT_PLAN_R1.md)
para medir la eliminación de redundancia antes de gastar otra cadena RTL.

## Dependencias y criterios

El replay propio de 970 y la [lectura por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_STAGE_TIMING_RESULTS_R2.md)
terminaron revisados y respaldados antes del banco aislado. Sus criterios
de muestras, comparación por lane, mutantes y evidencia quedaron cumplidos.
La decisión de integrar espera ahora las medidas de optimización comparables.

Una integración posterior requerirá las 183.058 filas RTL, síntesis
comparable y replay propio de la nueva netlist. Los scripts preparados no
integran automáticamente el candidato ni inician ruteo.

## Artefactos

Fuente: `fpga/rtl/ntrpu_pcie_symbolic_late_fault_leaves_distance_r1.sv`,
SHA-256 `3df0003b9f4bd467a10d05aad6ca08d4b633927cd83e0fda53fc57a33943743d`.
Generador: `prepare_pcie_symbolic_late_fault_leaves_distance_drafts_r1.py`.
Prueba: `pcie-symbolic-monitor-predecode-context-distance-r1-proof.json`,
SHA-256 `2104230b905fa26b991539e86a2fe198c2d195c2eedbea6039519b62ed733059`.
Scripts, cinco manifiestos y prueba local están en
`build/cloud-setup/20261002-board/`; el padre permanece intacto.
