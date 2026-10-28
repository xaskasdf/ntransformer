# Comparación de ruteo Quick y Default

## Estado al 2026-10-11 UTC

La referencia Default terminó por timeout de **1.800,006 s**, sin DCP ruteado
final; su diagnóstico está revisado y respaldado. El candidato está en curso
en `20261011T112248Z` (tanda 960). Todavía no hay comparación temporal Default.
El replay de la netlist Quick de
predecode (`20261011T101255Z`, tanda 957) aprobó 174.436 filas binarias y está
respaldado. El ensayo aislado también terminó y aprobó su revisión corregida.
La referencia `20261011T104556Z` (tanda 959) conserva sus fuentes, entrada y
logs. Véase [timeout y recuperación](PCIE_SYMBOLIC_MONITOR_PREDECODE_DEFAULT_TIMEOUT_R1.md).
Cada resultado completo tendrá reapertura; toda tanda tiene respaldo en D: y NFS.

## Pregunta y controles

Quick dejó WNS −11,835 ns en la referencia y −13,498 ns en predecode.
El peor camino E→D de predecode tiene 17,314 ns de datos, de los cuales
17,145 ns corresponden a interconexión. Estos resultados no aprueban timing.
La ayuda de `route_design` capturada en Vivado 2026.1 describe Quick como
no dirigido por timing; `-no_timing_driven` desactiva el algoritmo temporal
predeterminado. Se comparará `route_design -directive Default`.

Cada diseño parte de **su mismo DCP colocado utilizado por Quick**:

| Diseño | Tanda de origen | SHA-256 de `placed.dcp` |
| --- | --- | --- |
| Referencia | `20261011T062034Z` | `982dfbbf3992ea899dddf89c3b72e440d971b266de65f0e8dff2cc7ba06f5dbd` |
| Predecode | `20261011T093541Z` | `c5153f0903068ad1e6027c5bad9da2a1d1e1feeb1bb234071c7fbf7b2cce38b2` |

No se vuelve a sintetizar, optimizar ni colocar. Se mantienen VU5P -2,
reloj de 4 ns, un BUFGCE y restricciones OOC existentes. Se registran LOC,
BEL y primitivas antes y después: el router puede cambiar el diseño físico.
Por tanto, se controla la colocación inicial, sin afirmar que la final será
idéntica. No hay nuevas excepciones, pines ni cambios de severidad.

## Ejecución y dependencias

La cola original de `continue_symbolic_monitor_predecode_timing_route_r1.py`
continuó mediante `continue_symbolic_monitor_predecode_timing_route_recovery_r2.py`
tras un cierre SSH al recolectar 959. Su estado de error original permanece
conservado; R2 recuperó únicamente la recolección y lanzó el candidato ya
congelado. El estado compartido representa ahora esa recuperación para
mantener la prueba integrada detrás del par físico.
Exige replay Quick propio aprobado y respaldado para ambos diseños. El
ensayo de hojas es una dependencia de planificación; su aprobación no
valida estos DCP. Un fallo suyo ya respaldado permite continuar esta
comparación independiente. No se desplaza ninguna validación en curso.

Cada ruteo usa dos hilos y un límite de 1.800 s; cada reapertura, 900 s.
Las fuentes se congelan y respaldan antes del lanzamiento. Todo resultado,
incluido timeout o error, se conserva. Una implementación incompleta no
se presenta como resultado ruteado completo. No hay reintento automático.

## Evidencia y siguiente puerta

Guardar timing global, diez grupos de caminos —incluidos setup y hold
E→D—, rutas, reloj, restricciones, geometría, recursos, advertencias,
DRC/metodología, DCP, EDIF y netlist. Comparar con Quick dentro de cada
diseño y después comparar ambos diseños. Reabrir el DCP y comprobar
identidad de los informes e inventarios.

Los revisores existentes de Quick **no autorizan Default**. Se requiere
revisión independiente de los cambios del router, restricciones y avisos,
seguida de replay de cada nueva netlist antes de cualquier promoción.
Timing, SDF, PCIe nativo, CDC/reset, DDR y placa siguen sin calificar.

## Revisión local preparada

El lector `audit_ntrpu_symbolic_monitor_predecode_route_measurements_r2.py`
reprodujo los 164 caminos de los diez grupos del resultado Quick certificado
de predecode, sus recursos y sus cambios de primitivas. Rechazó doce
corrupciones de informes e inventarios. Un aviso crítico desconocido queda
registrado y nunca concede autorización de replay. La geometría LOC/BEL
todavía necesita ejecutarse contra los informes Default reales.

La primera versión exigía una sola aparición de cada recurso en todo el
informe; Vivado lo repite en tablas de distribución y SLR. R2 selecciona
las secciones principales por su encabezado y exige una sola fila allí.
El verificador R3 también comprueba los hashes de los informes E→D antes
de comparar sus campos temporales con la revisión previa. Las versiones
fallidas y sus correcciones se conservan. Prueba:
`pcie-symbolic-monitor-predecode-route-measurements-r3-verification.json`,
SHA-256 `bd7d64fa88914be856f1bd1f720fc60d956a832cea2452cfee411b82c6916f80`.

`continue_symbolic_monitor_predecode_timing_route_measurements_r2.py`
espera los respaldos de cada reapertura. Enlaza los informes al archivo
certificado, compara geometría y mediciones, y ejecuta la auditoría EDIF
de ciclos y conectividad del reloj con sus controles heredados. No ocupa
el turno de Vivado. Estos pasos preparan la revisión física completa;
no aprueban advertencias, equivalencia funcional ni timing.
El observador R1 detenido también se conserva. R2 mantiene sus algoritmos;
el resultado incompleto de referencia se registra como omitido, sin métricas
finales ni reapertura inventadas.
