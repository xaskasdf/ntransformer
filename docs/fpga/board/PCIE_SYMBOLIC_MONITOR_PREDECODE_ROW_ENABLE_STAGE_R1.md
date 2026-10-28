# E/D: comparación temporal por etapas pendiente

## Propósito y dependencia

Medir el efecto de registrar las habilitaciones por fila en E, frente al
DCP RQ 984. La tanda espera la aprobación y el respaldo del replay binario
de la netlist E/D propia. No comparte el DCP de otra variante.

Se reabren ambos DCP con el mismo Tcl y parser calificados en la tanda
987. No se ejecutan síntesis, optimización, colocación, ruteo ni cambios
de restricciones. Son estimaciones sobre diseños sintetizados.

## Mediciones y criterios

Se atribuyen todos los FDRE a siete grupos disjuntos: C, E, D, A, B, R y
TOP. Para cada uno de los 49 pares ordenados se registra presencia o
ausencia de rutas setup/hold y hasta veinte rutas peores. La cobertura
de registros debe coincidir con el inventario de la síntesis propia.

El revisor verifica identidades, endpoints, delays, niveles y primitivas;
diez alteraciones de evidencia deben rechazarse. E→D y B→R deben tener
rutas reportadas. Las rutas de E/S permanecen en el reporte global.
Máximo: dos hilos y 900 segundos.

Al revisar los resultados, comparar E→D junto con los demás pares y el
incremento real de FF/LUT. Un mejor WNS global puede ocultar regresiones
en otras rutas. Tampoco una mejora de interconexión estimada demuestra
mejor colocación física; el ensayo ruteado anterior mostró que esa
distinción importa. No se promueve arquitectura ni frecuencia con esta
tanda aislada.

## Evidencia reproducible

Plan: `build/cloud-setup/20261002-board/pcie-symbolic-monitor-predecode-row-enable-stage-queue-r1.json`,
SHA-256 `438413edb385f59bf0a77840783a9d51aa95c212a8f3dcd8c7608a00072fff15`.
Contiene cuatro derivaciones de helpers y conserva el Tcl/parser
calificado. El preparador enlaza los bytes de cada DCP al archivo
certificado de síntesis y a su replay aprobado antes de ejecutar.

El runner usa SSH por stdin. La tanda y sus fuentes se respaldan en D: y
NFS; el respaldo R17 incluye los snapshots pendientes de esta etapa.
