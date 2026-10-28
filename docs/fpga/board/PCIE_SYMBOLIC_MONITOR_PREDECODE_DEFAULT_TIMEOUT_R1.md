# Default: dos timeouts y recuperación de evidencia

## Resultado observado

Tanda 959, `20261011T104556Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-timing-route-reference-r1`:
salida **124**, límite de 1.800 s, duración **1.800,006 s**. Terminó el
2026-10-11 a las 11:16:02 UTC durante `Phase 5.2 Global Iteration 1`.
La última cuenta publicada por la iteración anterior fue de 47 nodos con
solapamientos; no es un inventario final de rutas.

Vivado registró congestión global/corta de nivel 5 y temporal de nivel 6,
además de 30.310 pines con restricciones ajustadas de setup y hold.
La lista de pines queda limitada por Vivado a 10.000 entradas. Estos avisos
describen el intento; el timeout no demuestra imposibilidad de ruteo.

No se exportaron `route_attempt.dcp`, `routed.edf`, `routed_funcsim.v` ni
informes de timing final. Los checkpoints conservados son la entrada
colocada y su copia previa al ruteo. No se atribuyen WNS/WHS finales a esta
tanda ni se compara una estimación previa con el resultado Quick como mejora.

## Recolección y respaldo

El proceso remoto ya había terminado cuando una conexión SSH se cerró
durante la recolección. Se conservó el estado original del coordinador y se
ejecutó exclusivamente el tramo de recolección/archivo del lanzador congelado.
No se relanzó Vivado ni se modificaron los artefactos originales.

Archivo terminal verificado en D: y NFS: **79.127.766 bytes**, SHA-256
`29a218586a3f696d1cfcd429262f7865a666d64a36771896608a9d1fb883ccb2`.
La revisión independiente enlazó invocación, console, 22 informes, cuatro
artefactos y fuentes con el archivo certificado. Preserva los 102 avisos
emitidos sin aprobarlos como cierre físico.

Resultado `REVIEWED_DEFAULT_ROUTE_TIMEOUT_DIAGNOSTIC_R1`, revisión SHA-256
`dd0d4e4c62b6e16cb46b32ff65da0306ef1a14f7c348535d5683e82b7dc90cfc`.

## Continuación

El candidato, tanda 960 `20261011T112248Z`, también agotó el presupuesto:
salida **124**, **1.800,007 s**, última fase `Phase 5.1 Global Iteration 0`.
La última cuenta publicada fue 8.359 solapamientos; registró congestión
temporal de nivel 5 y 10.951 pines con restricciones ajustadas. Tampoco
exportó DCP ruteado, netlist funcional ni timing final. Las fases y cuentas
intermedias no establecen una clasificación de calidad entre arquitecturas.

Su archivo de **80.314.391 bytes** quedó verificado en D: y NFS, SHA-256
`f8b092596f75292ebb905d5c62e2f5fc0107345521a37eeb689331480ec29da8`.
La revisión independiente del timeout tiene SHA-256
`f67756d31b3a4cb9260e730f79968bef3ab684de029313e23b544a01f25ab0c4`.
Las reaperturas y mediciones finales se omitieron explícitamente para ambos
intentos: no hay un resultado Default con el que comparar Quick.

La cola avanzó al banco RTL integrado de 183.058 filas, tanda 961
`20261011T115750Z`. Síntesis y replay propios siguen condicionados a sus
revisiones; no se repite automáticamente ninguno de los ruteos.

Un intento más largo o una modificación física se decidirá con ambos
resultados. La cola actual no repite automáticamente la referencia ni
promueve una arquitectura por una simulación funcional o un timeout.
