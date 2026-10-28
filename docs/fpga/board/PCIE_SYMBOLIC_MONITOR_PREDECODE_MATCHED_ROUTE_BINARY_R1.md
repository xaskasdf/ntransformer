# Replay de las dos netlists ruteadas

Preparación 2026-10-12 UTC. Todavía no es un resultado de simulación.

## Dependencias

Esta tanda espera la [comparación física independiente R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_PHYSICAL_REVIEW_R2.md).
Primero deben terminar y quedar respaldados los bancos de pesos reales,
los relojes y replays optimizados de RQ984/E/D, las dos implementaciones
Quick y la reapertura de sus DCP. Conserva todos los trabajos anteriores.

El coordinador `continue_symbolic_monitor_predecode_matched_route_binary_r1.py`
ejecuta primero `rq-bitmap` y después `row-enable`, con dos hilos y
3.000 segundos por miembro. Espera hasta doce horas las dependencias;
un fallo detiene esta rama y conserva su diagnóstico.

## Entradas y aceptación

Cada congelador exige el archivo `routed_funcsim.v` de la implementación
del miembro correspondiente, su SHA-256, revisión física, identidad de
timing, reapertura y respaldos verificados. También conserva el vínculo
con su propio replay optimizado y la síntesis original. No intercambia
netlists entre diseños ni usa el antiguo monitor ruteado como referencia física.

Se mantienen los bancos RTL ya calificados de cada miembro: 16 bancos,
174.436 filas binarias, 447 bits públicos, diez sondas quietas con
restauración, seis mutantes de entrada y 34 corrupciones de evidencia.
Las funciones de comprobación no cambian. Se verifica la fuente de
BUFGCE antes y después, los puertos, las unidades compiladas y las
advertencias esperadas. No se demuestra la identidad interna de la
biblioteca precompilada de Vivado.

El coordinador respalda preparación, entradas y cada resultado terminal
en almacenamiento local y NFS. Solo publica éxito conjunto si ambas
netlists pasan su revisión independiente y tienen respaldo verificado.

## Alcance

Es equivalencia funcional binaria contra RTL de referencia, sin SDF.
Un PASS no aprueba setup/hold, PCIe nativo, DDR, placa ni inferencia
completa. La decisión de diseño deberá combinar estos resultados con
recursos y timing de la comparación física; no se promueve E/D de antemano.
