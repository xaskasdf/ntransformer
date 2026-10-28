# Contexto del reloj y límites de los reportes

Observaciones del ruteo en curso `20261008T213940Z`, 2026-10-08.
No son resultados finales ni una aprobación de la placa.

## Reloj interno y frontera externa

El diagnóstico contiene un BUFGCE real, cuya conectividad ya pasó la
auditoría y el replay previo a colocación. Durante el ruteo Vivado emitió
`Route 35-197`: el puerto `clk` carece de `HD.CLK_SRC`, por lo que limita
la precisión temporal y las comprobaciones del origen externo.
La ubicación y las rutas internas del buffer deberán revisarse desde
el checkpoint final, separadas de esa frontera.

La guía histórica
[UG905 2023.2, buffers de reloj](https://docs.amd.com/r/en-US/ug905-vivado-hierarchical-design/I/O-and-Clock-Buffers)
describe el ruteo global de buffers dentro del módulo OOC y las
estimaciones usadas cuando el buffer está en el diseño padre. Su sección
de [restricciones de contexto](https://docs.amd.com/r/en-US/ug905-vivado-hierarchical-design/Context-Constraints)
define `HD.CLK_SRC` como ubicación de un buffer **externo** al módulo.
Es documentación 2023.2; no atribuirle una revisión 2026.1.

Consecuencia para el siguiente diseño: obtener el origen del reloj y
la frontera del padre nativo antes de asignar ese contexto. La ubicación
del BUFGCE interno no demuestra la ubicación de un buffer externo.
No cambiar el ensayo en curso ni añadir una ubicación arbitraria para
silenciar el aviso. La evidencia actual tampoco demuestra que el reloj
sea la causa única de las rutas de datos largas.

## Listados truncados del router

El archivo `tight_setup_hold_pins.txt` contiene sólo **10.000 filas**.
El propio log declara **17.563 pines** en ese diagnóstico inicial y
advierte el límite. Por tanto, omite 7.563 entradas. Sus histogramas por
registro y pin describen únicamente la parte listada.

En R1 anterior, el mismo diagnóstico inicial informó **15.965 pines**
y también guardó 10.000. El aviso posterior `Route 35-468` de **51.664
pines críticos** corresponde a otra estadística y fase; no intercambiar
esas cifras. El pie que anuncia una lista completa no anula el aviso
explícito de truncamiento.

Las observaciones y su revisión de alcance se conservan con sufijos
`-tight-setup-hold-observation.json` y
`-tight-setup-hold-coverage-review.json` en
`build/cloud-setup/20261002-board/`. Al cerrar la tanda, cotejar el prefijo
del log y los bytes del archivo con el respaldo. Los slacks finales y
la calificación funcional/física requieren sus revisiones independientes.
