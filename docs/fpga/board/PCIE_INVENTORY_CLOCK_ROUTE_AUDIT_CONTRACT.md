# Auditoría del ruteo con BUFGCE interno

## Puerta y evidencia

Ejecutar después de que la implementación con reloj interno termine y
quede respaldada. El inspector debe cotejar archivo, fuentes, DCP de
entrada y reportes. La presencia de un DCP final/parcial y sus exportaciones
habilita lectura diagnóstica; no aprueba timing ni replay.

Usar `tools/ntrpu_board_pcie_inventory_clock_route_audit.tcl` con
`OPTIMIZED_CLOCK_DCP IMPLEMENTED_DCP NEW_OUTPUT`. Copiar y abrir ambos
DCP exactos. No optimizar, colocar, rutear, cambiar propiedades o generar
un bitstream. Mantener parte, top, reloj de 4 ns y ausencia de cajas negras
y latches. Comparar puertos mediante nombres materializados; consultar
propiedades sobre objetos de Vivado.

Exportar EDIF desde ambos DCP, inventarios, restricciones, propiedades de
reloj/buffer, conexiones y registros. Preservar la fuente BUFGCE instalada.
Comprobar I desde `clk`, CE desde VCC y O exclusivamente a todos los C
de FDRE. Registrar ubicación, rutas y estados de ambos tramos de reloj.

## Diagnóstico y aceptación posterior

Conservar timing global y subconjuntos FDRE→FDRE setup/hold, DRC y sus
objetos, fanout, flags de ruteo y listados completos de las categorías
de rutas incompletas. Comparar constraints antes/después sin introducir
excepciones. Distinguir el tramo externo del reloj de la distribución
interna y de las rutas de datos.

Auditar el EDIF recargado, incluyendo BUFGCE como elemento combinacional
en el DAG y sus controles de polaridad/habilitación/conectividad. Vincular
esa exportación al DCP archivado y al EDIF original; investigar cualquier
diferencia más allá de la fecha. Revisar cambios de primitivas derivados
de la implementación. No exigir igualdad de la huella lógica con el
checkpoint optimizado si el placer replicó o transformó lógica.

Sólo después de revisar restricciones, alertas, DRC, rutas y DAG se puede
reproducir la netlist implementada en los cinco bancos binarios R12.
Un resultado funcional no cierra timing, frontera nativa, reset/CDC,
DDR/512 GiB ni implementación de placa.
