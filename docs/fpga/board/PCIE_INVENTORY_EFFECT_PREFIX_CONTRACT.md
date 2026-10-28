# R15: composición de efectos en consultas reales RC

Contrato del 2026-10-09. Prerrequisito de la segmentación del inventario,
no implementación del pipeline nuevo ni afirmación de mejora de timing.

## Cambio permitido

Partir del productor R12 y del selector R3 exactos. Cambiar únicamente
`rc_prefix_state`: codificar los cuatro consumos elegidos por la máscara
de ancestría, componerlos en dos parejas ordenadas y aplicar el resumen
de 57 bits a la semilla del productor. Reutilizar el paquete de efectos
validado; no duplicar su implementación. La suma se limita a 2049 counts.

La máscara y los counts siguen procediendo del tráfico RC crudo. Las
semillas siguen siendo las filas propias del productor o el contexto
retenido elegido por R3. No alimentar estado de referencia al DUT.
Conservar la elección de contexto anterior, la liberación terminal,
actualizaciones EOP, admisiones RQ, validación y estados posteriores a fallo.

Conservar exactamente los registros, capacidad, throughput y latencia
pública de R12. R15 todavía no añade etapas ni versiones. El formato que
ahora alimenta las consultas permitirá estudiar forwarding entre etapas;
esa integración sigue su [plan separado](PCIE_METADATA_EFFECT_INTEGRATION_PLAN.md).

## Pruebas y aceptación

Exigir todos los bancos completos de R12, incluidas 22.339 filas/68
escenarios, comparación de las 256 filas del banco, captura nativa,
propiedad/ancestría, RQ, aritmética, máscaras X/Z, integración de prefijos
y suma por carril. Preservar los siete mutantes de integración y controles
del lector. La referencia R7 conserva su propio estado.

Añadir un productor real con el orden de prefijos invertido; debe fallar
al aplicar counts `[0,1]` a remaining 10/dirección 1. La recurrencia
correcta termina en remaining 0/dirección 11; el orden invertido termina
en remaining 7/dirección 4. No simular el mutante dentro del DUT normal.

Solo tras la revisión RTL archivada permitir síntesis OOC comparable a
4 ns, I/O cero, sin excepciones, y replay de la netlist propia en los cinco
bancos binarios heredados. Medir recursos y timing frente a R12/R14; un
resultado peor sigue siendo evidencia y no autoriza promover R15.

Generaciones/wrap, payload retenido entre etapas, retiro por secuencia,
revocación inmediata de vacío y monitor segmentado completo siguen
pendientes. Se mantienen los criterios nativos, DDR/512 GiB, cómputo y
diseño físico de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
