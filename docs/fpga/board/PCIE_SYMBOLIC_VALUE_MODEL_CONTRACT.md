# Contrato del modelo de metadatos por valor

Modelo de comportamiento R1 del 2026-10-09. Sustituye en el productor
R15 la tabla remaining/dirección y el contexto numérico retenido por
descripciones autocontenidas de 51 bits. No añade etapas ni se propone
para síntesis; debe validar la representación antes de segmentar el diseño.

## Estado propio y evaluación

La descripción conserva semilla de 25 bits y resumen canónico de consumos
de 26 bits, usando el paquete de efectos ya revisado. Las consultas parten
exclusivamente de la tabla simbólica o del contexto retenido propios.
R3 conserva la selección cruda y las prioridades. RQ inicializa una nueva
semilla; RC escribe el resumen seleccionado, sin reemplazar TC/atributos.

El contexto retenido guarda descripción, count y una condición de entrega
capturada en la cabecera. EOP borra propiedad, pero no borra los valores
retenidos de `original`, `delivered` y dirección en la referencia. Tampoco
un discontinue posterior recalcula el delivered anterior. Por ello la
evaluación no usa la propiedad ni el indicador normal actuales.

Las vistas numéricas de tabla/contexto sirven para comparar, validar y
calcular deltas. No alimentan la recurrencia de descripciones. Se conservan
las capacidades, el orden RC→RQ, la latencia pública y los contadores
existentes. La referencia R7 mantiene estado propio y no alimenta al modelo.

## Primer banco obligatorio

Ejecutar todos los bancos de R15: principal 22.339 filas/68 escenarios,
propiedad, ancestría, admisión RQ, aritmética de prefijos, máscaras X/Z,
integraciones crudas y suma por lane. Comparar las 256 palabras por fila
donde corresponde, incluidos todos los estados posteriores a fallo.
Conservar los siete mutantes de integración y el mutante real de orden
invertido en la función del nuevo modelo. Regenerar vectores y revisar
fuentes realmente compiladas, archivo y hashes de cada dependencia.

Un fallo termina esta tanda y se archiva como diagnóstico; corregir solo
en una revisión nueva con fuentes nuevas. Un PASS demuestra compatibilidad
finita de la representación en esos bancos, no un pipeline o timing nuevo.

## Gates posteriores

Añadir pruebas crudas dirigidas para retención prolongada, pérdida y
reemplazo de contexto, reutilización de tag, atributos independientes,
saturación y X/Z. Comprobar consultas, contexto y deltas explícitamente,
además del estado público; detectar un mutante que recargue la fila nueva
al EOP. Solo entonces fijar registros, profundidad y retiro del pipeline.

Se mantienen [el contrato completo](PCIE_INVENTORY_PIPELINE_CONTRACT.md),
[el plan por valor](PCIE_SYMBOLIC_CONTEXT_VALUE_PLAN.md) y
[la cola global](VIVADO_QUEUE.md). Cada ejecución, incluso fallida, conserva
fuentes, invocación, trazas y respaldo verificado en D:/NFS. R12 continúa
como referencia de implementación.
