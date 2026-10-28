# Integración siguiente: selección cruda y contexto de metadatos

Propuesta del 2026-10-09. La unidad de efectos de 57 bits cubre una
transformación de fila ya seleccionada. No reemplaza el inventario PCIe.
Continuar desde R12 y conservar R14 como comparación experimental.

## Conectar eventos reales antes de ampliar el pipeline

Derivar de `ntrpu_pcie_request_inventory_r7.sv` y del selector R3 cada
efecto y su procedencia: beat, posición, tag, origen de semilla y decisión
de propiedad. Conservar RC antes de RQ y posición creciente dentro de RC.
Un consume solo representa una actualización de metadatos que realmente
permite la recurrencia; la cabecera no equivale por sí sola a esa escritura.

En R7, una cabecera terminal con propiedad libera el tag al aceptarse.
Su EOP retira `coriginal` del contador global; **no escribe la fila del
tag**. Un EOP no terminal, normal y con propiedad actualiza remaining y
dirección usando el contexto capturado. La admisión RQ reemplaza los
30 bits; sus dos escritores se resuelven después de las actualizaciones RC.
TC/atributos participan en la validación de cabecera, aunque una actualización
RC conserve esos campos. Vincular estas reglas a hashes y trazas reales.

El contexto terminal retenido debe conservar tag, propiedad, original,
dirección, delivered, longitud y posición hasta EOP, separado de una nueva
admisión con el mismo tag. No volver a leer la fila nueva para retirar bytes
del paquete anterior. Mantener también los estados posteriores a fallo:
descartar entradas inválidas cambiaría la referencia existente.

## Versiones, forwarding y retiro

Antes de congelar el RTL integrado, fijar profundidad y latencia; después
derivar el ancho de versiones de su máximo de reemplazos en vuelo. Un
contador finito no puede justificarse con «el wrap nunca ocurrirá». El
payload retenido puede durar más que el pipeline, por lo que no debe usar
una versión vencida para escribir la fila de una nueva solicitud.

Ordenar las transformaciones pendientes contra una semilla registrada y
retirar una sola vez. Probar colisión de todos los accesos en un tag/banco,
RC4→RQ2, reemplazos consecutivos, wrap deliberado y referencias antiguas.
Conservar un beat RQ y otro RC por ciclo, 256 tags, 257 lecturas y 1.052.672
bytes; no introducir stalls para esconder dependencias.

## Criterios del siguiente banco

Comparar selección y aplicación derivadas de tráfico crudo contra R7,
incluidas las diferencias históricas 4131, 4140, 4291 y 4300. La referencia
debe mantener su propio estado, sin alimentar semillas al DUT. Reutilizar
las pruebas de propiedad/ancestría y añadir generaciones/reemplazos,
terminal retenido, X/Z, reset frío y actividad con todas las etapas ocupadas.
Mutantes reales deben detectar escritura del propietario antiguo, inversión,
omisión, duplicación y declaración prematura de vacío.

La aprobación de esta unidad permite ese trabajo, no integración nativa.
Aplican [el contrato completo](PCIE_INVENTORY_PIPELINE_CONTRACT.md) y
[la cola de placa](VIVADO_QUEUE.md): revisión RTL completa, síntesis y
replay propio, CDC/reset, DDR/512 GiB e inferencia siguen abiertos.
