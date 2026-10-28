# Después de R14: segmentar consulta y transición RC

Propuesta del 2026-10-09 UTC. No es una implementación ni una calificación.
R14 ya tiene síntesis revisada; su replay propio y censo por destinos deben
cerrarse antes de elegir la nueva base. Conservar R12 como referencia y
conservar R14 para comparar el cambio de registros.

## Motivo y alcance

R14 mejora WNS OOC de −7,068 a −6,442 ns, pero empeora TNS. La peor ruta
revisada mide 10,453 ns y 40 niveles, desde `rc_user[76]` hasta
`producer/byte_counter/saved_removed_reg[12]/D`. Su reporte atraviesa lógica
de selección, prefijos/metadatos, bytes entregados y reducción RC. Los nombres
de las celdas ayudan a localizar la estructura; no prueban causalidad ni
predicen el resultado de otro mapeo.

El siguiente trabajo debe abordar consulta, forwarding y transición como
parte de la [arquitectura segmentada](PCIE_INVENTORY_PIPELINE_CONTRACT.md).
Otro registro alrededor de la suma, por sí solo, no satisface ese contrato.
No repetir ruteo Default agotado ni promover una variante por WNS aislado.

## Primer entregable: transición componible y sus límites

Definir una representación explícita de la actualización de metadatos de
un tag: identidad/orden del evento, origen del contexto, cantidad recibida,
liberación terminal, actualización de dirección/remaining y nueva admisión
RQ. Distinguir transformación relativa del estado y reemplazo por una nueva
generación. El payload terminal retenido debe conservar su contexto antiguo.

Probar si las transiciones de beats todavía no retirados pueden componerse
desde una semilla registrada, sin encadenar el resultado aritmético completo
del beat anterior. Derivar la composición de las asignaciones de R7 y de
los anchos/signos efectivos; no suponer saturación convencional ni usar el
byte count del descriptor como metadato esperado. Conservar los casos
4131, 4140, 4291 y 4300 del [diagnóstico de integración](PCIE_INVENTORY_INTEGRATION_GAPS.md).

El primer banco debe comparar la composición con una recurrencia independiente
y ordenada de esas asignaciones: mismo tag en beats consecutivos, todos los
eventos al mismo banco, RC4→RQ2, reutilización terminal con payload vivo,
contadores fuera de rango, X/Z, reset frío y estado posterior a fallo.
Si una clase no admite la composición propuesta, registrar el contraejemplo
y resolverla sin introducir stalls ni descartarla del dominio.

## Integración y puertas de aceptación

Antes de ejecutar Vivado, congelar el formato, la profundidad de etapas,
los registros dueños de estado y la regla de forwarding/versiones/wrap.
Reutilizar el banco de metadatos existente; su timing aislado no valida
todavía la producción de sus actualizaciones.

La propuesta debe aceptar un beat RQ y uno RC por ciclo, incluso con
dependencias adversas. Documentar la latencia pública y comparar por secuencia;
mantener revocación inmediata de vacío, obligaciones en vuelo y cancelación
exclusiva por reset frío. No limitar tags, lecturas, bytes ni tráfico sostenido.

Un bloque aislado no autoriza la integración nativa. Exigir el monitor completo,
bancos heredados y adversariales adicionales, mutantes reales, síntesis a
4 ns sin excepciones, revisión de todos los grupos y replay de su propia
netlist. Mantener separados RTL, timing OOC, cierre físico y pruebas de placa.
Registrar evidencia y respaldo por tanda. Los demás requisitos de
[VIVADO_QUEUE.md](VIVADO_QUEUE.md), incluidos DDR/512 GiB, reset/CDC e inferencia,
siguen abiertos.
