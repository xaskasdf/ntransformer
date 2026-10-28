# Contrato para la arquitectura segmentada del inventario PCIe

## Estado y motivo

Diseño propuesto el 2026-10-07; **todavía no implementado ni validado**.
R7 conserva la referencia funcional, pero su peor camino estimado mide
15,249 ns para un período de 4 ns. Va de los metadatos de dirección de un
tag a la validación RC de longitud. Registrar solamente la guarda de rango
no resolvió este camino. Véase [el diagnóstico](PCIE_INVENTORY_TIMING_PLAN.md).

La nueva arquitectura debe admitir un beat RQ y otro RC cada ciclo, con
hasta dos cabeceras RQ y cuatro RC, 256 tags, 257 lecturas y 1.052.672 bytes
pendientes. El observador es pasivo: no puede frenar XDMA para resolver
conflictos internos. Una FIFO que eventualmente se llena no satisface el
ritmo sostenido. No se ha elegido aún una implementación de memoria.

## Etapas y obligaciones

| Etapa propuesta | Entrada y trabajo | Condición para avanzar |
| --- | --- | --- |
| Captura | Datos, TUSER, handshake y configuración del mismo flanco | Conservar cambios de perfil, stalls y pérdida de link/reset, incluso sin transferencia aceptada |
| Decodificación | Descriptores de posiciones fijas, SOP/EOP y máscaras | Mantener posición y orden de cada evento; no consultar todavía el banco de tags |
| Consulta | Metadatos y validez del tag, con actualizaciones anteriores adelantadas | Obtener la versión que corresponde a ese evento, nunca una versión futura |
| Validación y transición | Identidad, bytes, longitud, payload y nuevos metadatos | Separar comparaciones largas del multiplexor de lectura; conservar dependencias entre eventos |
| Retirada | Aplicar deltas y fallos en orden; publicar estado correlacionado | Retirar cada evento exactamente una vez y eliminar su obligación en vuelo |

Esta tabla define responsabilidades, no garantiza cinco ciclos ni timing.
Hay que medir la consulta y las transiciones con forwarding antes de fijar
el número de registros. Insertar registros alrededor del bloque R7 completo
dejaría intacto su camino combinacional interno.

## Orden, reutilización y memoria

Cada ciclo observado recibe una secuencia de diagnóstico; dentro del beat,
RC se procesa por posición creciente antes que RQ. El orden incluye los
eventos de EOP de un paquete que empezó en el ciclo anterior. La consulta
de un tag debe seleccionar la actualización anterior más reciente tanto
del mismo beat como de beats todavía no retirados.

Liberar el tag en la cabecera terminal RC no extingue el payload pendiente.
Su contexto —dirección, longitud, bytes, identidad y pertenencia— debe vivir
separado del nuevo uso del tag hasta EOP. La reutilización no puede cambiar
ese contexto ni descontar bytes del propietario nuevo. La generación o el
identificador interno elegido debe distinguir ambos usos y tener una regla
explícita de wrap; no se presupone que un contador finito nunca se repita.

Un beat puede presentar hasta cuatro actualizaciones RC y dos RQ. Los
escritores se resuelven por el orden anterior antes de modificar cada tag.
Una RAM de dos puertos no admite directamente esos accesos. Una propuesta
de bancos debe demostrar el caso de todos los accesos al mismo banco y la
repetición de un tag en ciclos consecutivos. Replicar lecturas tampoco
resuelve por sí solo los conflictos de escritura. No aceptar burbujas,
serialización a menor ritmo ni cambio de capacidad para hacer pasar timing.

## Vacío, fallos y reset

Los eventos capturados que pueden modificar estado son obligaciones
pendientes desde el flanco de captura. Un candidato a `inventory_empty`
requiere estado retirado vacío, ninguna obligación en vuelo, ausencia de
paquetes parciales/stalls y entradas actuales inactivas. El registro de
captura debe revocar vacío al admitir actividad, antes de que esa actividad
alcance el banco. Las burbujas sin efecto no deben impedir vacío eternamente.

Un fallo de perfil, link o reset revoca de inmediato la posibilidad de
certificar vacío; los fallos descubiertos después permanecen latched. La
latencia de los fallos y de los contadores debe formar parte de la interfaz
documentada y de su comparación con la referencia. No ocultar diferencias
posteriores a fallo: registrar y justificar cualquier cambio de semántica.
Solo el reset frío autorizado elimina obligaciones y fallos; un reset de
usuario no puede vaciar silenciosamente la tubería. No generar ACK desde
este monitor: siguen faltando los contratos de fuente y recepción host.

## Orden de implementación y pruebas

1. Congelar formato de evento, fases de contexto RC/RQ y regla de versiones.
   Ejercitar consulta con actualizaciones del mismo tag, primero en un banco
   de prueba aislado; eso no sustituye al monitor completo.
2. Medir con Vivado el banco y el forwarding al máximo de accesos, con 4 ns,
   retardos I/O explícitos y cero excepciones añadidas. Rechazar una solución
   que dependa de stalls del observador o de distribución favorable de tags.
3. Integrar decodificación y validación; comparar el estado retirado contra
   los 22.339 ciclos/68 escenarios R7 mediante secuencia, no por una espera
   arbitraria. Mantener los casos inválidos, X/Z y la captura nativa.
4. Añadir tráfico sostenido RQ2/RC4, mismo banco/tag en ciclos consecutivos,
   RC terminal con payload aún activo, reutilización RC→RQ, reset/link y
   fallo con todas las etapas ocupadas. Inyectar omisión, duplicación,
   inversión de eventos y adelanto de vacío para probar el checker.
5. Solo tras la revisión RTL, sintetizar el monitor completo, revisar todos
   los caminos, reproducir el dominio binario en su propio netlist y volver
   a conectar la observación nativa. Una prueba aislada o finita no establece
   equivalencia general, cierre físico ni la FPGA de 512 GiB.

Toda tanda conserva fuentes, configuración, evidencia y respaldo D:/NFS.
Los criterios globales permanecen en [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
