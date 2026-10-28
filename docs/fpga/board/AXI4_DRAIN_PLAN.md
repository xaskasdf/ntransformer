# Siguiente etapa: inventario retenido AXI4

## Alcance concreto

La [primera implementación aislada](AXI4_DRAIN_MONITOR.md) pasa 22 escenarios
y cuatro controles negativos. La [continuación de capacidad/CDC](AXI4_CDC_DRAIN.md)
prueba 32 escrituras, las 128 lecturas agregadas y ambos extremos del Clock
Converter AMD real, con síntesis y cruces de control revisados. Este documento
conserva el contrato completo para las etapas restantes: admisión activa,
contención causal de respuestas e integración de colas protegidas XDMA.
El [protector causal aislado](AXI4_RESPONSE_GUARD.md) ya pasa 19 escenarios
y tres controles RTL negativos. La [admisión retenida](AXI4_RETAINED_ADMISSION.md)
ya lo incorpora y pasa trece escenarios aislados y cuatro controles negativos.
Falta componer esa frontera con el CDC real, certificados y el endpoint.
La sección AXI-Lite aprobada conserva sus fuentes y bancos.

El perfil presente en `tools/ntrpu_board_guarded_native_reset.tcl` configura
dos Clock Converter AMD AXI4 con datos de 512 bits, dirección de 64 bits,
ID de 4 bits y tres etapas CDC. `ntrpu_ffn_async_fabric.sv` expone esos
campos y LEN/LAST. No reducir la primera implementación a ID 0, una sola
transferencia o solo escrituras. La frontera debe conservar WSTRB y todos
los bits de dirección antes de validar y seleccionar el canal DDR.

`ntrpu_axi_write_capacity_queue.sv` es una cola experimental de escritura
con ID downstream 0 y reset coordinado. `ntrpu_axi_capacity_guard.sv`
conserva respuestas en buffers, pero su reset borra inventarios. Ninguno
constituye por sí solo un inventario retenido de drenaje del endpoint.

## Reglas de protocolo que debe respetar

AW y W son independientes: W puede llegar primero. AXI4 ordena los datos
de escritura según las direcciones y elimina el interleaving W. B requiere
AW aceptada y la última transferencia W aceptada. Las lecturas deben
asociarse con AR; RLAST y el número de transferencias deben coincidir con
ARLEN. Conservar VALID y payload hasta el handshake. Referencia consultada
el 2026-10-05: [ARM IHI 0022D, A3.2–A3.4 y A5](https://documentation-service.arm.com/static/67ab5a186dbc975ccea91734).

## Diseño propuesto y puntos que deben probarse

1. Observador pasivo desde configuración; solo `cold_rst_n` inicializa sus
   inventarios. PERST y resets de transporte no borran obligaciones.
2. Descriptores AW ordenados con ID/LEN y seguimiento separado de W,
   incluyendo ráfagas completas o parciales recibidas antes de AW.
   Comprobar WLAST contra LEN; no inventar una asociación por llegada de B.
3. Créditos B únicamente para escrituras completas. Conservar orden por ID
   y permitir el reordenamiento legal entre IDs. El procesamiento interno
   de descriptores no puede clasificar como huérfana una B legal inmediata.
4. Colas AR por ID con transferencias restantes; soportar interleaving R
   entre IDs y retirar cada lectura únicamente en su final válido.
5. Capacidad configurable y explícita. Registrar overflow como inventario
   desconocido permanente; no envolver contadores. La capacidad elegida
   debe cubrir el límite impuesto por la admisión de esa interfaz.
6. ACK fresco de 64 bits, fence independiente, todas las obligaciones
   vacías y dos observaciones tranquilas. Cualquier fallo invalida ACK.
   Exponer fuentes registradas separadas para la posterior composición CDC.

Los monitores no imponen por sí mismos el fence ni contienen respuestas.
La frontera activa posterior debe conservar ofertas ya presentadas, terminar
el grupo finito y mantener errores locales sin resetearlos prematuramente.

## Banco y orden de ejecución

Primero XSIM aislado y una referencia de colas independiente que reconstruya
cada handshake: AW antes/después de W, varios IDs, respuestas reordenadas,
R intercaladas, B/R bloqueadas, aceptación/retiro simultáneos y LEN/LAST
incorrectos. Incluir longitudes 1/2/4/16/64/256 con direcciones y tamaños
legales; por ejemplo, 256 transferencias de 16 bytes caben en una página
alineada de 4 KiB. Variar datos y WSTRB, no solo contar respuestas.

Después: controles negativos, síntesis OOC con recursos y restricciones
revisados, ambos extremos del CDC AMD real y frontera activa retenida.
Finalmente conectar el productor y las colas reales XDMA. Guardar evidencia
y respaldo por tanda; no sustituir esa integración por un ACK constante.

## Condición para avanzar al reset global

Inventario AXI4 externo vacío no demuestra vacías las colas PCIe/DMA
protegidas. Se necesita un contrato observable de detención del productor,
propiedad invalidada y respuestas locales pendientes identificadas. Mantener
separados el mecanismo nativo de reset, Completion Timeout de 50 ms,
reenumeración y prohibición de reutilización NTRD. La integración V3/V4,
el DIMM exacto y el cierre físico siguen teniendo sus propias dependencias.
