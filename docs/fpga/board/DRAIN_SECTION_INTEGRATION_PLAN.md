# Integración del coordinador con secciones reales

## Dependencias y objetivo

Continúa la cola R27 y el [coordinador de siete certificados](DRAIN_CERTIFICATE_JOIN.md).
La siguiente tanda debe conectar hardware de las secciones ya revisadas,
conservar payload y respuestas bajo cierre, y mostrar que falta el permiso
global cuando no existe un proveedor externo. No sustituir las secciones
por constantes ni interpretar un proveedor del banco como driver real.

## Composición requerida

1. Instanciar AXI-Lite `ntrpu_axil_fence_section_r3`, Shutdown Manager y
   Clock Converter AMD, junto con admisión/protector/certificado AXI4 R2
   y su Clock Converter. Conservar sus revisiones selladas.
2. Usar el reloj de gestión común. Conectar solicitudes 3/4 del coordinador
   a `req_valid` de cada sección y su identidad a `generation/tag`.
   La identidad de ACK debe corresponder a la solicitud efectivamente
   certificada. Demostrar que cambiarla suprime el ACK anterior antes de
   aceptar otro; no añadir etiquetas nuevas a un ACK viejo registrado.
3. Crear certificación separada para respuestas locales. `local_idle` es
   del dominio host y no puede conectarse directamente a gestión. Su
   proveedor requiere quiescencia aguas arriba estable, identidad fresca,
   cruce coherente y revocación ante una obligación tardía. Cerrar la ruta
   reenviada no impide por sí solo nuevas respuestas de error locales.
4. Mantener ausente el contrato XDMA protegido en el perfil de integración
   real. Un segundo perfil explícitamente de fixture puede aportar los
   contratos externos para comprobar el recorrido positivo del coordinador;
   sus resultados conservan ese alcance.

## Criterios de aceptación

- Tráfico distinguible AXI-Lite y AXI4, con B/R retenidas en cada sección;
  reconstruir dirección, identificadores, bytes y respuestas de las trazas.
- `section_ack` puede estar alto con respuestas locales pendientes:
  demostrar que eso no produce `complete_ack`.
- Cubrir las dos velocidades D=5/37 ns, ausencia/revocación de proveedor,
  cambio de identidad con mailbox ocupado y pérdida de readiness.
- Mantener el envío XPM hasta terminar su handshake y conservar alarmas
  aunque el reloj de origen se detenga. No forzar estado interno.
- Un control RTL negativo debe detectar omisión de una respuesta local
  pendiente o asociación de un certificado con una identidad incorrecta.
- Auditar CDC de los adaptadores nuevos y sus fuentes registradas antes
  de conectar el actuador de reset. El CDC-15 nativo existente sigue abierto.

El endpoint release sequencer depende todavía del inventario externo y
del drenaje completo. La autorización de reset debe preservar propiedad
invalidada y retención independiente; no reutilizar sesiones NTRD.
Cada tanda usa carpeta nueva, límites de recursos y respaldo comprimido
verificado local/NFS. DIMM exacto, cuatro canales y cierre físico siguen
las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
