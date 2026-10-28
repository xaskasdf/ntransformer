# Proveedor de respuestas locales R2

Revisión pendiente de simulación y síntesis nativas; no califica el drenaje
global ni autoriza reset del endpoint.

## Cambio de hardware

`ntrpu_local_reply_certificate_r2` reutiliza `section_fault_mgmt`, la alarma
ya sincronizada y retenida de su sección real. Los fallos de admisión siguen
participando en `observed_idle` en el dominio host; no se sincronizan de nuevo
en este proveedor. Solo su revocación registrada propia cruza mediante otra
instancia XPM. `fault_mgmt[0]` representa la alarma compartida y
`fault_mgmt[1]` la revocación propia; ambas se retienen en gestión.

La conexión es obligatoria: el puerto debe incluir todos los fallos de
admisión, provenir del mismo reloj de gestión y conservarlos durante las
operaciones de cancelación o cambio de identidad. En AXI4 se conecta al OR
de `ntrpu_axi4_section_certificate_r2.fault_mgmt`; en AXI-Lite al OR de
`ntrpu_axil_fence_section_r3.fault_mgmt`. El wrapper de síntesis usa las fuentes
registradas reales y el Clock Converter AMD.

## Pruebas en cola

1. Recuperar las cuatro capturas del hardware anterior mediante el Tcl R3.
2. Sintetizar el nuevo proveedor en la sección real y comprobar que desaparecen
   los 12 CDC-11 de duplicación, sin ocultarlos mediante excepciones. Revisar
   también el CDC-15 nativo y las restricciones de interfaz aún pendientes.
3. Ejecutar los cuatro perfiles integrados con el proveedor R2, conservando
   tráfico AXI-Lite/AXI4, respuestas locales retenidas, identidades obsoletas,
   ausencia del contrato XDMA y la revocación por petición tardía.
4. Inyectar una respuesta R huérfana por la interfaz pública del modelo de
   dispositivo, a través del Clock Converter real. Detener primero gestión,
   esperar el fallo registrado del protector host y detener ese reloj. Al
   reanudar gestión, exigir el bit de admisión sincronizado y su llegada al
   proveedor local sin más flancos host ni revocación propia que pueda ocultar
   una conexión ausente. Comprobar retención al cancelar y reemplazar identidad.

La revisión independiente debe correlacionar logs y VCD y añadir controles
negativos de RTL; los mensajes HDL PASS y un CDC sin errores no bastan por sí
solos. Las revisiones originales permanecen intactas. El controlador
`queue_drain_sections_followup.py` conserva trabajos activos, aplica el
preflight existente, limita cada tanda a 600 segundos y respalda cada resultado
antes de comenzar la siguiente. Una tanda fallida detiene la cola para revisión.
