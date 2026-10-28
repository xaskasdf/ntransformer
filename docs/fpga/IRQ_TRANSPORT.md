# Transporte de finalización hacia IRQ XDMA

Actualizado el 2026-10-03. **Bloque aislado aprobado en simulación, síntesis
y auditoría de inicialización.** `fpga/rtl/ntrpu_irq_transport.sv` conserva
una notificación hasta completar atención por software y transporte MSI/MSI-X.
Los pares del banco son modelos de protocolo: todavía no conecta el control
FFN ni la IP XDMA. Se conservan sus respectivas validaciones anteriores.

AMD requiere mantener `usr_irq_req` hasta recibir `usr_irq_ack` y hasta
que el host atienda y limpie la interrupción. El primero acredita envío al
bloque PCIe; no acredita ejecución del ISR.
[PG195, revisión 4.2](https://docs.amd.com/r/en-US/pg195-pcie-dma/MSI-and-MSI-X-Interrupts).

## Contrato del bloque

| Interfaz | Responsabilidad |
| --- | --- |
| `event_valid/ready` | Captura secuencia, época e indicación de IRQ; un evento pendiente como máximo. |
| `service_valid/ready` | Presenta la identidad atendida por MMIO. Una identidad incorrecta produce `service_rejected` y permite reintentar. |
| `usr_irq_req/ack` | Dominio host; conserva la solicitud hasta ambas confirmaciones, en cualquier orden. |
| `retire_valid/ready` | Devuelve secuencia/época tras vaciar ambos handshakes y rearmar el host. Retiene el resultado bajo backpressure. |

Evento y atención usan dos XPM_CDC_HANDSHAKE, de 65 y 64 bits; un
XPM_CDC_SINGLE devuelve disponibilidad. Todos tienen tres etapas de
sincronización. El ACK host debe volver a cero antes de rearmar: un ACK
retenido no se aplica al evento siguiente. Un ACK espurio durante una
solicitud activa no tiene identidad en la interfaz XDMA; se presupone un
par conforme al protocolo, no autenticación del ACK.

La indicación de IRQ se captura al aceptar el evento. Cambiar las entradas
después no cancela una notificación presentada. En modo polling no se
emite IRQ ni se espera ACK de transporte, pero sí atención y retiro.
El productor mantiene VALID y datos hasta READY. El bloque no impone
timeout: si falta una confirmación queda pendiente hasta atención o reset.
Las secuencias/épocas son identidades del llamador, no direcciones host.

## Reset: fallo encontrado y corrección

La primera revisión RTL alcanzó el marcador del banco, pero produjo ocho
errores **XPM_CDC_HANDSHAKE S-2** al retirar `src_send` por reset antes de
recibir `src_rcv`. Se rechazó íntegramente esa tanda; no se desactivaron
las aserciones XPM. El handshake exige completar el intercambio antes de
reutilizarlo. [XPM_CDC_HANDSHAKE](https://docs.amd.com/r/en-US/pg382-xpm-cdc-generator/XPM_CDC_HANDSHAKE).

Ahora los dos envíos y 129 bits de payload tienen inicialización de FPGA
a cero y conservan su estado durante el reset de aplicación. El destino
drena las transferencias antiguas sin emitir IRQ; el origen retira cada
envío solo al recibir su confirmación. La admisión espera que ambos
handshakes estén vacíos y el host se haya rearmado. La auditoría del
checkpoint confirma **131 FDRE con INIT=0**.

El banco mantiene ambos resets solapados, con 64 flancos del origen y
64 del destino; al detener el reloj host lo reanuda mientras ambos siguen
en reset. No califica reset unilateral, supervisor sintetizable, detección
de pérdida de reloj ni recuperación del DIMM. La futura integración debe
coordinar también todos los pares AXI.
[PG059: reset de interfaces](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).

## Evidencia funcional y métricas

Tres perfiles: **45 casos, 69 eventos admitidos, 48 retiros, 21 abortos,
24 resets y seis identidades de atención obsoletas rechazadas**. Se prueban
ambos órdenes de ACK, coincidencia en destino, polling, ACK retenido,
ausencia prolongada de ACK, backpressure del retiro, siguiente evento
retenido correctamente y siete puntos de reset con recuperación por perfil.

El revisor independiente reconstruye los pines en cada flanco y comprueba
identidad, estabilidad, orden, retención y contabilidad de ambos relojes.
Rechaza **15 mutaciones por perfil**. Una inyección separada retira la IRQ
tras el ACK de transporte; falla en el caso 1 antes de la atención por MMIO.

| Origen / host (ns) | Aceptación → primer muestreo IRQ (ns) | Aceptación → retiro (ns) |
| --- | ---: | ---: |
| 5 / 4 | 25,2–28,2 | 100–355 |
| 5 / 6,2 | 36,7–41,6 | 125–520 |
| 6,2 / 3,4 | 23,5–26,5 | 111,6–322,4 |

Rangos de eventos completados en **simulación RTL**; el segundo incluye
retardos deliberados de los pares. No son latencia PCIe/ISR física ni MTBF.

Síntesis: **78 LUT, 486 FF, cero DSP/BRAM/URAM**. WNS estimado +2,467 ns,
WHS +0,077 ns; sin latches, cajas negras ni endpoints internos sin reloj.
Los cinco CDC-3 visibles tienen tres etapas. Los dos buses mantienen
`set_max_delay` y `set_bus_skew` de 12 ns, margen skew estimado +11,852 ns;
los cinco false paths están acotados a primeras etapas de sincronización.

El XDC conserva **dos waivers CDC-15 internos de AMD**. La consulta simple
`get_waivers` devuelve cero, pero no invalida esas dos entradas `-internal`
exportadas. Sin excepciones propias ni temporales ignoradas. Faltan relojes
padre y retardos de 135 entradas/133 salidas; TIMING-18 y los avisos OOC
persisten. Todavía no hay colocación/ruteo del bloque.

## Tandas reproducibles y respaldo

Todas están archivadas y verificadas en NFS y `artifacts/ntrpu-board-vivado/`:

| ID | Resultado y tiempo |
| --- | --- |
| `20261003T172351Z-V7-irq-transport-sim` | Fallo Tcl previo a XSIM, 12,575 s; corregido con `set_property -dict`. |
| `20261003T172457Z-V7-irq-transport-sim-r2` | Fallo XPM S-2 durante reset, 28,968 s; conservado como fallo. |
| `20261003T172721Z-V7-irq-transport-sim-r3` | Tres perfiles revisados PASS, 49,901 s. |
| `20261003T173040Z-V7-irq-transport-corrupt` | Retirada prematura detectada, salida 1 esperada, 28,208 s. |
| `20261003T173051Z-V7-irq-transport-synth` | Síntesis revisada, 83,284 s. |
| `20261003T173352Z-V7-irq-transport-audit` | Checkpoint reabierto sin modificar; INIT y constraints revisados, 49,479 s. |

RTL SHA-256: `3bb99e6276a2d99c51e60c12934fcd7964b64b7ef483a648cdabfb10883ac41e`.
Checkpoint: `4fd7e69545ec1bd6682fd4bdfd17c7abe8403755214dea2ec0d134ca33f25b4f`.
Fuentes: simulación `source-20261003T172654Z`; negativo/síntesis
`source-20261003T173027Z`; auditoría `source-20261003T173337Z`. Comparten
RTL/banco/lanzador funcional e identidades XPM; cada `VERIFIED.json`
identifica archivo, fuente y SHA-256 completos.

```bash
vivado -mode batch -source tools/ntrpu_board_irq_transport.tcl \
  -tclargs sim build/IRQ_NUEVO
python3 tools/check_ntrpu_irq_transport.py build/IRQ_NUEVO/profile-0.log --profile 0
# Repetir el revisor para perfiles 1 y 2; sim_corrupt debe fallar.
vivado -mode batch -source tools/ntrpu_board_irq_transport.tcl \
  -tclargs synth build/IRQ_SYNTH_NUEVO
vivado -mode batch -source tools/ntrpu_board_irq_transport_audit.tcl \
  -tclargs build/IRQ_SYNTH_NUEVO/irq_transport_synth.dcp build/IRQ_AUDIT_NUEVO
```

## Integración con el CSR y siguiente paso

La [ABI 2.0 opcional](FFN_IRQ_CONTROL.md) ya conecta este transporte al CSR:
54 casos PASS en tres perfiles de reloj, incluidos error inmediato,
polling, ACK duplicado, dueño incorrecto y reset. START exige notificador
disponible; el descriptor único reserva el espacio hasta publicar y retirar
su finalización. ACK registra atención con identidad y las lecturas siguen
disponibles mientras START está bloqueado. El modo IRQ se captura en la
finalización; una habilitación posterior afecta trabajos futuros.

Los wrappers FFN conservan la [ABI 1.0](FFN_HOST_CONTROL.md) por defecto.
Con `EXTERNAL_RETIRE=1`, la [FFN/HLS asíncrona](FFN_IRQ_INTEGRATION.md)
ya pasa 126 casos pequeños y seis FFN Brandon en dos perfiles, con ambas
confirmaciones y polling. Faltan modelo XDMA/MSI y supervisor; DDR sigue
sus dependencias V2/V3/V7.
