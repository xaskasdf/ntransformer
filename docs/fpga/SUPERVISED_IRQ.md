# Interrupciones con supervisor de reset

Actualizado el 2026-10-04 UTC. Esta tanda conecta el
[supervisor](RESET_SUPERVISOR.md) al transporte real `ntrpu_irq_transport`.
Valida esa isla de notificaciones; no integra todavía los conversores AXI,
XDMA, FFN ni DDR. Sigue abierta la dependencia V7 de
[la cola](board/VIVADO_QUEUE.md).

## Contrato implementado

`ntrpu_supervised_irq_transport.sv` usa tres relojes y las confirmaciones
registradas de ambos extremos del transporte. `peers_held` exige que no
queden envíos, recepciones ni confirmaciones pendientes bajo reset.
`peers_ready` indica que ambos extremos abandonaron BOOT; no depende de
`event_ready`, que baja normalmente durante una notificación.

Los indicadores cruzan a gestión mediante cuatro XPM de tres etapas.
La admisión de eventos, servicio, retiro e IRQ permanece cerrada hasta RUN.
El transporte conserva los envíos y datos pendientes durante reset y
drena los handshakes XPM antes de admitir una nueva época.

La primera integración reveló cuatro errores **XPM_CDC_SINGLE S-1**:
`host_reset_drained` podía generar un pulso corto antes de que llegara
una petición en vuelo. Ahora exige ocho flancos locales consecutivos sin
actividad; el supervisor mantiene además al menos 64 flancos reales bajo
reset en ambos dominios. Esta revisión se prueba con gestión de 10 ns y
períodos locales mínimos de 3,4 ns. No generalizar el filtro a otras
relaciones de reloj sin nueva validación. El reloj de gestión debe seguir
funcionando; el bloque no detecta la pérdida de su propio reloj.

## Simulación revisada

Tres perfiles motor/host: 5/4, 5/6,2 y 6,2/3,4 ns. El verificador reconstruye
las transacciones desde los pines y cuenta los flancos reales de cada dominio:

- **63 casos:** 105 eventos aceptados, 66 retirados y 39 abortados por reset.
- **42 recuperaciones**, incluidas 18 por pérdida automática de uno o ambos
  relojes durante envío, IRQ, servicio, ACK o retiro con backpressure.
- Reset observado **635,001–685,001 ns** después de detener los relojes en
  esos escenarios. Ninguna notificación de la época anterior reaparece.
- 63 alteraciones de logs rechazadas. Una prueba negativa de hardware
  observado retira prematuramente la IRQ y falla en el caso 1 como se exige.
- La regresión anterior de 45 casos conserva exactamente sus trazas públicas.

Los pares de servicio/ACK son modelos de protocolo. Estos resultados no
miden latencia física de interrupciones, MTBF ni rendimiento de inferencia.

## Síntesis y revisión estructural

Vivado 2026.1, `xcvu5p-flva2104-2-i`: **149 LUT, 597 FF**, sin DSP ni RAM.
WNS +2,467/WHS +0,077 ns son estimaciones de síntesis, sin colocación/ruteo.
La auditoría confirma 12 FDCE de liberación con INIT=0/ASYNC_REG, tres FDPE
de reset externo sin LUT intermedia, 131 registros persistentes inicializados
a cero y las cuatro fuentes de estado registradas. Rechaza 12 alteraciones
de evidencia y verifica identidad del checkpoint, fuentes y modelos AMD.

CDC-3=17 y CDC-9=6 son informativos. **CDC-26** permanece visible en
`external_reset_async → supervisor/management_reset/arststages_ff_reg[0]/PRE`:
la descripción LUTRAM no concuerda con las primitivas inspeccionadas, pero
no se elimina ni se considera resuelta. Hay 135 entradas y 144 salidas sin
retardos de placa; cero extremos internos sin restricción y cero DRC OOC.
Se conservan 20 excepciones XPM, cuatro excepciones propias sobre 12 CLR
y dos waivers CDC-15 internos del fabricante para los handshakes. No se
añaden waivers propios ni excepciones generales entre relojes.

## Evidencia y reproducción

Cada ID está en `artifacts/ntrpu-board-vivado/<ID>/`, con fuentes, logs,
revisión, `FILES.json` y `VERIFIED.json`; existe copia verificada en NFS.

| ID | Resultado |
| --- | --- |
| `20261004T021425Z-V7-irq-status-legacy-regression` | 45 casos, sin cambio de trazas |
| `20261004T021809Z-V7-supervised-irq-sim` | Fallo de compilación del banco; conservado |
| `20261004T021809Z-V7-supervised-irq-synth` | Síntesis anterior al filtro; no califica funcionamiento |
| `20261004T022104Z-V7-supervised-irq-sim-r2` | Fallo XPM S-1; conservado íntegro |
| `20261004T022710Z-V7-supervised-irq-sim-r3` | 63 casos y 63 alteraciones revisados |
| `20261004T022710Z-V7-supervised-irq-synth-r2` | Síntesis vigente; advertencia CDC abierta |
| `20261004T022710Z-V7-irq-status-legacy-r2` | 45 casos y trazas idénticas al banco anterior |
| `20261004T023621Z-V7-supervised-irq-audit` | Checkpoint y estructura revisados |
| `20261004T023621Z-V7-supervised-irq-negative` | Fallo esperado por retirada prematura de IRQ |

SHA-256 del comprimido de simulación vigente:
`cb3f0c9e89dc3d622310afebb27d496bc0cb147da333533348d87da5cd1da488`.
Checkpoint sintetizado auditado:
`127bf1fd84b4145a1170cb9e4049e3a3c7d8556c9287573aea7791a71138e2a0`.

```bash
vivado -mode batch -source tools/ntrpu_board_supervised_irq.tcl \
  -tclargs sim build/SUPERVISED_IRQ_NUEVO
python3 tools/check_ntrpu_supervised_irq.py \
  build/SUPERVISED_IRQ_NUEVO/profile-0.log --profile 0
# Repetir perfiles 1 y 2; synth y sim_corrupt usan carpetas nuevas.
vivado -mode batch -source tools/ntrpu_board_supervised_irq_audit.tcl \
  -tclargs /ruta/supervised_irq_synth.dcp build/SUPERVISED_IRQ_AUDIT_NUEVO
```

## Siguiente integración

La [FFN supervisada](SUPERVISED_FFN.md) ya propaga esas observaciones y
coordina los pares AXI del banco: 126 casos, 30 recuperaciones y 1.344
FP32 exactos, con regresión del modo anterior. Este avance cubre resets
externos con relojes activos; el DMA del host permanece inactivo.

Conectar el supervisor al camino AXI/FFN requiere confirmaciones propias
del iniciador, conversores y memoria, además de las del transporte IRQ.
No basta conectar sus resets a esta isla. Deben cerrarse la admisión de
AW/W/AR y las respuestas durante recuperación, reiniciarse todos los pares
y probarse transacciones partidas o pendientes en cada canal. Propagar las
observaciones IRQ a través de control/dispositivo/fabric, conservando una
regresión del modo anterior. Exigir una FFN numéricamente exacta después
de cada recuperación, sin respuestas ni resultados de la época anterior.
Añadir también resets repetidos durante RELEASE y arranque: los casos
actuales no califican todos los instantes posibles de un reset externo.
