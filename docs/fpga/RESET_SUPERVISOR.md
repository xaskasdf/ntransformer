# Supervisor de reset y pérdida de reloj

Actualizado el 2026-10-04. **Bloque aislado aprobado en simulación**;
síntesis e INIT auditados, sin CDC críticos. Queda un aviso CDC del reset externo,
implementación e integración con los pares AXI. No sustituye
el circuito PERST, el arranque DCI ni la recuperación del DIMM.

## Contrato de integración

`fpga/rtl/ntrpu_reset_supervisor.sv` usa un `mgmt_clk` continuo e independiente.
AMD advierte que `axi_aclk` de XDMA puede desaparecer; no sirve como reloj
global de supervisión. [PG195](https://docs.amd.com/r/en-US/pg195-pcie-dma/Clocking-and-Resets).

| Señal | Responsabilidad del integrador |
| --- | --- |
| `external_reset_async` | Reunir solicitudes/fallos externos que requieren reset coordinado, incluida la caída del reset de la interfaz XDMA. |
| `peers_reset_req` | Poner en reset todos los extremos AXI participantes; no conectar directamente a PERST o MIG sin su protocolo específico. |
| `peers_held` | Confirmar en el dominio de gestión que los pares están en reset y que la notificación/transporte anterior ya puede recuperarse. No atar a uno en la integración. |
| `peers_ready` | Confirmar recuperación de todos los pares después de liberar reset. Su caída durante operación reinicia la secuencia. |
| `rearm/rearm_ready` | Solicitar rearme cuando está permitido. Un pulso prematuro no queda pendiente; mantener `rearm` alto permite aceptarlo posteriormente. |
| `host_reset_n/device_reset_n` | Resets locales: aserción asíncrona, liberación en tres flancos del dominio correspondiente. |
| `host_admit/device_admit` | Autorizar tráfico solo después de observar ambos resets liberados y todos los pares disponibles. |

AMD exige reset solapado de los conversores y sus extremos; recomienda
al menos 16 ciclos del reloj más lento. Este bloque exige **64 flancos reales
en cada dominio**, además de confirmaciones externas. El número de flancos
no demuestra por sí solo que una transacción se haya drenado.
[PG059](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).

## Secuencia y límites

Cada reloj responde a una solicitud de un bit que permanece estable hasta
completar su ida y vuelta mediante XPM de tres etapas. Así no se pierde
actividad por alias de un toggle libre. Los monitores siguen funcionando
durante el reset de aplicación. El timeout predeterminado es 64 ciclos
de gestión; debe superar el recorrido normal para las frecuencias reales.

El reset externo alimenta directamente el XPM de gestión. Las señales de
pérdida de reloj ya están registradas en gestión y se atienden en el flanco
siguiente: no se combinan con el reset asíncrono en una LUT. Después las
solicitudes registradas asertan los resets locales aunque sus relojes paren.

La secuencia borra los contadores, observa ambas confirmaciones a cero,
cuenta flancos bajo reset y espera rearme. Después libera los resets,
comprueba disponibilidad y habilita admisión. Una pérdida de reloj provoca
reset aunque el dominio afectado esté detenido. No hay detección de pérdida
del propio `mgmt_clk`: exige una fuente independiente y supervisión de placa.

El primer banco pasó sus aserciones tardías, pero el verificador independiente
rechazó el arranque: `XPM_CDC_ASYNC_RST` tiene INIT desasertado en el RTL AMD
instalado, visible durante GSR. La corrección usa cuatro cadenas de liberación
con **INIT=0 explícito** y revisa las salidas desde la primera observación.
Los XPM internos y sus modelos permanecen intactos. La liberación síncrona
de resets sigue el principio de [PG382](https://docs.amd.com/r/en-US/pg382-xpm-cdc-generator/XPM_CDC_ASYNC_RST);
las cadenas propias necesitan auditoría de INIT, CDC y constraints.

## Evidencia y reproducción

`20261004T015030Z-V7-reset-supervisor-sim-r4`: 36 casos, 51 arranques,
15 paradas de reloj y 21 alteraciones de evidencia rechazadas. Gestión 10 ns;
host/motor 4/5, 6,2/5 y 31/23 ns. En operación, las pérdidas de reloj
producen bloqueo en 574,999–684,999 ns; el caso que ya estaba bajo reset
se observa bloqueado tras 4,999 ns. Son medidas del banco RTL.

```bash
vivado -mode batch -source tools/ntrpu_board_reset_supervisor.tcl \
  -tclargs sim build/SUPERVISOR_NUEVO
python3 tools/check_ntrpu_reset_supervisor.py \
  build/SUPERVISOR_NUEVO/profile-0.log --profile 0
# Repetir perfiles 1 y 2. El modo synth requiere otra carpeta nueva.
```

Tanda vigente respaldada local/NFS: SHA-256
`c56e1fc25e5aca48f62684ea3f4a48f77d782c31ecd1469369c9540d93e892a1`.
El fallo inicial `20261004T010834Z` también está conservado y respaldado.

## Correcciones conservadas

La síntesis inicial `20261004T012414Z` conserva seis CDC-10 críticos por
lógica combinacional antes de sincronizadores, pese a cumplir su timing
estimado. La revisión r3 registra las solicitudes en gestión y añade
excepciones solo a los 12 pines CLR de las cadenas de liberación. Sus
trazas públicas son idénticas a la r2 en los tres perfiles. La síntesis
`20261004T013229Z-V7-reset-supervisor-synth-r2` y la auditoría
`20261004T013548Z-V7-reset-supervisor-init-audit-r2` están respaldadas:
**48 LUT, 87 FF**, sin DSP/BRAM; WNS +3,092/WHS +0,079 ns son estimaciones
de síntesis. Los 12 registros de liberación son FDCE, INIT=0, ASYNC_REG=1.
CDC informa ocho cruces de un bit y siete resets sincronizados, todos
informativos. Se conservan las 11 excepciones XPM y cuatro excepciones
propias limitadas a CLR; cero excepciones ignoradas y cero DRC.
El revisor rechaza diez alteraciones de evidencia.

En esa revisión permanece **LUTAR-1** en el combinador de fallo externo/relojes que alimenta
los tres PRE de `management_reset`; requiere revisar glitches y el camino
de reset. TIMING-18 corresponde a cuatro entradas y nueve salidas sin
retardos de placa. No se añadieron waivers ni excepciones entre relojes.
SHA-256 de síntesis:
`fc87da95bc4c2f741ee0c100cfa661e53cf068885445a49f9b32451637022688`;
auditoría: `3941eceb9abe64aee2c60da12e4f22bb3b716858df5e5d466ea5b4004f79b00c`.

Una auditoría anterior se detuvo al interpretar ASYNC_REG=1 como distinto
de TRUE; se preserva ese fallo de lectura y no se cuenta como auditada.

## Revisión vigente sin combinador de reset

La síntesis `20261004T015040Z-V7-reset-supervisor-synth-r3` ocupa
**51 LUT y 87 FF**, sin RAM/SRL/DSP. Elimina LUTAR-1 y conserva
WNS estimado +3,092/WHS +0,079 ns. Auditorías `20261004T015259Z` y
`20261004T015604Z`: los 12 registros de liberación mantienen INIT=0;
el puerto externo llega directamente a PRE de tres FDPE ASYNC_REG, con
cadena D/Q y reloj de gestión verificados. No hay LUT en ese recorrido.

Vivado 2026.1 reporta ocho CDC-3 y seis CDC-9 informativos, más un
**CDC-26 Warning** desde `external_reset_async` a PRE de la etapa 0.
Su descripción de colisión LUTRAM no concuerda con las primitivas
inspeccionadas. Se conserva como anomalía abierta, sin waiver ni cambio
de severidad; la auditoría estructural no equivale a calificación eléctrica
del puerto externo. AMD define CDC como análisis estructural y delimita
su cobertura de E/S en [UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_cdc).
Las 15 excepciones permanecen idénticas; TIMING-18 mantiene 13 fronteras
sin retardos. El revisor rechaza 13 alteraciones de evidencia.

La [integración con el transporte IRQ](SUPERVISED_IRQ.md) ya reemplaza las
confirmaciones simuladas por observaciones de sus dos extremos: 63 casos
revisados, incluidas pérdidas de reloj y descarte de épocas antiguas.
La [FFN supervisada](SUPERVISED_FFN.md) lo extiende a los conversores,
cómputo y pares conductuales: 126 casos revisados con relojes activos.
Faltan pérdida de reloj en el conjunto, DMA interrumpido y pares DDR/XDMA
reales. Ninguna de estas islas califica todavía recuperación de placa.
