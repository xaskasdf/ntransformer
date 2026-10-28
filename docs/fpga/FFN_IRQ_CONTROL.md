# Control FFN con confirmación de interrupciones: ABI experimental 2.0

Validado el 2026-10-03 en Vivado 2026.1, `xcvu5p-flva2104-2-i`.
`ntrpu_ffn_irq_control.sv` conecta el CSR existente con
`ntrpu_irq_transport.sv`. AXI-Lite y el par de cómputo pertenecen a `clk`;
`irq` y `usr_irq_ack` pertenecen a `host_clk`. Un host en otro dominio
necesita además el conversor AXI-Lite exterior. El banco usa pares de
cómputo y transporte de protocolo. La [integración FFN/HLS asíncrona](FFN_IRQ_INTEGRATION.md)
ya usa este wrapper y conserva su ABI; XDMA, DDR y placa siguen pendientes.

## Versiones y registros

`ntrpu_ffn_control` conserva por defecto `EXTERNAL_RETIRE=0`: la
[ABI 1.0](FFN_HOST_CONTROL.md) y su IRQ de nivel siguen vigentes en los
wrappers FFN existentes. Los wrappers también propagan `EXTERNAL_RETIRE=1`
para seleccionar este control y su transporte IRQ.
Es una versión mayor porque cambia el efecto de ACK. Conserva los registros,
alineación, respuestas, `WSTRB` y snapshots AXI-Lite anteriores, con estas
extensiones:

| Dirección | Acceso | ABI 2.0 |
| --- | --- | --- |
| `0x04` | R | Versión `0x00020000` |
| `0x08` | R | Bit 3: retiro externo y ACK con identidad; bits 0–2 anteriores |
| `0x0c` | R | Estado anterior en bits 0–4; 5: servicio solicitado; 6: evento aceptado por el notificador; 7: fallo de notificación; 8: notificador disponible |
| `0x38` | RW | Secuencia de la finalización que el software atiende; respeta `WSTRB` |
| `0x3c` | RW | Época de esa finalización; respeta `WSTRB` |
| `0x64` | R | Secuencia del descriptor admitido por START |
| `0x68` | R | Época del descriptor admitido por START |

Los bits 0–4 de estado siguen siendo descriptor pendiente, motor ejecutando,
finalización pendiente, cancelación ofrecida y cancelación solicitada.
`0x4c/0x50` conservan las identidades **devueltas por el motor**, incluso si
son incorrectas. Para retirar un resultado con esos errores se usa el dueño
admitido de `0x64/0x68`; `0x60` conserva el diagnóstico. Retirar un resultado
fallido permite recuperar el canal, pero no lo convierte en resultado válido.

## Secuencia del controlador host

1. Comprobar versión/capacidad. Serializar el acceso privilegiado a los
   registros y asignar un par secuencia/época distinto a cada trabajo.
   No reutilizar identidades dentro de una sesión: no hay protección contra
   un ACK antiguo que coincida con una identidad reutilizada.
2. Esperar estado sin bits 0–2 ni 7 y con bit 8 activo. Programar descriptor
   e IRQ; escribir START=`1` en `0x10`. START falla con `SLVERR` mientras el
   motor, la finalización o el notificador impidan admitir otro trabajo.
3. Al observar finalización, leer metadatos, diagnóstico y datos publicados.
   El modo IRQ se captura en el handshake de finalización. Cambiar `0x14`
   después afecta trabajos futuros. Si la escritura de habilitación coincide
   con ese handshake, se captura su valor anterior al flanco.
4. Escribir la identidad admitida en `0x38/0x3c` y ACK=`4` en `0x10`.
   Un ACK duplicado, sin finalización, con identidad distinta o durante fallo
   recibe `SLVERR`. El ACK válido registra la atención del software; **no
   borra todavía la finalización**. También se acepta antes de que el
   notificador pueda admitir el evento.
5. El transporte mantiene IRQ hasta recibir atención e `usr_irq_ack`, en
   cualquier orden. En polling solo necesita atención. El CSR libera el
   espacio al recibir un retiro con la identidad correcta, una vez drenados
   los handshakes. Lecturas, escrituras de configuración y ACK continúan
   funcionando mientras START está bloqueado. Una lectura ya aceptada
   conserva sus datos aunque se retire el trabajo y se admita otro.

Los registros de finalización mantienen su último contenido tras el retiro;
solo el bit de pendiente indica validez. Los shadows de ACK también
persisten hasta otra escritura o reset. El ACK PCIe no lleva identidad:
el par XDMA debe respetar su protocolo; este bloque no autentica un ACK
espurio presentado durante una IRQ nueva.

## Errores y reset

Un servicio rechazado por el notificador o un retiro inesperado/de otra
identidad fija el bit 7 hasta reset. La finalización se conserva y START
queda bloqueado. No existe timeout que retire silenciosamente eventos.

Ambos dominios y los pares de cómputo/host requieren reset coordinado.
Las transacciones XPM se drenan sin retirar prematuramente sus envíos;
131 registros de datos/envío conservan INIT de configuración cero.
Con reloj host detenido, el rearme espera a que vuelva y complete el
drenaje. El supervisor de reset/reloj sigue pendiente; no se admite reset
unilateral ni se demuestra recuperación automática de una pérdida de reloj.

## Evidencia funcional y regresiones

La ABI2 se verifica con `OUTPUT_FP32=1`; la regresión binary64 de esta
tanda corresponde a ABI1. El banco final ejecuta 18 casos en cada perfil `clk/host_clk`: 5/4,
5/6,2 y 6,2/3,4 ns. Comprueba los dos órdenes de confirmación, polling,
cambio de máscara, servicio anterior al evento, identidades antiguas,
metadatos erróneos, cancelación, error inmediato, lectura retenida,
seis fases de reset y dos fallos internos inyectados con recuperación.

El revisor Python reconstruye desde muestras de interfaz **798 escrituras
y 291 lecturas**, incluidos tres snapshots retenidos manualmente. Verifica
respuestas, máscaras, admisión, dueño, eventos, IRQ y retiro sin consultar
los estados internos de las máquinas. Son 27 resets, 78 trabajos aceptados,
75 finalizaciones/eventos, 57 retiros y 18 abortos. Rechaza 45 alteraciones
de evidencia. Otra simulación retira deliberadamente la IRQ observada tras
ACK PCIe y antes de atención: el banco detecta el fallo.

| Relojes, ns | Evento aceptado → retiro, ns | Pausas observadas del reloj host, ns |
| --- | --- | --- |
| 5 / 4 | 100–655 | 310 y 130 |
| 5 / 6,2 | 130–705 | 310 y 133,3 |
| 6,2 / 3,4 | 105,4–787,4 | 382,5 y 156,4 |

Son latencias simuladas con pausas deliberadas de los pares, no rendimiento
PCIe físico. La regresión ABI1 conserva exactamente 4.440 líneas de traza
FP32/binary64 y sus 54 grupos. La regresión conectada a FFN/HLS reales
conserva 9.298 líneas, 63 casos, 8.052 escrituras MMIO y 71.232 bytes
numéricos comparados, incluidos errores, cancelaciones y resets. Esta última
sigue usando ABI1; no constituye integración FFN de la ABI2.

## Síntesis y alcance temporal

Conjunto CSR/notificador: **2.093 LUT, 4.927 FF**, sin DSP/BRAM/URAM,
latches ni cajas negras. WNS estimado +1,988 ns, WHS +0,066 ns,
WPWS +1,725 ns; ningún endpoint interno sin restricción.

La auditoría reabre el mismo checkpoint sin modificarlo: cinco CDC-3 de
tres etapas, dos grupos max-delay/bus-skew de 12 ns con margen estimado
11,852 ns y cinco false paths limitados a primeras etapas XPM. Se conservan
**dos waivers CDC-15 internos AMD**, aunque `get_waivers` por defecto
devuelva cero. Los 131 FDRE de envío/datos tienen INIT=0.

Faltan relojes padre y retardos de 255 entradas/1.981 salidas. TIMING-18
persiste; su listado se limita a 1.000 hallazgos. PROT no se usa en este
bloque privilegiado y permanece la advertencia de granularidad para
floorplanning. DRC=0 tiene alcance OOC. Estas cifras corresponden al wrapper
aislado; la [FFN integrada](FFN_IRQ_INTEGRATION.md) ya tiene ruteo exploratorio
y auditoría propios, sin cierre temporal de placa.

## Tandas y reproducción

Todas tienen archivo, fuentes y miembros verificados en NFS y en
`artifacts/ntrpu-board-vivado/<id>/`; incluye los intentos fallidos.

| ID | Resultado; duración |
| --- | --- |
| `20261003T175212Z-V7-ffn-irq-control-sim` | Fallo de banco: force/release no reevalúa `event_ready`; watchdog, 82,069 s. Sustituido por ACK host retenido real. |
| `20261003T175212Z-V7-ffn-control-abi1-regression` | ABI1 FP32, 27 grupos PASS; 49,655 s. |
| `20261003T175212Z-V7-ffn-control-abi1-fp64-regression` | ABI1 binary64, 27 grupos PASS; 49,637 s. |
| `20261003T175349Z-V7-ffn-irq-control-sim-r2` | ABI2, 54 casos PASS; aviso de declaración implícita del cable del banco; 54,688 s. |
| `20261003T175413Z-V7-ffn-host-abi1-regression` | FFN real/ABI1, 63 casos PASS; 391,349 s. |
| `20261003T175525Z-V7-ffn-irq-control-corrupt` | Retirada prematura detectada, salida 1 esperada; 30,727 s. |
| `20261003T175525Z-V7-ffn-irq-control-synth` | Síntesis revisada; 268,517 s. |
| `20261003T180052Z-V7-ffn-irq-control-audit` | Auditoría de INIT/constraints; 51,070 s. |
| `20261003T180226Z-V7-ffn-irq-control-sim-r3` | Banco final sin declaración implícita, 54 casos PASS; 54,773 s. |

Las 24.715 líneas NTRPU de r2/r3 son idénticas. Solo se adelantó la
declaración explícita de `raw_irq` en el banco; RTL sin cambios. Fuente final
simulada `source-20261003T180217Z`, SHA-256
`c173d66ee2234dd002e833fb3a7e51401ea51d1fe5964abdccac67f390556767`.
Checkpoint sintetizado/auditado:
`b01d525e5cfe7c25961c313d917f52952741a791e327c1a9282a0a37bab47234`.
Los hashes completos de cada archivo y fuente figuran en `VERIFIED.json`.

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_irq_control.tcl \
  -tclargs sim build/CSR_IRQ_NUEVO
python3 tools/check_ntrpu_ffn_irq_control.py \
  build/CSR_IRQ_NUEVO/profile-0.log --profile 0
# Repetir revisión para perfiles 1 y 2. sim_corrupt debe fallar.
vivado -mode batch -source tools/ntrpu_board_ffn_irq_control.tcl \
  -tclargs synth build/CSR_IRQ_SYNTH_NUEVO
vivado -mode batch -source tools/ntrpu_board_ffn_irq_control_audit.tcl \
  -tclargs build/CSR_IRQ_SYNTH_NUEVO/ffn_irq_control_synth.dcp build/CSR_IRQ_AUDIT_NUEVO
```

## Siguiente integración

La opción ya está propagada por dispositivo, fabric y wrapper asíncrono:
[126 casos pequeños y seis FFN Brandon](FFN_IRQ_INTEGRATION.md) pasan
en dos perfiles de reloj, con síntesis integrada revisada. Sigue conectar
el modelo XDMA/MSI y el supervisor de reset. La calificación del DIMM y el
reset DDR siguen bloqueando el cierre V2/V3/V7 sobre memoria real.
