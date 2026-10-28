# FFN completa con retiro de interrupciones y host asíncrono

Actualizado el 2026-10-03. La [ABI 2.0](FFN_IRQ_CONTROL.md) ya controla
el motor FFN RTL/HLS a través de los conversores AMD AXI4/AXI4-Lite.
Dos perfiles de reloj pasan tanto el banco numérico/de errores como tres
FFN con pesos de Brandon cargados y recuperados por el host.
En esos bancos el ACK lo proporciona un par de protocolo. La integración
posterior [PCIe/XDMA y MSI-X](PCIE_FFN_INTEGRATION.md) pasa una FFN
sintética pequeña con ACK y paquetes del XDMA real simulado, además de
dos descriptores inválidos. DDR y el supervisor de reset siguen pendientes.

## Composición y compatibilidad

`EXTERNAL_RETIRE=1` selecciona `ntrpu_ffn_irq_control` en
`ntrpu_ffn_device`, y se propaga por `ntrpu_ffn_fabric` y
`ntrpu_ffn_async_fabric`. En este modo, `irq` e `usr_irq_ack` pertenecen
a `host_clk`: la IRQ ya atraviesa el transporte XPM y no vuelve a pasar
por el sincronizador de nivel de ABI1. El control AXI-Lite conserva la
[recodificación reversible RRESP](FFN_ASYNC_INTEGRATION.md).

Solo cambian esos tres wrappers RTL respecto del CSR/notificador previamente
validado. Motor, árbitro, conversores numéricos, HLS y HDL/XDC AMD conservan
sus hashes. Por defecto, `EXTERNAL_RETIRE=0` mantiene ABI1; sus entradas
host/ACK no utilizadas deben conectarse explícitamente. Una nueva regresión
de 63 casos conserva las 9.298 líneas NTRPU de la referencia ABI1.

## Contrato ejercitado por el host

Cada descriptor comprueba versión/capacidad, disponibilidad, identidad
admitida, contadores y metadatos. El host alterna tres modos:

1. ACK de transporte primero: IRQ continúa activa hasta servicio MMIO;
   cambiar la máscara o presentar una identidad errónea no la retira.
2. Servicio primero: resultado e IRQ permanecen pendientes hasta ACK de
   transporte. START y el servicio duplicado reciben `SLVERR`.
3. Polling: no se genera IRQ ni se exige ACK de transporte. Habilitar IRQ
   después de finalizar no cambia el modo capturado para ese trabajo.

El host lee los resultados antes de escribir secuencia/época y ACK en
`0x38/0x3c/0x10`. Solo inicia otro trabajo tras observar retiro y rearme.
La identidad sigue siendo un contrato del host privilegiado; no autentica
un ACK PCIe espurio ni admite reutilizar identidades en una sesión.

## Pruebas y revisión independiente

La tanda separada de [medición por etapa](FFN_PERFORMANCE.md) añade un
observador pasivo para distinguir ocupación MAC/HLS y esperas AXI. Sus
12 etapas pasan la revisión y conservan íntegramente las trazas de esta
referencia, incluidos tiempos; el respaldo está verificado.

Motor a 5 ns; host a 4 o 6,2 ns con desfase de 700 ps. Cada perfil ejecuta
tres semillas de 21 casos: **63 casos, 12 cancelaciones, 12 resets durante
cómputo y 51 finalizaciones retiradas**. Los resets iniciales son adicionales.
Se comparan resultados físicos, errores AXI, descriptores inválidos y
recuperación. Un reset puede interrumpir la reescritura de registros sombra:
el revisor exige un prefijo exacto posterior a START, sin finalización del
trabajo abortado. No exige completar escrituras que el reset interrumpió.

El banco pequeño conserva tres avisos VRFC 10-3380 por declarar `clk`
después de la tarea de reset. La declaración explícita de módulo resuelve
las tres referencias; los flancos observados se comprueban por separado.
No se dispensan incompatibilidades de ancho ni entradas DUT sin conectar.

Por cada perfil Brandon: **tres FFN D=256/H=720, 768 FP32 exactos**,
1.777.152 bytes cargados por AXI en RAM inicialmente vacía, 42.048 bytes
escritos por el cálculo y 3.072 bytes finales recuperados por el host.
Los pesos y los 37 archivos del fixture coinciden con la referencia sellada;
las entradas son sintéticas. No es inferencia completa del modelo.

`check_ntrpu_ffn_irq_async.py` combina la referencia numérica independiente
con trazas MMIO completas, handshakes del motor/notificador, identidad,
flancos de ambos relojes, duración del cálculo y orden de retiro. Rechaza
57 alteraciones de evidencia por perfil pequeño y 27 por perfil Brandon.
Una inyección en XSIM retira IRQ después del ACK de transporte y antes del
servicio: el banco falla en la primera FFN como se esperaba.

| Medida simulada por FFN Brandon | Host 4 ns | Host 6,2 ns |
| --- | ---: | ---: |
| Ciclos de cómputo, de aceptación a finalización | 308.795–308.953 | 308.781–308.977 |
| Carga host, µs | 382,3662–382,7672 | 483,0218–483,9816 |
| Programación/cómputo/polling, µs | 1.551,8078–1.552,7368 | 1.553,8532–1.555,5594 |
| Lectura final, µs | 0,7122–0,7332 | 0,9394–0,9670 |
| Evento aceptado → retiro, ns | 2.555–2.910 | 3.215–3.840 |
| Servicio MMIO aceptado → retiro, ns | 90–470 | 120–615 |

Los intervalos incluyen las pausas programadas del banco. Los modelos FIFO
asíncronos AMD omiten retardos de sincronización; estas cifras no miden
latencia PCIe ni ancho de banda físico.

## Síntesis e implementación

Síntesis: **44.585 LUT, 45.370 FF, 93 DSP, 39 RAMB36 y tres RAMB18**,
sin URAM. WNS estimado +0,768 ns; WHS +0,013 ns; WPWS +1,468 ns.
No hay endpoints internos sin restricciones. Conserva 17 grupos bus skew,
28 CDC-3, diez CDC-9, un CDC-15 visible y 21 waivers internos AMD.
Los cinco sincronizadores del transporte IRQ tienen tres etapas.

Los avisos de DSP/RAM del motor permanecen; los dos handshakes nuevos
eliminan sus registros XPM `dest_req_ext_ff` no utilizados en modo ACK
externo. Faltan retardos de 1.404 entradas/1.333 salidas y relojes padre;
TIMING-18 permanece.

El ruteo exploratorio termina con **44.472 LUT, 45.219 FF y 93 DSP**;
mantiene BRAM/URAM. WNS +0,104 ns, WHS +0,010 ns y WPWS +1,468 ns,
sin endpoints que fallen setup, hold o pulso. Sus **80.297 redes enrutables**
están completas, sin errores; los 17 grupos bus skew tienen margen mínimo
+3,389 ns. TIMING-37 y Route 35-470 siguen ausentes.

La auditoría independiente reabre el mismo checkpoint y confirma los
131 FDRE de payload/envío IRQ con INIT=0. Los 33 registros variables de
respuesta AXI-Lite alimentan un destino cada uno; la captura restante
recibe GND. Los 21 waivers internos AMD y todos los grupos max-delay/bus-skew
se conservan. Desaparecen solo tres false paths de reset-busy eliminados
por optimización, como en la referencia; CDC-3 cambia de 28 a 25.

Los 1.192/155 alias PARTIAL corresponden exactamente a los dos relojes
padre. Quedan esos dos huecos explícitos, retardos ausentes de
1.398 entradas/1.333 salidas y un CDC-15 visible dentro del FIFO AMD.
No se añadieron excepciones globales ni se rebajaron reglas DRC.
Estos márgenes internos no constituyen cierre temporal de placa.
El revisor posrutado rechaza 15 alteraciones de evidencia, incluidas INIT
incorrecta, waiver retirado, ruta incompleta y fuente RRESP compartida.

## Tandas y reproducción

Cada tanda conserva invocación, fuentes, logs, revisión y manifiestos en
NFS y `artifacts/ntrpu-board-vivado/<id>/`. `VERIFIED.json` acredita el
respaldo terminado; no debe confundirse salida cero con aprobación.

| ID | Resultado; duración de Vivado |
| --- | --- |
| `20261003T181642Z-V7-ffn-irq-async-errors` | 63 casos PASS, host 4 ns; 476,914 s |
| `20261003T181652Z-V7-ffn-irq-wrapper-legacy` | 63 casos ABI1 PASS, trazas idénticas; 397,386 s |
| `20261003T181921Z-V7-ffn-irq-async-errors-slow` | 63 casos PASS, host 6,2 ns; 479,129 s |
| `20261003T181921Z-V7-ffn-irq-brandon` | Tres FFN/768 FP32 PASS; 829,373 s |
| `20261003T181921Z-V7-ffn-irq-brandon-slow` | Tres FFN/768 FP32 PASS; 833,702 s |
| `20261003T181949Z-V7-ffn-irq-async-corrupt` | Retirada prematura detectada, salida 1 esperada; 86,789 s |
| `20261003T181949Z-V7-ffn-irq-async-synth` | Síntesis revisada; 551,078 s |
| `20261003T183700Z-V7-ffn-irq-async-route` | Ruteo exploratorio, timing interno positivo; 803,280 s |
| `20261003T185027Z-V7-ffn-irq-async-audit` | INIT, clocks, rutas y constraints revisados; 100,720 s |

Simulaciones y síntesis usan `source-20261003T181633Z`, SHA-256
`5bbdb96e0891f0470214932ca8428e79f6cb5f84b8490094c603a05528eaf609`.
Checkpoint de síntesis:
`df4896096955c6f08ed50c9806e5a65bcdc1bc5d28ad2aab7f045e40f71cea75`.
Checkpoint enrutado:
`8413ae4a429e5de5a8eae5ffa2f759ce3d16ecb30adf19ea7fb0798f51f5f72d`.
Ruteo/auditoría usan `source-20261003T183631Z`, SHA-256
`af8a34f3c871f3baf2eaa441b0fc7ad7c499745d54c78255ed2ed77ce3614268`;
el motor proviene del checkpoint de síntesis anterior, sin regenerarlo.
Los revisores añadidos después acompañan cada archivo con su propio hash.

```bash
# MAC/SWIGLU: raíces HLS fijadas en invocation.json; salidas nuevas siempre.
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC" "$SWIGLU" sim_irq_errors_fp32 build/IRQ_ERRORS_NUEVO
python3 tools/check_ntrpu_ffn_irq_async.py \
  build/IRQ_ERRORS_NUEVO/seed-1.log --seed 1 --host-period-ps 4000
# Repetir semillas 7/42; sim_irq_errors_slow_fp32 usa 6200 ps.
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC" "$SWIGLU" sim_irq_async_fp32 build/IRQ_BRANDON_NUEVO "$FIXTURE"
python3 tools/check_ntrpu_ffn_irq_async.py \
  build/IRQ_BRANDON_NUEVO/seed-1.log --seed 1 --host-period-ps 4000 \
  --fixture "$FIXTURE"
```

También existen `sim_irq_async_slow_fp32`, `sim_irq_errors_corrupt`
(fallo esperado) y `synth_irq_async_fp32`.

## Siguiente dependencia

La variante quad/FIFO y SwiGLU optimizado pasa otros **126 casos**:
`20261004T002842Z-V7-ffn-quad-irq-errors` y su pareja `-slow`, con host
de 4/6,2 ns, motor de 5 ns y desfase de 700 ps. Cada tanda verifica
63 casos, 12 resets durante cómputo, 12 cancelaciones, 51 eventos de
finalización/retiro y rechaza 57 alteraciones de evidencia.
La compilación confirma quad/FIFO; todos los RTL/HLS coinciden con
Brandon nativo optimizado. Se mantienen las limitaciones del modelo CDC.

Ambas tandas tienen respaldo local/NFS, 368 archivos cada una.
SHA-256 rápido: `d12ab489b644357d97b50cffbe8f618cb4b24d0020ff13e5d2d15929d67c9380`;
lento: `feb193e04d03e33708c8a1d7a3ab986f0f59873659aaad610800dd249e61389b`.
Modos reproducibles: `sim_irq_errors_quad_fp32_fifo` y
`sim_irq_errors_quad_slow_fp32_fifo`. La síntesis integrada
`20261004T004355Z-V7-ffn-quad-irq-synth` termina en 640,792 s y pasa
la revisión: **55.254 LUT, 51.190 FF, 153 DSP, 48 RAMB36 + 3 RAMB18**.
Su modo es `synth_irq_async_quad_fp32_fifo`. WNS +0,173/WHS +0,013 ns
son estimaciones de síntesis; los 17 grupos de bus skew cumplen, con
margen mínimo de 3,851 ns. Los cinco sincronizadores de IRQ conservan
tres etapas; CDC-15 sigue visible y no se añaden excepciones generales.
El verificador rechaza 13 alteraciones de fuentes, HLS, timing, CDC y avisos.
Respaldo local/NFS verificado: 117.468.065 bytes y 232 archivos, SHA-256
`518125540a59123d4f04c822a1a2722a4dbe1c359e359d3a9f1896172e34f4d5`.

`20261004T005107Z-V7-ffn-quad-irq-brandon` y su pareja `-slow` pasan y están
respaldados: tres FFN y 768 FP32 exactos por tanda, 42.048 bytes intermedios
y finales, 1.777.152 bytes cargados y 3.072 leídos por el host. Cada tanda
rechaza 27 alteraciones de evidencia. Los ciclos de cómputo medidos en el
CSR son 138.493/138.524/138.650 (host 4 ns) y 138.574/138.367/138.510
(host 6,2 ns); las cifras incluyen las pausas de su banco, no tokens/s.
Se conservan ambos órdenes de ACK y polling. Duraciones: 744,147/765,829 s.

Con esos respaldos, `20261004T011231Z-V7-ffn-quad-irq-route` y la auditoría
`20261004T013503Z-V7-ffn-quad-irq-route-audit` terminan y están respaldadas.
**FAIL de setup: WNS −0,003 ns, un endpoint; WHS +0,010 ns** a 5/4 ns.
Recursos: 54.969 LUT, 50.946 FF, 153 DSP, 48 RAMB36 + 3 RAMB18.
Las 98.530 nets de datos están completas; quedan los dos huecos de reloj
padre y fronteras sin calificar. Los 17 grupos de skew cumplen; se
conservan INIT=0 de los 131 registros IRQ y los 33 destinos únicos de
respuesta AXI-Lite. El revisor rechaza 15 alteraciones de evidencia.
BUFGCE_X0Y60/X0Y61 siguen siendo contexto provisional.
SHA-256: `bdd421df77b07844d4d99a2b38b900041c92ce888ea259589b8c93fd29582921`
y `03c832702e6a92532f9eae53503b96ae6add00236eb8ef3b7a94c979e1497c2f`.

El peor camino atraviesa detección de error del MAC hasta los acumuladores:
14 niveles, 4,748 ns de datos (72,556% ruteo).
`20261004T014712Z-V7-ffn-quad-irq-physopt` desde el checkpoint auditado,
sin cambiar RTL, períodos ni excepciones, mejora setup a **+0,003 ns**,
con hold **+0,010 ns** y recursos iguales. Vivado optimiza una net y no
añade ni elimina celdas; tarda 225,691 s. La auditoría `20261004T015319Z`
confirma el resultado en 121,140 s; rechaza 15 alteraciones de evidencia.
Los 17 grupos de skew tienen margen mínimo +2,844 ns. Ambas tandas están
respaldadas: `fd722f479c5f279caff7b23ee1a2ddb550439a8a119325ac143dc4ffed667bb2`
y `ff41e08c19c9f7e119c3de61d94752f6806d3370bd7987e875f09396d011ab92`.
El fallo original permanece; **3 ps es un margen exploratorio mínimo**,
con dos relojes padre y fronteras aún sin calificar en placa.
El primer dueño de cola rechazó una copia desactualizada antes de crear
un run remoto; se conserva su diagnóstico, sin duplicar implementación.

La conexión `irq/usr_irq_ack` a XDMA/MSI-X ya conserva ambas
confirmaciones en el [banco PCIe pequeño](PCIE_FFN_INTEGRATION.md).
Extenderlo a carga/lectura de Brandon. El fixture pequeño ya pasa sondeo
MMIO concurrente con DMA y recepción CQ/RC separada, con revisión de
paquetes y 33 alteraciones rechazadas. Brandon dual ya pasa y está respaldado;
la nueva integración Brandon quad/FIFO optimizada también pasa y está respaldada:
256 FP32 exactos y 136.909 ciclos de cómputo.
Integrar el
[supervisor](RESET_SUPERVISOR.md) que coordine host, motor, conversores y memoria
ante reset o pérdida de reloj. La [isla IRQ supervisada](SUPERVISED_IRQ.md)
ya pasa 63 casos con confirmaciones reales de drenaje y recuperación;
la [FFN supervisada](SUPERVISED_FFN.md) incorpora ahora conversores AXI,
cómputo y pares conductuales: 126 casos y 30 recuperaciones con relojes
activos. Faltan pérdidas de reloj en ese conjunto y DMA interrumpido.
La cobertura combinada actual usa resets durante cómputo
en matrices pequeñas; los resets en fases de notificación se verificaron
en el banco aislado del CSR/transporte, no en el fixture Brandon.
V2/V3 siguen abiertos para DIMM exacto y reset DDR; no habilita V8/ECAD.
