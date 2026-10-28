# Host AXI-Lite conectado a la FFN real

La tanda `20261003T122927Z-V7-ffn-host-memory` valida
`ntrpu_ffn_device`: [control por registros](FFN_HOST_CONTROL.md) conectado
al [motor FFN FP32](FFN_FP32.md), con MAC y SwiGLU HLS reales y memoria
AXI simulada. **63/63 casos**, 71.232 bytes físicos contrastados con una
referencia independiente y 8.052 escrituras AXI-Lite revisadas.

## Frontera del dispositivo

```mermaid
flowchart LR
  Host[Host simulado] -->|AXI-Lite 32 bits| CSR[ntrpu_ffn_control]
  CSR -->|descriptor y cancelación| FFN[ntrpu_ffn_memory]
  FFN -->|finalización| CSR
  CSR -->|IRQ y registros| Host
  FFN <-->|AXI 512 bits, direcciones 64 bits| RAM[Memoria AXI simulada]
```

El top RTL nuevo, `fpga/rtl/ntrpu_ffn_device.sv`, une ambos bloques en el
mismo reloj/reset. No agrega traducción de direcciones ni estrecha sus 64 bits.
Expone un AXI-Lite esclavo, un AXI maestro de memoria y la IRQ de nivel.
`activation_selected` identifica el dueño del puerto de memoria para diagnóstico.
Los 18 archivos RTL anteriores de la FFN se conservan sin cambios.

`fpga/tb/ntrpu_ffn_host_bfm.sv` es exclusivamente un conductor de simulación:
convierte las solicitudes del banco en accesos MMIO. No conduce las entradas
internas del motor. La observación jerárquica de fase pertenece al scoreboard;
el host obtiene finalización, identidad y métricas mediante lecturas de registros.

## Qué se comprueba

El banco original usa `NTRPU_FFN_HOST` para seleccionar el nuevo recorrido.
Las semillas 1, 7 y 42 recorren 21 casos cada una: cuatro FFN correctas,
errores RRESP y BRESP en cada etapa, cancelación en cada etapa, reset en cada
etapa y un descriptor de salida demasiado corto rechazado antes del tráfico.
Los ocho buffers comienzan por encima de 4 GiB.

El host escribe todos los campos y emite START. Inmediatamente modifica todas
las sombras, de modo que los resultados solo pueden ser correctos si el
descriptor activo quedó capturado. AW y W llegan en ambos órdenes y juntos;
B y R permanecen retenidos por el host durante dos ciclos. Al finalizar,
START debe responder SLVERR hasta ACK; la IRQ y los registros se conservan
mientras el consumidor retiene la finalización.

Para cancelación, el banco retiene la primera respuesta B de la etapa durante
al menos 100 ciclos, permitiendo enviar CANCEL por MMIO mientras existe una
escritura pendiente. Exige drenaje completo antes de finalizar. La referencia
anterior utiliza 25 ciclos con su conexión directa; ambos contratos quedan
separados. Los cuatro resets coordinan dispositivo y modelo AXI y permiten
otra operación. **No resuelven el fallo de warm reset del modelo DDR4**.

El revisor `tools/check_ntrpu_ffn_host.py` reconstruye independientemente las
direcciones, tamaños, épocas, forma, tiles, comandos y respuestas MMIO; cruza
las finalizaciones con cada byte realmente escrito. Rechaza doce corrupciones
de evidencia, incluyendo dirección alta truncada, identidad, datos, máscara,
ACK ausente y START indebidamente aceptado.

## Medidas del banco sintético

Vivado/XSIM 2026.1, `xcvu5p-flva2104-2-i`, reloj simulado de 5 ns.
Duración de pared: 392,839 s para las tres semillas.

| Semilla | Ciclos del banco | Bytes físicos revisados | Escrituras MMIO | Resets / cancelaciones |
| --- | ---: | ---: | ---: | ---: |
| 1 | 195.088 | 23.744 | 2.684 | 4 / 4 |
| 7 | 195.076 | 23.744 | 2.684 | 4 / 4 |
| 42 | 195.077 | 23.744 | 2.684 | 4 / 4 |

Ejemplos correctos de la semilla 1:

| D / H | Programación hasta respuesta de START | Contador del motor | Servicio host hasta ACK y comprobación de reposo |
| --- | ---: | ---: | ---: |
| 32 / 32 | 451 ciclos | 5.356 ciclos | 5.890 ciclos |
| 64 / 96 | 452 ciclos | 19.378 ciclos | 19.914 ciclos |
| 64 / 80 | 452 ciclos | 18.512 ciclos | 19.051 ciclos |

El contador del motor mide desde la aceptación del descriptor hasta la
finalización en el CSR. El servicio host también incluye programación,
consulta, consumo y ACK. Las escrituras que alteran las sombras se solapan
con el cómputo. El polling usa este bus ideal de simulación: estas cifras
**no miden MMIO por PCIe ni latencia de un driver físico**. Tampoco son
tokens/s ni inferencia completa.

## Regresión y procedencia

`20261003T122927Z-V7-ffn-host-bank-regression` pasa 63/63 casos directos en
373,494 s. Todas sus líneas `NTRPU_*` coinciden byte a byte con la referencia
`20261003T113156Z-V7-ffn-large-bank-regression`, incluidos ciclos y bytes.
Conserva 71.040 bytes físicos revisados; la diferencia de prefijos de reset
frente al banco host deriva de las pausas distintas y no se presenta como
una mejora de cómputo.

Ambas tandas usan `source-20261003T122856Z`, SHA-256
`1b1c4fe4146cd5544e483756b6dd2522f486e4bcd597a6eedd5f328fd566a8a3`.
La copia de los RTL HLS se comprueba byte a byte contra sus co-simulaciones
calificadas. Cada respaldo incluye fuentes, logs, revisión, manifiesto y hashes.
Los avisos de compilador XSIM y límites del visor de arrays quedan registrados.

## Pesos reales y síntesis del conjunto

`20261003T123822Z-V7-ffn-host-brandon` pasa tres FFN Brandon de capa 0,
`D=256,H=720`, en 656,075 s de pared. Se reutilizan los 37 archivos del
fixture identificado en [FFN_FP32](FFN_FP32.md): pesos reales, entrada
sintética compartida por gate/up y referencia escalar independiente.
Los 42.048 bytes físicos coinciden exactamente, incluidos los 768 valores
FP32 finales de la referencia Q8 nativa. Cada muestra lee 638.720 bytes y
escribe 14.016; 418 AR, 9980 R, 82 AW, 219 W y 82 B. Doce alteraciones de
evidencia son rechazadas también en este recorrido.

| Muestra | Contador del motor | Servicio host completo | Error L2 frente a FFN FP32 |
| --- | ---: | ---: | ---: |
| 1 | 307.219 ciclos | 307.755 ciclos | 1,149197% |
| 2 | 307.134 ciclos | 307.669 ciclos | 1,369425% |
| 3 | 307.157 ciclos | 307.696 ciclos | 1,503418% |

El error L2 compara la aritmética Q8 con la referencia FP32, no indica
desacuerdo entre RTL y Q8. Los ciclos difieren del banco directo porque cambia
la secuencia de pausas; no se atribuye aceleración al registro de control.
La prueba realiza 384 escrituras y 184.108 lecturas MMIO, incluyendo polling.

`20261003T123822Z-V7-ffn-host-synth` sintetiza el conjunto con **41.802 LUT,
38.981 FF, 93 DSP, 39 RAMB36 y tres RAMB18**. Hay 168 LUTRAM y 244 SRL
incluidos en el total LUT. El coste adicional respecto a la síntesis FFN
previa es 1.395 LUT y 4.364 FF; BRAM/DSP no aumentan.

WNS estimado +0,683 ns y WHS +0,039 ns bajo reloj de 5 ns. El peor camino
va de `control/running_reg` al enable de `core/matrix/controller/done_rows`.
Sin latches ni cajas negras; cero endpoints internos sin reloj/restricción.
Faltan retardos de 636 entradas y 769 salidas y el contexto del reloj padre.
Se conservan los avisos de aritmética HLS, cachés paralelas en registros,
puertos sin carga y recomendaciones DSP/RAM; DRC/metodología no reportan
errores ni avisos críticos. **La síntesis no demuestra timing posrutado**.

Estas dos tandas usan `source-20261003T123750Z`, SHA-256
`2d931214286b5c606c4ec768d10f8871d71d00c2139db2e2f741cd3958c9d0cc`.
Los 20 RTL coinciden con la simulación sintética. DCP sintetizado:
`3c66cceb350a2c65f4f0990153a618f12d805a901f0305bf042033d8bc718971`.

## Implementación y auditoría del reloj

`20261003T124752Z-V7-ffn-host-route` coloca y rutea el conjunto en 673,419 s.
El reporte final muestra **41.536 LUT, 38.775 FF, 93 DSP y 40,5 BRAM**;
84 LUTRAM y 175 SRL están incluidos en las LUT. A 5 ns, WNS **+0,099 ns**,
WHS **+0,010 ns** y margen de pulso **+1,958 ns**, sin endpoints internos
que fallen. El peor camino une la FSM de un MAC HLS con el enable de uno
de sus registros de entrada.

Vivado cuenta 70.799 nets enrutables completas y cero errores de ruteo,
pero conserva **un hueco explícito del reloj padre**. La auditoría de solo
lectura `20261003T125921Z-V7-ffn-host-route-audit` (79,427 s) identifica
1.053 alias jerárquicos PARTIAL con exactamente la misma ruta que `clk`.
No son 1.053 fallos de datos distintos. Confirma cero latches, un único reloj
y ninguna excepción temporal añadida.

El contexto `HD.CLK_SRC=BUFGCE_X0Y60` sigue siendo un supuesto de ubicación,
no un buffer padre instanciado. Faltan posiciones de frontera y retardos de
636 entradas/769 salidas; CDC omite esas fronteras sin restricciones. No se
ha completado el ruteo del reloj ni el timing de placa. DRC mantiene las
recomendaciones DSP, RTSTAT-10 y metodología SYNTH-6/9 y TIMING-18.
No se rebajan severidades ni se introducen false paths para ocultarlos.

Fuente de ruteo/auditoría: `source-20261003T124720Z`, SHA-256
`637d050b589b54f10a1c0968d1da41f9f720a263a5ce9b9ce694631744c262fb`.
Entrada DCP contrastada con la síntesis calificada y los 20 RTL de simulación;
checkpoint de ruteo:
`037a66f7f5ce662dfe5e4a5d5c9f2c2e358d4846cb3f964d049abcb74dd2e862`.

## Reproducción y pendientes

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_RUN" "$SWIGLU_RUN" sim_host_fp32 build/HOST_RUN_NUEVO
# El quinto argumento opcional selecciona el fixture Brandon previamente verificado.
# synth_host_fp32 exporta ffn_device_synth.dcp y los reportes del conjunto.
vivado -mode batch -source tools/ntrpu_board_ffn_memory_route.tcl \
  -tclargs "$SYNTH_DCP" BUFGCE_X0Y60 build/HOST_ROUTE_NUEVO
vivado -mode batch -source tools/ntrpu_board_ffn_memory_route_audit.tcl \
  -tclargs "$ROUTED_DCP" build/HOST_AUDIT_NUEVO
```

Las seis tandas, con sus revisiones y respaldos, se registran por separado en
[VIVADO_STATUS](board/VIVADO_STATUS.md).
La FFN completa 2048/8192 sigue siendo una tanda independiente.

En las seis tandas anteriores los pesos y entradas están precargados en el
modelo de memoria. El [ensayo posterior de memoria compartida](AXI_MEMORY_FABRIC.md)
ya valida carga AXI desde el host, arbitraje y lectura final con tres muestras
Brandon. Faltan carga DMA
desde el host, conexión DDR, puentes CDC, driver y prueba
PCIe integrada. Las dependencias V2–V6 y la calificación eléctrica de placa
siguen vigentes; este bloque no convierte el BAR en memoria coherente.

La siguiente integración debe conectar ese camino al XDMA ya probado y
calificar sus relojes/reset, conservando aparte la dependencia DDR. El árbitro
de dos entradas ya mide un beat por ciclo en su banco aislado; su revisión
integrada se registra en el documento de memoria compartida.
