# SwiGLU y cuantización entre regiones AXI

Estado al 2026-10-03 UTC: primer banco integrado **105/105 PASS** con el
RTL HLS real. `fpga/rtl/ntrpu_swiglu_memory.sv` conecta el
[lector AXI](AXI_READER.md), el [operador local](SWIGLU_Q8_BLOCK.md) y el
[escritor ABI1](ACTIVATION_WRITER.md). No integra todavía la secuencia
gate/up/down del [motor de matrices](MATRIX_MEMORY.md), DDR ni DMA PCIe.

## Flujo y contrato

Un job procesa de 1 a 8192 parejas de elementos gate/up binary64
little-endian. Cada descriptor contiene dirección, base, longitud y época;
se captura junto con el número de elementos y la identidad del job.
Las cuatro épocas —job y tres buffers— deben coincidir con `expected_epoch`.
Las entradas deben permanecer inmutables hasta terminar el job.

Las direcciones se alinean a 64 bytes. Cada entrada necesita
`ceil(elements/8) × 64` bytes accesibles; la salida necesita
`ceil(elements/32) × 64`. Se revisan sumas de 65 bits y regiones completas
antes de emitir AR. El intervalo escrito no puede solaparse con ninguno
de los intervalos leídos; gate y up pueden compartir memoria de solo lectura.
El padding leído del último beat no participa en el cálculo.

Por cada bloque de hasta 32 elementos:

1. Leer hasta 256 bytes de gate y hasta 256 de up, conservando direcciones
   de 64 bits y dividiendo las lecturas en límites de página de 4 KiB.
2. Esperar la terminación correcta de ambas lecturas antes de arrancar HLS.
   Sus puertos `ap_memory` leen buffers privados con latencia de un ciclo.
3. Capturar las ocho palabras de salida y su mapa de presencia. Rechazar
   palabras duplicadas, faltantes o `ap_return` distinto de cero.
4. Entregar un registro completo al escritor. Esperar su B correcta antes
   de avanzar a los siguientes 32 elementos.

El destino permanece **privado hasta la completion del vector entero**.
`done_blocks` cuenta bloques escritos y completados correctamente;
`done_bytes` incluye bytes W de un bloque que terminó con error.
Ninguno autoriza publicar un prefijo si el estado raíz es distinto de cero.
`done_hls_status` conserva el retorno de la última operación HLS terminada.

## Fallos y recuperación

| Estado raíz | Causa |
| --- | --- |
| 0 | Vector completo correcto |
| 1 / 2 | Forma, rango, alineación o solapamiento / época |
| 3 | Lectura AXI, conteo o delimitación incorrectos |
| 4 | Escritura, respuesta o metadatos del escritor incorrectos |
| 5 | Error HLS o captura incompleta/duplicada de sus palabras |
| 7 / 8 | Cancelación / watchdog |

El primer fallo queda retenido. Cancelación o timeout bloquean las fases
futuras y drenan el lector, HLS o escritor ya arrancados. Un comando de
lectura pendiente puede terminar sus fragmentos de página; un AW/W
pendiente mantiene valid/payload hasta handshake. El watchdog no abandona
transacciones. Un peer permanentemente detenido requiere reset coordinado;
este debe resetear o aislar AXI y descartar el vector privado. El módulo
y sus subbloques usan un único reloj, sin CDC añadidos.

## Evidencia inicial

`20261003T080726Z-V7-swiglu-memory-unit`: 35 casos por semilla 1/7/42,
270,515 s. Incluye 1/31/32/33/720 elementos, registros parciales,
padding NaN ignorado, direcciones altas, cruces de página, descriptores
alterados tras aceptación, errores RRESP/HLS/B tardíos, cinco cancelaciones,
timeouts y cinco fases de reset por semilla. El siguiente job se recupera.

Por semilla: 141 AR, 365 beats R, 41 W y 39 B; dos resets interrumpen
escrituras antes de B. Son conteos de una mezcla de éxitos y fallos,
no throughput. Los 123 registros W de las tres semillas coinciden byte
por byte con una referencia Python independiente; el banco comprueba
también memoria final y guardas. Trece alteraciones de evidencia se rechazan.

El fixture inicial usa gate=257 y up entero variable, con up[0]=127
en cada bloque. En FP32, `SiLU(257)` redondea a 257; la referencia cerrada
produce escala 257 e INT8 iguales a up. Comprueba transporte, ensamblaje,
colas y control con el HLS real. La variedad numérica general del operador
está cubierta por sus bancos locales; aquí no se usan pesos de un modelo.
La regresión `20261003T081935Z-V7-swiglu-memory-varied` cambia gate a
`257 + 2 × bloque` y pasa otros 105/105 casos, con los mismos conteos
y 123 registros W exactos. Verifica la recarga de ambos vectores entre
bloques; no modifica el RTL sintetizado. Dura 271,495 s.

| Elementos | Ciclos del job, semillas 1 / 7 / 42 | AR / R / W |
| --- | --- | --- |
| 1 | 3702 / 3704 / 3707 | 2 / 2 / 1 |
| 31 | 3836 / 3834 / 3831 | 4 / 8 / 1 |
| 32 | 3842 / 3836 / 3838 | 4 / 8 / 1 |
| 33 | 7535 / 7543 / 7530 | 6 / 10 / 2 |
| 720 | 87.978 / 88.005 / 87.984 | 50 / 180 / 23 |

`20261003T081935Z-V7-swiglu-memory-8192` pasa la forma máxima, semilla 1:
**980.075 ciclos**, 544 AR, 2048 beats R y 256 AW/W/B. Lee 131.072 bytes
y escribe **16.384 bytes exactos** en 256 registros, con guardas intactas.
Los 8192 INT8 y las 256 escalas coinciden con la referencia independiente;
nueve alteraciones de evidencia se rechazan. Dura 388,948 s de herramientas.
A 5 ns por ciclo son 4,900375 ms simulados; no es tiempo físico ni tokens/s.

## Implementación del adaptador

Síntesis `20261003T080726Z-V7-swiglu-memory-synth`, 243,679 s:
**17.424 LUT, 14.060 FF, 37 DSP, 2 RAMB36 + 1 RAMB18**; WNS estimado
+1,747 ns y WHS +0,039 ns con reloj de 5 ns. Los buffers gate/up con ocho
escrituras paralelas se implementan en registros (`Synth 8-4767`); los
totales incluyen ese coste. Se conservan recomendaciones DSP/BRAM de HLS.
No hay loops ni endpoints internos sin restricción. Faltan retardos de
1316 entradas y 745 salidas, y contexto del reloj padre.

El ruteo `20261003T081935Z-V7-swiglu-memory-route` usa **17.024 LUT,
13.956 FF, 37 DSP, 2 RAMB36 + 1 RAMB18**, con WNS +0,567 ns y
WHS +0,010 ns bajo las restricciones internas actuales. Vivado cuenta
24.495 nets ruteables completas y cero errores de ruteo; queda un gap
explícito del reloj. Dura 267,076 s.

La auditoría `20261003T082422Z-V7-swiglu-memory-route-audit`, 117,673 s,
identifica 624 aliases jerárquicos `PARTIAL` con idéntica ruta del reloj.
`HD.CLK_SRC=BUFGCE_X0Y60` es provisional; no hay buffer padre ni retardos
externos calificados. No se añadieron excepciones ni dispensas. Los márgenes
OOC no cierran timing de placa. Se conservan recomendaciones DSP/BRAM,
incluidos tres registros de salida no absorbidos por BRAM en HLS.

## Reproducción y siguiente integración

Partir del directorio de una co-simulación HLS terminada y respaldada:

```bash
vivado -mode batch -source tools/ntrpu_board_swiglu_memory.tcl \
  -tclargs "$SWIGLU_HLS_RUN" sim NEW_SIM
vivado -mode batch -source tools/ntrpu_board_swiglu_memory.tcl \
  -tclargs "$SWIGLU_HLS_RUN" large NEW_8192
vivado -mode batch -source tools/ntrpu_board_swiglu_memory.tcl \
  -tclargs "$SWIGLU_HLS_RUN" synth NEW_SYNTH
```

El lanzador copia los RTL/IP HLS sin modificarlos. `sim` ejecuta tres
semillas; `large` ejecuta únicamente 8192 elementos, semilla 1.
El reloj del banco es de 5 ns y su watchdog de 4096 ciclos; el RTL de
producción utiliza 65.536. El verificador independiente reconstruye el
plan AR y todos los bytes ABI1. Las ejecuciones usan carpetas nuevas.

La [FFN raíz](FFN_MEMORY.md) ya conecta matrices gate/up, este vector y
down: 63 casos numéricos sintéticos/error/reset con memoria compartida.
También pasan tres cadenas con pesos reales de Brandon y 45.120 bytes
exactos con salida binary64. El [escritor FP32 integrado](FFN_FP32.md)
ya pasa sus bancos numéricos y de fallos; sus métricas se registran aparte.
La publicación debe esperar el éxito de down y del controlador raíz.
