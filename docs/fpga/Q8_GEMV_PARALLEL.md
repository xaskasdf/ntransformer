# GEMV Q8 con dos motores MAC

Estado al 2026-10-03 UTC: dos MAC HLS aceleran las proyecciones reales de
Brandon **1,916–1,917×** frente al controlador escalar.
Las 5088 salidas comparadas mantienen diferencia numérica cero frente a la
referencia Q8. Son resultados de simulación RTL con memoria AXI sintética;
no miden DDR física, tokens/s ni una FFN encadenada.

## Arquitectura y formato

El [wrapper de memoria](ACTIVATION_MEMORY.md) añade carga AXI de
activaciones y arbitraje por fases sin modificar este núcleo. Conserva
la caché privada y solo admite pesos tras validar el vector completo.

`fpga/rtl/ntrpu_q8_gemv_parallel.sv` conserva el transporte y los contratos
del [GEMV escalar](Q8_GEMV_STREAM.md), que permanece como referencia separada.
Instancia dos copias del MAC HLS validado, asignando filas pares al motor 0
e impares al motor 1. Cada fila conserva su acumulador FP64 y orden K
creciente entre tiles; no se suman subtotales ni se reasocia la reducción.

El nuevo comando exige **`job_layout=2`**. Dentro de cada tile, el orden es
`[bloque K][fila][34 bytes Q8]`; layout 1 recorre `[fila][bloque K]` y se
rechaza antes de emitir AXI. Un empaquetador debe declarar el formato, no
reutilizar los bytes del layout escalar. Se mantienen 1–32 filas, K hasta
8192 elementos, tiles de hasta 32 bloques K y la alineación existente.

Cada motor retiene pesos, activaciones, acumulador anterior y fila mientras
ejecuta. Los resultados vuelven a filas diferentes; el contador admite dos
completions simultáneas. Cancelación y primer error se comparten: se drenan
AXI y ambos motores antes de completion, sin publicar filas fallidas.

La caché de activaciones usa dos puertos explícitos. El puerto A escribe
durante carga o lee para el motor 0; B lee para el motor 1. La primera
descripción con selección dinámica de salida infería LUTRAM. Compartir
explícitamente la dirección de escritura/lectura de A recuperó BRAM sin
añadir ciclos ni cambiar las salidas.

## Medición con Brandon

Se reutiliza el fixture inmutable, revisión del modelo y referencia descritos
en [GEMV escalar](Q8_GEMV_STREAM.md#proyecciones-completas-con-pesos-de-brandon).
Por semilla: 54 jobs, 1696 filas, 17.408 bloques MAC y 591.872 bytes de pesos.
Los tiles ocupan direcciones distintas. Las entradas de gate/up/down son
sintéticas e independientes; SwiGLU y el encadenamiento quedan pendientes.

| Medida por semilla 1 / 7 / 42 | Un MAC | Dos MAC |
| --- | ---: | ---: |
| Ciclos de los jobs | 418265 / 418335 / 418348 | 218195 / 218310 / 218328 |
| Ciclos de la suite, incluidas pausas entre jobs | 418427 / 418497 / 418510 | 218357 / 218472 / 218490 |
| Ciclos ocupados por motor | 382976 | 191488 cada uno |
| Ciclos con ambos motores ocupados | 0 | 182784 |
| Ciclos Q8 bloqueado | 380424 | 179804 |
| Bytes útiles por ciclo de job | 1,415 | 2,711–2,713 |

Cada job cuenta seis fases disjuntas cuya suma debe coincidir con su latencia.
Ocupación, pausas Q8 y espera AXI se solapan; no se suman como fases.
`tools/check_ntrpu_gemv_metrics.py` comprueba esas identidades, los bytes
AXI, bloques, escrituras de activaciones y actividad de ambos motores.
El MAC escalar está ocupado aproximadamente el 91,5% del tiempo; esta
medición motivó la réplica. En dual, Q8 aún espera al cómputo el 82,4%.

Con reloj **simulado** de 5 ns, el caudal útil pasa aproximadamente de
0,283 a 0,542 GB/s. No equivale al ancho máximo del bus ni permite extrapolar
cuatro canales o Llama. El error L2 frente a FP32 sigue siendo 0,792496%
gate, 0,718751% up y 0,716756% down, debido a la representación cuantizada.

## Síntesis y validación

| Síntesis OOC VU5P, reloj 5 ns | LUT | FF | RAMB36 + RAMB18 | DSP | WNS estimado |
| --- | ---: | ---: | ---: | ---: | ---: |
| Escalar segmentado | 10396 | 7704 | 32 + 3 | 26 | +1,785 ns |
| Dual inicial, caché LUTRAM | 18126 | 10869 | 28 + 2 | 52 | +1,070 ns |
| Dual con puertos BRAM | 16258 | 10567 | 37 + 2 | 52 | +1,163 ns |

La versión BRAM no tiene LUTRAM ni cajas negras. Conserva avisos de puertos
sin carga de operadores generados y recomendaciones de pipeline DSP/RAM.
Faltan retardos de 1240 entradas y 282 salidas; el reloj del padre y las
fronteras deben calificarse al integrar. La síntesis no acredita ruteo.

`20261003T031546Z-V7-gemv-dual-route` completó opt/place/route en
320,399 s: **16.318 LUT, 10.489 FF, 37 RAMB36 + 2 RAMB18 y 52 DSP**.
El informe final da **WNS +0,271 ns y WHS +0,010 ns**, sin fallos internos
de setup/hold. Cuenta 26.231 nets enrutables completas y cero errores,
pero una net de reloj mantiene gaps explícitos. La auditoría
`20261003T032242Z-V7-gemv-dual-route-audit` (177,141 s) identifica 401
alias jerárquicos de la misma ruta `clk` parcial, todos con idéntico hash.
El contexto supone `HD.CLK_SRC=BUFGCE_X0Y60` sin instanciar el driver padre.
Conserva los 1240/282 retardos ausentes y ningún endpoint interno sin
restricción. No se agregaron excepciones ni se aprobó timing de tarjeta.

Avisos conservados: 100 Route 35-198 (límite de mensajes), cuatro DRC
23-814, 108 recomendaciones de pipeline DSP, RTSTAT-10=1 y 32 AVAL;
metodología registra 39 SYNTH-6, 16 SYNTH-9 y 1000 TIMING-18 reportados.
Los hashes confirman que el RTL sintetizable actual coincide con el de
síntesis, simulaciones BRAM y checkpoint implementado.

Evidencia revisada y respaldada por tanda:

- `20261003T024712Z-V7-gemv-metric-regression`: regresión escalar 72/72.
- `20261003T024712Z-V7-gemv-scalar-occupancy`: 162 jobs Brandon y ocupación.
- `20261003T025308Z-V7-gemv-dual-regression`: 75/75; añade rechazo de layout.
- `20261003T025308Z-V7-gemv-dual-brandon`: 162 jobs con la caché inicial.
- `20261003T025620Z-V7-gemv-dual-synth`: identifica inferencia LUTRAM.
- `20261003T030230Z-V7-gemv-dual-bram-synth`: confirma la corrección BRAM.
- `20261003T030438Z-V7-gemv-dual-bram-brandon-r2`: 162/162, 358,169 s;
  mismos ciclos/salidas tras corregir la RAM. Seis avisos VRFC de una rama
  de inyección no ejecutada quedan registrados; no validan esa rama.
- `20261003T032531Z-V7-gemv-dual-order-boundary`: **87/87 PASS**, 128,130 s;
  patrón K32/K33 que distingue acumulación ordenada de subtotales,
  error/cancelación con ambos motores activos y reset coordinado de ambos.
  Ocho alteraciones de métricas y ocho de cobertura funcional rechazadas.

El banco ampliado comprueba error temprano del motor 1 y cancelación con
ambos motores activos, reset coordinado de ambos y una reducción sensible
al orden entre tiles. Sus intentos fallidos se conservan: dos errores de
elaboración del banco, inyección sin el solapamiento requerido y una
constante numérica mal representada. La calificación ampliada se registra
en [VIVADO_STATUS](board/VIVADO_STATUS.md).

La reproducción aislada `20261003T031738Z-V7-fp64-reference-probe-r2`
demuestra que esta instalación de XSIM interpreta `9007199254740992.0`
como `433ffffffffffffe`. El banco usa ahora
`$bitstoreal(64'h4340000000000000)` para fijar exactamente 2^53.
La secuencia 2^53 + 1 − 2^53 da cero. Para distinguir además la suma de
subtotales, el banco coloca 2^53 en K0 y la unidad/cancelación en K32/K33:
la referencia ordenada da cero y los subtotales de tiles dan uno. Comprueba
ambas referencias antes de aceptar el resultado RTL. El banco anterior
`20261003T031837Z-V7-gemv-dual-stress-r4` pasó 87 casos con K31/K32, pero
esa distribución no distinguía el agrupamiento por tiles; se conserva con
ese límite de cobertura; la tanda `032531` pasa la prueba corregida.
El primer probe devolvió código Tcl 0 pese al Fatal; se conserva como FAIL.

## Reproducción y siguiente integración

### Formas máximas y filas impares

Las tandas `20261003T033430Z-V7-gemv-scalar-max` (277,028 s) y
`20261003T033440Z-V7-gemv-dual-max` (268,583 s) pasaron cuatro formas por
semilla 1/7/42: **12/12 cada variante**, 288 salidas cada una con diferencia
cero frente a una referencia Python FP64 independiente. Son pesos sintéticos
y dimensiones útiles para Llama; no pesos Meta ni una proyección completa
con todas sus filas.

| Forma del job | Ciclos escalar | Ciclos dual |
| --- | ---: | ---: |
| 32×2048 | 48622–48771 | 25076–25220 |
| 32×8192 | 194757–194887 | 100548–100713 |
| 1×8192 | 6462–6497 | 6462–6497 |
| 31×2048 | 47187–47290 | 25106–25213 |

La forma máxima acelera aproximadamente 1,94×. Una sola fila conserva la
dependencia de acumulación y no usa el segundo motor: no se espera mejora.
Los 31 renglones ejercitan el reparto desigual entre motores. Por semilla
se comprueban 12.480 bloques, 136 ráfagas y 424.320 bytes útiles; los
contadores de fases/ocupación también se reconciliaron. Los revisores
rechazan 15 alteraciones de logs del escalar y 16 del dual.

El [empaquetador de host](GEMV_WEIGHT_LAYOUT.md) materializa el layout 2 y
descriptores, con rangos y hashes; su banco adicional consume los bytes
inmutables desde una imagen, sin reconstruirlos durante las lecturas AXI.
Usar `- dual max` como argumentos finales del Tcl para repetir el banco
sintético de formas máximas.

```bash
vivado -mode batch -source tools/ntrpu_board_gemv_stream_sim.tcl \
  -tclargs RUTA_HLS_MAC NUEVA_SALIDA - dual
vivado -mode batch -source tools/ntrpu_board_gemv_stream_sim.tcl \
  -tclargs RUTA_HLS_MAC OTRA_SALIDA RUTA_FIXTURE dual
python tools/check_ntrpu_gemv_stream.py RUTA_LOG --seed 1 --lanes 2
python tools/check_ntrpu_gemv_metrics.py RUTA_LOG --lanes 2
vivado -mode batch -source tools/ntrpu_board_gemv_stream_synth.tcl \
  -tclargs RUTA_HLS_MAC NUEVA_SINTESIS dual
```

Conservar el formato y orden numérico al añadir carga DMA de activaciones,
registro de buffers, escritura de resultados y secuenciador FFN. Incorporar
el reloj real y las interfaces registradas antes de calificar timing de
sistema. DDR/PCIe y la placa siguen las dependencias V2–V9; no están
validados por este banco.
