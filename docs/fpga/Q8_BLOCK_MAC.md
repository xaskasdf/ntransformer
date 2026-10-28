# Cómputo por bloque Q8 para V7

Estado al 2026-10-02: implementación, pruebas CPU y **317 operaciones C/RTL
correctas en VU5P**, tanto antes como después de optimizar la conversión FP16.
Es el núcleo aritmético
para conectar el [flujo de pesos](Q8_UNPACKER.md) con acumuladores por fila.
El [controlador GEMV](Q8_GEMV_STREAM.md) ya conecta este núcleo con el lector
AXI/Q8, activaciones y acumuladores privados; pasó 72 casos RTL. La memoria
es sintética y todavía no está conectado a DDR física.

## Interfaz y orden numérico

`fpga/hls/q8_block_mac/ntrpu_q8_block_mac.cpp` recibe un bloque desempaquetado:
cuatro palabras de 64 bits de pesos, cuatro de activaciones INT8, escala
FP16 de pesos, escala FP64 de activaciones y acumulador FP64 previo.
El byte menos significativo representa el primer lane de cada palabra.

Calcula los 32 productos INT8, reduce en INT32 y aplica
`acumulador + ((escala_pesos × escala_activaciones) × dot)`. Conserva el
orden de la referencia GEMV; sumar subtotales de tiles cambiaría el redondeo.
Las escalas de pesos aceptan ceros, subnormales y signos; NaN/Inf se rechazan.
La escala de activación debe estar entre cero y `FLT_MAX / 127` y el
acumulador debe ser finito. Los errores no escriben el resultado C.

El contrato HLS usa `ap_ctrl_hs`, entradas escalares `ap_none` y salida
`ap_vld`. El controlador debe mantener las entradas estables desde inicio
hasta finalización y aceptar el resultado solo con estado `OK`. La forma
generada y la ausencia de pulso de resultado en errores requieren revisión RTL.

## Acumulación y publicación en la integración

El núcleo no conserva estado entre llamadas. El adaptador GEMV guarda el
acumulador de cada fila y lo realimenta **sin reiniciar en límites K**.
Por ejemplo, K=8192 comprende 256 bloques, aunque se transporten en ocho
tiles de K=1024. Cada bloque debe coincidir con la activación de su columna.

Los resultados intermedios son privados: solo se publica tras validar todos
los bloques y las confirmaciones `load_done` y `decode_done`, con etiquetas,
conteos, `sequence` y `model_epoch` correctos. Un error tardío descarta toda
la operación; reset/cancelación deben coordinar lector, buffers y MAC.

## Validación reproducible

```powershell
cmake --build build/ntrpu-sim --config Release --target test_ntrpu_hls_block_mac
ctest --test-dir build/ntrpu-sim -C Release -R '^ntrpu_hls_block_mac_' -V
```

CPU: **317 operaciones** con referencia escalar independiente (`ldexp` e
INT64), posiciones de bytes, extremos INT8/FP16, K=8192 con continuidad,
una secuencia no asociativa, errores y recuperación. Otras **65.536**
operaciones verifican exhaustivamente los patrones FP16, incluidos rechazos.
Las comparaciones numéricas son exactas; las comprobaciones no dependen de
`assert` ni se desactivan en Release.

```bash
python3 scripts/run_ntrpu_hls.py --kernel q8_block_mac \
  --part xcvu5p-flva2104-2-i --clock-ns 5 --stage cosim
```

El banco C/RTL usa las 317 operaciones; la barrida FP16 exhaustiva queda
separada en CPU. No usa pesos reales ni mide inferencia Llama.

## Medición y optimización en VU5P

Vitis/XSIM 2026.1, `xcvu5p-flva2104-2-i`, reloj solicitado 5 ns:

| Medida | Conversión por multiplicaciones | Conversión exacta por bits |
| --- | ---: | ---: |
| Run | `20261002T194824Z-V7-q8-block-mac-cosim` | `20261002T195413Z-V7-q8-block-mac-bitcast` |
| C/RTL | 317 PASS | 317 PASS |
| Latencia por bloque válido | 193 ciclos | **21 ciclos** |
| Intervalo por bloque válido | 194 ciclos | **22 ciclos** |
| Banco completo | 59.399 ciclos | 6.767 ciclos |
| LUT estimadas HLS | 5.662 | 2.908 |
| FF estimados HLS | 4.100 | 2.024 |
| DSP estimados HLS | 27 | 27 |
| BRAM/URAM estimadas | 0/0 | 0/0 |
| Periodo estimado HLS | 3,913 ns | 3,419 ns |
| Duración real de herramientas | 161,230 s | 129,466 s |

Hay 306 operaciones válidas, ocho errores de escala a 2 ciclos y tres errores
de acumulador a 3 ciclos. Ambas versiones conservan el orden numérico, la
continuidad K y la recuperación. La versión actual construye exponente y
mantisa FP64 con enteros y usa `memcpy` para reinterpretar bits; evita la
cadena de multiplicaciones por dos sin cambiar los valores FP16 representados.

Los 3,419 ns estimados quedan por debajo del presupuesto HLS efectivo de
3,650 ns (5 − 1,35). No son timing posrutado. Se retienen tres avisos de flujo
Tcl obsoleto y uno del entorno del compilador XSIM, sin errores ni avisos de
conversión. Las fuentes y resultados de ambas versiones quedan archivados;
Los ciclos del adaptador, las proyecciones reales de Brandon y su ruteo OOC
se documentan en [Q8_GEMV_STREAM](Q8_GEMV_STREAM.md). El contexto aislado
mantiene el reloj del padre parcial y los puertos sin calificar; falta la
implementación integrada y su aprobación temporal.
