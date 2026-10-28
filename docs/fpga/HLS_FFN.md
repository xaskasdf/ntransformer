# Controlador local de FFN Q8_0 y SwiGLU

Estado al 2026-10-02: **SwiGLU pasó C simulation, síntesis y 14 transacciones C/RTL**. **La FFN completa pasó 23 transacciones C/RTL en ZU7EV**, incluida una muestra real de Brandon. Para el VU5P provisional de la placa ya terminaron C simulation, síntesis HLS y síntesis independiente de su RTL en Vivado; la validación RTL específica de VU5P se registra por separado abajo.

## Interfaz y secuencia

`fpga/hls/ffn_q8_0/ntrpu_ffn_q8_0.cpp` acepta matrices gate/up Q8_0 de `hidden × dim`, down de `dim × padded`, entrada FP32 y salida FP32. Admite `dim=32..256` en múltiplos de 32, `hidden=2..720` y `padded=ceil(hidden/32)*32`. Son límites del prototipo Brandon.

1. Cuantizar la entrada una vez y reutilizar INT8/escalas para todos los tiles de gate y up.
2. Convertir cada resultado FP64 a FP32, comprobando desbordamiento.
3. Aplicar `SiLU(gate) * up` con intermediarios FP64 y salida FP32.
4. Rellenar explícitamente el padding con ceros y cuantizar ese vector una vez.
5. Calcular down por tiles y conservar toda la salida en scratch hasta terminar sin errores.

Para Brandon, `256 → 720 → 256`, son 23 tiles gate, 23 up, ocho down y dos cuantizaciones. Los pesos ocupan 591,872 bytes por FFN en este formato, incluido padding. El caso completo tardó 6,698,058 ciclos en co-simulación RTL; falta medir el tráfico AXI útil y el rendimiento físico.

Retornos: `0` correcto; `1` forma inválida; `2` entrada no finita; `3` escala de peso inválida; `4` desbordamiento numérico. En error, la salida completa permanece intacta. El llamador debe proporcionar buffers válidos, no superpuestos y del tamaño documentado: este top no valida direcciones DMA ni implementa descriptores PCIe, épocas o un controlador DDR.

## Contrato numérico

SwiGLU usa la forma estable `exp(-abs(g))`; a magnitudes mayores de 256 omite la exponencial porque la rama negativa redondea a cero FP32 incluso con `up=FLT_MAX`, y la positiva redondea su sigmoid a uno. Se rechazan NaN, infinitos y resultados fuera del rango finito FP32. Las conversiones por campos IEEE conservan subnormales y aplican redondeo al par.

La compilación local usa `std::exp`. Los Tcl definen `NTRPU_USE_HLS_MATH` en simulación C y síntesis para usar `hls::exp`. AMD documenta que su biblioteca permite alinear los resultados matemáticos de C simulation y C/RTL; por eso el resultado local no sustituye esa validación. No se relajan los límites de error para ocultar diferencias de implementación. [AMD UG1399, precisión de la biblioteca matemática](https://docs.amd.com/r/en-US/ug1399-vitis-hls/HLS-Math-Library-Accuracy).

La auditoría CPU comprueba los 16,777,216 patrones de subnormales y ceros con signo, y 600,000 conversiones alrededor de puntos medios entre FP32, además de extremos. Los bancos usan comprobaciones activas en Release.

Resultado: **63/63 pruebas CTest en Windows/MSVC y en la nueva VM Ubuntu 22.04/GCC 11.4**. Se explicitó `std::vector<UnitConfig>` al construir `Fabric` para evitar una ambigüedad de GCC 11. En la tanda local anterior, WSL/GCC 13.3 pasó los bancos nuevos con AddressSanitizer y UndefinedBehaviorSanitizer: tamaños exactos de buffers, casos numéricos, errores tardíos, conversiones y ocho muestras de una FFN real.

La compilación directa contra `hls_math.h` detectó una colisión con el namespace global `detail` del SDK; las llamadas a las conversiones ahora están calificadas completamente. Con `hls::exp` y las bibliotecas de simulación AMD, los bancos SwiGLU/FFN y las 96 muestras de Brandon pasaron en CPU. Estas últimas mantienen diferencia cero frente a Q8 NumPy. Tras actualizar la licencia también pasaron los bancos de Vitis C simulation; la tanda RTL de FFN usa solo una de las muestras reales, además de casos sintéticos.

## Resultados AMD y límites actuales

Referencia: `xczu7ev-ffvc1156-2-e`, reloj solicitado de 5 ns e incertidumbre HLS de 1.35 ns. Los recursos y períodos siguientes son **estimaciones de síntesis HLS**, no resultados posrutado.

| Top | Período estimado | BRAM18K | DSP | FF | LUT | C/RTL |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SwiGLU | 3.650 ns | 6 | 40 | 6,823 | 10,716 | PASS, 14 transacciones |
| FFN Q8_0 | 4.172 ns | 40 | 92 | 20,616 | 42,279 | PASS, 23 transacciones |

SwiGLU: ejecución `20261002T052530Z-cosim-7cuv199c`, latencias de 3 a 133,161 ciclos; el máximo corresponde a 1,024 elementos. Se comprobaron extremos, subnormales representativos, formas inválidas y conservación de la salida en error. La auditoría exhaustiva de conversiones sigue siendo una prueba CPU separada. Archivo completo verificado por SHA-256 en `artifacts/ntrpu-rtl-20261002/` y en el volumen persistente.

FFN: ejecución `20261002T062547Z-cosim-sc8kn0wf`, modo `full`, **PASS en las 23 transacciones**, con latencias de 4 a 6,698,058 ciclos. La muestra real mantuvo diferencia máxima cero frente a Q8, error relativo L2 de 1.14920248439% y error absoluto máximo de 0.0240252017975 frente a FP32. Es una FFN aislada, no inferencia completa. Archivo de 95,387,074 bytes verificado en el volumen persistente y en `artifacts/ntrpu-rtl-20261002/`, SHA-256 `a701147da9016fb8706d5851ae6e328dd4ed477e8370c54980ad0d0287aaa9a3`. `VERIFIED.json` registra fuentes, fixture, latencias y comparación RTL.

La primera tanda (`20261002T052530Z-cosim-kn_kk9br`) se detuvo manualmente para acotar costes; su salida de herramienta es `failed` por interrupción, sin veredicto numérico. Su diagnóstico mostró avance de los contadores y paso de gate a up en el primer tile. El banco completo se relanzó con las directivas corregidas y terminó sin interrupción. Ambos intentos mantienen fuentes e informes separados.

Vitis advierte que algunas rutas exceden el presupuesto efectivo de 3.650 ns. Se corrigió el orden de argumentos de las directivas `ALLOCATION`, colocando `function` inmediatamente después de la directiva, como requiere [AMD UG1399](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Allocation). La nueva síntesis ya no reporta ese error, pero conserva dos instancias especializadas de GEMV y dos de cuantización, con los mismos recursos estimados. Reutilizar lógicamente las cuantizaciones no demuestra una sola instancia física. Los límites de latencia de síntesis con bucles variables no son benchmarks.

El banco completo tarda considerablemente más que las pruebas de bloques aislados. `--test-mode smoke` permite una tanda reducida de 21 transacciones pequeñas y de errores, sin el caso máximo ni pesos reales; queda identificado en el manifiesto y rechaza `--fixture`. No sustituye el modo `full`, predeterminado y usado en esta ejecución aprobada.

## Pruebas locales

```powershell
cmake -S . -B build/ntrpu-sim -DNTRANSFORMER_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build/ntrpu-sim --config Release --parallel 8
ctest --test-dir build/ntrpu-sim -C Release --output-on-failure
```

`test_ntrpu_hls_ffn` acepta una fixture `NTRFFN01` existente. El generador auditado también puede ejecutar el nuevo top sobre las 12 FFN y registrar ambas referencias:

```powershell
build/ntrpu-reference-venv/Scripts/python.exe tools/run_ntrpu_brandon.py --model-dir build/models/brandon-tiny-ca6329d --exe build/ntrpu-sim/src/ntrpu/Release/test_ntrpu_hls_ffn.exe --output-dir build/ntrpu-brandon/new-ffn-run --device numpy --samples 8
```

Elegir un directorio nuevo; el generador no sobrescribe resultados. Esta tanda obtuvo diferencia máxima cero frente a la referencia Q8 NumPy y error relativo L2 de 0.871624% a 1.554962% frente a FP32. Esto caracteriza FFN aisladas con entradas sintéticas, no la calidad de inferencia del modelo.

## Repetir la ejecución AMD

Primero validar SwiGLU por separado. Usar una fixture de una muestra y una capa (`--samples 1 --layer 0` en el generador anterior) para acotar el primer trabajo de FFN; el banco añade casos sintéticos, formas inválidas y errores tardíos.

```bash
python3 scripts/run_ntrpu_hls.py --kernel swiglu --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage cosim
python3 scripts/run_ntrpu_hls.py --kernel ffn_q8_0 --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage cosim --fixture /path/to/ffn_00.bin
```

Añadir `--dry-run` permite preparar fuentes y hashes sin herramientas AMD. El runner incluye las tres dependencias y distingue la fixture FFN de los tiles GEMV. La referencia ZU7EV no es la selección de hardware para 512 GB. Quedan pendientes la corrección y medición de la reutilización física, la FFN integrada con AXI bajo backpressure y timing posrutado.

## Preparación V7 sobre VU5P

`20261002T182015Z-V7-ffn-vu5p-synth`: C simulation completa sintética y
síntesis HLS correctas en `xcvu5p-flva2104-2-i`, 135,011 s reales.
No incluyó fixture de Brandon ni co-simulación. Estimación: 38.726 LUT,
18.589 FF, 86 DSP, 22 BRAM18K; periodo 3,913 ns frente a 5 ns solicitados
y 1,35 ns de incertidumbre. Supera en 0,263 ns el presupuesto efectivo HLS.

El aviso **HLS 214-300 persiste en esta tanda**: se ignora `ALLOCATION`
porque cuantización está clonada (214-366). La corrección de sintaxis anterior
no demuestra que el compilador aplique reutilización física. Se conservan
también avisos de límites de bucles, renombrado, flujo obsoleto y puertos fijos.
Los tres AXI maestros usan datos de **32 bits** y direcciones de 64 bits;
no están conectados al lector de 512 bits ni al flujo Q8 desempaquetado.

`20261002T182535Z-V7-ffn-vu5p-ooc-synth`: síntesis Vivado independiente,
306,833 s, sin cajas negras. Recursos mapeados: **27.918 LUT, 22.961 FF,
55 DSP, 6 RAMB36 + 6 RAMB18**, cero URAM. Son recursos del bloque aislado;
no sumar directamente estimaciones HLS para dimensionar réplicas.

WNS +1,747 ns/WHS +0,009 ns son anteriores a colocación. `HD.CLK_SRC` no
está fijado; faltan retardos en 127 entradas y 280 salidas. Se retienen 427
TIMING-18, 239 avisos de pipeline DSP y 54 recomendaciones AVAL; DRC OOC
no cubre toda la conectividad. No acredita cierre físico a 200 MHz.

`20261002T193955Z-V7-ffn-vu5p-cosim-smoke` pasó **21/21 transacciones**
cortas de valores/errores en 535,836 s reales, con cuatro CPU. Total:
564.333 ciclos; latencias individuales de 3 a 123.474 ciclos según caso.
Conserva los avisos HLS anteriores y uno del entorno de XSIM. No reemplaza
el banco completo, la muestra real ni la integración con memoria. Reproducir:

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_cosim.tcl \
  -tclargs /ruta/Vitis/bin/vitis-run NUEVA_CARPETA smoke
```

Las tandas usan fuentes identificadas y carpetas independientes de la cola
de placa; sus revisiones y respaldos figuran en [VIVADO_STATUS](board/VIVADO_STATUS.md).
