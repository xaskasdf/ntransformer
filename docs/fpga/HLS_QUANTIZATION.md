# Cuantización de activaciones Q8_0

> Estado actualizado (2026-10-01): licencia instalada; INT8 pasa co-simulación C/RTL. Q8 y cuantizador siguen en validación; el cuantizador no cumple 5 ns. Véase [evidencia HLS](validation/hls-20261001.md). Las menciones anteriores a licencia pendiente describen el bloqueo histórico.

`fpga/hls/quantize_q8_0/ntrpu_quantize_q8_0.cpp` es un tercer top candidato a síntesis. Recibe activaciones FP32 y produce INT8 y escalas por bloque compatibles con el [GEMV Q8_0](HLS_Q8_0.md). Las dos fuentes se ejecutan consecutivamente en el simulador para entradas de hasta 1024 columnas; todavía no constituyen un sistema RTL conectado.

## Contrato

Cada bloque de 32 valores usa `dX = max(abs(x)) / 127` en doble precisión. Se divide cada activación por `dX`, se redondea al entero más próximo con empates alejándose de cero y se limita a −127…127. Un bloque cero produce enteros y escala cero. La parte entera y la fracción se comparan explícitamente: sumar 0,5 antes de convertir puede alterar el resultado cerca de un empate.

Se permiten 32–1024 columnas, múltiplo de 32. `0` indica éxito, `1` dimensiones inválidas y `2` entrada NaN/Inf. Un error deja intactas ambas salidas; una forma inválida no accede a memoria. Los buffers deben ser válidos y no solaparse. Solo se leen y escriben los elementos solicitados.

Hay tres masters AXI y control AXI-Lite. El top almacena resultados temporalmente y publica después de validar las entradas. La doble precisión es una referencia inicial costosa; el número de operadores, II, recursos y latencia se medirán antes de optimizar. No compilar con fast-math.

## Evidencia disponible

El 2026-10-01 pasaron **53/53 pruebas CPU en Windows/MSVC y en la VM Linux/GCC 13.3, Release**, incluyendo cuatro casos nuevos: tamaños y llamadas repetidas, todos los empates ±0,5…±126,5 con sus vecinos FP32, patrones aleatorios de bits finitos, subnormales, máximo FP32, NaN/Inf tardíos y salidas intactas. Se compara contra `std::round` y la cota de error de media escala. Los bancos RTL por defecto siempre suministran buffers completos, incluso con dimensiones inválidas; las comprobaciones de punteros nulos se ejecutan solo en la invocaci?n CPU del caso `invalid`.

Con las fuentes nuevas de cuantización y escalado, las 12 FFN únicas de Brandon y ocho entradas por FFN conservaron error absoluto cero frente a Q8 NumPy. El error L2 frente a FP32 continúa entre 0,8716 % y 1,5550 %. [Informe con hashes](validation/brandon-q8-quantize-hls-cpu-20261001.json). Las entradas son sintéticas, no activaciones obtenidas de prompts.

**No hay validación RTL.** La licencia Vivado Basic sigue pendiente. En co-simulación se deben comprobar especialmente subnormales y conversiones FP32/FP64; la coincidencia CPU no demuestra el comportamiento de los operadores flotantes generados.

## Reproducir

```bash
ctest --test-dir build/ntrpu-sim -R ntrpu_hls_quantize --output-on-failure
source ~/vivado/AMDDesignTools/2026.1/Vitis/settings64.sh
python3 scripts/run_ntrpu_hls.py --kernel quantize_q8_0 \
  --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage csim
# Después de csim correcto, repetir con --stage cosim.
```

En Windows agregar `-C Release` a CTest. `--dry-run` prepara fuentes y manifiesto sin invocar AMD. La selección de esta pieza sigue siendo una referencia del informe recuperado, no el dispositivo final de 512 GB. Después faltan validar y conectar estos bloques, llevar SwiGLU al hardware y completar memoria/PCIe y el backend de inferencia.
