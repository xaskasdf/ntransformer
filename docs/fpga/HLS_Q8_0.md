# Núcleo HLS Q8_0 con escalas por bloque

> Estado actualizado (2026-10-01): licencia instalada; INT8 pasa co-simulación C/RTL. Q8 y cuantizador siguen en validación; el cuantizador no cumple 5 ns. Véase [evidencia HLS](validation/hls-20261001.md). Las menciones anteriores a licencia pendiente describen el bloqueo histórico.

`fpga/hls/gemv_q8_0/ntrpu_gemv_q8_0.cpp` añade un top independiente para calcular productos INT8, decodificar escalas FP16 Q8_0 y acumular bloques escalados. Es fuente candidata a síntesis: **todavía no hay síntesis ni co-simulación RTL**. La primera ejecución del top INT8 con Vitis 2026.1 se detuvo por falta de licencia Vivado antes de compilar.

## Contrato y precisión

- Pesos: bytes Q8_0 contiguos, por filas; cada bloque contiene una escala FP16 little endian y 32 valores INT8.
- Entrada: activaciones ya cuantizadas y una escala `double` por bloque. La cuantización `maxabs / 127` y el redondeo tienen ahora un [top separado](HLS_QUANTIZATION.md), llamado como C++ desde el adaptador host.
- Tamaño: 1–32 filas y 32–1024 columnas, múltiplo de 32; las filas mayores se reparten en tiles.
- Cálculo: dot INT32 exacto por bloque, seguido de `double(dW) * dX * dot`, acumulado en orden de bloques. La salida es `double`; el simulador convierte después a FP32 como antes.
- Escalas de pesos: acepta todos los FP16 finitos, incluidos negativos y subnormales. Rechaza Inf/NaN. Escalas de activaciones: `0 <= dX <= FLT_MAX / 127`, sin NaN.
- Errores: `1` para dimensiones inválidas, `2` para escalas inválidas; no modifica ninguna salida al fallar. Dimensiones inválidas no acceden a memoria. El llamador garantiza buffers válidos y no solapados.

Se solicita AXI de 64 bits para direcciones, cuatro masters (`weights_mem`, `input_mem`, `scales_mem`, `output_mem`) y control AXI-Lite. La suma local se publica después de validar todas las filas. Esto no protege contra fallos físicos AXI ni sustituye permisos DMA.

La doble precisión conserva la referencia numérica existente y facilita aislar errores de cuantización. **No es una decisión final de microarquitectura**: medir DSP/LUT/latencia antes de elegir FP32, punto fijo o acumulación mixta. Tampoco hay doble buffer ni rendimiento DDR medido.

## Integración y comprobaciones

Para entradas de hasta 1024 columnas, `q8_0_gemv_hls_model()` llama a este top como C++ normal; matrices más anchas conservan el camino INT8 anterior con escalado host. Las FFN de Brandon usan anchos 256 y 736, por lo que ya recorren la nueva fuente. Esto verifica su integración funcional, no ejecución FPGA.

La suite CPU pasó **49/49 pruebas en Windows/MSVC y en la VM Linux/GCC 13.3, Release**. Incluye referencia independiente INT64/`ldexp`, tamaños máximos, bloques con escalas diferentes, cancelación, extremos, errores en la última fila, salidas intactas y una comprobación host de las 65.536 codificaciones FP16. La prueba exhaustiva no se incluye en el banco RTL por defecto.

Las 12 FFN únicas de Brandon, ocho entradas por FFN, conservaron error absoluto cero frente a la referencia Q8 NumPy y L2 relativo de 0,8716–1,5550 % frente a FP32. Ver [informe CPU](validation/brandon-q8-scaled-hls-cpu-20261001.json). Las entradas son sintéticas; no demuestra calidad de inferencia ni logits correctos.

## Ejecutar cuando esté instalada la licencia

```bash
source ~/vivado/AMDDesignTools/2026.1/Vitis/settings64.sh
cd ~/ntrpu/current
python3 scripts/run_ntrpu_hls.py --kernel gemv_q8_0 \
  --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage csim
# Solo después de csim correcto:
python3 scripts/run_ntrpu_hls.py --kernel gemv_q8_0 \
  --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage cosim
```

El lanzador conserva fuentes, hashes y resultados por ejecución. El top anterior sigue siendo el predeterminado (`--kernel gemv_int8`). Después de validar los tops y conectarlos, falta llevar SwiGLU al hardware y conectar memoria, colas y PCIe. La cuantización ya dispone de fuente candidata; su RTL sigue pendiente.
