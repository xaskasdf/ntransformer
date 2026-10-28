# RMSNorm para el circuito Llama 3.2 1B

Estado inicial: referencia nativa probada; Vitis HLS pendiente en la cola.
Este bloque añade la normalización que precede a atención y FFN y la norma
final. No convierte la FFN aislada en un transformer completo.

## Interfaz y contrato numérico

`fpga/hls/rmsnorm_fp32_r1/ntrpu_rmsnorm_fp32_r1.cpp` admite 1–2048 elementos
con entrada, peso y salida codificados como palabras FP32 de 32 bits.
Tres puertos `ap_memory` y `ap_ctrl_hs` permiten conectarlo después a una
memoria local. No hay aún wrapper AXI, épocas, cancelación ni DDR.

Calcula `y[i] = (x[i] / sqrt(sum(x²)/N + epsilon)) * weight[i]`, con suma
FP64 en orden ascendente, inversa compartida y redondeo final FP32.
Reutiliza las conversiones IEEE enteras ya probadas para conservar
subnormales y ceros con signo. No es el orden de reducción del núcleo CUDA
ni se declara equivalencia bit a bit con BF16. La constante real del GGUF
es `epsilon_bits=0x3727c5ac` (aproximadamente 1e-5).

Retornos: 0 correcto; 1 forma inválida; 2 epsilon no positivo/no finito;
3 entrada o peso no finito; 4 salida fuera del rango FP32 finito.
Valida y calcula en buffers privados antes de escribir la salida completa.
El integrador debe aportar buffers válidos e independientes y coordinar
reset/publicación; el núcleo no valida direcciones externas.

Los arreglos privados solicitan BRAM; recursos, latencia y período deben
medirse en síntesis/C-RTL. Se conservan las operaciones ordenadas sin
optimización matemática asociativa, siguiendo las opciones de
[compilación AMD](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Compile).
Las [interfaces de memoria](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Memory-Property-on-Interface)
son locales; no equivalen por sí solas a un maestro AXI.

## Pesos y validación

El fixture contiene los **33 vectores FP32 originales**: attn/ffn norm
de las 16 capas y output norm. Revisión del GGUF:
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`. Las activaciones son sintéticas.
SHA-256 de `llama32_norm_r1.bin`:
`66573f15c4070efaf91d5baf1c3c64fb58441a9d84ad4790e115c8d8a9ee3e60`.

MSVC Release pasó **60 casos, 44 correctos y 78.208 valores**; 67.584 usan
pesos reales. Todos coinciden con Python a cero ULP. Las referencias reales
también coinciden con Decimal de 80 dígitos tras redondear a FP32.
Se prueban dimensiones impares, ceros, subnormales, extremos finitos,
errores tardíos, precedencia de validación, recuperación y salida intacta
en error. Un revisor externo reconstruye la entrada y rechaza siete
alteraciones del transcript.

El criterio fijado antes de HLS es como máximo 1 ULP FP32 respecto a la
referencia ordenada, con ceros y signos exactos. La comparación C/RTL debe
ser exacta contra C compilado con `hls::sqrt`; AMD documenta diferencias
posibles entre su biblioteca y libm, y la alineación de C/RTL al usar
[la misma biblioteca matemática](https://docs.amd.com/r/en-US/ug1399-vitis-hls/HLS-Math-Library-Accuracy).
No se amplía la tolerancia si aparece un fallo.

## Reproducción y cola

```powershell
cmake -S fpga/hls/rmsnorm_fp32_r1 -B build/ntrpu-rmsnorm-r1 `
  -DNTRPU_RMSNORM_FIXTURE=D:/ruta/llama32_norm_r1.bin
cmake --build build/ntrpu-rmsnorm-r1 --config Release --parallel 2
ctest --test-dir build/ntrpu-rmsnorm-r1 -C Release --output-on-failure
```

La tanda HLS exige el replay de máscaras de la cola recuperada antes de
ocupar Vivado/Vitis. Usa VU5P, 5 ns solicitados, afinidad máxima de dos CPU
y límite de 2.400 segundos, con C simulation → síntesis → C/RTL.
Cada terminal se revisa y respalda, incluidos fallos. Un resultado HLS no
cierra timing físico; la integración de memoria y el bloque de atención
siguen pendientes.
