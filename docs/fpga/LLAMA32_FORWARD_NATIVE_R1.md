# Recorrido numérico completo de Llama 3.2 1B

Fecha: 2026-10-16. Estado: **recorrido completo aprobado con matemáticas
estándar y AMD, con revisión y restauración verificadas**. Esta prueba no
utiliza el slot Vivado.

## Alcance y contrato

Un token de entrada, ID 0 en posición 0, atraviesa embedding, las 16 capas,
normalización final y los 128.256 logits. Usa el GGUF Q8_0 fijado a revisión
`b69aef112e9f895e6f98d7ae0949f72ff09aa401`, SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
Los 147 tensores se contrastan con la auditoría original. La salida comparte
pesos con `token_embd.weight`; no existe un tensor `output.weight` separado.

Se aplica el contrato NTRPU: activaciones Q8/ABI1, acumulación FP64 ordenada,
redondeo a FP32 y residual exacto. La referencia Python/NumPy y el banco C++
leen los mismos pesos originales. El banco ejecuta once unidades fuente HLS;
los resultados esperados sirven únicamente para comparar. Cada capa consume
la salida nativa anterior. En posición cero, RoPE usa coseno 1/seno 0 y la
atención tiene una sola clave: no ejercita atención multi-token ni KV acumulada.

## Evidencia disponible

- Generador: `tools/prepare_ntrpu_llama32_forward_reference_r1.py`.
  Tanda `20261016T110932Z-llama32-forward-reference-r1`.
  Manifiesto `b64a48681d404dfd9255bd527bbf3c0c6000f3e15c92f6220970d9a40c18be5a`.
- C++ estándar: `20261016T112027Z-llama32-forward-native-r2`, aprobado:
  372 comparaciones y 6.700.032 bytes exactos; 38.617.088 bloques MAC.
  Salida de capa 0 idéntica al vector RTL aprobado; argmax del logit: token 475.
- AMD: `20261016T112359Z-llama32-forward-native-r4` alcanzó el límite de
  120 segundos tras completar las capas 0–2, sin divergencias registradas.
  Es un diagnóstico parcial. La tanda `20261016T112851Z-llama32-forward-native-r5`
  completó desde ese archivo nativo las capas 3–15 y todos los logits, una
  capa por invocación limitada a 120 segundos, CPU 2–3, con respaldo y
  restauración antes de avanzar. Cada capa tardó unos 40 segundos en CPU;
  ese tiempo no representa latencia FPGA.
- Revisión: `llama32-forward-native-r1-review.json` en
  `build/cloud-setup/20261002-board/`, estado `PASS_LLAMA32_FULL_FORWARD_NATIVE_R1`.
  Ambas variantes cumplen las mismas 372 comparaciones, 6.700.032 bytes y
  cobertura completa de las 16 capas/16 segmentos. Once alteraciones de
  logs fueron rechazadas. El control nativo negativo detectó el byte
  513.023 del último logit y terminó con código 1, como correspondía.

SHA-256 de los 513.024 bytes FP32 finales:
`6d64b6ac5ae3696db987551eb5a59206c87319ac60cade7bc4372e26f071939e`.
La revisión verifica explícitamente que cada continuación lee el archivo
producido por la ejecución anterior, además de comprobar sus hashes.

Las fuentes HLS AMD se compilan en C++17 y el banco en C++20, con
`-O2 -fno-fast-math -ffp-contract=off`. Se vinculan las bibliotecas matemáticas
de Vitis 2026.1; se registran 69 dependencias de cabecera y diez bibliotecas
de ejecución. Los diagnósticos anteriores de nombre ambiguo, cabeceras C++20
y enlace quedan conservados; no se modificaron kernels ni tolerancias.

## Consecuencias para el diseño FPGA

`ntrpu_matrix_sequencer` admite como máximo 8.192 filas por trabajo. La cabeza
de vocabulario necesita **16 trabajos: quince de 8.192 filas y uno de 5.376**.
Son 64 bloques Q8 por fila, 2.176 bytes por fila y 279.085.056 bytes de pesos.
La salida completa ocupa 513.024 bytes FP32 o 1.026.048 bytes FP64.

Los offsets del manifiesto describen filas originales del GGUF. Antes de DMA
hay que empaquetar cada segmento al layout compacto del secuenciador mediante
`tools/ntrpu_matrix_layout.py`; el GGUF crudo no tiene ese orden. Para salida
FP32 debe habilitarse `FP32_CAPABLE` y solicitar `job_output_fp32`; el valor
por defecto es FP64. Mantener pesos/activación inmutables y los 16 destinos
privados hasta completar y drenar todas las respuestas. Un fallo o cancelación
debe invalidar la publicación conjunta, sin presentar logits parciales.

Pendientes: controlador de las 16 capas y de la cabeza completa, pruebas RTL
de continuidad/cancelación, contexto mult-token, tokenizer, muestreo y memoria
DDR compartida. Este resultado nativo no certifica equivalencia BF16/CUDA,
Vitis csim/cosim, latencia RTL, consumo ni rendimiento físico.

## Respaldo

Los cinco intentos terminados tienen `results.zip`, `restored/` y
`VERIFIED.json` en `artifacts/ntrpu-board-vivado/<tanda>/`, respaldados en D:
y NFS. Incluyen la referencia completa y fuentes fijadas. El modelo permanece
externo, con copias de igual SHA-256 en D: y NFS. La continuación conserva un
ZIP y recibo por capa. Su archivo completo contiene 558 archivos restaurados
en D:/NFS, SHA-256
`5a35b2f9d7f44353a179f2c817ef7132598b5e0b1c843e1671ff18a1ad5389a8`.
