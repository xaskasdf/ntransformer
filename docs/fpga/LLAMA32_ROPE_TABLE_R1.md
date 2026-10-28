# RoPE Llama GGUF: núcleo con tabla de fases

El núcleo `ntrpu_rope_table_fp32_r1` pasó CPU y Vitis C/síntesis/C-RTL.
La revisión archivada `PASS_ROPE_TABLE_R1_C_RTL` corresponde a la tanda
`20261013T154001Z-V7-llama32-rope-hls-r1`; se verificó nuevamente contra
su respaldo el 15 de octubre. La integración con memoria AXI sigue pendiente.

## Contrato del modelo y del circuito

El GGUF fijado contiene `rope_freqs.weight`: 32 factores FP32, SHA-256
`d25a6d176d9313062b8f86440c88649f9e568d573ea66bfa4e85e2424cf35633`.
La fase del par `j` es `position * 500000^(-j/32) / factor[j]`.
No basta con usar theta=500000 y omitir esos factores.

El circuito consume cabezas consecutivas de 64 elementos y rota parejas
adyacentes `(2j,2j+1)`. Es el orden de los Q/K almacenados por el
[conversor Llama de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/conversion/llama.py);
su [operador RoPE](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/ggml/src/ggml-cpu/ops.cpp)
aplica los factores como divisores. Se conserva una copia identificada de
estas fuentes y su licencia para revisar la interpretación del formato.

Admite 1–40 cabezas: las 32 Q y ocho K caben juntas, con Q antes de K.
La tabla contiene 32 parejas FP32 `(cos,sin)`, compartidas por las cabezas.
La entrada y salida son palabras FP32; hay tres puertos `ap_memory` y
control `ap_ctrl_hs`, sin wrapper AXI ni acceso directo a DDR.

Se elevan entradas y coeficientes a FP64, se ejecutan cuatro productos y
una suma/resta por pareja, y se redondean las salidas a FP32. No se permite
FMA ni reasociación. Las conversiones IEEE conservan subnormales.
Este contrato no promete igualdad binaria con los operadores FP32 de
llama.cpp, CUDA o el checkpoint BF16; esa comparación requiere integración.

## Tabla, errores y publicación

El generador calcula cada fila de fases antes de la ejecución. Son
256 bytes por posición; una tabla de 131.072 posiciones ocuparía 32 MiB.
El fixture contiene posiciones seleccionadas, no una tabla completa.
El runtime deberá comprobar identidad/hash y cargar la fila correspondiente;
el núcleo solo comprueba el rango y la coincidencia de los dos campos de
posición. Un encabezado correcto no prueba que la fase pertenezca a ese token.

Retornos: 0 correcto, 1 forma inválida, 2 posición inválida/desfasada,
3 coeficientes inválidos, 4 entrada no finita, 5 desbordamiento FP32.
Los coeficientes deben ser finitos, estar en [-1,1] y satisfacer
`abs(c*c+s*s-1) <= 2^-22`; posición cero exige `(1,+0)` exactos.
La precedencia sigue ese orden. Se valida toda la entrada antes de calcular
y se retienen resultados privados hasta terminar sin errores.

La salida lógica se publica al recibir `ap_done` con retorno cero. El
integrador deberá resolver reset, épocas, exclusión de buffers y escrituras
AXI parciales; estas responsabilidades no quedan cubiertas por memoria local.
Se solicitan BRAM para 20 KiB de entrada FP64, 10 KiB de resultados y
512 bytes de fases. HLS informó 25 BRAM18K, 38 DSP, 3.928 FF y 7.227 LUT;
el reloj estimado de 3,611 ns no constituye timing después de ruteo.

## Validación y reproducción

MSVC pasó **48 casos, 28 operaciones correctas y 61.760 valores exactos**.
Incluye doce posiciones de 0 a 131.071, dimensiones de cabezas 1/3/8/32/40,
ceros con signo, subnormales, extremos, errores tardíos y recuperación.
Seis casos usan Q/K completos de capas 0/15, calculados en CPU a partir
de sus pesos Q8 originales y activaciones INT8 sintéticas. Esto no califica
esas proyecciones en RTL ni constituye estados de una inferencia completa.

Los coeficientes y rotaciones del fixture coinciden tras redondear con una
referencia Decimal de 90 dígitos; el revisor también usa aritmética racional.
Ocho corrupciones del transcript y una referencia binaria alterada fueron
rechazadas. La auditoría de puertos rechazó ocho alteraciones de trazas
construidas; estas últimas prueban el revisor, no el hardware.

```powershell
cmake -S fpga/hls/rope_table_fp32_r1 -B build/ntrpu-rope-r1 `
  -DNTRPU_ROPE_FIXTURE=D:/ruta/llama32_rope_r1.bin
cmake --build build/ntrpu-rope-r1 --config Release --parallel 2
ctest --test-dir build/ntrpu-rope-r1 -C Release --output-on-failure
```

La tanda Vitis conservó VU5P, 5 ns solicitados y dos CPU. Se revisaron
48 transacciones y 61.760 palabras en C/RTL, con PASS Verilog. El informe
registra latencia mínima/media/máxima de 1/14.730/27.511 ciclos, incluyendo
rechazos tempranos; no es una medida de tokens por segundo. Se rechazaron
ocho corrupciones numéricas y ocho de trazas. Los dumps binarios de dos
estados no prueban ausencia de X/Z; una traza de error sin escrituras
tampoco prueba la conservación de toda la RAM.

Archivo SHA-256:
`07290498a89bc0da4841bb5b31363b276a8e6bdc1687c06154b522dd46c5fb6b`.
La revisión está incorporada en el archivo y coincide con la copia local.
Faltan la propiedad de buffers integrada, el cargador de fases, la unión
QKV→RoPE→KV, AXI/DDR y timing físico del conjunto.
