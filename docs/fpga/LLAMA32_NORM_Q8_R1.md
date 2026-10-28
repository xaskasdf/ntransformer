# RMSNorm conectada a activaciones Q8 ABI1 — R1

Actualización 2026-10-15: **C/RTL del núcleo aprobado y respaldado** en la
tanda `20261013T155333Z-V7-llama32-norm-q8-hls-r1`. Revisión independiente
de 61 transacciones y 22.880 palabras de salida; diez alteraciones de traza
rechazadas. Estimaciones HLS: 27 BRAM de 18 Kb, 11 DSP, 3.678 FF, 9.210 LUT
y periodo de 3,647 ns; no son timing ruteado. El archivo verificado tiene
SHA-256 `411d90a513f6d68acb7735e617cd0451a5fc531fd38529b5fe585ae1b1433be4`.
El [propietario de RAM](LLAMA32_INPUT_NORM_OWNER_R1.md) y el recorrido AXI ya
pasaron protocolo nativo. La [integración con el HLS original](LLAMA32_INPUT_NORM_REAL_R1.md)
está encolada en Vivado; su ejecución y la conexión viva con Q/K/V siguen pendientes.
Los apartados siguientes conservan el alcance y preparación original.

Preparada el 2026-10-12 UTC. `fpga/hls/norm_q8_abi1_r1/` conecta en
un top HLS la RMSNorm FP32 de 2048 elementos y el cuantizador existente,
sin cambiar sus fuentes. El resultado es el formato físico que consume
`ntrpu_activation_loader`, previo al GEMV. La transferencia AXI entre
ambos bloques y el secuenciador de capas todavía no están integrados.

## Operación y buffers

Entradas: dos RAM de 2048 palabras de 32 bits, longitud y epsilon FP32.
La longitud debe ser múltiplo de 32, entre 32 y 2048. RMSNorm conserva
su reducción FP64 ordenada, epsilon positivo y redondeo final a FP32.
Cada bloque de 32 resultados pasa al cuantizador original:
`escala=max(abs(x))/127` en FP64, división FP64 y redondeo de INT8 con
empates alejándose de cero, limitado a −127…127.

La salida contiene hasta 64 registros de 64 bytes, expresados como
512 palabras de 64 bits. Por registro: INT8[32], escala binary64
little-endian y 24 bytes cero. El top solicita 8 KiB para el vector
normalizado y 4 KiB para registros privados, además de los temporales
internos de las funciones llamadas. Recursos reales pendientes de síntesis.

Estados: `0` éxito; `1` forma; `2` epsilon; `3` NaN/infinito activo;
`4` desbordamiento de RMSNorm; `5` fallo interno del cuantizador.
La última guarda no es alcanzable con los contratos válidos actuales.
Los resultados se publican únicamente después del éxito completo.
Propiedad de buffers, cancelación, épocas, reset y transporte externo
corresponden al adaptador aún pendiente.

## Pruebas y alcance

El banco nativo MSVC pasa **61 transacciones**, 49 correctas y **2.860
registros ABI1**: 183.040 bytes exactos, incluidos escalas y reservados.
La referencia independiente reproduce los redondeos FP64 por operaciones
racionales y contrasta la raíz con Decimal de 90 dígitos.

Incluye los 33 vectores de norma reales con activaciones sintéticas;
cuatro filas reales de embeddings con la norma de atención de capa 0;
subnormales, extremos, vecinos de empates y errores tardíos. No son
logits ni inferencia completa. Diez corrupciones del transcript y un
resultado serializado alterado son rechazados. Las guardas y salidas
intactas ante errores se comprobaron en CPU; los puertos HLS inactivos
no bastarán para certificar toda la RAM en RTL.

```powershell
cmake --build build/ntrpu-norm-q8-r1 --config Release
ctest --test-dir build/ntrpu-norm-q8-r1 -C Release --output-on-failure
```

## Tanda HLS

`queue_llama32_norm_q8_hls_r1.py` espera la suma residual aprobada y
respaldada. Ejecutará C, síntesis y cosimulación Verilog en VU5P a 5 ns,
con dos CPU y límite de 2400 s. Exige 61 estados, 22.880 palabras reales,
sus ciclos, revisión independiente y respaldo local/NFS. Los revisores
ya rechazaron diez alteraciones sobre trazas construidas; no constituyen
todavía evidencia RTL del núcleo integrado.
