# Suma residual FP32 para Llama — R1

Estado revisado el 2026-10-15: **C, síntesis HLS y cosimulación RTL
aprobadas y respaldadas** en la tanda del 2026-10-13. Núcleo local
`fpga/hls/residual_fp32_r1/`, con dos entradas y una salida de hasta
2048 palabras binary32. Suma el vector preservado y la rama de atención
o FFN; todavía falta conectarlo al transporte AXI y al secuenciador de capas.

## Contrato del hardware

`ntrpu_residual_fp32_r1` recibe bits IEEE, longitud y memorias independientes.
Opera mediante enteros, alineación y bits de guarda/redondeo/sticky;
no utiliza operaciones FP64. Solicita una RAM privada de 8 KiB para resultados.
La síntesis debe determinar recursos, latencia y viabilidad del II solicitado.

Usa redondeo al más cercano, empates al par y subnormales. La cancelación
exacta produce `+0`; `-0 + -0` conserva `-0`. El desbordamiento se decide
después del redondeo: un resultado exacto algo mayor que el máximo finito
puede redondear a ese máximo y seguir siendo válido.

Estados por prioridad: `1` longitud fuera de 1–2048; `2` NaN/infinito activo;
`3` resultado que redondea a infinito; `0` éxito. No lee la cola inactiva.
El código retiene los resultados hasta validar todo el vector; los errores
no escriben salida. La prueba nativa comprueba esta propiedad y guardas.
Las trazas HLS de puertos inactivos, por sí solas, no certifican conservación
de toda la RAM. Reset, propiedad de buffers y publicación externa siguen pendientes.

## Evidencia nativa

43 transacciones, 33 correctas y **51.556 palabras exactas**, contrastadas
con una referencia independiente de enteros arbitrarios en unidades de
`2^-149`. Incluyen empates, signos de cero, subnormales, extremos, longitudes
y prioridad de errores. Cuatro casos usan filas de embeddings del GGUF
Q8_0 fijado, con ramas sintéticas; no son activaciones de una capa completa.

El ensayo adicional compara contra la suma FP32 de la CPU: 2.097.152 parejas,
1.937.408 resultados comparados y 78 transacciones con desbordamiento y salida
intacta. MSVC 19.51 pasa ambos bancos; ocho corrupciones del transcript son
rechazadas. Alterar un resultado serializado hace fallar el ejecutable.

```powershell
cmake --build build/ntrpu-residual-r1 --config Release
ctest --test-dir build/ntrpu-residual-r1 -C Release --output-on-failure
```

## Evidencia HLS y transporte pendiente

Tanda `20261013T154948Z-V7-llama32-residual-hls-r1`, revisada por el
coordinador R7: 43 transacciones reales C/RTL y 51.556 palabras exactas.
Los vectores completos correctos tardaron 4.110 ciclos; los errores de
longitud, un ciclo, y los errores de contenido de longitud 2.048, 2.062.
Son latencias del núcleo en cosimulación, sin transporte AXI.

Estimaciones HLS: 4.067 LUT, 1.387 FF, cuatro BRAM de 18 Kb y cero DSP;
periodo estimado 3,644 ns con objetivo 5 ns. No equivalen a timing ruteado.
Archivo D:/NFS: `9cb7806bea34935c788cad00af7cf4dc22843561efcb6bc26bdf93c145818108`.
Revisión: `09faf097028fa806157b962259d8b6a7b97b6d6fcb9223899e5fb55bae9e9a2f`.

El [cargador de dos operandos](LLAMA32_RESIDUAL_LOADER_R1.md) ya pasó
pruebas nativas y está encolado en Vivado. Siguen pendientes propietario
de RAM, escritor con confirmación B, unión a FFN/post-O y validación de
inferencia completa y placa.
