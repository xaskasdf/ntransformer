# Atención causal por tiles para Llama 3.2 1B

## Contrato del bloque

`fpga/hls/attention_tile_fp32_r1/` procesa un cabezal Q de dimensión 64
contra 1–64 posiciones K/V consecutivas, desde cero hasta la posición
de consulta. Acepta 16 capas, 32 cabezales Q y ocho K/V, con
`kv_head = query_head / 4`; el contexto máximo es 131.072 posiciones.
Q/K/V son FP32. Cada score usa 64 productos y sumas FP64 ordenados,
escalados por 1/8. El softmax incremental conserva máximo, masa y
64 acumuladores ponderados FP64 entre tiles.

El estado ocupa 104 palabras de 64 bits: firma, secuencia, época,
posición, capa/cabezal, cantidad procesada, máximo, masa, 32 palabras
con los bits exactos de Q y 64 acumuladores. Una continuación exige
identidad y posición coincidentes. Es estado privado y confiable del
trabajo; estas comprobaciones no autentican un estado arbitrario.

Las memorias deben ser distintas y los insumos inmutables. Sólo se
escribe estado tras completar correctamente el tile; la salida se
escribe en el último. El consumidor publica al observar `ap_done`.
No hay adaptador AXI, validez de caché KV, cancelación ni reset integrado.

Estados: 0 correcto; 1 forma; 2 identidad; 3 continuación inválida;
4 entrada no finita; 5 fallo numérico. Forma e identidad preceden al
estado; el estado inválido precede a entradas no finitas al continuar.
El inicio ignora el estado previo. Para absorber redondeo al extremo
FP32 se admite un exceso relativo máximo de 2^-32 antes de redondear
a `FLT_MAX`. Este límite no es una garantía general de precisión.

## Evidencia nativa

- 4.368 llamadas: continuidad con distintas particiones, 43 rechazos
  con salidas intactas y recorrido completo de 131.072 posiciones.
- 143 casos numéricos, 291 tiles, 30.264 palabras de estado y 9.152
  salidas. Incluyen los 32 cabezales para cuatro prefijos de los tokens
  `[0, 42, 128000, 128255]`, embeddings/norma reales, cuantización ABI1,
  proyecciones Q/K/V completas de capa 0 y RoPE del GGUF original.
- Scores comprobados con aritmética racional y softmax independiente
  Decimal90: 0 ULP de diferencia final; máximo error escalado de
  estado 3,595e-16, frente al criterio 2e-12. Doce corrupciones del
  transcript y una referencia alterada ejecutada fueron rechazadas.

El banco detectó y corrigió un falso desbordamiento de media convexa
con 63 valores `FLT_MAX`; el diagnóstico original queda conservado.

## Reproducción y siguiente tanda

```powershell
cmake -S fpga/hls/attention_tile_fp32_r1 -B build/ntrpu-attention-tile-r1 -DNTRPU_ATTENTION_FIXTURE=<ruta>/llama32_attention_r1.bin
cmake --build build/ntrpu-attention-tile-r1 --config Release
ctest --test-dir build/ntrpu-attention-tile-r1 -C Release --output-on-failure
```

El fixture tiene SHA-256
`6c992c47e5a3e9f2d6fc95c3db291b6d83a96cf3483bb2ba5e96f3f54976f361`.
La tanda HLS espera norma→Q8 aprobada y respaldada: VU5P, objetivo
5 ns, dos CPU y límite de 2.400 segundos, con C→síntesis→cosimulación.
El revisor exige estado, retorno y salidas reales C/RTL y ciclos para
291 transacciones; pasó diez corrupciones de trazas construidas.
Los rechazos y el contexto completo siguen siendo pruebas nativas.
No se acredita todavía RTL, memoria KV externa, inferencia completa
ni frecuencia o rendimiento de una placa física.
