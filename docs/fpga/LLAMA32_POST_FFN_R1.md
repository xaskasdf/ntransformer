# Continuación post-O → FFN → segunda suma residual

Estado 2026-10-15: **referencia y ejecución nativa aprobadas** para cuatro
prefijos de capa 0. La unión RTL de estos tres bloques sigue pendiente.

## Datos y contrato numérico

Se usan las cuatro salidas de [post-O](LLAMA32_POST_ATTENTION_REAL_R1.md),
asociadas a tokens 0, 42, 128000 y 128255. La entrada FFN conserva exactamente
sus 64 registros ABI1: 4.096 bytes, sin reconstruir FP32 ni recuantizar.
El residual FP32 de post-O permanece separado e inmutable hasta la segunda suma.

Cada caso ejecuta gate/up de 8192×2048, SwiGLU/cuantización y down de
2048×8192 con los 53.477.376 bytes Q8_0 originales de capa 0. Se conserva
el orden de acumulación binary64 y los estrechamientos FP32 del hardware.
Después se suma down FP32 al residual retenido. La referencia de esta suma
usa enteros exactos en unidades de 2⁻¹⁴⁹ y redondeo al par.

## Evidencia nativa

Tanda `20261015T144526Z-V7-post-ffn-native-r1`, CPU2/3. Ambos ejecutables,
matemática estándar y AMD, aprobaron los cuatro casos:

- 122.880 valores intermedios/finales comparados por ejecutable;
  6.291.456 invocaciones al MAC Q8 y 1.024 registros SwiGLU ABI1.
- 8.192 valores de salida final por ejecutable revisados de nuevo desde
  las trazas: 32.768 bytes exactos, además de procedencia ABI1 y residual.
- Seis ejecuciones corruptas fallaron explícitamente: entrada ABI1,
  referencia gate y segunda suma, en ambos backends.
- Dieciséis corrupciones de las trazas fueron rechazadas por el revisor.

Los tiempos CPU observados son 0,135–0,137 s por caso con stdlib y
7,441–7,482 s con AMD; **no son latencia FPGA ni tokens/s**.

| Artefacto | SHA-256 |
| --- | --- |
| Manifiesto de referencias | `f2e3edda6ce1e3b5a6fca72886501e118584cd6004412772c279b7c615df5d09` |
| Archivo nativo D:/NFS | `fb3a47fc34e653820e7edc642900fd5897c1fea3f4a1a74c9e1368fd49e50170` |
| Revisión independiente | `39b27795d66ffb183023e6257345c7daa1719691c113b9a2a128b547fb641fec` |

Referencias externas: `D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/post-ffn-reference-r1/`.
Fuentes: `prepare_ntrpu_llama32_post_ffn_reference_r1.py`,
`test_ntrpu_llama32_post_ffn_native_r1.cpp` y
`check_ntrpu_post_ffn_native_r1.py`. La FFN RTL aislada con otra entrada
ya tiene [resultado y métricas](LLAMA32_FULL_FFN_METRICS_RESULTS_R1.md).

## Adaptador y FFN real preparados

La FFN existente acepta secuencia/época de 32 bits; atención/post-O usan
identidades completas de 64 bits. No se debe truncar la identidad pública.
El [adaptador R3](LLAMA32_FFN_OWNER_R1.md) conserva ambos campos públicos y
asigna una etiqueta interna privada de 64 bits, dividida en secuencia/época
del hijo. Reserva cuatro secuencias alineadas y rechaza agotamiento sin
reutilizar etiquetas. Pasó 161 casos nativos, 119 controles del revisor y
26 mutantes compilados. Su síntesis/XSim está encolada detrás de la
optimización física existente. Un reset exige reiniciar hijo y peer AXI.

El padre retendrá la respuesta de post-O hasta terminar FFN y segunda suma.
Antes de lanzar FFN, comprobará que sus destinos privados 4–7 no solapen el
residual retenido; el secuenciador existente verifica las ocho regiones FFN.
La entrada de buffer 0 se tomará del descriptor ABI1 confirmado de post-O.
Cancelación o cambio de época completa deberán drenar el hijo y revocar
la publicación, incluidos los eventos tardíos durante una respuesta retenida.
Una respuesta ajena no podrá liberar la propiedad del buffer.

La [FFN real con propietario](LLAMA32_OWNED_FFN_R1.md) está conectada y
encolada en Vivado, con el ABI1 exacto de referencia y un banco que exige
confirmación B antes de leer intermedios. Falta calificar esta ejecución
y unir el productor post-O vivo. La segunda suma debe consumir
down confirmado por B y publicar su resultado sólo tras confirmar sus
propias escrituras. Estas referencias no sustituyen esas pruebas, la
inferencia completa ni la implementación de DDR4/PCIe y cierre temporal.

El [cargador de los dos operandos residuales](LLAMA32_RESIDUAL_LOADER_R1.md)
ya pasó 114 casos nativos, 78 controles y once mutantes RTL. Su síntesis/XSim
espera detrás de la FFN real. Falta unirlo a la aritmética y al escritor;
esa prueba usa datos sintéticos, separada de los cuatro casos numéricos anteriores.
