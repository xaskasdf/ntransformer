# Atención Llama 3.2 1B: resultados C/RTL R1

Tanda `20261013T165530Z-V7-llama32-attention-hls-r1`:
**PASS_ATTENTION_R1_SUCCESS_C_RTL**, finalizada el 2026-10-13 a las 17:28 UTC
y respaldada a las 17:30 UTC.

## Cobertura y resultados

Se revisaron 143 casos, incluidos 128 con datos reales de embedding,
normalización, QKV y RoPE. Las 291 transacciones exitosas comparan
30.264 palabras de estado y 9.152 palabras de salida C/RTL.
La referencia independiente obtiene 0 ULP en las salidas y error escalado
máximo de estado `3,595e-16`, por debajo del límite `2e-12`.

Los archivos binarios C y RTL coinciden para estado y salida; también se
verificaron los ecos de query, K, V y estado de entrada. El revisor rechazó
12 corrupciones numéricas y 10 corrupciones de las transacciones.
La serialización es de dos estados: esos archivos no prueban ausencia de X/Z.

| Métrica del banco | Valor |
| --- | ---: |
| Latencia mínima / media / máxima | 1.454 / 7.700 / 58.266 ciclos |
| Intervalo mínimo / medio / máximo | 1.455 / 7.713 / 58.267 ciclos |
| Ejecución total del banco | 2.241.204 ciclos |
| Recursos estimados HLS | 20 BRAM18K, 74 DSP, 7.265 FF, 15.227 LUT |
| Periodo estimado HLS | 3,641 ns |

Con el reloj solicitado de 5 ns, la mayor latencia observada corresponde
a 291,33 µs por transacción. No es latencia por token ni timing físico cerrado.
Los rechazos, el recorrido de contexto completo, DDR externa y la integración
AXI con propiedad del caché quedan fuera de esta calificación.

## Cierre y respaldo

La herramienta terminó con código 0 después de 1.959,686 s, sin reinicio.
Se había ampliado el presupuesto de 2.400 a 5.400 s; finalmente acabó dentro
del presupuesto original. El supervisor reanudó correctamente al padre,
pero su última lectura coincidió con la escritura no atómica de
`invocation.json` y produjo `JSONDecodeError`. La revisión terminal separada
verificó la invocación definitiva, el resultado numérico y el archivo completo;
el registro del supervisor se conserva sin corregir retrospectivamente.

- Archivo local/NFS: `e6e29fd059d92e84728f88adb6b40fb33f3fc2a3bdb11ec81d8930df2d75fc0e`.
- Revisión C/RTL: `6cfef3c60d61b66dae2b622cf8dea916bd19bed7dc4bb15c776f59521d0e5e05`.
- Revisión de cierre: `d6f75b96e4549e6d0d4be73a80b360abb1789f98c5e4c1e1a57133fdd736bc04`.

La cola continúa por cargador KV, append, caché, RAM de atención, camino KV,
contexto, memoria y motor de atención con el núcleo generado real.
