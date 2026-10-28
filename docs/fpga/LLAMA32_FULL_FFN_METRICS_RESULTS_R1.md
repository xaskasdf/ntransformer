# FFN Llama con pesos reales: ciclos y tráfico

La capa 0 completa pasó nuevamente la revisión numérica desde el archivo
cerrado. Usa pesos Q8 originales y la entrada sintética número 2.
Las métricas siguientes son ciclos de simulación, no tokens/s en placa.

| Fase | Ciclos | Porcentaje | Bytes R | Bytes W | Bytes de pesos/ciclo |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate | 6454374 | 31.74% | 18874368 | 65536 | 2.7618 |
| up | 6455250 | 31.75% | 18874368 | 65536 | 2.7614 |
| swiglu_q8 | 980311 | 4.82% | 131072 | 16384 | 0.0000 |
| down | 6442729 | 31.69% | 18874368 | 8192 | 2.7668 |

Total: **20332664 ciclos**, 56754176 bytes R y 155648 bytes W.
Los ciclos de fase incluyen control y esperas. Las muestras cada 100.000
ciclos acotan las transiciones y concuerdan con los contadores; no son
una medición independiente de cada transición al ciclo exacto.
Las fracciones de beats y pausas pueden solaparse y no miden ocupación del MAC.

Se reconciliaron 49 entradas de compilación de la aplicación y once unidades
de simulación generadas, imágenes, copias HLS y el log dentro del archivo.
Catorce corrupciones de métricas fueron rechazadas. La semilla única no
describe una distribución general de rendimiento. Atención, inferencia
completa, DDR física y cierre de timing permanecen fuera de este resultado.

Tanda: `20261012T010512Z-V7-llama32-full-ffn-r3`. Revisión: `20261012T010512Z-V7-llama32-full-ffn-r3-independent-full-ffn-metrics-review-r4.json`.
SHA-256 de revisión: `9f0c309360fd2af7f19516598fa825cdcabab1c75be6778574c91c75d9ad45a6`.
Archivo: `818a2bce67b8021dab959e820ec253d92dc1886a671f7598fb43391c34a97179`.
