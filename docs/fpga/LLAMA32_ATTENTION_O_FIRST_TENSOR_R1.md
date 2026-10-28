# Primera proyección O real completa (R1)

Observación 20261013T223134Z: **un tensor O completo coincide bit a bit**.
La tanda `20261013T221343Z-V7-llama32-attention-projection-memory-vivado-r1` sigue activa; no se declara aprobado
el banco completo, su síntesis ni la integración con DDR.

## Evidencia numérica

Se usa el primer vector real de atención y la matriz O Q8_0 de la capa 0
de Llama 3.2 1B Instruct. El circuito contiene el MAC dual generado por HLS,
el motor GEMV real y el escritor FP32; la memoria externa es un BFM AXI.

- **2.048 acumulaciones FP64** idénticas a la referencia CPU revisada.
- **2.048 resultados FP32 / 8.192 bytes escritos y publicados**, también exactos.
- 1.344 AR, 73.728 R, 66 AW, 128 W y 66 B: se verificaron los conteos completos.
- SHA-256 FP64: `ec907a637100f04f7620f32bb065b03ff82fce3a5439a04c06505b38d4d3fa66`.
- SHA-256 FP32: `828e991453ff06c7637ba2d58bf0439e89988b30e362c2a81d661aca7104383f`.

Las referencias y fuentes se vinculan por los hashes del paquete sellado,
sin regenerarlas a partir de la salida del simulador. El [oráculo CPU](LLAMA32_ATTENTION_OUTPUT_NATIVE_RESULTS_R1.md)
y el [contrato del banco](LLAMA32_ATTENTION_PROJECTION_MEMORY_R1.md) describen
el orden de acumulación y el redondeo esperado.

## Ciclos y watchdog

Desde aceptación hasta respuesta de la matriz, contando ambos flancos:
**1.606.203 ciclos**, equivalentes a **8,031015 ms** a 5 ns. El intervalo
BEGIN→COMPLETE del banco suma 1.606.211 ciclos e incluye retención pública.
Son tiempos simulados de una proyección, no tokens/s ni mediciones en tarjeta.

El mayor intervalo sin handshake AXI fue **11.820 ciclos**. Se recalculó
desde los eventos AR/R/AW/W/B y los límites activos, y coincide con el contador
del testbench. Queda por debajo del watchdog de producción de 65.536 ciclos.
Esta medición responde a una duda real de integración: el trabajo del MAC
entre ráfagas no provocó un timeout del padre en este vector.

## Respaldo y alcance restante

Se congelaron los primeros 15.694.918 bytes del log hasta la primera completion,
junto con el script de captura y la revisión numérica. ZIP D:/NFS verificado:
`59c218ce1643f991a97aab54f40ebf6e467a3bfbcc44869bd92794ec1d00de41`. Prefijo de traza:
`4047ac3573a0ab84878b9833b705d3a1ea749e39c92d66b55ae9564ac31d3892`.

La captura no detuvo ni modificó la simulación. Faltan los otros vectores,
cancelación/límite de región/recuperación, revisión causal completa de AXI,
controles negativos y síntesis de esta tanda. El enlace cuantizador→O tiene
[pruebas de protocolo separadas](LLAMA32_ATTENTION_OUTPUT_PATH_RESULTS_R1.md);
su composición aritmética y la inferencia completa siguen pendientes.
