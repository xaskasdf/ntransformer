# Vocabulario completo: aritmética RTL R1

Actualización 2026-10-16 13:46 UTC: **banco compilado y encolado;
simulación Vivado pendiente**. Coordinador PID 23336, después de DDR R2.
Paquete de 49 archivos restaurado en D:/NFS, SHA-256
`3ec6c3c8589cfe7c47e9b8d91b9415cda0bb89bc8ba05184780849542cdf7478`.
La admisión preservó las 82 entradas anteriores y la ejecución Q/K/V.

Después de esta tanda se agregó la [integración de normalización final y
vocabulario](LLAMA32_FINAL_VOCAB_R1.md), para síntesis/protocolo separado.
Este banco y sus pesos permanecen intactos; el sucesor exige su aprobación
y respaldo antes de ocupar el slot.

Preparación: 2026-10-16.
Complementa la [síntesis y propiedad del vocabulario](LLAMA32_VOCAB_HEAD_R1.md).
No repite esa síntesis ni modifica las validaciones anteriores.

## Alcance y entradas

El banco usa el controlador existente, su envoltorio de memoria y los dos
MAC originales generados por HLS. Calcula las **128.256 filas** de la cabeza
compartida de Llama 3.2 1B Instruct Q8_0, en quince segmentos de 8.192 filas
y uno de 5.376. Compara todos los resultados FP64 y los 513.024 bytes FP32
publicados contra la referencia completa previamente contrastada en CPU.
La entrada ABI1 final procede de esa referencia; no se ejecutan aquí las
dieciséis capas del modelo ni su normalización en RTL.

El fixture contiene 279.085.056 bytes compactos de pesos. Cada imagen hex
tiene un prefijo de simulación independiente de 4.032 bytes: **no concatenar
esos prefijos al payload DMA**. La inversión independiente de la distribución
verificó todos los bytes contra el GGUF; también se decodificaron los hex.
El archivo de 47 miembros está restaurado en D: y NFS, SHA-256
`6895ac27e171c895d2b8e180e57ef2eb2c4dd574e97b4b50c76db22ee7e2ab5f`.
Se referencia externamente para evitar copiar otra vez los pesos en cada tanda.

## Ejecución y aceptación

Tres trabajos: región inválida sin tráfico, cancelación temprana y recuperación
con vocabulario completo. Se comprueban identidades de 64 bits, etiquetas,
segmentos, límites AXI, contrapresión, drenaje y retención de la publicación.
Se conservan la salida completa y dieciséis archivos por segmento.

El coordinador espera DDR R2 revisado y respaldado, además de la síntesis y
protocolo del vocabulario aprobados. Usa CPU 0–1, dos hilos y un límite de
16 horas, sin reintento automático. El coste se presupuestó con el avance
real de Q/K/V: aproximadamente 785 ciclos por fila. La estimación de unas
100 millones de iteraciones de reloj es orientativa; no es una medición
de throughput físico. Una pérdida de SSH observa la misma tanda.

El revisor verifica fuentes compiladas, pesos, las 128.256 comparaciones,
8.016 escrituras y publicación final. Fallo, timeout y éxito conservan logs,
revisión, archivo y restauración en D:/NFS antes de liberar el slot.
Los ciclos de simulación usan un reloj de banco de 5 ns; no califican timing,
potencia, placa ni inferencia completa.

## Evidencia previa

La compilación de interfaz pasó con 16 fuentes; su firma MAC auxiliar **no se
ejecutó** ni forma parte de la simulación Vivado. Quince controles negativos
del revisor pasaron en Windows/Linux. Son transcripciones sintéticas con
los bytes de referencia reales, no resultados RTL. El borrador de revisor R1
queda preservado; R2 corrige la expectativa de cancelación a `revoked=1`.

Configuración y estado se registran en
`build/cloud-setup/20261002-board/vocab-arithmetic-vivado-r1-{package,coordinator}.json`.
