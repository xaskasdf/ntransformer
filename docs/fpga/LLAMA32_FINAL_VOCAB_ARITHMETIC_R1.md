# Normalización final y vocabulario: banco aritmético conjunto R1

Estado 2026-10-16 14:21 UTC: **encolado y compilado; ejecución Vivado pendiente**.
La entrada es el residual FP32 de la capa 15 de la referencia CPU para
token 0, posición 0. Esta prueba no ejecuta las dieciséis capas en RTL.

## Ruta que debe ejecutar el banco

`ntrpu_final_vocab_real_r1` conecta RMSNorm/Q8 HLS, escritura AXI del ABI1
y el motor original de vocabulario con dos MAC. Se mantienen las 128.256
filas, los 16 segmentos y las 5.376 filas del último segmento. El ABI1
comienza envenenado y sólo recibe datos de escrituras reales confirmadas
con B OKAY; las lecturas del vocabulario usan esa memoria, no el vector
esperado. El banco registra las 256.576 lecturas ABI1 de sus tres trabajos.

Los trabajos son: rechazo de una región de salida insuficiente sin tráfico,
cancelación después de al menos 70 lecturas de la primera matriz y
recuperación con todos los logits. Se exigen dos normalizaciones completas,
1.024 palabras HLS, 128 confirmaciones de escritura ABI1, 128.256 resultados
FP64 y 8.016 beats de logits FP32. El revisor contrasta también identidad,
segmentación, orden de publicación, retención del productor y drenaje.

## Entradas y verificaciones previas

El fixture pequeño conserva el residual y `output_norm.weight` del GGUF
fijado. Su ABI1 coincide byte a byte con la referencia completa y con el
fixture de vocabulario ya respaldado. No se duplican sus 279.085.056 bytes
de pesos compactados. Manifiesto pequeño:
`1c171df5d2ba592e400563b835b0b692f75685b79477ad88b4c240c81b411439`.

Preflight `20261016T141706Z-final-vocab-arithmetic-preflight-r1`: 26 fuentes
compiladas con Icarus en CPU 2–3; las dos firmas HLS auxiliares no se
ejecutaron. Veinte controles de corrupción del revisor pasaron sobre una
traza sintética completa. Esto no acredita aritmética RTL ni latencia.
Respaldo restaurado en D: y NFS, 56 archivos, ZIP SHA-256
`4e7b23425da366aa87f80346d6e481fa1b54123189ef0903d942acffa4d19820`.

## Ejecución y aceptación

La tanda debe ir después de la síntesis/protocolo de
[normalización final y vocabulario](LLAMA32_FINAL_VOCAB_R1.md), conservando
las dependencias anteriores. Usa 59 fuentes reales más las IP generadas,
CPU 0–1 y un máximo de 16 horas, sin repetición automática. Los pesos son
una dependencia externa inmutable, verificada antes y después.

Aceptar sólo con revisión independiente, archivos de salida exactos,
identidad de fuentes compiladas y restauración del respaldo en D:/NFS.
Un fallo conserva logs y diagnóstico. La prueba no califica DDR física,
temporización de placa, consumo, múltiples tokens ni equivalencia BF16/CUDA.

## Admisión y respaldo

Coordinador PID 17532, estado
`waiting_for_final_vocab_synthesis_review_and_backup`. Paquete
`20261016T142035Z-final-vocab-arithmetic-vivado-r1`: 103 archivos,
244.972 bytes comprimidos, restaurados en D: y NFS. SHA-256:
`e2263fda215c48aea6b703bc9a92410ec5fa5c679ac4bcbfd829260a293abeb7`.
Configuración: `build/cloud-setup/20261002-board/final-vocab-arithmetic-vivado-r1-package.json`.

La admisión conservó las 84 entradas anteriores y la tanda Q/K/V activa.
Auditorías R82/R22: 32 propietarios, 38 paquetes y 2.023 vínculos intactos.
El traspaso documental queda en
`artifacts/ntrpu-board-vivado/final-vocab-arithmetic-handoff-20261016-r1/`,
con referencias a los respaldos previos, sin duplicar los pesos.
