# Cargador AXI de tiles KV FP32

## Layout e interfaz

`ntrpu_kv_tile_loader_r1.sv` conecta el lector AXI existente con una
caché privada para atención. Conserva direcciones y etiquetas de
secuencia/época de 64 bits; todos sus puertos usan el mismo reloj.
El layout es `[capa][K/V][cabezal KV][posición][dimensión]`, con 16
capas, dos planos, ocho cabezales y dimensión 64. Cada posición/cabezal
ocupa 256 bytes FP32; un tile de 64 posiciones consume 16 KiB por plano.

La capacidad reservada es configurable entre 1 y 131.072 posiciones.
La reserva completa ocupa `capacity * 65.536` bytes: **8 GiB** al máximo
en esta versión FP32. La previsión anterior de 4 GiB corresponde a KV
FP16/BF16, todavía sin conversión ni calificación en este camino.

El descriptor registra base de caché, región autorizada, capacidad,
prefijo publicado de la capa, posición de consulta, tile, capa,
`query_head`, `kv_head`, secuencia y épocas. Exige
`kv_head == query_head / 4`; el tile debe quedar dentro del prefijo
causal. Comprueba toda la reserva y sumas de 65 bits antes del primer AR.
La base se alinea a 64 bytes; cada ráfaga respeta páginas de 4 KiB.
Primero entrega K y después V, en beats de 512 bits con plano e índice.

## Publicación y recuperación

El prefijo y las etiquetas de caché son entradas de una autoridad
confiable. El escritor/publicador KV que las producirá sigue pendiente;
este cargador no demuestra que un descriptor arbitrario diga la verdad.

Los datos transferidos permanecen privados. Sólo `done_status == 0`
habilita atención. El contador cuenta beats entregados, incluidos los
prefijos privados de jobs fallidos; no acredita resultados publicables.
Cada beat se comprueba para rechazar NaN e infinitos FP32.

Estados: 0 correcto, 1 descriptor/rango, 2 identidad/época, 3 AXI,
4 formato, 7 cancelación y 8 watchdog. El primer fallo permanece
retenido. Cancelación y watchdog impiden nuevas ráfagas, conservan
señales detenidas y drenan lo ya emitido. Un peer permanentemente
detenido requiere reset o aislamiento coordinado, incluyendo caché
consumidora y peer AXI. No hay CDC en este módulo.

## Evidencia previa a Vivado

Icarus 11.0 ejecutó tres semillas con watchdog 32 y una suite con el
valor de producción 65.536: **340 casos y 20 resets**. La revisión
independiente reconstruyó descriptores, AR, respuestas R, fallos
inyectados y cada byte entregado. Cada suite produjo 1.159.680 bytes
de jobs correctos y 707 beats privados de jobs fallidos/reset.

Los tiles completos K+V transportaron 32.768 bytes en **833–920 ciclos
simulados** con pausas. No son medidas DDR ni rendimiento físico.
Doce corrupciones del transcript por suite y cinco variantes RTL
compiladas fueron rechazadas: truncamiento de dirección, plano erróneo,
prefijo sin publicar, época incorrecta y aceptación de datos no finitos.

Tanda Icarus:
`20261012T040158Z-V7-kv-loader-native-r1`, archivo SHA-256
`ea5587c69a10052ea3326ab01e0031ed7fa5fe70b053cd80b87c847e3ce22f0e`,
verificado en D: y NFS. Se conservan dos diagnósticos iniciales de
elaboración y la primera suite de 255 casos sin trazas R completas.

## Tanda en Vivado y límites

`tools/ntrpu_board_kv_tile_loader_r1.tcl` repetirá las cuatro suites
en XSIM y sintetizará el cargador con watchdog de producción para
VU5P, reloj objetivo 5 ns. El coordinador espera atención HLS revisada
y respaldada; limita la tanda a 1.200 segundos y dos CPU/hilos.
La síntesis OOC y sus recursos se revisarán por separado del timing
físico. Faltan publicación/escritura KV, adaptación a las RAM de HLS,
integración de atención, DDR real e inferencia completa.
