# Caché KV con publicación propia y concesión de lectura

## Contrato del bloque

`fpga/rtl/ntrpu_kv_cache_r1.sv` conecta, sin modificar sus fuentes selladas,
el escritor/publicador KV y el lector de tiles FP32. Conserva una asignación
de memoria y una sola consulta activa por reloj. El layout externo es
`[16 capas][K/V][8 cabezas KV][capacidad][64 valores FP32]`.

El escritor conserva los 16 prefijos publicados. Una concesión captura
capa, cabeza de consulta, posición, prefijo, secuencia, época e identificador;
el lector deriva la cabeza KV mediante GQA (`query_head / 4`). El consumidor
no proporciona la base ni un prefijo de confianza en cada lectura.

Los tiles deben recorrer el prefijo de la consulta en orden, con 1–64
posiciones por tile. Sólo una finalización correcta avanza esa posición;
un error permite reintentar. Identificadores de concesión no nulos y
estrictamente crecientes evitan reutilizaciones hasta un reset coordinado.
Se permite añadir tokens mientras una consulta conserva su prefijo anterior.
Una adquisición nueva espera a que el escritor termine, incluidas sus
respuestas B y la finalización retenida.

Reasignar memoria exige que la concesión, su respuesta y todas las lecturas
estén drenadas. `release` es el certificado del consumidor de que ya no usa
los datos para atención; puede cancelar una consulta entre tiles. Un cambio
de época rechaza nuevas lecturas, pero permite liberar la concesión anterior.
Reset exige detener también los pares AXI y descartar datos privados del
consumidor. Los datos de un tile con error no pueden publicarse a atención.

## Evidencia nativa

La tanda `20261013T055226Z-V7-kv-cache-native-r1` ejecutó Icarus con semillas
1, 7 y 42, y repitió la semilla 1 con watchdog de producción (65536 ciclos).
Cada suite aprobó **252 operaciones y seis resets**: siete vinculaciones,
77 escrituras, 109 comandos de concesión y 59 lecturas. Total: 1008 operaciones
y 24 resets. Se comprobaron las 32 cabezas de consulta, capas 0/2/15,
fragmentación en 4 KiB, direcciones mayores que 4 GiB, tiles de 64+1 posiciones,
AXI concurrente, reintentos y revocación de época.

El modelo AXI conserva los bytes escritos, incluso cuando devuelve B con
error. El revisor independiente reconstruye direcciones, contenido físico,
respuestas, prefijos y datos entregados. Por suite verificó 307200 bytes
publicables de escritura y 53760 bytes de lecturas exitosas; otros 320 bytes
de salida pertenecen a lecturas fallidas y permanecen privados.

La lectura de 64 posiciones tardó **1261–1304 ciclos** entre la solicitud y
el registro de finalización, incluidos cinco ciclos de retención de la
respuesta por el banco. Son ciclos de simulación con backpressure sintético,
no ancho de banda DDR medido ni frecuencia física aprobada.

Diecinueve alteraciones de evidencia por suite fueron rechazadas. También
fallaron seis variantes RTL compiladas: cabeza GQA incorrecta, reasignación
prematura, identificación de lectura ignorada, tile repetido, prefijo propio
ignorado e identificador de concesión reutilizado.

## Siguiente validación y límites

`tools/ntrpu_board_kv_cache_r1.tcl` prepara cuatro suites XSim y síntesis OOC
VU5P con reloj de 5 ns, dos CPU y límite de 1200 segundos. Se encola después
del escritor KV revisado y respaldado. Los bancos originales y sus fuentes
selladas se conservan. Un PASS nativo no equivale a un PASS de Vivado.

Todavía falta el adaptador de RAM hacia atención HLS, el secuenciador de
consulta completa, la integración con DDR real y la implementación física.
Este banco usa datos FP32 sintéticos; no ejecuta la inferencia completa de Llama.
