# Escritura y publicación del KV FP32

## Contrato de memoria y propiedad

`ntrpu_kv_append_r1.sv` administra una reserva para una secuencia activa,
con 16 contadores de tokens publicados, uno por capa. Comparte el layout
`[capa][K/V][cabezal KV][posición][64]` del
[lector KV](LLAMA32_KV_TILE_LOADER_R1.md): 65.536 bytes por posición entre
todas las capas y hasta 8 GiB para 131.072 posiciones FP32.

Una vinculación registra región, base, capacidad, secuencia y época de
64 bits. Comprueba sumas de 65 bits, alineación de 64 bytes y toda la
reserva. Requiere una época mayor que la última vinculación correcta;
una vinculación aceptada invalida la anterior y reinicia los contadores,
incluso si termina rechazada. Nunca se reinicializan los bytes DDR.

`maintenance_allowed` es un certificado del planificador: **no pueden
quedar lectores ni usuarios de la reserva anterior**. El bloque comprueba
esa entrada antes de aceptar una vinculación; generar el certificado y
conectarlo al control de leases continúa pendiente. Las escrituras anexan
datos después del prefijo publicado y no modifican tokens anteriores.

## Escritura de un token

El comando exige `position == committed_tokens[layer]`, dentro de capacidad,
y la identidad vigente. Recibe 64 beats de 512 bits: cuatro por cabezal,
primero los ocho K y después los ocho V. Una RAM privada de 4 KiB conserva
el token completo. Verifica orden, identidad, `last` y finitud FP32 antes
de aceptar un commit correcto; no emite AW/W durante esa validación.

Cada ráfaga respeta páginas de 4 KiB. AW y W avanzan independientemente,
con una sola respuesta pendiente. Sólo la última respuesta B correcta
incrementa el contador de la capa. Los contadores de otras capas permanecen
intactos. `done_bytes` cuenta handshakes W; un error puede dejar bytes
privados modificados, sin publicar el token. Se puede reintentar la misma
posición con un token completo después de drenar la tanda anterior.

Estados: 0 correcto, 1 descriptor/posición, 2 identidad, 3 AXI, 4 formato,
7 cancelación y 8 watchdog; un fallo del productor también se propaga.
El primer fallo permanece retenido. Cancelación/watchdog durante la carga
drenan hasta el commit del productor; durante escritura drenan AW/W/B
emitidos y detienen las siguientes ráfagas. Un peer permanentemente
detenido necesita aislamiento/reset coordinado. El reset invalida la
reserva y todos los prefijos; debe alcanzar también productores, lectores
y peer AXI. No hay CDC ni recuperación persistente de prefijos.

## Evidencia y siguiente tanda

Icarus ejecutó cuatro suites: semillas 1/7/42 con watchdog 32 y semilla 1
con el valor de producción 65.536. La revisión independiente aprobó
**116 vinculaciones, 312 comandos de escritura, 20 resets y 6.848 lecturas
de contadores**. Cada suite completó 23 tokens, 94.208 bytes publicables;
los tokens correctos tardaron **305–378 ciclos simulados** bajo pausas.
Se verifican datos físicos W, planes AW completos y respuestas B,
incluidos errores tardíos, épocas altas, capas aisladas y reintentos.
Trece corrupciones de evidencia por suite fueron rechazadas.
Seis variantes RTL compiladas también fallaron como se esperaba:
dirección truncada, BRESP ignorado, salto del prefijo, reutilización de
época, datos no finitos y avance sin esperar la respuesta B.

Tanda canónica: `20261012T043211Z-V7-kv-append-native-r1`, archivo SHA-256
`145e4b30c704ecf95dfd0caffefc3e918853b3ab2f53ab0d5c63b4241395e0bc`,
verificado en D: y NFS. Se conservan el primer diagnóstico de una
expectativa del banco para capa inválida y la primera ejecución correcta
sin las trazas adicionales de publicación. También se corrigió un bucle
del revisor al construir un plan para una reserva desalineada rechazada.

Vivado repitió las cuatro suites: 116 vinculaciones, 312 escrituras,
20 resets y 6.848 lecturas de contadores. Los tokens correctos tardaron
**312–387 ciclos en XSim**, con las pausas del banco. La tanda
`20261013T175629Z-V7-llama32-kv-append-vivado-r1` obtuvo
`PASS_KV_APPEND_R1_XSIM_SYNTH`; su revisión se contrastó nuevamente con
el respaldo el 15 de octubre. Archivo SHA-256:
`5cf5c4430449194b013fb0540a03f5cdb6d64a8068a9a7d1dd89a47f701fda2a`.

La síntesis OOC para VU5P informó 998 LUT, 1.131 FF, 7,5 bloques BRAM y
un DSP. Con reloj de 5 ns: WNS de setup 2,667 ns y hold 0,065 ns, sin
blackboxes, latches ni bucles combinacionales. Son estimaciones de síntesis;
no califican timing ruteado ni interfaces externas.

Falta conectar el productor Q/K/V y el control de lectores, ejercitar
toda la longitud del contexto en RTL e integrar DDR real. Estas cifras
no miden ancho de banda físico ni inferencia completa.
