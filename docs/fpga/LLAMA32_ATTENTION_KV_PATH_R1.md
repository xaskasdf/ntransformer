# Caché KV conectada a RAM de atención — R1

## Arquitectura y ownership

`ntrpu_attention_kv_path_r1` conecta, sin modificar sus fuentes, la caché
KV R1 y el adaptador RAM K/V R1. Conserva el escritor AXI y añade una
entrada de consulta con capa, cabeza, posición, secuencia, época y número
de concesión. Sólo administra una consulta a la vez, en un reloj común.

El controlador adquiere una concesión sobre el prefijo publicado. Recorre
desde cero hasta la posición inclusiva, en tiles de hasta 64 tokens.
Prepara la RAM antes de solicitar la lectura; sólo permite consumir un
tile cuya finalización e identidad hayan sido verificadas. El consumidor
recibe dos puertos RAM síncronos FP32 y los metadatos correspondientes.

La concesión permanece activa después de terminar AXI, durante la espera
del consumidor, su ejecución y los cambios de tile. Se libera al terminar
todos los tiles o después de drenar un fallo. `query_done_valid` aparece
tras retirar la respuesta de liberación. El estado y resultado numérico
del consumidor deben permanecer privados hasta una finalización global
exitosa; los contadores de tiles previos no convierten una consulta fallida
en un resultado publicable.

Append puede publicar tokens posteriores mientras una consulta consume
su prefijo fijo. Rebind queda bloqueado por la concesión. Una consulta
puede esperar a que terminen las respuestas B necesarias para publicar
su primer token. La cancelación antes de adquirir no cancela ese append.

## Cancelación, errores y reset

Se conserva el primer error, con prioridad de revocación de época sobre
cancelación cuando coinciden. Los estados incluyen descriptor inválido
(1), identidad (2), error AXI (3), formato (4), cancelación (7), timeout
(8), acceso RAM fuera de límites (9) y el estado del consumidor.

Una lectura ya admitida se emite y drena incluso si llega una cancelación
antes de aceptar AR. El watchdog de RAM empieza antes que la lectura;
cuando vence, cancela al lector, que informa 7, mientras la consulta retiene
el timeout 8. Un consumidor activo recibe `compute_abort` y debe confirmar
`finish` antes de liberar la concesión. Reset exige detener coordinadamente
AXI, consumidor y control; invalida los prefijos y las concesiones, sin
borrar físicamente la memoria. Un fallo inesperado de liberación mantiene
`query_ready` bloqueado mientras la caché conserve su concesión.

## Validación

Tanda nativa canónica:
`20261013T142928Z-V7-attention-kv-path-native-r1`, archivo SHA-256
`9247572844c3bfed92cad3b0982aaac8dda7dd8a946eec503edea83c62bb471d`.
Cuatro suites Icarus: semillas 1/7/42 con watchdog 32 y semilla 1 con
watchdog 65536. Por suite: seis binds, 78 appends, 70 consultas, cinco
resets, 57 inicios de consumidor y 67.364 palabras RAM comparadas.
Se publican 315.392 bytes de escritura; 64 bytes adicionales del intento
con B erróneo permanecen sin publicar.

Se prueban GQA completo, direcciones superiores a 4 GiB, cortes de página,
consultas de 65 tokens, append simultáneo y fallos antes, durante y después
del primer tile, usando valores FP32 sintéticos distinguibles por capa,
cabeza, posición y carril. Las consultas exitosas de 65 tokens consumen
5460–5499 ciclos del banco, con y sin append concurrente. El revisor
reconstruye los bytes de AW/W/B, coteja AR/R,
reconstruye los tiles y compara cada lectura RAM. Comprueba las concesiones
y su liberación después del consumidor. Incluye 25 corrupciones de evidencia
y seis defectos RTL compilados. Fuentes, resultados y diagnósticos previos
se conservan en D: y NFS.

## Alcance pendiente

El banco usa un consumidor que comprueba puertos RAM; no ejecuta aritmética
de atención. La tanda Vivado depende del adaptador RAM revisado y respaldado:
cuatro suites XSim y síntesis OOC VU5P, reloj de 5 ns, dos CPU, 1200 segundos.
Siguen pendientes el enlace con el RTL HLS generado, sus estados/resultados,
DDR físico, cierre de timing e inferencia completa. Los ciclos del banco
incluyen transferencia, lecturas del comprobador y esperas dirigidas; no
son latencia de atención ni tokens por segundo de hardware.
