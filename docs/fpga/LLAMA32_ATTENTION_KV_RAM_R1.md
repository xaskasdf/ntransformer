# RAM privada K/V para atención — R1

## Contrato

`ntrpu_attention_kv_ram_r1` recibe tiles FP32 de 1–64 tokens mediante
beats de 512 bits, primero K y después V. Almacena 32 KiB en 16 bancos
de 512 palabras de 32 bits. Cada banco usa direcciones 0–255 para K y
256–511 para V. La inferencia de BRAM debe comprobarse con síntesis;
la organización RTL por sí sola no demuestra el recurso físico.

El controlador propietario entrega posición, capa, cabeza de consulta,
secuencia, época y concesión. Este componente no autentica descriptores
arbitrarios ni concede acceso a la caché. `compute_kv_head=query_head/4`.
El controlador debe conservar la concesión KV durante todo el consumo.

Los datos permanecen privados hasta recibir una finalización exitosa,
con identidad, número de beats y bytes coincidentes. Se verifican orden,
plano, índice, marca final y valores finitos en cada beat. La finalización
puede coincidir con el último beat. Un fallo invalida el tile entero.

Tras `compute_valid && compute_ready`, dos puertos síncronos independientes
permiten leer K y V: `*_ce`, `*_address[11:0]`, `*_q[31:0]`. La respuesta
aparece después del flanco de lectura; `*_q_valid` indica su validez.
Fuera del consumo autorizado las salidas son cero. No se aceptan nuevas
cargas hasta `finish_valid && finish_ready` y el retiro de `done_valid`.

## Fallos y reset

Estados: 0 éxito, 1 descriptor inválido, 2 identidad/época, 4 formato,
7 cancelación, 8 espera de carga agotada y 9 dirección RAM fuera del tile.
Los errores de productor/consumidor se propagan cuando no existe un fallo
previo. Durante carga, cancelación y watchdog solicitan cancelar al origen
y drenan hasta su finalización. Durante cómputo, `compute_abort` exige que
el consumidor drene antes de confirmar `finish`; el almacenamiento sigue
retenido. La aritmética puede tardar sin accesos RAM y no activa el watchdog
de carga. Reset requiere detener coordinadamente todos los participantes:
invalida ownership, pero no borra físicamente los bancos.

## Evidencia nativa

Tanda `20261013T140411Z-V7-attention-kv-ram-native-r1`: cuatro suites Icarus,
semillas 1/7/42 con watchdog 32 y semilla 1 con watchdog 65536. Cada suite
comprueba 81 trabajos, cinco resets, 1824 beats de entrada, 45 inicios de
consumidor, 12.487 ciclos de lectura y 24.366 palabras habilitadas. Incluye
las 32 cabezas GQA, tiles 1/2/3/16/63/64, contrapresión, errores, cancelación,
revocación, lectura fuera de límites y reset en cada fase de ownership.

El revisor `tools/check_ntrpu_attention_kv_ram_r1.py` reconstruye cada tile
y coteja las respuestas con los bytes aceptados. Rechaza 18 alteraciones
de evidencia por suite. Seis variantes RTL compiladas con planos/carriles
erróneos, publicación prematura, concesión ignorada, reutilización temprana
y acceso fuera de límites también fueron rechazadas. Fuentes y resultados
están respaldados en D: y NFS; archivo nativo SHA-256
`7f47d4f089d293f236b83be213a1be87f6f269e79a8c8bd91257fe6b0d2863e1`.

## Siguiente validación

El paquete Vivado R1 requiere la caché KV revisada y respaldada. Ejecutará
las cuatro suites XSim y síntesis OOC VU5P, reloj de 5 ns, dos CPU y límite
de 1200 segundos. Debe comprobar BRAM, ausencia de latches/blackboxes y
reportar recursos y timing sin promover estimaciones a cierre físico.

El consumidor del banco es un comprobador de puertos RAM. **No es atención
HLS**: siguen pendientes el enlace con sus puertos generados y su latencia,
la integración con la caché real, la aritmética conectada, DDR físico y la
inferencia completa. Los pesos GGUF Q8_0 no cambian este contrato K/V FP32.
