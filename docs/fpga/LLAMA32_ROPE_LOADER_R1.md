# Carga AXI de Q/K y coeficientes RoPE

El [cargador R1](../../fpga/rtl/ntrpu_rope_loader_r1.sv) carga tres regiones
de memoria en las dos RAM que necesita el [propietario RoPE](LLAMA32_ROPE_OWNER_R1.md).
Pasó la validación Icarus con el lector AXI real y un peer de memoria
sintético. La unión con el núcleo aritmético y DDR físico sigue pendiente.

## Descriptores y representación

| Región AXI | Tamaño fijo | Destino privado |
| --- | ---: | --- |
| Q | 8.192 bytes / 128 beats | RAM QK, palabras 0–2.047 |
| K | 2.048 bytes / 32 beats | RAM QK, palabras 2.048–2.559 |
| Coeficientes | 256 bytes / 4 beats | RAM de fases, palabras 0–63 |

Los beats tienen 512 bits y dieciséis palabras FP32 en orden little-endian.
Se verifica finitud y se escribe una palabra por ciclo. Los descriptores
incluyen dirección, base y tamaño de reserva de 64 bits. Las tres regiones
se validan antes del primer AR: alineación de 64 bytes, extensión completa,
reserva no vacía y ausencia de desbordamiento mediante sumas de 65 bits.
Las regiones son de lectura; el bloque no exige exclusión entre ellas.

Se capturan secuencia/época de 64 bits y posición de 32 bits. La posición
debe ser menor que 131.072 y coincidir con los campos del productor Q/K y
de coeficientes. El llamador debe retener las regiones inmutables hasta
reconocer la respuesta. Los campos son una instantánea del descriptor,
no un handshake que libere automáticamente al productor original.

Las posiciones **no autentican la tabla de fases**. El runtime aún debe
verificar su identidad/hash y seleccionar la fila correcta. Este cargador
comprueba transporte y finitud; las comprobaciones de coeficientes unitarios
y su uso aritmético corresponden al núcleo RoPE.

## Errores y propiedad

El lector admite una ráfaga pendiente y evita cruzar páginas de 4 KiB.
Comprueba RRESP, RID, RLAST, bytes y número de beats de cada región. Sólo
una respuesta correcta y vigente publica las dos RAM completas. Las lecturas
públicas tardan un ciclo; CE inactivo o direcciones fuera de rango producen
`q_valid=0` y cero. Los bytes físicos no se borran al revocar.

Cancelación, cambio de época y watchdog retienen el primer fallo y drenan
la ráfaga emitida. No se reutiliza el puerto mientras quede pendiente una
respuesta. Un peer que nunca responda requiere reset coordinado. Se mantiene
la respuesta pública hasta ACK, con revocación irreversible durante la espera.

## Evidencia

El [banco](../../fpga/tb/tb_ntrpu_rope_loader_r1.sv) ejecutó tres semillas,
**153 escenarios, 147 respuestas, nueve resets y 70.848 palabras verificadas**.
La revisión independiente reconstruye direcciones y datos aceptados por AXI
y los compara con las RAM públicas, incluyendo la frontera Q/K y la última
palabra de coeficientes. Rechazó 90 alteraciones de traza. Los 19 mutantes
RTL compilados fallaron mediante aserciones explícitas, sin aceptar timeouts.

Las cargas correctas abarcaron 2.812–2.839 ciclos del banco desde la
liberación anterior hasta la nueva respuesta, sin incluir la lectura posterior
de las RAM. Es una medida de simulación con pausas sintéticas, no ancho de
banda DDR ni rendimiento de inferencia. La RAM recibe una palabra por ciclo.

Revisión: `build/cloud-setup/20261002-board/rope-loader-native-r1-review.json`,
estado `PASS_ROPE_LOADER_R1_NATIVE`. Ambas tandas están en D:/NFS, con
fuentes y resultados restaurados:

- Base `20261016T001129Z-V7-rope-loader-native-r1`, TAR SHA-256
  `efd1543592d63877e875906bbb224b5a174ee3b0c68ca7e34f4d450d43819cae`.
- Mutantes `20261016T001312Z-V7-rope-loader-mutants-r1`, TAR SHA-256
  `3e06fcf1854fad526c680a5a91220f2ad4019b94746dc8f2ad465e48438f536a`.

## Siguiente integración

Síntesis OOC y XSim están encolados después del propietario RoPE aprobado y
restaurado: coordinador PID 27644, dos CPU, reloj solicitado de 5 ns y límite
de 30 minutos, sin reintento automático. Paquete
`20261016T001557Z-rope-loader-vivado-r1`, 30 archivos y tres fuentes compilables,
restaurado en D:/NFS. ZIP SHA-256:
`2c1ff8566a14494a06f1a519a23a821a629cceb116ef64cb17862408a1e676ba`.
Todavía no hay medidas Vivado nuevas de este cargador. Cada tanda terminal
debe archivarse y restaurarse antes de liberar el slot.

Traspaso documental y cola:
`artifacts/ntrpu-board-vivado/rope-loader-queue-20261016-r1/`. La auditoría
R53 verificó catorce coordinadores y 607 vínculos inmutables. Se conservan
la FFN/residual activa y el turno de actividad DDR posterior a BRAM.

Las dos RAM ya están [conectadas al propietario R2](LLAMA32_ROPE_INPUT_PATH_R1.md)
y pasaron el banco de protocolo con HLS sintético. Falta añadir el RTL HLS
original y revisar bytes calculados contra el fixture RoPE. Después se
necesitan el publicador Q/K, la retención de V y el append KV coordinado.
Estas pruebas de carga no completan el recorrido QKV→RoPE→KV.
