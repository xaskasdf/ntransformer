# Propiedad de las memorias de RoPE

El [controlador R1](../../fpga/rtl/ntrpu_rope_owner_r1.sv) conserva las
memorias de Q/K y de coeficientes durante un cálculo RoPE. Su interfaz HLS
es abstracta: la aritmética original tiene [evidencia C/RTL](LLAMA32_ROPE_TABLE_R1.md),
pero todavía no está conectada a este controlador.

## Contrato y publicación

El plano 0 contiene 2.560 palabras FP32: 32 cabezas Q seguidas de ocho K,
de 64 elementos cada una. El plano 1 contiene 64 coeficientes FP32,
32 pares coseno/seno. Cada puerto RAM responde un ciclo después de CE.
El controlador rechaza direcciones de entrada fuera de 0–2.559 y 0–63
antes de reenviarlas; verifica respuestas presentes y valores finitos.

Un comando captura secuencia y época de 64 bits y posición de 32 bits.
Exige posición menor que 131.072 y coincidencia con ambos campos del
productor. Los metadatos deben permanecer estables mientras las memorias
estén prestadas. Esto **no prueba que los coeficientes correspondan a la
fase**: el cargador debe verificar una tabla inmutable identificada por hash.

La salida privada tiene 2.560 palabras FP32. Sólo se publica tras recibir
exactamente todas, en orden, finitas, y `ap_done`/`ap_ready` con retorno
cero. La lectura pública también tiene un ciclo de latencia; una dirección
fuera de rango devuelve cero con `result_q_valid=0`, sin cambiar el estado
de la respuesta retenida. No se borran físicamente los bytes al revocar.

Estados: 0 correcto, 1 posición fuera de rango, 2 identidad/posición
incompatible, 4 memoria/formato, 5 error HLS, 7 cancelación, 8 watchdog
y 11 contrato del productor/HLS. Los errores previos del productor se
propagan. Cancelación, cambio de época o fallo quedan retenidos y drenan
el HLS antes de liberar la fuente. Una identidad ajena no se reconoce;
un productor permanentemente detenido necesita reset coordinado.

## Pruebas y evidencia

El [banco R2](../../fpga/tb/tb_ntrpu_rope_owner_r2.sv) pasó Icarus con las
semillas 1/7/42: **297 trabajos, 279 respuestas, 407.040 palabras FP32 y
21 resets**. El revisor independiente rechazó 23 alteraciones de traza por
semilla. Son pruebas de propiedad con productores y HLS sintéticos, no
rotaciones calculadas ni rendimiento de inferencia.

La primera tanda conservada agotó el watchdog global porque el banco
esperaba incorrectamente un arranque tras rechazar la posición 131.072.
R2 corrigió el alcance de esa espera sin modificar el RTL. Con R2, los
**29 mutantes RTL compilados** fallaron por aserciones explícitas; un timeout
no cuenta como detección. Se conserva la primera tanda de mutantes, donde
el error anterior del banco impidió calificar seis de esas detecciones.

Revisión: `build/cloud-setup/20261002-board/rope-owner-native-r1-review.json`,
estado `PASS_ROPE_OWNER_R1_NATIVE`. Cuatro tandas archivadas en D:/NFS y
restauradas completamente, incluyendo fuentes y diagnósticos. Tanda correcta:
`20261015T235619Z-V7-rope-owner-native-r2`, TAR SHA-256
`534e504a87c03ad756bc9d879222d535322845bd38a825635c18e3b594e0f9a2`.
Mutantes: `20261015T235851Z-V7-rope-owner-mutants-r2`, TAR SHA-256
`21e74253bd88e295e1354d9064767c3242218dc75b481e81d47b0af5a28ee24e`.
Los contadores de acceso del banco son diagnósticos; la revisión independiente
comprueba los datos de salida, identidades y orden de publicación/liberación.

## Turno Vivado y respaldo

Paquete `20261016T000621Z-rope-owner-vivado-r1`: 39 archivos, dos fuentes
compilables, copias D:/NFS y restauración verificadas. SHA-256 del ZIP:
`73878c89dc7fba35aab82410a5929f0bb69e5b2b783bfec03b046603b413a94a`.
El coordinador PID 27656 espera la aprobación y restauración de la
normalización→QKV aritmética real y un slot libre. Usa dos CPU y 30 minutos
para síntesis OOC a 5 ns y tres semillas XSim, sin reintento automático.
Todavía no hay recursos ni timing Vivado nuevos para este propietario.

La auditoría R52 verificó trece coordinadores y 581 vínculos inmutables.
FFN/residual continúa activa; actividad DDR conserva su turno después de
BRAM. Cada diagnóstico o resultado de RoPE se archivará y restaurará antes
de liberar el slot. El traspaso documental y de cola se respalda en
`artifacts/ntrpu-board-vivado/rope-owner-queue-20261016-r1/`, con dependencias
identificadas para las cuatro tandas y el paquete completo.

## Integración siguiente

El [cargador AXI R1](LLAMA32_ROPE_LOADER_R1.md) ya pasó su banco nativo:
captura Q (8.192 bytes), K (2.048 bytes) y la fila de fases (256 bytes),
verifica regiones completas y produce el contrato de las dos RAM anteriores.
La [composición de entrada](LLAMA32_ROPE_INPUT_PATH_R1.md) ya conecta ambos
y pasó su banco con HLS sintético. Usa propietario R2 para corregir la
precedencia de cancelación al aceptar una carga revocada; conserva este
R1 y su paquete. Falta conectar el RTL RoPE original y revisar sus bytes
calculados frente a la referencia independiente.

El propietario exterior debe conservar también V hasta confirmar el
[append KV](LLAMA32_KV_APPEND_R1.md): liberar aquí las RAM de entrada no
autoriza reutilizar los buffers originales QKV. Q rotado y K/V confirmado
sólo habilitarán atención cuando pertenezcan a la misma capa, secuencia,
posición y época. Siguen pendientes AXI integrado, lectores, DDR y timing
físico; este bloque no certifica el permiso de mantenimiento del KV.
