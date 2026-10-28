# Productor Q/K/V: contrato y referencia completa R1

Estado 2026-10-15: **referencia numérica y control nativo aprobados;
controlador encolado para síntesis/XSim; aritmética integrada pendiente**.
La atención actual recibe Q y K/V ya calculados. Este trabajo debe sustituir
esa entrada preparada por un productor conectado, conservando las mismas
convenciones numéricas y los protocolos de propiedad.

## Datos comprobados

El fixture externo `qkv-projection-r1` conserva las tres matrices originales
Q8_0 de capa 0: Q de 2.048 × 2.048; K y V de 512 × 2.048. Son 6.684.672
bytes de pesos, sin recuantización. El formato físico sigue
`[grupo de 32 filas][tile K][bloque K][fila][34 bytes]`, con 4.032 bytes
de guarda iniciales para ejercitar cruces de página AXI.

Cuatro entradas provienen de embeddings reales de los tokens
`0, 42, 128000, 128255`, normalizados con `blk.0.attn_norm.weight` y
cuantizados a ABI1. No son activaciones sintéticas ni una inferencia
multicapa: cada token atraviesa solamente este tramo de capa 0.

La revisión independiente recalcula las 12 proyecciones mediante productos
enteros y redondeos FP64 ordenados sobre aritmética racional. Compara 12.288
valores FP64 y sus 12.288 conversiones FP32. También verifica 10.240 valores
Q/K rotados, con fases Decimal de 90 dígitos y los factores del GGUF.
Q, K y V coinciden byte a byte con las entradas de los 128 casos reales
del banco de atención existente. Se rechazan 20 corrupciones, incluso
cuando se actualiza su hash dentro del manifiesto.

## Controlador implementado y validación

`fpga/rtl/ntrpu_llama32_qkv_control_r2.sv` implementa el contrato descrito
abajo. El banco R6 ejecutó **481 trabajos**: tres semillas con 158 casos
cada una y siete casos de límite, desalineación y agotamiento de etiquetas.
Incluye rangos de los siete descriptores, alias entre asignaciones aunque
sus datos útiles no se solapen, identidad de 64 bits, cancelación retenida,
respuestas ajenas, timeout, revocación y reset coordinado.

El revisor independiente reconstruyó fases, etiquetas, direcciones,
contadores y orden de liberación; rechazó **91 alteraciones de traza**.
Los **29 mutantes compilados** fallaron por aserciones específicas, incluidos
cancelación omitida, condicionada a READY o duplicada. En el primer banco
de mutantes, uno sólo fallaba por falta de progreso: R6 añadió la comprobación
continua de identidad capturada y la segunda tanda lo detectó directamente.
El RTL R2 no cambió entre ambas tandas.

Estas pruebas usan respuestas de matrices controladas por el banco. No
acreditan multiplicaciones Q/K/V ni rendimiento de inferencia. La respuesta
pública se consume siempre con VALID/READY; si hay un peer ajeno, el
controlador permanece drenando sin republicar ni admitir otro comando.
El consumidor debe comprobar `result_valid` y `completion_revoked` además
del estado: una revocación tardía invalida el conjunto sin cambiar el estado
de una respuesta ya retenida.

Paquete `20261015T215026Z-qkv-control-vivado-r1`: 35 archivos restaurados y
verificados en D: y NFS, coordinador PID 25708. Ejecutará síntesis OOC a
5 ns y los seis bancos XSim, con dos CPU y límite de 30 minutos.
Espera a que actividad DDR termine y tenga respaldo/restauración verificados;
esa dependencia sólo ordena el slot. Un diagnóstico DDR archivado no aprueba
potencia ni invalida el banco Q/K/V independiente. La cadena anterior permanece
intacta. **Todavía no hay recursos ni timing nuevos de Vivado para Q/K/V.**

## Contrato del controlador

Usar un único `ntrpu_q8_matrix_memory` con salida FP32 y fases Q, K, V.
Las tres fases leen el mismo ABI1 de 4.096 bytes; los resultados tienen
8.192, 2.048 y 2.048 bytes. Publicar el conjunto solo después de completar
las tres escrituras y sus respuestas B. Una salida parcial permanece privada.

Retener el productor de normalización hasta el ACK público final, con
secuencia y época completas de 64 bits. Validar tamaño, dirección y propiedad
del ABI1 al admitirlo y mientras esté en uso. Un error, cancelación o cambio
de época invalida el conjunto; revertir la época no debe revivirlo.

Capturar siete descriptores: activación, tres pesos y tres salidas. Comprobar
alineación, sumas de 65 bits y pertenencia a regiones antes de emitir trabajos.
Ninguna asignación de salida puede solaparse con las asignaciones de entradas
ni con otra salida. Serializar estas comprobaciones para evitar otro camino
combinacional largo en la admisión.

Reservar 192 etiquetas internas alineadas a 64 por comando admitido: 64 para
cada fase, suficientes para los grupos Q/K/V de 64/16/16 filas. Rechazar
agotamiento sin reutilizar etiquetas. Mantener VALID y descriptores estables
durante backpressure; nunca reconocer una respuesta de identidad ajena.
La cancelación debe drenar el hijo aceptado antes de publicar el error.
El reset coordinado también debe vaciar productor, motor y memoria AXI.

## Banco aritmético integrado y encolado

El envoltorio `fpga/rtl/ntrpu_llama32_qkv_memory_r1.sv` conecta el control R2
con `ntrpu_q8_matrix_memory` original. Su banco usa matrices completas y
cuatro activaciones ABI1 reales preparadas, con memoria AXI y backpressure.
Incluye nueve trabajos: cinco proyecciones completas (una revocada antes
del ACK y una de recuperación), rechazo por límites y cancelación durante
lecturas de cada fase Q/K/V. Comprueba resultados FP64, bytes FP32 escritos,
drenaje de respuestas B y retención de la fuente hasta la liberación pública.

El alcance esperado es 21 matrices, 627 grupos, 19.968 resultados FP64 y
79.872 bytes FP32 escritos, incluido trabajo descartado. Son criterios del
banco, **todavía no resultados RTL**. El revisor independiente rechazó
53 alteraciones sobre una traza sintética; este ensayo verifica el revisor.
La jerarquía de 16 fuentes compiló con Icarus usando una firma MAC que nunca
se ejecutó. Se conservan 19 avisos de símbolos enum implícitos de módulos
anteriores; sólo la elaboración vendor resolverá la calificación nativa.

Paquete `20261015T221001Z-qkv-memory-vivado-r1`: 145 archivos, 89.064.902 bytes,
restauración verificada en D: y extracción/hash verificados en NFS. Compilará
23 fuentes, incluido el MAC HLS original y cuatro IP AMD; no incluye stubs.
Coordinador PID 19224, después de control Q/K/V aprobado y respaldado.
Síntesis OOC y XSim a 5 ns, dos CPU, límite de ocho horas, sin reintento
automático. Cada resultado o diagnóstico se archiva y restaura antes de
liberar el slot. La cadena anterior, incluida actividad DDR, permanece intacta.

## Integración y criterios pendientes

Falta ejecutar ambas tandas Q/K/V en Vivado, revisar todos los bytes y medir
ciclos por fase. La revisión también exige las fuentes compiladas exactas,
las cuatro IP y `glbl`, cero cajas negras y ausencia de bucles combinacionales.
La síntesis OOC no acredita timing ruteado ni rendimiento físico.

La [normalización viva unida a QKV](LLAMA32_INPUT_QKV_REAL_R1.md) está
sellada y encolada, todavía sin ejecución aritmética conjunta. El
[propietario RAM de RoPE](LLAMA32_ROPE_OWNER_R1.md) inicia el siguiente
tramo. RoPE consume Q seguido de K; V conserva su orden. La propiedad se transfiere al lector de atención
solo cuando Q rotado y K/V confirmado pertenecen al mismo token/posición/época.
Todas las capas, logits, tokens y transporte DDR/PCIe físico siguen pendientes.

## Evidencia local

Generador: `tools/prepare_ntrpu_llama32_qkv_fixture_r1.py`.
Revisor: `tools/check_ntrpu_llama32_qkv_fixture_r1.py`.
Resultado: `build/cloud-setup/20261002-board/llama32-qkv-fixture-r1-review.json`.
Manifiesto SHA-256:
`f5b4413ece6c6bd9da2fa772fab69502eef3ce1c39d802afe4d1c44087af9626`.
Son 79 archivos externos; todavía no acreditan aritmética Q/K/V en RTL.

Control: `tools/check_ntrpu_qkv_control_r1.py` y
`build/cloud-setup/20261002-board/qkv-control-native-r1-review.json`.
Tanda positiva `20261015T214628Z-V7-qkv-control-native-r6`, TAR SHA-256
`2189e0cb7c58ba116ffc51f29d8372185872f3308371c5e563c5f5519b8ff301`.
Mutantes `20261015T214708Z-V7-qkv-control-mutants-r2`, TAR SHA-256
`47fe2b1ab722073e5b90fff066d6730a43132a61e3326371717d67621e516ae6`.
Ambas tandas y sus predecesoras están archivadas y restauradas en
`D:/AI/ntransformer/artifacts/ntrpu-qkv-control-native/`, con copias en NFS.
ZIP de Vivado SHA-256:
`f60af4561cf3722280aaf879815dac1ba677ef09f97aaec144cc77d538915259`.

Banco real: `tools/check_ntrpu_qkv_memory_r1.py` y
`build/cloud-setup/20261002-board/qkv-memory-vivado-r1-package.json`.
Preflight restaurado: `20261015T215906Z-V7-qkv-memory-compile-r1`, TAR SHA-256
`aaba620e1db75311a2d7393981bcac87663bb6078062ad8d5da7620efdde554a`.
ZIP real de Vivado SHA-256:
`02acc56f2629968a8c5c02460a6cd092a2526d6f1ad76b6dac9c155ffb7febda`.
Traspaso y documentación: `artifacts/ntrpu-board-vivado/qkv-memory-queue-20261015-r1/`.
