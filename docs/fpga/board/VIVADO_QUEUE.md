# Cola de trabajo Vivado para el agente principal

## Corte local tras perder acceso — 2026-10-16

El usuario informa pérdida de acceso a la VM. La cola compartida preserva sus
dependencias, pero los coordinadores no están activos: DDR R2 se detuvo antes
de captura al requerir inspección de `vocab-memory`; el análisis SAIF/DCP R1
se detuvo antes de captura porque su dependencia de token requiere inspección.
La aritmética real de normalización terminó revisada y respaldada; la tanda
Q/K/V posterior quedó como diagnóstico respaldado. Los PID remotos de registros
antiguos no confirman procesos actuales. No hay vatios nuevos. Véase el
[traspaso por pérdida de acceso](VIVADO_ACCESS_LOSS_HANDOFF_20261016.md).

Actualización 2026-10-16 15:17 UTC: [correspondencia SAIF/DCP R1](POWER_SAIF_MAPPING_R1.md)
encolada después de `token-guard-vivado-r1`, PID 28148. Conserva 86 entradas
previas: 87 en total y 31 propietarios auditados. Dos CPU, 1.200 segundos,
sin reintento; abre un DCP nuevo por ventana y exporta asociaciones e
inventarios, sin `report_power`. Catorce controles sintéticos pasaron en
Windows/Linux. Veinte fuentes restauradas y tres entradas grandes
verificadas en D:/NFS; 2.064 vínculos intactos en cuarenta paquetes.
Normalización aritmética sigue en el mismo XSim. DDR R2 conserva su lugar.

Actualización 2026-10-16 15:01 UTC: [control de token de dieciséis capas](../LLAMA32_TOKEN_GUARD_R1.md)
añadido después de `final-vocab-arithmetic-vivado-r1`, PID 24452. Dos CPU,
1.800 segundos, sin reintento: síntesis OOC y tres semillas de protocolo
con pares sintéticos. Paquete fuente restaurado; 85 entradas previas
intactas, 86 en total. Auditorías R84/R24: treinta propietarios y 2.047
vínculos en 39 paquetes. Normalización aritmética real ocupa el slot.

Entrada/QKV conserva su paquete y turno; el observador separado
`queue_input_qkv_ancestor_recovery_r1.py`, PID 11980, corrige los ancestros
R1 detenidos hacia las tandas R3 con fuentes idénticas. Exige revisión y
restauración de normalización y el suplemento respaldado de Q/K/V;
dieciocho controles rechazan metadatos incorrectos. El diagnóstico Q/K/V
original y la cobertura RVALID pendiente siguen explícitos. No se cambió
el banco ni su revisor. DDR R2 mantiene su lugar después de vocabulario;
la nueva estimación de consumo todavía no está encolada.

Actualización 2026-10-16 14:21 UTC: [aritmética conjunta de normalización final
y vocabulario](../LLAMA32_FINAL_VOCAB_ARITHMETIC_R1.md) encolada después de
su síntesis/protocolo, PID 17532. Conserva las 84 entradas previas: ahora
85 entradas y 32 propietarios. Compilación de 26 fuentes y veinte controles
del revisor aprobados; fuentes y fixture pequeño restaurados en D:/NFS.
La futura simulación usa HLS/MAC reales y lee exclusivamente el ABI1 escrito
por normalización, con las 128.256 filas. Máximo 16 horas, sin reintento.
Auditoría de 38 paquetes y 2.023 vínculos aprobada. Q/K/V sigue ejecutándose,
trabajo 9, ciclo 13.555.463 observado. DDR R2 conserva su lugar; sin consumo nuevo.

Actualización 2026-10-16 14:03 UTC: [normalización final y vocabulario](../LLAMA32_FINAL_VOCAB_R1.md)
añadidos después del banco aritmético completo, PID 8740. Protocolo Icarus:
540 trabajos, 33 mutaciones RTL y 174 controles de traza; jerarquía de
25 fuentes compilada sin ejecutar firmas auxiliares. La tanda Vivado usa
58 fuentes reales para síntesis y un proyecto separado de protocolo:
dos CPU, una hora máxima y ninguna repetición automática. Paquete y
diagnósticos restaurados en D:/NFS. Treinta y un propietarios, 84 entradas
históricas y 1.976 vínculos de 37 paquetes verificados. Q/K/V sigue en XSim,
ciclo 10.875.678 observado; DDR R2 conserva su turno. Sin potencia nueva.

Actualización 2026-10-16 13:46 UTC: [aritmética del vocabulario completo](../LLAMA32_VOCAB_ARITHMETIC_R1.md)
añadida después de DDR R2, PID 23336. Preserva las 82 entradas previas;
83 entradas históricas, treinta propietarios y 1.930 vínculos de 36 paquetes
verificados. Dos MAC originales, 128.256 filas, máximo 16 horas sin reintento.
Fixture, compilación y quince controles Windows/Linux restaurados en D:/NFS;
la ejecución aritmética está pendiente. Q/K/V mantiene el slot.

El observador local de Q/K/V murió por falta de espacio en C:, mientras
XSim siguió ejecutándose. Se trasladaron cuatro checkpoints antiguos a D:
con hashes verificados y enlaces que conservan sus rutas. El observador
PID 25464 retoma **la misma tanda y PID remoto 952238**, sin lanzar Vivado
de nuevo ni cambiar paquetes o dependencias. Auditoría R80 aprobada;
[recuperación y evidencia](OBSERVER_RECOVERY_20261016_R1.md).

Actualización 2026-10-16 13:06 UTC: [captura DDR R2 con comandos por rank](POWER_ACTIVITY_R2.md)
añadida al final, después del vocabulario, PID 26696. Nueve controles
Python y seis Tcl pasaron en Windows/Linux; 415 fuentes restauradas en
D:/NFS, 394 heredadas de R1 sin cambios. CPU 0–1, 1.200 segundos, sin
repetición automática. Memoria Q/K/V sigue en XSim. Veintinueve propietarios,
82 entradas históricas y 1.896 vínculos de 35 paquetes verificados.
R1 permanece respaldada; **R2 aún sin simulación nativa ni consumo nuevo**.

Actualización 2026-10-16 12:46 UTC: **DDR R1 ejecutada y respaldada**, con
[dos SAIF y revisión suplementaria](POWER_ACTIVITY_RESULTS_R1.md). La tanda
original y sus diagnósticos quedan intactos. Q/K/V control aprobó y liberó
su turno; memoria Q/K/V está activa. Veintiocho propietarios, 81 entradas
históricas y 34 paquetes mantienen el orden. Las revisiones de actividad,
identidad y finales de línea se hicieron fuera del slot. No se encoló una
nueva captura ni potencia; siguen pendientes sus criterios de admisión.

Actualización 2026-10-16 12:25 UTC: BRAM terminó y quedó revisado,
respaldado y restaurado. **DDR R1 ocupa el slot**, tanda
`20261016T122423Z-V8-ddr-activity-vivado-r3`; integración nativa en curso,
dos CPU, 20 minutos y sin reintento automático. Q/K/V conserva el turno
posterior a su revisión y respaldo. Treinta propietarios verificados;
81 entradas históricas. Todavía sin SAIF ni estimación nueva de consumo.

Actualización 2026-10-16 12:02 UTC: [control del vocabulario completo](../LLAMA32_VOCAB_HEAD_R1.md)
añadido al final, después de la comparación de cuatro MAC, PID 14796.
Protocolo nativo: 679 trabajos, 35 variantes RTL defectuosas y 91 controles
de transcripción; interfaces compiladas y restauradas. Síntesis OOC y
protocolo XSim separado: dos CPU, una hora, sin reintento automático.
Treinta y un propietarios, 81 entradas históricas y 1.870 vínculos de
34 paquetes verificados. BRAM continúa en implementación física y DDR
conserva el siguiente turno; sus 394 archivos coinciden en D:/NFS.
**Sin SAIF ni nueva estimación de consumo.**

Actualización 2026-10-16 10:48 UTC: [comparación de cuatro MAC](../LLAMA32_ATTENTION_FFN_QUAD_R1.md)
añadida al final, después de cancelación registrada, PID 22664. Compilación
de interfaces aprobada: dos motores y ocho MAC; aritmética todavía pendiente.
Dos CPU, ocho horas máximas, respaldo y restauración antes de liberar el slot.
Treinta propietarios y 1.832 vínculos de 33 paquetes verificados. BRAM sigue
en XSim (ciclo 1.710.051 observado); DDR conserva su turno inmediatamente
posterior. Sus 394 archivos coinciden en D:/NFS. **Sin SAIF ni potencia nueva.**

Comprobación 2026-10-16 10:31 UTC: BRAM ejecuta XSim, ciclo 541.774
observado; DDR sigue inmediatamente después de su revisión y respaldo.
Se añadió únicamente el [observador de identidad DDR](POWER_ACTIVITY_IDENTITY_R1.md),
PID 3696, sin acceso al slot ni cambios de dependencias. Se mantiene
el paquete de captura R1 y no se agregó otra tanda Vivado.

Comprobación 2026-10-16 10:19 UTC: se conserva el orden BRAM → actividad
DDR → Q/K/V. BRAM está en síntesis. El observador independiente de
restauración, PID 23628, espera su respaldo terminal; no ocupa Vivado
ni altera los coordinadores existentes. No se agregó trabajo al slot.
La captura DDR continúa pendiente de elaboración y SAIF.

Actualización 2026-10-16 10:12 UTC: [atención/FFN/residual real](../LLAMA32_ATTENTION_FFN_REAL_R1.md)
aprobada: 22.282.102 ciclos y 8.192 bytes finales exactos; respaldo D:/NFS
restaurado y verificado. BRAM: `running`. DDR conserva su turno
posterior y el observador complementario sigue esperando la captura.
El [candidato de cancelación registrada](../LLAMA32_POST_FFN_CANCEL_R1.md)
se agregó al final, PID 29932: 1.566 trabajos de comparación,
43 variantes defectuosas y 324 controles de traza. Su efecto sobre WNS
debe medirse en Vivado; dos CPU y una hora máxima. **Sin SAIF ni potencia nueva.**

Actualización 2026-10-16 09:52 UTC: [banco de capa completa](../LLAMA32_LAYER_REAL_R1.md)
preparado y encolado tras su síntesis/protocolo, PID 10788.
Compilación de interfaces, ocho casos del servicio AXI y 33 controles
sintéticos Windows/Linux aprobados; aritmética conjunta todavía pendiente.
Una capa/vector, ocho HLS reales, 239 fuentes, dos CPU y máximo 12 horas,
sin repetir síntesis ni reintentar. Veintinueve propietarios mantienen
su orden. DDR sigue después de BRAM, con sus 394 archivos intactos;
el observador complementario espera el respaldo de la captura.
**Sin SAIF ni nueva estimación de consumo.**

Actualización 2026-10-16 09:14 UTC: [capa completa R1](../LLAMA32_LAYER_R1.md)
integrada y añadida al final, después del validador de memoria, PID 11872.
Control Icarus: 165 trabajos, 36 controles de transcripción y 12 mutaciones
detectadas por aserciones explícitas. Nueve tandas restauradas. Vivado
deberá sintetizar 238 fuentes con ocho familias HLS reales y ejecutar el
banco de control separado; dos CPU, 3.600 segundos, sin reintento automático.
Veintiocho propietarios verificados. Atención/FFN sigue activa, ciclo
19.775.825 observado a las 09:13 UTC. DDR conserva su turno después de
BRAM; sus 394 archivos volvieron a verificarse en D: y NFS. **Sin SAIF,
potencia nueva ni validación aritmética de la capa completa.**

Actualización 2026-10-16 08:38 UTC: [admisión de las 24 regiones de una capa](../LLAMA32_LAYER_PREFLIGHT_R1.md)
añadida al final, después de entrada/atención aritmética, PID 25120.
Icarus: 1.743 trabajos, 75 trazas corruptas rechazadas y 18 variantes RTL
detectadas. Vivado: síntesis OOC y tres semillas XSim, dos CPU, 30 minutos.
El wrapper de capa completa aún debe integrarse. Se conservan los turnos
existentes: atención/FFN sigue activa, ciclo 18.033.464 a las 08:37 UTC;
BRAM y actividad DDR permanecen a continuación. Sin SAIF ni potencia nueva.

Actualización 2026-10-16 08:13 UTC: [revisión complementaria DDR](POWER_ACTIVITY_DETAIL_R1.md)
conectada a un observador del respaldo terminal, PID 16980. No ocupa Vivado,
modifica la cola ni adelanta potencia. Quince pruebas sintéticas de integridad
y postprocesamiento pasaron en Windows/Linux. Se mantienen los 26 propietarios
y el orden atención/FFN → BRAM → DDR. Atención/FFN alcanzó 16.830.230 ciclos;
la captura DDR todavía no tiene elaboración ni SAIF.

Actualización 2026-10-16 07:56 UTC: [entrada→atención aritmética](../LLAMA32_INPUT_ATTENTION_REAL_R1.md)
añadida al final, tras síntesis/protocolo del wrapper, PID 24836. Ocho casos,
cuatro embeddings, cuatro HLS originales, 126 fuentes y 21 IP; XSim con
dos CPU y límite de ocho horas. Compilación de interfaces y 65 controles
sintéticos aprobados; aritmética conjunta aún pendiente. Veintiséis
propietarios, 75 entradas históricas, 28 paquetes y 1.373 vínculos verificados.
Atención/FFN sigue activa: 15.940.191 ciclos a las 07:55 UTC.
DDR conserva su turno tras BRAM y sus 394 archivos intactos.

Actualización 2026-10-16 07:27 UTC: [revisor complementario DDR](POWER_ACTIVITY_DETAIL_R1.md)
preparado y probado con datos sintéticos en Windows/Linux: 39 rechazos y
22 comprobaciones diagnósticas. Aplicarlo a la captura restaurada después
de la revisión R2 del coordinador existente; no añade propietario ni cambia
dependencias. Atención/FFN sigue activa, ciclo 14.370.338 a las 07:24 UTC.
DDR permanece después de BRAM, con sus 394 archivos intactos y sin SAIF.

Actualización 2026-10-16 07:06 UTC: [entrada→atención](../LLAMA32_INPUT_ATTENTION_R1.md)
cableada con productor, retención y consumidor reales. Protocolo Icarus:
894 trabajos, 150 controles y 16 mutantes detectados; cuatro tandas
restauradas. Síntesis de 125 fuentes y 21 IP originales, más XSim separado
del protocolo, al final después de consulta KV; PID 13140, dos CPU,
60 minutos. Todavía sin ejecución aritmética de esta composición.
Veinticinco propietarios, 74 entradas históricas, 27 paquetes y 1.276
vínculos intactos. Atención/FFN: 13.402.910 ciclos a las 07:05 UTC, activa.
DDR conserva su turno tras BRAM y sus 394 archivos verificados.

Actualización 2026-10-16 06:34 UTC: [consulta independiente KV](../LLAMA32_CACHE_QUERY_R1.md)
aprobada en Icarus: 97.920 consultas, 54 controles y ocho mutantes
detectados. Dos tandas restauradas. Síntesis del guard y XSim añadidos al
final tras el control de retención, PID 22492: 36 archivos, tres fuentes,
dos CPU, 30 minutos. Los wrappers sólo tienen compilación nativa.
Veinticuatro propietarios, 73 entradas históricas, 26 paquetes y 1.188
vínculos verificados. Atención/FFN sigue activa: 11.606.693 ciclos a las
06:29 UTC. DDR conserva su turno después de BRAM, con 394 archivos intactos.

Actualización 2026-10-16 06:13 UTC: [retención Q/K/V → RoPE/atención](../LLAMA32_QKV_ROPE_HANDOFF_R1.md)
aprobada en Icarus: 1.044 trabajos, 321 controles y 25 mutantes detectados.
Siete tandas restauradas. Síntesis y tres semillas XSim al final, después
de RoPE/atención aritmética, PID 2936: 46 archivos, dos fuentes, dos CPU,
30 minutos. Pares sintéticos; integración aritmética aún pendiente.
Veintitrés propietarios, 72 entradas históricas, 25 paquetes y 1.156
vínculos verificados. Atención/FFN sigue activa: 10.651.945 ciclos a las
06:09 UTC. DDR mantiene su turno tras BRAM y sus 394 archivos intactos.

Actualización 2026-10-16 05:40 UTC: [RoPE → KV → atención aritmética](../LLAMA32_ROPE_ATTENTION_REAL_R1.md)
encolada después de su síntesis, PID 18736. Ocho trabajos con cuatro
embeddings reales, cinco éxitos previstos y 160 cabezas; compilación de
interfaces aprobada y revisor sintético con 168 corrupciones rechazadas.
Paquete de 113 archivos, 67 fuentes y diez IP originales; sólo XSim,
dos CPU, 60 minutos. **Todavía sin resultado aritmético conjunto**.
Veintidós propietarios, 71 entradas históricas, 24 paquetes y 1.124 vínculos
verificados. Atención/FFN continúa activa: 9.131.528 ciclos observados a
las 05:38 UTC. DDR conserva su turno tras BRAM y sus 394 archivos intactos.

Actualización 2026-10-16 05:13 UTC: [unión RoPE/atención](../LLAMA32_ROPE_ATTENTION_R1.md)
con 151 puertos y arbitraje bind/comando. Admisión nativa: 18 rechazos
previstos, 45 vinculaciones, 189 controles y cinco mutantes detectados;
una variante redundante sobrevive y se documenta. Tres tandas restauradas.
Síntesis completa añadida después de reutilización KV, PID 26844: 66
fuentes, diez IP originales, dos CPU y 30 minutos. Aritmética conjunta
pendiente. Veintiún propietarios, 70 entradas históricas, 23 paquetes y
1.071 vínculos verificados. Atención/FFN sigue activa, ciclo 7.863.026 a las
05:12 UTC. DDR permanece tras BRAM, con 394 archivos revalidados y sin SAIF.

Actualización 2026-10-16 04:52 UTC: [reutilización KV entre capas](../LLAMA32_QKV_LAYER_REUSE_R1.md)
aprobada en Icarus: 174 trabajos, 313.344 palabras, 864 controles y seis
variantes RTL defectuosas detectadas. Dos tandas restauradas. Se añade XSim
al final, después de RoPE aritmético integrado, PID 26856: 40 archivos,
ocho fuentes, dos CPU y 30 minutos, sin repetir síntesis. Veinte propietarios,
69 entradas históricas, 22 paquetes y 1.023 vínculos verificados.
Atención/FFN sigue activa: ciclo 6.682.171 a las 04:49 UTC. DDR conserva
su turno después de BRAM; 394 archivos revalidados, sin SAIF nuevo.

Actualización 2026-10-16 04:25 UTC: [RoPE aritmético en captura/despacho](../LLAMA32_ROPE_QKV_REAL_R1.md)
queda encolado tras su síntesis exacta, PID 21452. Banco de 82 trabajos:
39 vectores originales, prefijos cortos poblados por append real y rechazo
obligatorio de prefijos lejanos incompletos. Compilación de interfaces y
revisor sintético verificados; **todavía no hay simulación aritmética nueva**.
Paquete de 76 archivos, 28 fuentes, cuatro IP; XSim solamente, dos CPU y
60 minutos. Diecinueve propietarios, 68 entradas históricas, 21 paquetes
y 987 vínculos intactos. Atención/FFN avanza hasta 5.425.966 ciclos a las
04:23 UTC. DDR conserva su turno tras BRAM; sus 394 archivos están intactos.

Actualización 2026-10-16 04:04 UTC: [RoPE/captura/despacho](../LLAMA32_ROPE_QKV_DISPATCH_R1.md)
aprueba 105 escenarios de protocolo, 324 controles y once fallos RTL
deliberados; una variante redundante en estos casos sobrevive y se registra.
Cinco tandas restauradas. La **síntesis con HLS RoPE original** se añade al
final, después de captura/despacho R1, PID 24044: 62 archivos, 24 fuentes,
cuatro IP, dos CPU, 30 minutos. No ejecutará simulación aritmética en esta
tanda. Dieciocho propietarios, 67 entradas históricas, 20 paquetes y 943
vínculos verificados. Atención/FFN sigue activa, ciclo 4.358.041 observado
a las 04:02 UTC; DDR mantiene su posición tras BRAM, sin SAIF nuevo.

Actualización 2026-10-16 03:33 UTC: [captura/despacho integrados](../LLAMA32_QKV_CAPTURE_DISPATCH_R1.md)
**aprobados en Icarus y encolados al final**, tras despacho R2. 93 trabajos,
49.152 palabras públicas reconstruidas desde los retornos de RAM/V,
258 controles de traza y diez mutantes. Se retiene cancelación en el traspaso
y exclusión de caché hasta liberar la fuente RoPE. Tres tandas restauradas.
Paquete de 44 archivos, ocho fuentes, dos CPU y 30 minutos; PID 9824.
La síntesis medirá captura/lector/despacho; RoPE y aritmética siguen simulados
por pares del banco. Diecisiete propietarios verificados, 66 entradas históricas,
19 paquetes y 912 vínculos intactos. Atención/FFN sigue en XSim; DDR conserva
su posición después de BRAM, con 394 archivos revalidados y sin SAIF nuevo.

Actualización 2026-10-16 03:04 UTC: [despacho Q/K/V](../LLAMA32_QKV_DISPATCH_R1.md)
**aprobado en Icarus y añadido al final de la cola**, detrás de captura Q/K/V.
144 escenarios, 267 controles de traza y 25 mutantes; cuatro tandas y fuentes
restauradas. El banco R4 añade aserciones que detectan el paso indebido de NaN
antes de que los hijos lo rechacen. RTL de producción intacto.
Paquete de 54 archivos, cinco fuentes, dos CPU y 30 minutos; PID 3996.
La síntesis será del despachador, con simulación del append/guardia/secuenciador
y pares aritméticos sintéticos. Dieciséis coordinadores, 65 entradas históricas,
18 paquetes y 874 vínculos verificados. Atención/FFN continúa; DDR conserva
su posición después de BRAM, con 394 archivos intactos y sin SAIF nuevo.

Actualización 2026-10-16 02:31 UTC: **cola reanudada tras corregir el banco**.
XSim reproducía una llamada a la función `void` de aserción aun con CE/WE
en cero. Comprobaciones directas conservan las 139 condiciones; doce casos
XSim/Icarus verifican guardas y errores deliberados. Cinco tandas restauradas.
Atención/FFN R4, PID 27592, ejecuta la simulación completa con banco R6;
reutiliza síntesis exacta respaldada y todavía no acredita aritmética integrada.
Quince coordinadores activos, 64 entradas históricas y 828 vínculos de
17 paquetes verificados. BRAM R3 exige ahora los 82 controles del revisor.
DDR R3, PID 11608, sigue después de BRAM: mismas 394 fuentes, sin SAIF nuevo.
Recuperación: `attention-macro-queue-recovery-20261016-r1`, SHA-256
`0c1b86d0558d49777525e9a01103f81199ac702ad210a3fced8aaf7987c21d92`.

Actualización 2026-10-16 02:03 UTC: atención/FFN R2 terminó con un fallo
de arranque en el ciclo 6; **no aprobada**. Diagnóstico restaurado completo
(775 resultados y 308 fuentes, SHA del TAR
`54c5dee36a0ba5cdf25ca3602b1bfaca46d0f2fd994208c02652f732cc91c28f`).
Los catorce sucesores se detuvieron sin ejecución. Se agregó una repetición
diagnóstica corta, sin síntesis y con observadores pasivos, como entrada 46;
PID 15640, tanda `20261016T020310Z-V7-attention-startup-probe-r2`.
Las 45 entradas anteriores permanecen. BRAM → actividad DDR → QKV → norma
→ RoPE → captura Q/K/V conservan su orden y sus requisitos de aprobación.
DDR no produjo SAIF ni potencia nueva.

Actualización 2026-10-16 01:37 UTC: [captura Q/K/V](../LLAMA32_QKV_SNAPSHOT_R1.md)
**aprobada en Icarus y encolada después de RoPE real**. 321 escenarios,
35.712 tramas verificadas, 177 corrupciones rechazadas y 23 mutantes
detectados por aserciones. Cuatro tandas restauradas. Paquete de 39 archivos,
tres fuentes, dos CPU, 30 minutos; PID 28052. El productor Q/K y la memoria
son sintéticos; falta conectar el append KV y el envío Q a atención.
R57 confirma quince coordinadores y una tanda atención/FFN activa;
17 paquetes y 767 vínculos intactos. DDR conserva su turno, sin SAIF nuevo.

Actualización 2026-10-16 01:23 UTC: [handoff post-O](../LLAMA32_POST_FFN_HANDOFF_R1.md)
**aprobado y respaldado**, 1.566 escenarios XSim y 309 controles de traza.
La composición aritmética atención/FFN ya inició
`20261016T012306Z-V7-attention-ffn-real-vivado-r2`, PID local 12600.
Le siguen BRAM y actividad DDR R1; el orden restante no cambió.

Actualización 2026-10-16 01:20 UTC: [FFN/residual real](../LLAMA32_FFN_RESIDUAL_REAL_R1.md)
**aprobada en XSim tras revisión R2, sin repetir herramientas**. 8.192 bytes
finales exactos y 64 controles negativos; WNS OOC −1,770 ns, timing pendiente.
El diagnóstico inicial y los quince coordinadores detenidos se conservan.
Los sucesores R2 retoman el mismo orden y reutilizan las fuentes selladas.
Handoff post-O activo (`20261016T011916Z-V7-llama32-post-ffn-handoff-vivado-r2`);
siguen atención/FFN, BRAM, actividad DDR, QKV, norma y RoPE. La recuperación
está respaldada/restaurada en D:/NFS. Auditoría R56: quince coordinadores;
16 paquetes y 736 vínculos verificados. DDR R2, PID 17120, conserva sus
394 fuentes y el límite inicial de veinte minutos; **sin SAIF todavía**.

Actualización 2026-10-16 00:57 UTC: [RoPE HLS con carga AXI](../LLAMA32_ROPE_INPUT_REAL_R1.md)
**sellado y encolado tras el protocolo RoPE**, PID 24812. Interfaces
compiladas, 43 controles sintéticos del revisor y 61 controles de puertos.
Paquete de 70 archivos, 21 fuentes y cuatro IP originales, D:/NFS con
restauración verificada. Los 69 casos y 107.648 valores calculados son
cobertura prevista; aún sin ejecución aritmética integrada. Dos CPU,
una hora y sin reintentos. R55 confirma dieciséis coordinadores y 686
vínculos intactos. FFN/residual sigue en XSim; DDR conserva su turno tras
BRAM, sin SAIF ni nueva estimación de potencia.

Actualización 2026-10-16 00:44 UTC: [cargador→propietario RoPE](../LLAMA32_ROPE_INPUT_PATH_R1.md)
**protocolo aprobado y encolado después del cargador**, PID 4840. Corrige
cancelación en el traspaso; 195 casos, 122.880 palabras, 99 controles de
traza y diez mutantes. Cinco tandas restauradas. Paquete de 50 archivos,
cinco fuentes, dos CPU y 30 minutos; núcleo HLS sintético. R54 confirma
quince coordinadores y 647 vínculos intactos. FFN/residual sigue activa.
Actividad DDR conserva su coordinador y turno tras BRAM; sus 394 archivos
se revalidaron en D:/NFS. Aún sin elaboración DDR, SAIF ni nueva potencia.

Actualización 2026-10-16 00:17 UTC: [cargador AXI RoPE R1](../LLAMA32_ROPE_LOADER_R1.md)
**nativo aprobado y encolado después del propietario RoPE**. Carga Q de
8 KiB, K de 2 KiB y fases de 256 bytes en dos RAM. 153 escenarios,
70.848 palabras contrastadas contra bytes AXI, 90 controles de traza y
19 mutantes detectados por aserciones. Dos tandas restauradas D:/NFS;
paquete de 30 archivos y tres fuentes, PID 27644, dos CPU y 30 minutos.
Espera aprobación/restauración del propietario. HLS y DDR físico siguen
pendientes. R53 confirma catorce coordinadores y 607 vínculos intactos.
FFN/residual sigue en XSim y actividad DDR conserva su turno tras BRAM.

Actualización 2026-10-16 00:07 UTC: [propiedad RAM RoPE R1](../LLAMA32_ROPE_OWNER_R1.md)
**nativo aprobado y Vivado encolado al final**, tras normalización→QKV real
aprobado/restaurado. 297 trabajos, 407.040 palabras FP32, 69 alteraciones
de traza y 29 mutantes compilados detectados por aserciones. Cuatro tandas
restauradas; paquete de 39 archivos en D:/NFS, PID 27656, dos CPU y límite
de 30 minutos. Los pares HLS del banco son sintéticos; la aritmética RoPE
original todavía no está conectada al propietario. Auditoría R52: trece
coordinadores y 581 vínculos intactos. FFN/residual sigue como única tanda
Vivado; actividad DDR conserva su lugar después de BRAM, todavía sin SAIF.

Actualización 2026-10-15 23:43 UTC: [normalización→Q/K/V con AXI real R1](../LLAMA32_INPUT_QKV_REAL_R1.md)
**sellada y encolada después del traspaso**, PID 7880. También exige
aprobación aritmética y restauración de ambos productores. Paquete de
215 archivos, 59 fuentes y once IP originales, verificado en D:/NFS.
Compilaron las interfaces; el revisor rechazó 85 corrupciones sintéticas
y tres formatos inválidos. Los doce trabajos y 112.640 bytes de escritura
son cobertura prevista, aún sin ejecución conjunta. Dos CPU, ocho horas
y ningún reintento automático. Auditoría R51: doce coordinadores vivos y
550 vínculos intactos. FFN/residual sigue como única ejecución Vivado;
la reserva DDR posterior a BRAM permanece sin cambios y todavía sin SAIF.

Actualización 2026-10-15 23:23 UTC: [traspaso normalización→Q/K/V R1](../LLAMA32_INPUT_QKV_HANDOFF_R1.md)
**control nativo aprobado y síntesis/XSim encolados tras normalización real**.
822 trabajos, 417 respuestas aceptadas, 174 alteraciones de traza rechazadas
y 28 mutantes detectados por aserciones. Siete tandas restauradas en D:/NFS;
paquete de 46 archivos, PID 26112, dos CPU y límite de 30 minutos. Requiere
aprobación/restauración del predecesor. Los hijos del banco son sintéticos;
aritmética y AXI conectados siguen pendientes. La FFN activa y la reserva
de actividad DDR posterior a BRAM se conservan; todavía sin SAIF nuevo.
Auditoría R50 a las 23:23 UTC: once coordinadores vivos, 485 vínculos de
once paquetes sin cambios. FFN/residual avanzó al ciclo 10.817.624 de fase 1.

Actualización 2026-10-15 22:58 UTC: [normalización aritmética sobre AXI R1](../LLAMA32_INPUT_NORM_REAL_R1.md)
**sellada y encolada después del recorrido de protocolo**. Paquete de
93 archivos, 36 fuentes reales y siete IP originales, restaurado en D:/NFS.
Firmas e interfaces comprobadas sin ejecutar sustitutos; 58 alteraciones
de puertos y 51 corrupciones sintéticas rechazadas. Los 20 trabajos,
8.704 palabras calculadas y 49.152 bytes aceptados son cobertura prevista,
todavía no resultados RTL. PID 9808, dos CPU y límite de dos horas; exige
aprobación/restauración del predecesor y preserva la reserva DDR.
Auditoría R49: diez coordinadores vivos, FFN/residual como única ejecución
Vivado; diez paquetes anteriores/nuevos conservan sus archivos vinculados.

Actualización 2026-10-15 22:47 UTC: [normalización por AXI R1](../LLAMA32_INPUT_NORM_PATH_R1.md)
**protocolo nativo aprobado y Vivado encolado tras su propietario RAM**.
198 trabajos, 430.080 bytes ABI1 aceptados, 66 controles de traza y 14
mutantes detectados por aserciones. Dos tandas restauradas en D:/NFS;
paquete de 51 archivos y siete fuentes, PID 16564. Espera aprobación y
respaldo del propietario; dos CPU y 30 minutos para síntesis/XSim.
Auditoría R48: nueve coordinadores vivos, 412 vínculos de nueve paquetes
sin cambios. FFN/residual sigue en XSim, ciclo 7.363.026 de fase 1.
Actividad DDR conserva su lugar posterior a BRAM; todavía sin SAIF.
El peer HLS reproduce referencias: la aritmética conectada sigue pendiente.

Actualización 2026-10-15 22:28 UTC: [propiedad de normalización de entrada](../LLAMA32_INPUT_NORM_OWNER_R1.md)
**nativo aprobado y Vivado encolado después de Q/K/V aritmético**. Banco R4:
243 trabajos, 66.048 palabras ABI1, 63 controles de traza y 21 mutantes
detectados por aserciones. Seis tandas restauradas, incluida la primera
tanda de mutantes insuficiente. Paquete de 40 archivos D:/NFS, PID 26776;
dos CPU y 30 minutos para síntesis/XSim de protocolo. La dependencia Q/K/V
ordena el slot y exige respaldo/restauración, sin aprobar un diagnóstico.
Auditoría R47: ocho coordinadores vivos; FFN/residual sigue activa y la
reserva DDR permanece después de BRAM. Norm/AXI/QKV conectado aún pendiente.

Actualización 2026-10-15 22:11 UTC: [Q/K/V aritmético R1](../LLAMA32_QKV_PRODUCER_R1.md)
**sellado y encolado después de control Q/K/V**. Paquete de 145 archivos,
23 fuentes reales y cuatro IP AMD; D:/NFS y restauración verificados.
Preflight de jerarquía compilado sin ejecutar la firma MAC; revisor probado
con 53 corrupciones sintéticas. Los 19.968 valores FP64 y 79.872 bytes FP32
son cobertura esperada, no medidas RTL. Coordinador PID 19224; requiere
aprobación y respaldo del control, usa dos CPU y límite de ocho horas.
Auditoría R46: siete propietarios vivos; FFN/residual real sigue como única
tanda activa. Actividad DDR conserva su lugar después de BRAM, sin SAIF aún.

Actualización 2026-10-15 21:51 UTC: [control Q/K/V R1](../LLAMA32_QKV_PRODUCER_R1.md)
**aprobado en control nativo y encolado** después de actividad DDR. RTL R2,
banco R6: 481 trabajos, 91 alteraciones de traza rechazadas y 29 mutantes
compilados detectados. Cuatro tandas conservadas y restauradas. Paquete de
35 archivos verificado en D:/NFS, coordinador PID 25708. Síntesis OOC a
5 ns y seis suites XSim pendientes, dos CPU y límite de 30 minutos.
La dependencia DDR sólo ordena el slot y exige cierre/respaldo/restauración;
su diagnóstico no se convierte en aprobación de potencia. Auditoría R45:
seis coordinadores vivos; FFN/residual real sigue como única tanda activa.

Actualización 2026-10-15 21:39 UTC: [actividad DDR R1](POWER_ACTIVITY_R1.md)
**sellada y encolada después de BRAM**. Paquete completo de 394 archivos,
restaurado localmente y verificado en NFS. Coordinador PID 25756; espera
revisión y respaldo del predecesor y slot vacío. La FFN/residual real
`20261015T211706Z` sigue en XSim. Se preservó toda la cadena anterior.
Dos CPU, límite de 20 minutos para la invocación inicial, sin reintento;
éxito parcial, fallo y timeout se archivan y restauran. El revisor SAIF R2
pasó doce controles sintéticos. **Sin elaboración, SAIF ni nueva potencia
todavía**; refresh/ranks y correspondencia con el DCP siguen pendientes.

Actualización 2026-10-15 21:07 UTC: [potencia DDR](POWER_PROBE_RESULTS_R1.md)
exportada, revisada y respaldada; tanda `20261015T205842Z-V8-ddr-power-probe-r2`.
Estimación genérica 3,015 W, confianza baja, sin SAIF; no califica la placa.
R1 falló por comparar NAME con TOP y permanece archivado. El sucesor R2
conservó DCP, parte y 154 puertos. Tres intentos preservados en D:/NFS;
44 archivos de la tanda válida restaurados y verificados. A las 21:07:39 UTC
se liberó el slot y se reanudó el coordinador residual real PID 23708.
Los seis sucesores mantienen paquetes, dependencias y 316 archivos vinculados.

Reserva 2026-10-15 20:47 UTC: ejecutar el
[diagnóstico de potencia DDR R1](POWER_PROBE_HANDOFF_R1.md) desde el hilo
principal tras terminar, revisar y respaldar el escritor residual R1
(`20261015T204443Z`). Únicamente se retuvo el coordinador sucesor, todavía
en espera (PID 23708); la validación activa continúa. No se modificaron
fuentes, paquetes ni dependencias y no se creó otro coordinador.
Tras archivar y restaurar el diagnóstico se libera el sucesor y continúa
la misma cadena. Registro operativo:
`build/cloud-setup/20261002-board/power-probe-r1-main-20261015T204728Z/reservation.json`.

Actualización 2026-10-15 20:36 UTC: [FFN completa con propiedad R4](../LLAMA32_OWNED_FFN_RESULTS_R4.md)
**aprobada y respaldada**: 20.332.665 ciclos, 155.648 bytes escritos
verificados y salida final de 8.192 bytes. El cargador residual R2 inició
`20261015T203547Z-V7-llama32-residual-loader-vivado-r2` tras cerrar el
respaldo. Sigue intacta la cadena hasta atención/FFN/residual y BRAM.
La [referencia Q/K/V completa](../LLAMA32_QKV_PRODUCER_R1.md) también pasó
revisión numérica; su productor RTL está pendiente y aún no está encolado.

Actualización 2026-10-15 20:26 UTC: [lectura privada de BRAM R1](../LLAMA32_PROJECTED_ROOT_BRAM_R1.md)
**sellada y encolada** después de atención/FFN/residual real R1. Propietario
R3 aprobado en 219 trabajos, 1.191.819 muestras de equivalencia pública,
99 controles y seis mutantes. Jerarquía sucesora elaborada; aritmética y
ruteo nuevos pendientes. Diez coordinadores activos, PID nuevo 10204.
La tanda exige XSim numérico y revisión independiente antes de ruteo y
physopt a 5 ns. Conserva la FFN activa, el único slot y respaldo por tanda.

Actualización 2026-10-15 20:06 UTC: [atención/FFN/residual real R1](../LLAMA32_ATTENTION_FFN_REAL_R1.md)
**sellado y encolado** detrás de post-O handoff R1. Banco R5, 179 fuentes
originales (56 RTL, 122 HLS y TB), 27 IP y pesos Q8 completos. Revisor
integrado probado con 78 alteraciones sintéticas; ejecución real pendiente.
Coordinador PID 9668; nueve propietarios locales confirmados a las 20:05
UTC. Exige revisiones y respaldos previos, síntesis OOC a 5 ns y XSIM
completo con dos CPU y límite de ocho horas. La FFN R4 sigue activa.

Actualización 2026-10-15 19:50 UTC: [banco integrado atención/FFN/residual R3](../LLAMA32_ATTENTION_FFN_REAL_R1.md)
escrito y elaborado con firmas aritméticas (71 fuentes, sin ejecución).
Fixtures del mismo token emparejados byte a byte; revisor upstream probado
contra una traza vendor real y 70 alteraciones. Faltan revisión integrada,
IP vendor y paquete real antes de encolar tras post-O handoff R1.
La auditoría de las 19:48 UTC confirma los ocho coordinadores activos y
la FFN R4 en ejecución. Se conservan dependencias, único slot y respaldos.

Actualizaci?n 2026-10-15 19:27 UTC: [wrapper atenci?n ? FFN/residual R4](../LLAMA32_ATTENTION_FFN_CONTROL_R1.md)
aprobado en control nativo: 3.102 trabajos, 222 controles y 34 mutantes.
Vivado elabor? su jerarqu?a con 70 fuentes y cinco firmas aritm?ticas;
no ejecut? aritm?tica. El inventario para el banco real re?ne 178 fuentes
sin sustitutos. La tanda integrada est? **en preparaci?n**, todav?a sin
paquete ejecutable ni coordinador. Debe seguir al handoff post-O ya
en cola y conservar revisi?n y respaldo entre etapas. Ocho coordinadores
siguen activos; FFN R4 avanz? a fase 3 a las 19:24 UTC.

Actualización 2026-10-15 18:58 UTC: [propiedad post-O → FFN/residual](../LLAMA32_POST_FFN_HANDOFF_R1.md)
**aprobada en control nativo y encolada** después de FFN/residual real R1.
RTL R2 y banco R4: 1.566 trabajos, 309 controles y 41 mutantes detectados
por aserción directa. Protege ocho asignaciones del productor frente a
cinco destinos y retiene ambos propietarios hasta el ACK final.
Coordinador PID 23840; espera revisión y respaldo de su predecesor.
Mantiene un único slot, síntesis OOC a 5 ns, tres semillas XSim y respaldo
por tanda. No acredita aún productor vivo, aritmética ni transporte AXI.

Actualización 2026-10-15 18:35 UTC: [FFN → residual aritmético R1](../LLAMA32_FFN_RESIDUAL_REAL_R1.md)
**encolado** después del traspaso R1. El coordinador exige aprobación y
respaldo de éste y de la FFN R4; después ejecuta síntesis OOC a 5 ns y
XSim completo, dos CPU y límite de ocho horas. Paquete sellado con 68
fuentes reales, pesos Q8 completos, fixture post-O ABI1 y residual exacto.
Sin resultado integrado todavía. Se conservan las validaciones activas,
los criterios V0–V9 y la evidencia/respaldo por tanda.

Actualización 2026-10-15 18:17 UTC: el [wrapper FFN → residual R3](../LLAMA32_FFN_RESIDUAL_CONTROL_R1.md)
superó el banco nativo de control: 288 trabajos, 186 controles de trazas
y 20 mutantes RTL. La etapa aritmética completa sigue **en preparación**:
requiere TB con pesos originales, lecturas del resultado FFN confirmado
por B, revisor independiente y paquete sin peers sustitutos. Una vez
sellada, debe depender del traspaso R1 ya encolado. No es todavía una
etapa ejecutable ni una aprobación de aritmética. Se conserva la FFN R4
activa y toda su cadena de dependencias y respaldos.

Ejecución iniciada el 2026-10-02: consultar [estado, evidencias y respaldos](VIVADO_STATUS.md). Los criterios siguientes se conservan como contrato de la cola; el estado actual se registra allí.

Extensión V7 encolada el 2026-10-15: [propiedad FFN de 64 bits y residual
retenido](../LLAMA32_FFN_OWNER_R1.md), después de unión atención/post-O,
revisión del resultado archivado y optimización física de la raíz. Mantiene
un único slot Vivado, preflight nativo con mutantes y respaldo por tanda.
No sustituye las dependencias DDR4/PCIe de V5–V8.

Actualización 2026-10-15 16:46 UTC: atención/post-O y
[optimización física raíz R5](../LLAMA32_PROJECTED_ROOT_PHYSOPT_R1.md)
revisadas y respaldadas. Esta última conserva setup negativo de −0,034 ns.
Cadena vigente: propietario FFN Vivado R4 → [FFN real con propietario R4](../LLAMA32_OWNED_FFN_R1.md)
→ [cargador residual R2](../LLAMA32_RESIDUAL_LOADER_R1.md)
→ [propietario residual R1](../LLAMA32_RESIDUAL_OWNER_R1.md)
→ [escritor residual R1](../LLAMA32_RESIDUAL_STORE_R1.md).
Los sucesores conservan fuentes compiladas y fixtures; la revisión física
corregida usa el encabezado real `Physopt postRoute`, sin repetir herramientas.
Un solo slot, revisión y respaldo verificado entre etapas.

Avance 16:50 UTC: propietario FFN R4 aprobado y respaldado; FFN completa
R4 inició `20261015T164923Z`. El orden de los dos bloques residuales se conserva.

Extensión 17:04 UTC: escritor residual al final de la cadena, después de
246 casos nativos, 117 controles y 21 mutantes. Publica 8.192 bytes sólo
tras todos los B correctos. El banco usa resultados de referencia en RAM;
la unión al HLS real y a FFN viva es la próxima prueba de integración.

Extensión 17:28 UTC: [composición residual con HLS real](../LLAMA32_RESIDUAL_REAL_R1.md)
después del escritor R1. Preflight: 144 trabajos, 147 controles y 12
mutantes, con 17 fuentes compiladas y cuatro pares post-FFN reales.
Espera aprobación y respaldo del escritor; luego síntesis a 5 ns y tres
semillas XSim, con límite de 1.800 segundos. La FFN completa en curso
se conserva. Sigue pendiente conectar su salida viva a esta composición.

Extensión 17:46 UTC: [propiedad FFN → residual](../LLAMA32_FFN_RESIDUAL_HANDOFF_R1.md)
después de la composición residual real R1. Controlador con 342 casos
nativos, 165 controles y 24 mutantes; síntesis OOC y tres semillas XSim
con peers de completion controlados. La unión aritmética y el mux AXI
completo quedan como etapa posterior, manteniendo el caso FFN real.

Preflight residual: cargador con 114 casos, 78 controles y once mutantes;
propietario con 219 casos, 63 controles y 17 mutantes. Los bancos de estos
controladores usan peers sintéticos. La FFN completa consume ABI1 post-O
de referencia; sigue pendiente el productor vivo, la segunda suma integrada
y V7 sobre DDR real.

Preparada el 2026-10-02. **Pendiente de ejecución en Vivado**: esta conversación lateral hizo investigación, inventario de archivos AMD y herramientas locales. No ejecutó MIG, PCIe, DRC ni implementación de la tarjeta. Encolar estas tareas conservando las validaciones que ya administra el hilo principal.

## Orden lista para trasladar

> Ejecuta la cola de `docs/fpga/board/VIVADO_QUEUE.md` en orden de dependencias. Empieza por V0–V3: entorno, partes instaladas, DIMM y un canal DDR4. Usa proyectos y carpetas nuevos, conserva los trabajos en curso y respalda cada tanda localmente. Prioriza resolver el banco 65, PERST y la coexistencia DDR4/PCIe antes de una implementación larga. No des por validados los scripts probados con mocks ni el pinout documental. Registra resultados, bloqueos y evidencias por ID; continúa con las tareas independientes si falta un componente o modelo. La meta de esta cola es entregar un pinout implementado y reproducible para el prototipo de banco, más una ruta explícita de integración y ensayo físico.

Esta orden se entrega como documentación; no se envió a otro agente ni se programaron ejecuciones.

## Contexto y archivos de entrada

- **Formato confirmado:** prototipo de banco, con espacio para DDR4, refrigeración y medidas. Objetivo de diseño: 512 GiB mediante cuatro canales independientes de 128 GiB, cada uno x64 útil más ECC.
- **Primera topología a calificar:** dos RDIMM de 64 GiB 2Rx4 por canal. `MTA36ASF8G72PZ-3G2F1` es candidato documental, pendiente de ficha, SPD, modelo y soporte exactos. Un RDIMM 3DS de 128 GiB es alternativa separada; no intercambiar configuraciones.
- **FPGA:** empezar con `XCVU5P/FLVA2104`; considerar `XCVU9P/FLGA2104` si faltan recursos o no cierra implementación. Velocidad y temperatura siguen pendientes. El ZU7EV de las pruebas HLS no es el componente aprobado para esta placa.
- **PCIe:** objetivo Gen3 x16; arranque convencional por JTAG/QSPI x4, sin Tandem en el reparto actual. No se exige un BAR del tamaño de toda la DDR.
- `DDR0` a `DDR3` identifican canales; todos usan tecnología DDR4. «DDR2» en esta cola significa el canal número 2.
- Leer [arquitectura](ARCHITECTURE.md), [viabilidad de memoria](MEMORY_FEASIBILITY.md), [plan de E/S](IO_PLAN.md), [manifiesto](io_plan.json) y [paquete de inventario](VIVADO_HANDOFF.md).
- Consultar el estado vigente en [POC_ROADMAP](../POC_ROADMAP.md). Para integración: [lector AXI](../AXI_READER.md), [buffers](../WEIGHT_TILE_BUFFER.md), [camino Q8](../Q8_UNPACKER.md), [FFN](../HLS_FFN.md) y [Llama 1B](../LLAMA32_1B_HARDWARE.md). Verificar fuentes actuales; estos bloques pueden haber avanzado desde la preparación de la cola.

Ya se comprobaron 2.104 bolas y 832 E/S HP por candidato contra el ZIP AMD. Las hojas JSON generadas están en `build/ntrpu-board-study-20261002/`; esa carpeta puede no viajar con Git. Copiarla expresamente o regenerarla según el handoff. El script de inventario Vivado solo tiene pruebas de flujo con comandos simulados.

## Dependencias y prioridad

Todos los pasos comienzan **pendientes**, salvo la investigación local indicada arriba.

| ID | Trabajo | Depende de | Resultado que permite avanzar |
| --- | --- | --- | --- |
| V0 | Entorno, estado y respaldo | — | Herramientas y salida persistente identificadas |
| V1 | Inventario real de dispositivo/IP | V0 | Parte completa disponible y exportación revisada |
| V2 | Calificar DIMM y controlador | V1 | Configuración exacta o bloqueo reproducible |
| V3 | Un canal MIG: pines y simulación | V2 | Canal funcional en simulación y pinout aceptado |
| V4 | PCIe, PERST y banco 65 | V1; V2/V3 para cierre | Reset y bolas compatibles con DDR2 |
| V5 | Cuatro DDR4 + PCIe | V3, V4 | Top conjunto sin conflictos de interfaces |
| V6 | Implementación de interfaces | V5 | Colocación/ruteo y restricciones revisados |
| V7 | Integración del cómputo NTRPU | V3; estado RTL principal | Transferencias y resultados correctos |
| V8 | Implementación integrada y ECAD | V6, V7 | Paquete de pinout y diseño reproducible |
| V9 | Bring-up físico | V8 y hardware disponible | Resultados medidos en placa |

V4 puede comenzar con un PCIe aislado mientras se obtienen datos de memoria. V7 puede prepararse contra memoria simulada. Su cierre sobre DDR real y la implementación integrada requieren las dependencias restantes.

## V0 — Preparar una tanda aislada

1. Identificar VM, volumen montado, instalación y trabajos activos a partir del estado actual del hilo principal. `/lambda/nfs/vivado` y Vivado/Vitis 2026.1 constan en el historial; verificar su presencia sin asumir una IP SSH ni rutas vigentes.
2. Cargar el `settings64.sh` real de Vivado y registrar `vivado -version`, sistema, memoria/disco disponibles y versión de Python. El inspector de encapsulados necesita Python 3.11+; el Tcl no depende de Python.
3. Comprobar soporte Virtex UltraScale+ y licencia para la parte/IP seleccionadas. Conservar el diagnóstico y las restricciones de evaluación que informe la herramienta, sin incluir claves, tokens ni archivos `.lic` en los entregables versionados.
4. Trabajar en una copia identificada del árbol, incluyendo archivos nuevos no seguidos por Git. Guardar commit, diferencias y hashes de fuentes: el commit solo no identifica este trabajo en curso.
5. Elegir carpeta nueva por tanda, límites de recursos y presupuesto de ejecución según la máquina disponible. No lanzar síntesis/ruteo pesados hasta pasar los filtros anteriores; no cancelar ni reiniciar el banco FFN existente para hacer sitio.

## V1 — Ejecutar el inventario real

Ejemplo **Bash desde la raíz del repositorio**, después de cargar Vivado. Sustituir `RUN_ID_NUEVO` y la parte; usar un identificador nuevo en cada intento:

```bash
set -euo pipefail
NTRPU_BOARD_RUN="build/ntrpu-board-vivado/RUN_ID_NUEVO"
mkdir -p build/ntrpu-board-vivado
mkdir "$NTRPU_BOARD_RUN"
vivado -mode batch -source tools/ntrpu_board_inventory.tcl \
  -log "$NTRPU_BOARD_RUN/discover.log" \
  -journal "$NTRPU_BOARD_RUN/discover.jou" \
  -tclargs discover "$NTRPU_BOARD_RUN/parts" \
  > "$NTRPU_BOARD_RUN/discover.console.log" 2>&1
```

Leer `parts/available_parts.txt`. Seleccionar una parte exacta justificando velocidad/temperatura; no tomar automáticamente la primera de la lista.

```bash
NTRPU_PART='COPIAR_PARTE_COMPLETA_DEL_INVENTARIO'
vivado -mode batch -source tools/ntrpu_board_inventory.tcl \
  -log "$NTRPU_BOARD_RUN/inventory.log" \
  -journal "$NTRPU_BOARD_RUN/inventory.jou" \
  -tclargs "$NTRPU_PART" "$NTRPU_BOARD_RUN/inventory" \
  > "$NTRPU_BOARD_RUN/inventory.console.log" 2>&1
```

Revisar parte, bancos, SLR, propiedades de bolas y catálogo DDR4/PCIe/XDMA exportados. Comparar con `io_plan.json` y los archivos AMD; investigar diferencias, incluidos pines que Vivado no representa como E/S de usuario. `INVENTORY_COMPLETE.txt` acredita exportación, no viabilidad de la placa. Si el script falla en Vivado, conservar el error y corregirlo en una nueva revisión antes de continuar.

## V2 — Cerrar los parámetros de memoria

Preparar una ficha de configuración por población: fabricante/parte/revisión, capacidad en bytes, tipo RDIMM, chips x4, densidad, ranks, slots, geometría, ECC, tasa, timings, reloj y reset. El objetivo analítico DDR4-2400 no es una tasa ya validada.

Obtener ficha, SPD y modelo de simulación correspondientes. Si falta una parte soportada, estudiar la opción personalizada documentada por AMD y registrar qué datos faltan; una memoria genérica puede servir para explorar recursos, pero no homologa el DIMM objetivo. Ver [PG150: partes personalizadas](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Basic-Tab?contentId=MGveBVBHGBtHBQNV7JSeVA).

Crear un proyecto de IP separado del inventario y de HLS. Consultar las propiedades expuestas por la IP instalada antes de escribir su Tcl; guardar VLNV, parámetros y `.xci`. Verificar capacidad, ECC y ancho AXI generado: 128 GiB por canal requieren 37 bits de dirección de byte; la interfaz útil propuesta es de hasta 512 bits. No reducir capacidad, ECC o slots silenciosamente para que la generación pase. Mantener configuraciones distintas para un módulo y para dos.

## V3 — Validar un canal y su colocación

Empezar con un canal sencillo; llevar después esa configuración al grupo DDR2 que contiene el banco 65. Usar el diseño de ejemplo generado por la versión instalada y el modelo correspondiente. Guardar scripts reproducibles de generación, simulación y síntesis fuera de contexto.

Verificar byte lanes/nibbles x4, DQ/DQS, dirección/control, clocks, reset, referencias y tensiones; comprobar que el planificador acepta la terna de bancos. Los bancos propuestos son reservas, no constraints definitivos. Exportar pinout/XDC y revisar todos los avisos antes de expandir a cuatro canales.

En simulación exigir calibración y comparación de datos, escrituras/lecturas, máscaras con ECC si están habilitadas, cambios de rank/slot, límites de ráfaga y backpressure. Comprobar direcciones por encima de 4 GiB y cerca del extremo de 128 GiB, evitando alias por truncamiento. Usar patrones dispersos y un modelo adecuado; documentar los límites si no puede representar toda la geometría. Probar corrección/detección ECC donde la IP y el modelo permitan inyección. Una inicialización acelerada de simulación no mide tiempo de arranque físico.

## V4 — Resolver PCIe y reset antes de rutear

Generar la IP PCIe elegida para el bloque del [plan](IO_PLAN.md): `PCIE4_X1Y0` en VU5P o `PCIE4_X1Y2` en VU9P, con selección x16 de lane 0 en quad 227. Confirmar los quads realmente ocupados, orden completo de lanes, referencia de reloj y pines del diseño de ejemplo. Elegir y documentar la capa DMA/control; un bloque PCIe integrado no aporta por sí solo todo el DMA de NTRPU.

**Cerrar específicamente banco 65 y PERST:** AM17 (`PERSTN0`) y AP17 (`PERSTN1`) están en el banco de DDR2. Determinar qué ruta usa la IP, si MIG deja la bola necesaria disponible y cómo se resuelve la tensión del reset host. Evaluar alternativas de colocación o ruta solo si las herramientas/IP las admiten; guardar evidencia conjunta. Si no cierra, registrar el conflicto y revisar bancos/encapsulado. Mantener Tandem desactivado en esta arquitectura; activarlo exige replanificar.

Definir BAR de control, ventanas si hacen falta, colas, interrupciones, anchos de direcciones DMA y comportamiento ante reset. La memoria interna total necesita 39 bits de dirección de byte; conservar el protocolo host de 64 bits y validar rangos antes de estrecharlos. Documentar carga inicial de pesos y lectura de resultados. Definir secuencia de configuración, reloj, `PERST#` y enumeración; comprobarla con el modelo PCIe y dejar el ensayo del host físico para V9.

## V5–V6 — Integrar e implementar las interfaces

Construir un top con cuatro controladores independientes, PCIe/DMA, arbitraje, generadores/verificadores de tráfico y estado de calibración/error. Dar actividad y observabilidad a cada canal para evitar que síntesis elimine interfaces. Revisar recursos remanentes para el motor NTRPU; un top de interfaces que cabe todavía no demuestra que cabe el cómputo.

Definir dominios de reloj y reset, puentes/FIFOs CDC, arbitraje y bloqueo de accesos hasta calibración. El objetivo de cuatro caminos paralelos no debe convertirse accidentalmente en un único puerto limitante. Probar aislamiento de canales, rangos fuera de memoria, tráfico simultáneo, errores y reinicios coordinados.

Sintetizar, colocar y rutear. Mantener las restricciones generadas por las IP y añadir las de placa con procedencia de clocks, jitter y retardos externos. Revisar setup/hold, pulsos, rutas sin reloj/restricción, CDC, cruces SLR, congestión y pines no asignados. No ocultar fallos con excepciones temporales generales o degradando DRC. Si faltan datos de PCB, etiquetar el resultado como provisional y listar qué comprobaciones dependen de ellos.

## V7 — Conectar el motor NTRPU

Coordinar con el hilo principal para reutilizar las revisiones RTL/HLS vigentes y sus bancos. Integrar primero un camino DDR → lector AXI → buffers → Q8 → cómputo; añadir carga/escritura desde el host, salida de resultados y control. El lector existente solo cubre AR/R: no asumir que implementa la escritura de pesos o resultados.

Revisar adaptación de anchos, acumulación entre tiles, CDC, rangos registrados, épocas, finalización y errores tardíos sin publicar resultados parciales. Definir timeout/cancelación y recuperación coordinada si todavía faltan. Comparar contra referencia numérica independiente; empezar con fixtures pequeños de Brandon y después dimensiones Llama 1B. Pesos reales y revisión del modelo deben quedar identificados; no denominar «inferencia Llama» a pruebas de transporte o matrices sintéticas.

Medir ciclos, pausas, ocupación y ancho útil por etapa en simulación. Replicar motores/canales a partir del cuello de botella observado. Mantener separados los techos analíticos, medidas de simulación y rendimiento físico.

## V8 — Evidencia, potencia y entrega a ECAD

Repetir implementación con el cómputo integrado. Guardar, según el estado abierto del diseño, `report_ip_status`, `report_utilization`, `report_io`, `report_drc`, `report_methodology`, `check_timing`, `report_timing_summary`, `report_clock_interaction`, `report_cdc`, `report_route_status` y `report_power`. Consultar `help` en la versión instalada para opciones y requisitos de cada comando. La [referencia UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands) documenta esos comandos; [report_ip_status](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_ip_status) permite registrar el estado de licencias de las IP.

Conservar Tcl de reconstrucción, fuentes, XCI/BD si aplica, XDC, checkpoints de síntesis/ruteo, configuración de simulación, logs y resultados numéricos. Exportar la tabla `bola → función → banco/SLR → puerto → net PCB → tensión → destino → estado de revisión`, incluyendo alimentación y pines dedicados. Mantener sincronizados todos los cambios de pines entre Vivado y ECAD.

Estimar potencia con actividad representativa y supuestos registrados; añadir aparte DIMM, reguladores, pérdidas y refrigeración. Preparar modelos IBIS y retardos de encapsulado cuando correspondan para SI; no fijar impedancias, longitudes o capas solo a partir del reporte de timing. Entregar a ECAD las restricciones y las incógnitas que debe devolver al diseño. Tras fijar esos datos de placa, revalidar restricciones e implementación antes de liberar fabricación.

Generar bitstream solo cuando la configuración y las licencias lo permitan y las revisiones aplicables estén resueltas. Registrar cualquier limitación de evaluación: crear un `.bit` no demuestra permiso de producción ni funcionamiento físico.

## V9 — Ensayo posterior en hardware

Seguir el [orden de encendido](IO_PLAN.md): alimentación, JTAG/configuración, un DIMM, dos DIMM, cuatro canales, PCIe/DMA y cómputo. Preparar ILA y registros para calibración, errores ECC, resets, enlace y transferencias. Medir capacidad/ranks, integridad de datos, enumeración repetida, caudal sostenido, consumo y temperatura. Estos resultados requieren una placa; no pueden cerrarse exclusivamente con la VM.

Dos tarjetas, P2P y agregación de pesos serán una tanda posterior según [NTRPU Fabric](../NTRPU_FABRIC.md). Sus BAR no equivalen automáticamente a memoria coherente ni a un enlace físico entre tarjetas.

## Registro y respaldo obligatorio de cada tanda

Crear `result.md` y un manifiesto de hashes por ejecución: ID V0–V9, estado `pendiente/en_curso/aprobado/bloqueado`, fecha, parte, versión/build de herramientas e IP, revisión/hash de fuentes y parámetros, comandos, código de salida, marcas de éxito, advertencias, duración y siguiente acción. Un código 0 sin comparación de resultados no cierra una prueba funcional.

Guardar primero en el volumen persistente confirmado por el agente principal y copiar después a una carpeta local nueva de `artifacts/ntrpu-board-vivado/<run_id>/`. Verificar hashes y legibilidad del archivo antes de considerar completo el respaldo. Conservar también intentos fallidos: no son evidencia de éxito, pero evitan repetir trabajo costoso. No apagar la instancia hasta que el hilo principal haya confirmado el respaldo de todas sus tandas, incluidas las ajenas a esta cola.

**Cierre de esta cola de diseño:** V0–V8 cumplen sus criterios con evidencia revisable y el paquete ECAD identifica pendientes eléctricos/físicos; V9 queda separado hasta disponer de hardware. Un resultado provisional por falta de restricciones de placa no cierra la implementación final: requiere revisión tras recibir esos datos. Si una combinación no es viable, entregar el bloqueo reproducible y la alternativa investigada sin marcar aprobada la tarjeta.
