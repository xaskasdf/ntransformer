# Ejecución de la cola Vivado

## Corte local tras perder acceso — 2026-10-16

El usuario informa que se perdió el acceso a la VM; no se realizó una nueva
comprobación remota. La rama DDR se detuvo antes de R2 (`vocab-memory` requiere
inspección) y la rama SAIF/DCP antes de asociar actividad (el controlador de
token requiere inspección). La aritmética real de normalización sí está
revisada y respaldada; Q/K/V terminó en diagnóstico con respaldo, sin métricas
aceptadas. No existe una estimación nueva de consumo. Consulte el
[traspaso](VIVADO_ACCESS_LOSS_HANDOFF_20261016.md).

Actualización 2026-10-16 15:17 UTC: normalización aritmética real mantiene
runner 1022813 y XSim 1025162; todavía sin resultado terminal. Las auditorías
verificaron 31 propietarios y 2.064 vínculos de cuarenta paquetes.
La [asociación SAIF/DCP](POWER_SAIF_MAPPING_R1.md) quedó al final de la cola,
con catorce controles Windows/Linux y entradas verificadas. Su futura
salida será evidencia de correspondencia, no una calificación de consumo.
**DDR R2 conserva el turno y aún no hay vatios nuevos.**

Actualización 2026-10-16 15:01 UTC: [Q/K/V terminó](../LLAMA32_QKV_MEMORY_RESULTS_R2.md),
con 19.968 resultados FP64 y 79.872 bytes FP32 exactos. La revisión
complementaria conserva el diagnóstico R1 y la cobertura RVALID pendiente;
captura y suplemento restaurados en D:/NFS. Control y ruta de normalización
pasaron y quedaron respaldados; ahora ejecuta normalización aritmética real.

El [controlador de token](../LLAMA32_TOKEN_GUARD_R1.md) pasó 1.182 trabajos,
42 controles de traza y veinte mutaciones RTL con pares sintéticos.
Síntesis/protocolo encolados al final, PID 24452: 86 entradas y treinta
propietarios auditados, 2.047 vínculos intactos en 39 paquetes. La reparación
de ancestros de entrada/QKV usa un observador separado y no cambia fuentes
selladas. **DDR R2 mantiene su turno; potencia aún sin encolar ni estimar.**

Actualización 2026-10-16 14:21 UTC: Q/K/V continúa en el trabajo 9, ciclo
13.555.463 observado; mismo runner y XSim, sin relanzamiento. Las auditorías
verificaron 32 propietarios y 2.023 vínculos de 38 paquetes. La
[normalización final con vocabulario completo](../LLAMA32_FINAL_VOCAB_ARITHMETIC_R1.md)
pasó compilación y veinte controles sintéticos; ejecución aritmética
encolada después de su síntesis/protocolo. Su buffer ABI1 se llenará con
escrituras reales confirmadas. Paquete de 103 archivos restaurado en D:/NFS.
**No hay aún resultados de esa simulación ni una estimación nueva de consumo.**

Actualización 2026-10-16 14:03 UTC: Q/K/V permanece activo, ciclo 10.875.678,
trabajo 7; conserva runner y simulador originales y observador recuperado.
Treinta y un propietarios verificados, sin fallos de supervisión nuevos.
La [conexión final norma/vocabulario](../LLAMA32_FINAL_VOCAB_R1.md) pasó
540 trabajos de control, 33 mutaciones y 174 controles de transcripción.
Síntesis/protocolo XSim encolados al final, después de la cabeza aritmética;
la aritmética conjunta aún no se ejecutó. **Sin estimación nueva de consumo.**

Actualización 2026-10-16 13:46 UTC: Q/K/V continúa en el ciclo 8.562.419.
Tres trabajos completaron y publicaron 3.072 resultados cada uno; el cuarto
está activo. Es progreso parcial, todavía sin revisión terminal del banco.
La supervisión se recuperó tras un fallo local de espacio, conservando la
misma simulación. Treinta propietarios verificados por la auditoría R80.

El [vocabulario aritmético completo](../LLAMA32_VOCAB_ARITHMETIC_R1.md) quedó
al final, después de DDR R2. Fuentes, fixture y controles restaurados en D:/NFS;
no hay resultados RTL nuevos de esa cabeza. DDR R1 conserva sus dos SAIF y
respaldo; DDR R2 espera su turno. **La estimación de consumo sigue pendiente.**

Actualización 2026-10-16 13:06 UTC: memoria Q/K/V continúa en XSim;
PID 954836 confirmado activo. [DDR R2](POWER_ACTIVITY_R2.md) quedó al final
de la cola para capturar comandos de los cuatro ranks junto a las ventanas
SAIF, sin modificar R1. Fuentes y controles restaurados en D:/NFS;
29 propietarios y 1.896 vínculos verificados. **R2 aún no se ejecutó**;
la estimación de consumo sigue pendiente.

Actualización 2026-10-16 12:46 UTC: [actividad DDR R1 terminada](POWER_ACTIVITY_RESULTS_R1.md),
dos SAIF de 20,48 µs y 577.280 señales, 585 archivos restaurados. Revisión
suplementaria de actividad aprobada; identidad verificada con puntos
abiertos de bibliotecas, modelo y asociación al DCP. Diagnósticos originales
conservados. Q/K/V control también terminó revisado y respaldado;
memoria Q/K/V está activa en `20261016T124000Z-V7-qkv-memory-vivado-r3`.
Veintiocho propietarios y 1.870 vínculos de 34 paquetes verificados.
**Sin estimación nueva de consumo ni calificación de la placa completa.**

Actualización 2026-10-16 12:25 UTC: **actividad DDR R1 en Vivado**, tanda
`20261016T122423Z-V8-ddr-activity-vivado-r3`, dos CPU y 20 minutos máximos.
Generación de simulación de la IP en curso; todavía sin SAIF. BRAM aprobó
aritmética y timing interno OOC a 5 ns: setup +0,056 ns y hold +0,010 ns.
Respaldo `af89519c5abd8258bcdbf289d41eac727194767ad12b035eac69c31f2a1164b6`;
447 resultados y 165 fuentes restaurados. Treinta propietarios mantienen
la cola. No se ha estimado nueva potencia ni calificado la placa completa.

Actualización 2026-10-16 12:02 UTC: BRAM continúa en implementación física;
actividad DDR conserva el siguiente turno. Sus 394 archivos se verificaron
en D:/NFS (`ddr-activity-handoff-r6-audit-20261016T120214Z.json`). El
[control del vocabulario](../LLAMA32_VOCAB_HEAD_R1.md) quedó al final de la
cola tras cuatro MAC: 679 trabajos nativos, 35 variantes defectuosas,
91 controles y compilación de interfaces aprobados. Todavía falta su
síntesis/protocolo XSim y la ejecución aritmética de la composición.
Treinta y un propietarios verificados. **DDR aún sin elaboración,
SAIF ni estimación nueva de consumo.**

Actualización 2026-10-16 11:43 UTC: BRAM continúa en implementación física;
DDR conserva el siguiente turno. Treinta propietarios verificados, sin
modificaciones de orden ni nuevas ejecuciones Vivado. El [recorrido completo
de Llama en CPU](../LLAMA32_FORWARD_NATIVE_R1.md) pasó con matemáticas estándar
y AMD: 16 capas, 128.256 logits y 372 comparaciones exactas en cada variante.
Control de corrupción del último logit rechazado; cinco tandas restauradas
en D:/NFS. Esto no certifica inferencia completa RTL ni rendimiento físico.
**La captura DDR sigue pendiente; no hay SAIF ni consumo nuevo.**

Actualización 2026-10-16 11:30 UTC: BRAM aprobó el control aritmético previo
al route y continúa en implementación física. El banco registró 3.823.015
ciclos, tres trabajos (incluida cancelación) y 4.096 valores finales. Falta
la revisión terminal y el respaldo. DDR conserva el siguiente turno; sus
394 fuentes se revalidaron en D:/NFS. Treinta propietarios mantienen orden.
El [recorrido completo de Llama en CPU](../LLAMA32_FORWARD_NATIVE_R1.md)
pasó con matemáticas estándar; AMD continúa por capas, fuera del slot Vivado.
**Todavía no hay SAIF ni una nueva estimación de consumo.**

Actualización 2026-10-16 11:00 UTC: BRAM continúa en XSim y DDR conserva
el siguiente turno. Treinta propietarios verificados; no se agregó trabajo
Vivado. Se probó el [revisor temporal de comandos DDR](POWER_RANK_TRACE_R1.md)
en Windows/Linux y se contrastó con un VCD nativo histórico: 325 comandos
seleccionados, tres REF y cobertura de comandos de un solo rank.
Esto no califica la nueva captura, todavía sin elaboración ni SAIF.

Actualización 2026-10-16 10:48 UTC: BRAM continúa en XSim; ciclo 1.710.051
observado, sin resultado final. DDR mantiene su turno posterior y sus 394
archivos siguen íntegros en D:/NFS. La [comparación de cuatro MAC](../LLAMA32_ATTENTION_FFN_QUAD_R1.md)
queda al final, tras cancelación registrada, PID 22664. Su compilación de
interfaces pasó y está restaurada; faltan síntesis y aritmética reales.
Treinta propietarios verificados. **No hay SAIF ni potencia nueva.**

Actualización 2026-10-16 10:31 UTC: BRAM está en XSim, ciclo 541.774
observado, sin resultado final. DDR sigue a continuación. El observador
de [identidad DDR](POWER_ACTIVITY_IDENTITY_R1.md), PID 3696, espera su
captura respaldada y complementa al revisor de actividad. Pasaron 26 y
15 comprobaciones sintéticas Windows/Linux, respectivamente. La cola
mantiene sus 29 propietarios; no se añade trabajo Vivado. **Sin SAIF
ni nueva estimación de consumo.**

Actualización 2026-10-16 10:19 UTC: BRAM sigue en síntesis; DDR permanece
inmediatamente después. Se añadió un observador de restauración de BRAM,
PID 23628: espera el respaldo terminal, extrae resultados y fuentes en D:,
verifica todos los hashes y conserva un recibo separado en NFS, incluso
si el resultado es diagnóstico. No cambia dependencias ni adquiere el slot.
Estado `projected-root-bram-restore-r1-monitor.json`: esperando respaldo.
**Todavía sin resultado BRAM, elaboración DDR, SAIF ni potencia nueva.**

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

Actualización 2026-10-16 09:14 UTC: [integración de capa completa](../LLAMA32_LAYER_R1.md)
aprobada sólo en su control nativo: 165 trabajos, 12 mutaciones explícitas
y 36 trazas corruptas rechazadas. Nueve tandas restauradas y verificadas.
Síntesis de los ocho bloques HLS reales y banco XSim separado encolados
tras preflight, PID 11872; no han comenzado. Atención/FFN sigue en la misma
simulación, ciclo 19.775.825 a las 09:13 UTC. DDR permanece después de BRAM,
con 394 archivos íntegros en D:/NFS y el observador complementario esperando.
**Todavía sin SAIF, potencia nueva ni aritmética completa de esta integración.**

Actualización 2026-10-16 08:38 UTC: [validador de memoria de capa](../LLAMA32_LAYER_PREFLIGHT_R1.md)
aprobado en Icarus y encolado al final, PID 25120. Revisa 24 regiones antes
de permitir escrituras iniciales: 1.743 trabajos, máximo 301 ciclos de
validación, 75 controles de traza y 18 variantes defectuosas detectadas.
No ejecuta aritmética ni AXI. Tres tandas restauradas en D:/NFS.
Atención/FFN continúa, ciclo 18.033.464 a las 08:37 UTC. DDR conserva su
turno tras BRAM y el observador complementario espera su respaldo.
**La captura DDR aún no tiene SAIF ni una nueva estimación de consumo.**

Actualización 2026-10-16 08:13 UTC: observador de
[revisión DDR](POWER_ACTIVITY_DETAIL_R1.md) activo, PID 16980, esperando la
captura y su respaldo verificado. Sus 15 pruebas sintéticas pasaron en
Windows/Linux; no acreditan actividad nativa. Conserva la cola y los 26
propietarios Vivado. Atención/FFN sigue activa, ciclo 16.830.230; BRAM y DDR
mantienen sus turnos. **Todavía sin SAIF ni una estimación nueva de consumo.**

Actualización 2026-10-16 07:56 UTC: [banco de entrada→atención](../LLAMA32_INPUT_ATTENTION_REAL_R1.md)
compilado y encolado después de la síntesis exacta del wrapper, PID 24836.
Coinciden 12.288 proyecciones de cuatro embeddings entre las referencias;
65 controles sintéticos verificaron causalidad y revisión numérica.
Eso no equivale a ejecutar los cuatro HLS juntos: queda pendiente XSim.
Auditoría R71: 26 propietarios; 28 paquetes y 1.373 vínculos intactos.
Atención/FFN llegó al ciclo 15.940.191 a las 07:55 UTC, sin resultado final.
DDR mantiene su lugar después de BRAM. **Sin SAIF ni consumo nuevo.**

Actualización 2026-10-16 07:27 UTC: [revisión detallada DDR](POWER_ACTIVITY_DETAIL_R1.md)
preparada, con 39 rechazos y 22 comprobaciones diagnósticas idénticas en
Python local/Linux. Son pruebas sintéticas del revisor; captura pendiente.
Atención/FFN sigue activa: 14.370.338 ciclos observados a las 07:24 UTC.
Se conservan los 25 propietarios y el turno DDR después de BRAM. El paquete
R1 sigue íntegro; no se inició otra sesión Vivado. **Sin SAIF ni consumo nuevo.**

Actualización 2026-10-16 07:06 UTC: [wrapper entrada→atención](../LLAMA32_INPUT_ATTENTION_R1.md)
cableado, con validación de asignaciones antes de permitir escrituras.
Pasaron 894 trabajos de protocolo, 150 controles y 16 mutantes. Las cuatro
tandas están restauradas; el RTL del propietario permaneció intacto.
Síntesis real de cuatro tipos de HLS y XSim del protocolo encolados al
final, PID 13140; no acreditan todavía aritmética conjunta. Auditoría R70:
veinticinco propietarios, 27 paquetes y 1.276 vínculos intactos.
Atención/FFN sigue activa, ciclo 13.402.910 observado a las 07:05 UTC.
DDR conserva el turno después de BRAM y los 394 archivos verificados.
**Todavía no hay SAIF ni consumo nuevo.**

Actualización 2026-10-16 06:34 UTC: [consulta de prefijos por capa](../LLAMA32_CACHE_QUERY_R1.md)
separada del comando interno retenido. Pasaron 97.920 consultas nativas,
54 controles y ocho mutantes; fuentes y resultados restaurados. Síntesis
del guard y XSim al final de la cola, PID 22492. La jerarquía completa
tiene sólo compilación con sustitutos HLS sin ejecutar. Auditoría R69:
veinticuatro propietarios, 26 paquetes y 1.188 vínculos intactos.
Atención/FFN alcanzó 11.606.693 ciclos a las 06:29 UTC, sin resultado final.
DDR permanece después de BRAM; 394 archivos revalidados con una observación
reciente de sus propietarios. **Sin SAIF ni consumo nuevo.**

Actualización 2026-10-16 06:13 UTC: [control de retención Q/K/V](../LLAMA32_QKV_ROPE_HANDOFF_R1.md)
aprobado en Icarus, con 1.044 trabajos, 321 corrupciones rechazadas y
25 mutantes detectados por aserciones. Siete tandas restauradas; síntesis
y XSim al final de la cola, PID 2936. Los extremos son pares sintéticos,
sin aritmética ni AXI en esta prueba. Auditoría R68: veintitrés propietarios,
25 paquetes y 1.156 vínculos intactos. Atención/FFN llegó a 10.651.945 ciclos
a las 06:09 UTC, todavía sin resultado final. DDR permanece después de BRAM;
394 archivos revalidados en D: y NFS, **sin SAIF ni consumo nuevo**.

Actualización 2026-10-16 05:40 UTC: [RoPE/atención con ambos HLS originales](../LLAMA32_ROPE_ATTENTION_REAL_R1.md)
compilado y encolado al final, después de su síntesis, PID 18736.
Las pruebas del revisor rechazaron 168 corrupciones de una traza sintética;
**no hay ejecución aritmética conjunta todavía**. La compilación está
respaldada y restaurada. Auditoría R67: veintidós propietarios;
24 paquetes y 1.124 vínculos intactos. Atención/FFN llegó a 9.131.528 ciclos
a las 05:38 UTC, sin resultado final. DDR sigue tras BRAM; los 394 archivos
R1 se revalidaron en D: y NFS. **Sin SAIF ni nueva estimación de consumo**.

Actualización 2026-10-16 05:13 UTC: [RoPE/caché/atención](../LLAMA32_ROPE_ATTENTION_R1.md)
unidos bajo admisión común. Pasaron 18 rechazos previstos, 45 vinculaciones
y 189 controles; cinco mutantes detectados y una variante redundante
superviviente. Tres tandas restauradas. Síntesis con ambos HLS originales
encolada al final, PID 26844; todavía no hay aritmética conjunta ejecutada.
Auditoría R66: veintiún propietarios, 23 paquetes y 1.071 vínculos intactos.
Atención/FFN avanza, ciclo 7.863.026 a las 05:12 UTC; no tiene resultado final.
Actividad DDR sigue esperando BRAM, **sin SAIF ni potencia nueva**.

Actualización 2026-10-16 04:52 UTC: [caché KV compartida entre capas](../LLAMA32_QKV_LAYER_REUSE_R1.md)
aprobada en Icarus y encolada para XSim al final, PID 26856. Se comprobaron
174 trabajos, 313.344 palabras, 864 corrupciones y seis mutantes; RTL intacto.
Ambas tandas están restauradas en D: con copias verificadas en NFS.
Auditoría R65: veinte propietarios; 22 paquetes y 1.023 vínculos intactos.
Atención/FFN avanza, ciclo 6.682.171 observado a las 04:49 UTC; la simulación
sigue incompleta. Actividad DDR R1 espera BRAM, **sin SAIF ni potencia nueva**.

Actualización 2026-10-16 04:25 UTC: [banco RoPE/captura/despacho real](../LLAMA32_ROPE_QKV_REAL_R1.md)
compilado y encolado, PID 21452, después de su síntesis exacta. Son 82
trabajos previstos; el revisor pasó 51 casos sintéticos y 101 corrupciones.
**No se ha ejecutado todavía la aritmética de esta composición**. La tanda
de compilación está restaurada. Auditoría R64: diecinueve propietarios;
21 paquetes y 987 vínculos verificados. XSim atención/FFN sigue activo,
ciclo 5.425.966 a las 04:23 UTC, todavía sin resultado final. Los 394
archivos DDR se verificaron otra vez. **Sin SAIF ni potencia nueva.**

Actualización 2026-10-16 04:04 UTC: [RoPE/captura/despacho](../LLAMA32_ROPE_QKV_DISPATCH_R1.md)
tiene protocolo aprobado: 105 trabajos, 36.864 palabras públicas y 324
controles de traza. Once variantes defectuosas detectadas; una comprobación
redundante en este banco sobrevive y se documenta. Cinco tandas restauradas.
La síntesis con el HLS original queda al final, PID 24044; el banco nativo
usó un modelo de identidad, por lo que todavía no valida aritmética RoPE.
Auditoría R63: dieciocho propietarios; 20 paquetes y 943 vínculos intactos.
Atención/FFN sigue en XSim (ciclo 4.358.041, 04:02 UTC), sin aprobación aún.
Se verificaron otra vez los 394 archivos DDR: **sin SAIF ni potencia nueva**.

Actualización 2026-10-16 03:33 UTC: [captura/despacho Q/K/V](../LLAMA32_QKV_CAPTURE_DISPATCH_R1.md)
aprobados en Icarus: 93 escenarios, 6.873 tramas, 49.152 palabras públicas,
258 controles de traza y diez mutantes detectados por aserciones. Los datos
atraviesan captura/lector/append/guardia/secuenciador reales; RoPE y cabeza
aritmética son pares sintéticos. Tres tandas restauradas. Síntesis/XSim al
final de la cola, PID 9824. Auditoría R62: diecisiete propietarios; 19 paquetes
y 912 vínculos intactos. Atención/FFN permanece en la única ejecución Vivado
(ciclo 2.260.261 observado a las 03:20 UTC, aún incompleta). Los 394 archivos
DDR se revalidaron; R1 sigue detrás de BRAM. **Sin SAIF ni potencia nueva.**

Actualización 2026-10-16 03:04 UTC: [despachador Q/K/V](../LLAMA32_QKV_DISPATCH_R1.md)
aprobado en Icarus: 144 escenarios, 55.296 palabras públicas contrastadas,
267 controles negativos de traza y 25 mutantes detectados por aserciones.
Cuatro tandas restauradas; síntesis/XSim encolados después de captura Q/K/V,
PID 3996. Los pares aritméticos siguen siendo sintéticos.
Auditoría R61 verifica dieciséis propietarios por PID y hash de su script;
18 paquetes y 874 vínculos intactos. Atención/FFN conserva la única ejecución
Vivado: ciclo 1.307.658 a las 03:01 UTC, sin fatal observado, aún incompleta.
DDR espera después de BRAM; **no hay SAIF ni nueva estimación de consumo**.

Actualización 2026-10-16 02:31 UTC: atención/FFN R4 está en simulación,
después de aislar un fallo del helper de aserción condicional en XSim.
El banco R6 conserva sus 139 comprobaciones; doce casos XSim/Icarus
validan la forma corregida. Cinco tandas de diagnóstico restauradas.
La síntesis previa se conserva: 120.523 LUT, 106.453 registros, 159,5 BRAM,
238 DSP, WNS OOC −10,467 ns a 5 ns. No certifica timing ni aritmética.
La cola retoma el orden con quince coordinadores; auditoría R58 y 828
vínculos verificados. DDR R3 espera BRAM, con sus 394 archivos intactos.
**Aún sin captura SAIF ni nueva potencia**. Detalles en
[atención/FFN](../LLAMA32_ATTENTION_FFN_REAL_R1.md) y
[actividad DDR](POWER_ACTIVITY_R1.md).

Actualización 2026-10-16 01:37 UTC: [captura privada Q/K/V](../LLAMA32_QKV_SNAPSHOT_R1.md)
aprobada en Icarus: 321 escenarios, 2.285.568 bytes públicos contrastados,
177 controles de traza y 23 mutantes. Cuatro tandas respaldadas/restauradas;
síntesis y XSim encolados tras RoPE real, PID 28052. La RAM retiene 12 KiB
de datos; recursos físicos todavía sin medir. Atención/FFN continúa en
Vivado y DDR mantiene su posición tras BRAM. Auditoría R57: quince
coordinadores, 17 paquetes y 767 vínculos intactos. No hay SAIF nuevo.

Actualización 2026-10-16 01:23 UTC: [handoff post-O](../LLAMA32_POST_FFN_HANDOFF_R1.md)
**aprobado y respaldado**: 1.566 escenarios XSim, 309 controles, 2.480 LUT,
3.106 registros, WNS OOC +1,571 ns. La composición aritmética atención/FFN
está en Vivado (`20261016T012306Z-V7-attention-ffn-real-vivado-r2`).
BRAM y actividad DDR esperan ese resultado; todavía no hay SAIF nuevo.

Actualización 2026-10-16 01:20 UTC: [FFN/residual real](../LLAMA32_FFN_RESIDUAL_REAL_R1.md)
**terminó y pasó la revisión R2**: 20.345.899 ciclos, 56.770.560 bytes
leídos, 163.840 escritos, 8.192 bytes finales exactos; 64 controles rechazados.
Última escritura HLS y `done` coinciden en un flanco válido; se corrigió
el revisor y se conservaron RTL, banco, diagnóstico y archivo original.
WNS OOC −1,770 ns: falta cierre temporal. Restaurados 334 archivos de
resultado y 137 de fuentes. Quince coordinadores sucesores R2 reanudaron
la cola, con post-O en Vivado; se verificaron 736 vínculos de 16 paquetes.
Actividad DDR mantiene su turno después de BRAM, PID 17120, con sus
394 archivos revalidados D:/NFS. **No hay SAIF ni potencia nueva.**

Actualización 2026-10-16 00:57 UTC: [RoPE aritmético por AXI](../LLAMA32_ROPE_INPUT_REAL_R1.md)
encolado después del protocolo, PID 24812. Compilación de interfaces
aprobada y restaurada; 43 corrupciones del revisor rechazadas. El paquete
conserva las cuatro IP originales y 21 fuentes, con respaldo D:/NFS.
Los 69 casos previstos todavía no corrieron en RTL. R55 verifica dieciséis
coordinadores y 686 vínculos. FFN/residual continúa en XSim; a las 00:53 UTC
alcanzó el ciclo 19.654.072, fase 3. Actividad DDR conserva su posición
después de BRAM y sigue sin SAIF ni nueva potencia.

Actualización 2026-10-16 00:44 UTC: [entrada RoPE compuesta](../LLAMA32_ROPE_INPUT_PATH_R1.md)
aprobada en Icarus: 195 casos, 183 respuestas, 15 resets, 122.880 palabras,
99 controles de traza y diez mutantes. Corrección de cancelación reproducida;
cinco tandas restauradas. Síntesis/XSim encolados después del cargador,
PID 4840; aritmética RoPE conectada todavía pendiente. R54 verifica quince
coordinadores y 647 vínculos intactos. FFN/residual continúa en XSim; a las
00:38 UTC llegó al ciclo 18.156.638, fase 3. DDR conserva su lugar tras BRAM;
394 archivos revalidados y traspaso respaldado/restaurado. No hay SAIF ni
estimación nueva de consumo.

Actualización 2026-10-16 00:17 UTC: [carga Q/K/fases por AXI](../LLAMA32_ROPE_LOADER_R1.md)
aprobada en Icarus: 153 escenarios, 70.848 palabras, 90 alteraciones de
traza y 19 mutantes. Ambas tandas restauradas. Síntesis/XSim encolados
después del propietario RoPE, PID 27644; paquete de 30 archivos D:/NFS.
R53 confirma catorce coordinadores y 607 vínculos intactos. FFN/residual
continúa en XSim; el registro de las 00:15 UTC llegó al ciclo 15.916.871
de fase 3. Actividad DDR mantiene su turno tras BRAM; sigue sin SAIF ni
nueva estimación de consumo. Falta unir cargador, propietario y RoPE HLS.

Actualización 2026-10-16 00:07 UTC: [control de memorias RoPE](../LLAMA32_ROPE_OWNER_R1.md)
aprobado en Icarus: 297 trabajos, 407.040 palabras, 69 controles de traza y
29 mutantes. Cuatro tandas restauradas, incluidas las fallidas. Síntesis/XSim
encolados tras normalización→QKV real, PID 27656, paquete D:/NFS de 39
archivos. Son pruebas de protocolo; la unión con aritmética RoPE aún falta.
R52 confirma trece coordinadores y 581 vínculos intactos. FFN/residual sigue
activa en XSim; a las 00:03 UTC alcanzó el ciclo 14.700.000 de fase 3.
Actividad DDR mantiene su turno tras BRAM, sin SAIF ni nueva potencia.

Actualización 2026-10-15 23:43 UTC: [normalización y Q/K/V unidos por AXI](../LLAMA32_INPUT_QKV_REAL_R1.md)
preparados y encolados tras el traspaso, PID 7880. Copias D:/NFS restauradas;
215 archivos, 59 fuentes y once IP. Compilación de interfaces aprobada y
85 controles sintéticos del revisor; aritmética conjunta todavía pendiente.
Auditoría R51 confirma doce coordinadores y 550 vínculos intactos. La FFN
continúa en XSim; a las 23:41 UTC alcanzó el ciclo 12.544.397 de fase 1.
Actividad DDR conserva su turno tras BRAM, sin SAIF ni nueva potencia.

Actualización 2026-10-15 23:23 UTC: [control de traspaso normalización→Q/K/V](../LLAMA32_INPUT_QKV_HANDOFF_R1.md)
aprobado en 822 trabajos nativos, con 417 respuestas aceptadas, 174 controles
de traza y 28 mutantes. Siete tandas restauradas; síntesis/XSim encolados
tras normalización real, PID 26112. El banco usa hijos sintéticos y no
acredita aritmética integrada. Se conservan la FFN/residual en XSim y el
turno DDR posterior a BRAM. No hay todavía SAIF ni nueva estimación de potencia.
Auditoría R50: once coordinadores vivos, 485 vínculos intactos; FFN/residual
continúa en XSim y alcanzó el ciclo 10.817.624 de fase 1 a las 23:23 UTC.

Actualización 2026-10-15 22:58 UTC: [normalización HLS real por AXI](../LLAMA32_INPUT_NORM_REAL_R1.md)
encolada tras el recorrido de protocolo, PID 9808. El paquete conserva el
RTL HLS aprobado y siete IP, con respaldo D:/NFS y restauración verificada.
Pasaron la compilación de interfaces, 58 controles de puertos y 51 controles
sintéticos del revisor; aún sin ejecución aritmética integrada ni recursos
nuevos. Auditoría R49 confirma diez coordinadores y la FFN/residual activa
en XSim. Actividad DDR mantiene su turno después de BRAM; sigue sin SAIF.

Actualización 2026-10-15 22:47 UTC: [recorrido de norma por AXI](../LLAMA32_INPUT_NORM_PATH_R1.md)
aprobado en 198 trabajos nativos, 430.080 bytes aceptados, 66 controles y
14 mutantes. Síntesis/XSim encolados detrás del propietario RAM, PID 16564;
paquete y ambas tandas restaurados y verificados en D:/NFS. El HLS del banco
reproduce referencias y no acredita aritmética integrada. Auditoría R48:
nueve coordinadores vivos; FFN/residual sigue como única ejecución Vivado,
ciclo 7.363.026 de fase 1. La actividad DDR R1 mantiene su reserva después
de BRAM, sin SAIF ni nueva estimación de consumo.

Actualización 2026-10-15 22:28 UTC: [propietario de normalización](../LLAMA32_INPUT_NORM_OWNER_R1.md)
aprobado en 243 trabajos nativos, 66.048 palabras, 63 controles y 21 mutantes.
Síntesis/XSim de protocolo encolados detrás de Q/K/V aritmético, PID 26776;
paquete y seis tandas respaldados y restaurados. Auditoría R47 confirma ocho
coordinadores y una sola ejecución, FFN/residual `20261015T211706Z` en XSim,
avanzando por el ciclo 5.585.259 de fase 0. Actividad DDR sigue esperando su
turno posterior a BRAM; sin SAIF ni nueva estimación de consumo.

Actualización 2026-10-15 22:11 UTC: [Q/K/V aritmético R1](../LLAMA32_QKV_PRODUCER_R1.md)
encolado detrás de su control, PID 19224. Paquete de 145 archivos en D:/NFS
con restauración verificada; 23 fuentes y cuatro IP originales. Compilación
de jerarquía y 53 controles sintéticos del revisor completados; aritmética
XSim, recursos y timing nuevos pendientes. Auditoría R46 confirma siete
coordinadores vivos y una sola ejecución: FFN/residual `20261015T211706Z`.
Se conserva la cola previa y la captura DDR posterior a BRAM; aún sin SAIF.

Actualización 2026-10-15 21:51 UTC: [control Q/K/V](../LLAMA32_QKV_PRODUCER_R1.md)
pasó 481 trabajos nativos, 91 controles de traza y 29 mutantes compilados.
Síntesis/XSim encolados tras actividad DDR, PID 25708; todavía sin medidas
de recursos/timing ni aritmética Q/K/V integrada. Paquetes y cuatro tandas
nativas respaldados y restaurados. Auditoría R45 confirma seis propietarios
y la FFN/residual `20261015T211706Z` activa en XSim; cola anterior intacta.

Actualización 2026-10-15 21:39 UTC: [captura de actividad DDR R1](POWER_ACTIVITY_R1.md)
encolada detrás de BRAM. Paquete de 394 archivos restaurado y verificado
en D: y NFS; coordinador PID 25756. Auditoría R44: cinco coordinadores vivos,
FFN/residual `20261015T211706Z` como única tanda activa, en XSim. La captura
espera revisión y respaldo del predecesor y slot vacío. Invocación inicial
limitada a 20 minutos y dos CPU, sin reintento automático. Sin SAIF ni
nueva potencia; la revisión sintética del parser no sustituye ejecución RTL.

Actualización 2026-10-15 21:07 UTC: el [traspaso de potencia DDR](POWER_PROBE_RESULTS_R1.md)
está ejecutado. Exportación R2 en 98 s, salida cero, revisión independiente
y restauración verificada de 44 archivos. SHA-256 del TAR:
`733dc47a231e5cede530a1b2bc531f85fe55d11e8fffd984733dcdca606cfbbc`.
Resultado: 3,015 W on-chip, 1,349 W dinámica y 1,666 W estática;
confianza baja, sin SAIF y con avisos de reset. No califica potencia de
placa ni el DIMM. Intentos fallidos también archivados. El escritor
residual previo quedó aprobado y respaldado; su sucesor PID 23708 fue
reanudado a las 21:07:39 UTC, conservando toda la cadena y sus dependencias.

Actualización 2026-10-15 20:36 UTC: [FFN completa R4](../LLAMA32_OWNED_FFN_RESULTS_R4.md)
aprobada y respaldada: 20.332.665 ciclos, 56.754.176 bytes leídos y
155.648 escritos, salida final exacta de 8.192 bytes. Síntesis OOC:
41.040 LUT, 36.160 FF, 40,5 BRAM, 93 DSP; setup estimado +0,755 ns.
No es timing ruteado. El cargador residual R2 empezó a las 20:35:47 UTC.
Se preservan los nueve coordinadores restantes y su orden.
[Q/K/V R1](../LLAMA32_QKV_PRODUCER_R1.md) cuenta con 12 proyecciones de
referencia, 12.288 valores FP64/FP32, 128 entradas de atención enlazadas
byte a byte y 20 controles; todavía sin controlador RTL ni tanda Vivado.

Actualización 2026-10-15 20:26 UTC: [experimento BRAM R1](../LLAMA32_PROJECTED_ROOT_BRAM_R1.md)
encolado tras atención/FFN/residual real R1. Control nativo aprobado:
219 trabajos, 59.904 palabras, 1.191.819 muestras de equivalencia,
99 controles y seis mutantes. Propietario R3 y padres sucesores;
38 fuentes elaboradas con firmas, sin aritmética nueva todavía.
Paquete real de 88 fuentes y 14 IP sellado; el revisor numérico debe
aprobar antes del ruteo/physopt. Diez coordinadores activos; único slot
ocupado por FFN R4. A las 20:27 UTC ésta avanza en fase 3, offset
16.707.648 de 17.825.792 bytes. Último timing físico sigue en −0,034 ns.

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

Actualización 2026-10-15 18:35 UTC: **composición aritmética FFN → residual
R1 sellada y encolada**. Conserva la geometría 2048/8192, las matrices
Q8 originales y la entrada ABI1 exacta. El residual debe consumir los
bytes FFN confirmados por B. Paquete con 68 fuentes efectivas, sin peers
aritméticos sustitutos; preflight de interfaces y 60 controles del revisor
aprobados. [Contrato y evidencia](../LLAMA32_FFN_RESIDUAL_REAL_R1.md).
No hay todavía resultado XSim ni síntesis para la cadena completa.

Auditor R40: siete coordinadores vivos y una tanda principal. FFN R4
continúa en XSim; el nuevo coordinador espera detrás del traspaso R1.
Cada etapa exige revisión y respaldo de su predecesora. La FFN activa
se conserva sin cambios.

---

Actualización 2026-10-15 18:17 UTC: **wrapper FFN → residual R3 aprobado
en control nativo**: 288 trabajos, 186 alteraciones de trazas rechazadas
y 20 mutantes RTL detectados. Se corrigió la pérdida de la causa de un
cambio de época durante FFN. El mux AXI conserva la propiedad hasta el
ACK final. [Evidencia y límites](../LLAMA32_FFN_RESIDUAL_CONTROL_R1.md).
La aritmética de esta prueba usa peers controlados; falta el banco
completo con ambos motores reales antes de sellar y encolar esa etapa.

Auditor R39: seis coordinadores vivos, una tanda principal. FFN R4
continúa en XSim, fase 1. La cadena en espera sigue: cargador residual
R2 → propietario R1 → escritor R1 → residual real R1 → traspaso R1.
No se interrumpieron procesos ni se alteraron paquetes encolados.

---

Actualización 2026-10-15 17:46 UTC: **control de propiedad FFN → residual
aprobado en nativo y encolado**: 342 trabajos con peers controlados,
165 controles de trazas y 24 mutantes RTL. Conserva ambos productores
de respuesta hasta el ACK público final y evita confirmar identidades
ajenas. [Contrato, evidencia e integración siguiente](../LLAMA32_FFN_RESIDUAL_HANDOFF_R1.md).
La aritmética y el mux AXI de la cadena completa siguen pendientes.

Auditor R39: seis coordinadores vivos, una tanda principal. FFN completa
R4 continúa en XSim. Cadena: cargador residual R2 → propietario R1 →
escritor R1 → composición residual real R1 → traspaso FFN/residual R1,
con revisión y respaldo entre etapas. No se reinició la FFN activa.

---

Actualización 2026-10-15 17:28 UTC: **composición residual con HLS real
aprobada en nativo y encolada en Vivado**. Diecisiete fuentes efectivas,
incluidos nueve Verilog originales; 144 trabajos, 178.176 palabras FP32
exactas, 147 controles de trazas y 12 mutantes RTL. Se ofrecieron 87
vectores; 78 se aceptaron sin revocación/reset. HLS: 4.110 ciclos;
primer AR a último B: 12.941–13.000 ciclos con pausas del banco.
[Contrato, medidas y evidencia](../LLAMA32_RESIDUAL_REAL_R1.md).

Auditor R38: cinco coordinadores vivos, una sola tanda principal. FFN
completa R4 sigue en XSim; esperan cargador residual R2 → propietario
R1 → escritor R1 → composición residual real R1. Cada etapa exige
revisión y respaldo de la anterior. La nueva composición usa operandos
post-FFN retenidos; la conexión a la FFN viva sigue pendiente.

---

Actualización 2026-10-15 17:04 UTC: **escritor residual nativo aprobado y
encolado**: 246 trabajos, 132 vectores publicados, 1.081.344 bytes exactos,
117 controles de revisión y 21 mutantes RTL rechazados. Transferencia
RAM → último B: 4.441–4.513 ciclos con pausas del banco; no es latencia
de inferencia. [Contrato y evidencia](../LLAMA32_RESIDUAL_STORE_R1.md).

La FFN completa R4 (`20261015T164923Z`) continúa en XSim; auditor R37
verificó `xsimk` vivo. Esperan cargador residual R2 → propietario residual
R1 → escritor residual R1. Cuatro coordinadores, una sola tanda principal.
Se preparó el adaptador de 18 puertos del HLS residual original, con 55
controles de contrato; compilación y unión funcional siguen pendientes.

---

Actualización 2026-10-15 16:50 UTC: **propietario FFN aprobado en Vivado R4
y respaldado**. Cinco bancos, 161 trabajos, 119 controles de trazas y
siete de identidad. Síntesis: 1.339 LUT, 2.084 FF, WNS +1,545 ns,
WHS +0,068 ns a 5 ns; cero BRAM/DSP. Es protocolo con peer controlado,
no aritmética ni cierre físico. [Evidencia terminal](../LLAMA32_FFN_OWNER_R1.md).

La [FFN completa R4](../LLAMA32_OWNED_FFN_R1.md) inició automáticamente
`20261015T164923Z`; conserva el caso real y sus 51 fuentes compiladas.
Detrás esperan cargador residual R2 y propietario residual R1.
Auditor R36: tres coordinadores vivos, una sola tanda principal.

---

Actualización 2026-10-15 16:46 UTC: **propietario residual nativo aprobado**:
219 trabajos, 245.760 palabras comparadas, 63 controles y 17 mutantes RTL
rechazados. [Contrato y evidencia](../LLAMA32_RESIDUAL_OWNER_R1.md).
El banco R2 añade detección directa de admisión con identidad truncada;
no se contabilizan timeouts como detección.

La [optimización física](../LLAMA32_PROJECTED_ROOT_PHYSOPT_R1.md) terminó:
setup −0,034 ns, hold +0,003 ns, 86.884/86.884 nets completos. **No cierra
timing**. La revisión corregida reconoce el estado exacto `Physopt postRoute`,
conserva los demás criterios y respalda los resultados sin repetir Vivado.

Cola actual: propietario FFN R4 en ejecución (`20261015T164528Z`) → FFN
completa R4 → cargador residual R2 → propietario residual R1.
Auditor R36: cuatro coordinadores vivos y una sola tanda principal.
Pendientes: HLS residual unido al transporte, escritor confirmado por B,
integración completa y cierre físico de placa.

---

Actualización 2026-10-15 16:24 UTC: **cargador residual nativo aprobado y
encolado en Vivado**: 114 trabajos, 86.016 palabras RAM contrastadas con
AXI, 78 corrupciones rechazadas y once mutantes RTL detectados por aserción.
[Contrato, alcance y hashes](../LLAMA32_RESIDUAL_LOADER_R1.md).
La cola conserva optimización física R5 → propietario FFN R3 → FFN completa
R3 → cargador residual R1. Auditor R35: cuatro coordinadores vivos, una sola
tanda principal. El cargador usa datos sintéticos; aritmética, escritor y
conexión al resultado FFN real todavía no están integrados en ese banco.

---

Actualización 2026-10-15 16:11 UTC: **optimización física R5 activa**,
`20261015T161045Z`, seguida por propietario FFN R3 y FFN completa R3.
R4 rechazó el checkpoint antes de optimizar porque consultaba una propiedad
vacía; su diagnóstico quedó respaldado. La [corrección](../LLAMA32_PROJECTED_ROOT_PHYSOPT_R1.md)
verifica las 86.884 rutas completas y cero errores mediante el informe
real, sin cambiar checkpoint, reloj ni constraints. Auditor R34:
tres coordinadores vivos y una sola tanda principal. Los sucesores FFN
conservan los archivos compilados y fixture de sus paquetes anteriores.

---

Actualización 2026-10-15 16:05 UTC: **atención → O → post-O aprobada y
respaldada**: tres trabajos, dos pares exactos, 24.576 bytes publicados,
4.194.333 ciclos del banco y 96 controles rechazados. Síntesis: 63.581 LUT,
57.943 FF, 111 BRAM, 145 DSP; WNS −1,052 ns, WHS +0,030 ns a 5 ns.
[Revisión terminal y hashes](../LLAMA32_ATTENTION_POST_REAL_R1.md).

La cola se recuperó de un fallo de sangría en el verificador remoto del
suplemento; se verificaron los mismos bytes sin repetir esa simulación.
Está ejecutándose `20261015T160507Z`, optimización física R4 del checkpoint
raíz original. Después: propietario FFN Vivado R2 y
[FFN completa con propietario](../LLAMA32_OWNED_FFN_R1.md) R2.
Auditor R33: tres coordinadores vivos, una sola tanda principal.
Fuentes y paquetes sucesores verificados en D:/NFS; ninguna fuente
compilada previa se modificó.

La nueva FFN incorpora el núcleo real sin cambios y usa ABI1 post-O
de referencia, todavía sin productor vivo. Pasó compilación vendor del
top con firmas HLS vacías; eso no acredita aritmética. La ejecución
completa encolada usa HLS real y revisa causalidad AW/W/B, bytes y fases.
Siguen pendientes unión completa, segunda suma, timing y placa DDR4/PCIe.

---

Actualización 2026-10-15 15:25 UTC: **adaptador de propiedad FFN aprobado
en nativo y encolado en Vivado**, detrás de la revisión y optimización
física ya existentes. Sus 161 trabajos, 119 controles independientes y
26 mutantes compilados están [documentados y respaldados](../LLAMA32_FFN_OWNER_R1.md).
Conserva identidad pública de 64 bits y protege el residual retenido;
la conexión con la aritmética FFN/AXI real sigue pendiente.

La unión atención → O → post-O continúa en XSim: primer caso publicado
con 12.288 bytes, cancelación siguiente con estado 7 y cero bytes, tercer
caso en proyección O. Última observación: ciclo 2.627.582. Son observaciones
parciales; la revisión terminal sigue pendiente. Auditor R32: cuatro
coordinadores vivos, una sola tanda Vivado principal, sin reinicios.

---

Actualización 2026-10-15 14:47 UTC: **unión raíz → post-O R2 en XSim**,
con síntesis observada de 63.581 LUT, 57.943 FF, 111 BRAM y 145 DSP.
WNS de síntesis −1,052 ns, WHS +0,030 ns a 5 ns; revisión terminal pendiente.
En CPU2/3, la [continuación post-O → FFN → segunda suma](../LLAMA32_POST_FFN_R1.md)
aprobó cuatro casos con ABI1 exacto en stdlib y AMD: 122.880 valores por
backend, seis ejecuciones corruptas detectadas y dieciséis controles de
revisión rechazados. Archivo nativo verificado en D:/NFS. Aún falta unir
esos bloques en RTL; no se declara inferencia completa ni rendimiento físico.

---

Actualización 2026-10-15 14:30 UTC: **unión raíz → O → post-O en Vivado R2**,
ejecución `20261015T142808Z`. El controlador pasó 165 trabajos nativos,
36 controles de revisión y 16 mutantes RTL. El banco real compila 131
fuentes y consume O únicamente desde escrituras confirmadas por B.
La primera tanda falló por una entrada omitida en la lista Tcl; diagnóstico
respaldado y corregido sin cambios de RTL/banco. [Detalle y hashes](../LLAMA32_ATTENTION_POST_REAL_R1.md).

El [ruteo raíz → O](../LLAMA32_PROJECTED_ROOT_ROUTE_R1.md) terminó:
86.884/86.884 conexiones, WNS −0,079 ns, WHS +0,010 ns a 5 ns;
48.175 LUT, 44.887 FF, 84,5 BRAM y 134 DSP. **No cierra timing**.
La revisión R2 corrigió un control negativo de formato sin cambiar las
métricas; archivo y suplemento están verificados en D:/NFS.

Detrás de la unión real se encoló `phys_opt_design -directive Explore`
sobre el checkpoint ruteado original, conservando 5 ns y restricciones.
Paquete físico R3: `0c8fb3bc0b1a12848a23af04746c50675e6e2ffe832599fe93c2e43cf3db9cbc`.
Coordinador `queue_llama32_attention_projected_root_physopt_vivado_r3.py`;
auditoría R31. Su antecesor espera el resultado archivado y lo revisa con
el verificador R2: las dos marcas de finalización del banco están separadas
por siete ciclos de respuesta retenida. La revisión exige esa diferencia
y añade un control negativo, sin cambiar ni reiniciar la simulación.
Una sola tanda principal, revisión/respaldo entre etapas.
Siguen pendientes FFN integrada, inferencia completa y cierre de placa.

---

Actualización 2026-10-15 09:00 UTC: **post-O con HLS y transporte AXI reales
aprobado y respaldado**. Ocho trabajos, seis pares y 73.728 bytes exactos;
cancelaciones durante cálculo/escritura y ambas recuperaciones correctas.
Se rechazaron catorce controles independientes. Máximo intervalo HLS sin
progreso: 165.137 ciclos frente al watchdog 262.144. Síntesis OOC a 5 ns:
13.131 LUT, 11.389 FF, 26,5 BRAM, 11 DSP, WNS +0,953 ns, WHS +0,039 ns;
sin cajas negras, latches o lazos. [Evidencia](../LLAMA32_POST_ATTENTION_REAL_R1.md).
Los 245 miembros del archivo de resultados están verificados en D:/NFS.
La tanda de [ruteo raíz → O](../LLAMA32_PROJECTED_ROOT_ROUTE_R1.md)
`20261015T090048Z` comenzó después del respaldo. Auditoría R28; una sola
tanda principal. Aún falta unir raíz → post-O → FFN, cerrar implementación
integrada y ejecutar inferencia completa.

---

Actualización 2026-10-15 08:53 UTC: **post-O real en XSim**, con dos
transacciones completas observadas; revisión final aún pendiente.
Cada una tardó 173.353 ciclos HLS y tuvo 165.137 ciclos máximos sin acceso
externo, por debajo del watchdog 262.144. Síntesis observada: 13.131 LUT,
11.389 FF, 26,5 BRAM, 11 DSP, WNS OOC +0,953 ns a 5 ns.
Se encoló el [ruteo de la raíz → O](../LLAMA32_PROJECTED_ROOT_ROUTE_R1.md)
para medir su camino crítico de control, WNS de síntesis −0,203 ns,
sin modificar RTL ni repetir el banco funcional. Fuente/checkpoint y
configuración están verificados en D:/NFS. Auditoría R28: dos coordinadores,
una sola tanda principal. Los valores observados no reemplazan la revisión
terminal ni constituyen cierre temporal de la tarjeta.

---

Actualización 2026-10-15 08:41 UTC: **cargador, propietario y escritor
post-O aprobados en Vivado y respaldados**. Sus tres bancos verificaron
135, 255 y 330 trabajos respectivamente; el escritor confirmó 2.211.840
bytes exactos. Los tres tienen WNS OOC positivo a 5 ns y cero cajas negras,
latches o lazos; no son resultados de ruteo.
El [banco integrado con HLS real](../LLAMA32_POST_ATTENTION_REAL_R1.md)
comenzó como `20261015T084117Z`, con ocho trabajos, seis pares esperados
y cancelación/recuperación. Sus 44 fuentes provienen del HLS aprobado y
los controladores reales. El controlador de transporte detectó doce
mutantes compilados. Auditoría R27: una tanda principal y trabajos externos
preservados; resultado aritmético integrado todavía pendiente.

---

Actualización 2026-10-15 08:21 UTC: **post-O HLS aprobado en C/RTL y
respaldado**. Sus 26 transacciones verifican 8.480 FP32 y 2.120 palabras
ABI1 exactas; los quince errores conservan ambas salidas intactas.
Latencia máxima medida: 173.353 ciclos. Recursos estimados HLS:
37 BRAM18, 11 DSP, 5.349 FF y 13.882 LUT; periodo estimado 3,647 ns,
sin cierre físico. El [resultado](../LLAMA32_POST_ATTENTION_R1.md) incluye
21 controles de revisión. El cargador AXI R3 comenzó automáticamente
como `20261015T081949Z`; propietario R2 y escritor R3 siguen en espera.
Se generó el [adaptador HLS](../LLAMA32_POST_ATTENTION_PATH_R1.md) desde
el archivo aprobado; falta simular esa unión aritmética con transporte.

---

Actualización 2026-10-15 08:15 UTC: **raíz de atención → cuantización → O
aprobada y respaldada**. Dos proyecciones completas: 4.096 FP64 y 16.384
bytes FP32 exactos, cancelación/recuperación y 69 controles. La revisión
R4 corrigió una desigualdad temporal del verificador: B y la respuesta
registrada K/V comparten contador de ciclo. Se añadió comprobación del
orden de eventos; no se repitió simulación ni se modificó su archivo.
WNS OOC −0,203 ns a 5 ns: cierre temporal pendiente.
[Métricas y evidencia](../LLAMA32_PROJECTED_ROOT_REVIEW_R4.md).

HLS post-O R3 ejecuta cosimulación. Le siguen cargador R3, propietario R2
y escritor R3; auditoría R26 con cuatro coordinadores vivos y una tanda
principal. La [unión de transporte](../LLAMA32_POST_ATTENTION_PATH_R1.md)
pasó 144 trabajos nativos y 39 controles, con 1.069.056 bytes confirmados.
Su sustituto HLS reproduce referencias; la aritmética integrada sigue pendiente.

---

Actualización 2026-10-15 07:48 UTC: **escritor AXI post-O aprobado en banco
nativo y encolado en Vivado**. Pasó 330 trabajos y publicó 180 pares
residual/ABI1: 2.211.840 bytes exactos, visibles solo después de B correcto.
Se detectaron 20 mutantes compilados y 126 corrupciones de traza. El ajuste
R2 afecta únicamente un control negativo del verificador que antes podía
producir un retraso legal de ACK; RTL y banco permanecen intactos.
Fuente y paquete nuevos están verificados en D:/NFS. Auditoría R24:
cinco coordinadores vivos y una sola tanda principal, la raíz completa R3,
que continúa en XSim. La unión real de cargador, HLS, propietario y escritor
todavía requiere su banco integrado.
[Contrato, evidencia y criterios](../LLAMA32_POST_ATTENTION_STORE_R1.md).

---

Actualización 2026-10-15 07:14 UTC: **propietario de RAM post-O implementado,
probado y encolado**. Sus 255 casos nativos verifican 276.480 palabras
residuales y 69.120 palabras ABI1; 75 controles y 17 defectos de RTL
introducidos deliberadamente fueron detectados. Mantiene ambas salidas
privadas hasta completar HLS y bloquea la liberación temprana de la entrada.
Se encoló síntesis/XSim de protocolo después del cargador AXI R2.
Auditoría R23: cuatro coordinadores vivos y una sola tanda principal.
La raíz completa R3 ya sintetizó y ejecuta XSim: confirmó 64 registros ABI1
en el ciclo 214.308 y comenzó O en el 214.310; no ha terminado su regresión.
[Contrato, evidencia y unión pendiente](../LLAMA32_POST_ATTENTION_OWNER_R1.md).

---

Actualización 2026-10-15 06:50 UTC: **O R3 y cuantización → O R3 aprobados
en simulación y respaldados; raíz completa corregida en ejecución**.
O entregó 10.240 FP64 y 40.960 bytes FP32 exactos; cuantización → O,
6.144 FP64 y 24.576 bytes. Esta última tiene WNS OOC −0,787 ns a 5 ns:
su aprobación funcional no implica cierre temporal. La raíz R2 se detuvo
por `INITIAL_QUANT_TAG=0xffffffff`, incompatible con reservas alineadas
de 64 etiquetas. El banco R2 usa `0xffffffc0` y conserva las comprobaciones
anteriores; RTL y aritmética permanecen iguales. La tanda R3 está activa.
Post-atención HLS R2 y cargador AXI R2 esperan en secuencia. El cargador
pasó 135 casos nativos, 129.024 palabras RAM, 81 controles y once mutantes.
Los paquetes nuevos están verificados en D:/NFS; se conservan las tandas
externas y los diagnósticos originales.
[Evidencia y cola vigente](../LLAMA32_PROJECTED_ROOT_TAG_RECOVERY_R3.md) ·
[Cargador post-O](../LLAMA32_POST_ATTENTION_LOADER_R1.md).

---

Actualización 2026-10-14 04:45 UTC: **bloque posterior a O implementado y encolado**.
La suma residual y normalización/Q8 de entrada a FFN pasaron 26 casos nativos:
8.480 palabras FP32 y 2.120 palabras ABI1 exactas, once controles y cinco
mutantes detectados. Ambos buffers permanecen intactos ante errores, incluso
fallos tardíos de normalización. Fuentes y resultado verificados en D:/NFS.
La nueva tanda HLS espera a la raíz completa R2; no ha ejecutado RTL todavía.
Se conservan O R3 activo, cuantizador → O R3 y raíz → O R2 en espera.
Auditoría vigente R20: cuatro coordinadores vivos y una sola tanda de Vivado.
[Contrato, pruebas, cola y siguiente transporte](../LLAMA32_POST_ATTENTION_R1.md).

---

Actualización 2026-10-14 04:26 UTC: **raíz → cuantización → O real encolada**.
El paquete R2 incorpora 34 módulos RTL propios y 53 fuentes Verilog HLS;
el banco completo vuelve a compilar con el watchdog corregido. Espera
PASS y respaldo de cuantización → O R3, detrás del O R3 actualmente en XSim.
Los tres coordinadores están vivos; solo la tanda O ocupa Vivado.
La nueva etapa exige dos resultados completos, cancelación/recuperación,
materialización K/V y ABI1, revisión de 67 controles y respaldo por tanda.
Auditoría vigente R19. [Paquete, alcance y criterios](../LLAMA32_ATTENTION_PROJECTED_ROOT_REAL_QUEUE_R2.md).

---

Actualización 2026-10-14: **O original PASS numérico; lazo de watchdog detectado
y corregido en una nueva revisión**. Se verificaron 10.240 resultados FP64,
40.960 bytes FP32 y 32 controles; el banco y el diagnóstico de síntesis están
respaldados en D: y NFS. El controlador corregido pasó 145 casos nativos y
83 controles. La tanda `20261014T041103Z` ya pasó síntesis con cero lazos
(19.981 LUT, 17.996 FF, 38 BRAM, 54 DSP; WNS OOC +1,019 ns) y ejecuta
el banco XSim completo. La integración cuantización → O R3 espera
su revisión y respaldo; R1/R2 quedaron como antecedentes detenidos.
El banco de raíz completa compila y dispone de verificador preparado, aún
sin ejecución aritmética. Auditoría vigente R18.
[Evidencia, cambio y criterios](../LLAMA32_ATTENTION_O_WATCHDOG_RECOVERY_R3.md).

---

Actualización 20261013T230847Z: **unión raíz de atención → cuantización → O implementada**.
Protocolo nativo: 93 escenarios, diez mutantes de RTL detectados y 78 controles
de traza rechazados. La gráfica de 34 módulos RTL reales compila con tres
interfaces HLS vacías; no constituye validación aritmética.
[Diseño, pruebas y próximo banco](../LLAMA32_ATTENTION_PROJECTED_ROOT_NATIVE_R1.md).
Se conserva el O real activo y la etapa 24 en espera. El banco real del nuevo
wrapper está planificado, todavía no encolado; auditoría vigente R16.

---

Actualización 20261013T224949Z: **cuantización real → memoria AXI → O real encolado**.
Cuatro cuantizaciones, tres proyecciones completas y cancelación durante el
traspaso; O leerá exclusivamente los datos publicados por el escritor AXI.
Compilación sintáctica y esquema estático aprobados; aritmética integrada pendiente.
Fuentes verificadas en D:/NFS; 37 controles negativos preparados para la traza real.
[Contrato, dependencias y respaldo](../LLAMA32_ATTENTION_OUTPUT_PATH_REAL_QUEUE_R1.md).
La etapa 24 espera al O real activo, revisión y respaldo; auditoría R16.

---

Actualización 20261013T223134Z: **primer tensor O real completo, bit a bit**:
2.048 FP64 y 8.192 bytes FP32; 1.606.203 ciclos de matriz (8,031015 ms a 5 ns).
Intervalo AXI inactivo máximo verificado: 11.820 ciclos, inferior a 65.536.
Captura respaldada en D:/NFS; el banco de siete trabajos continúa y aún no
tiene veredicto completo. [Evidencia y límites](../LLAMA32_ATTENTION_O_FIRST_TENSOR_R1.md).

---

Actualización 20261013T222735Z: **cuantizador HLS → publicación AXI aprobado y respaldado**:
24 trabajos, 61.440 bytes válidos exactos, 30 corrupciones rechazadas;
7.221 LUT, 6.566 FF, 3,5 BRAM, 3 DSP; WNS de síntesis +0,807 ns.
El enlace ABI1 → O pasó 105 escenarios nativos con sustitutos, nueve mutantes
y 57 controles negativos; composición aritmética todavía pendiente.
[Resultados, métricas y contratos](../LLAMA32_ATTENTION_OUTPUT_PATH_RESULTS_R1.md).
La tanda O real continúa; su observador R2 corrige metadatos de procedencia
sin reiniciar Vivado. Auditoría vigente R15; 23 etapas.

---

Actualización 20261013T220812Z: **proyección O con GEMV y MAC HLS reales encolada**.
Siete trabajos sobre cuatro activaciones de atención y pesos Q8_0 reales;
comparación FP64/FP32, límite de región, cancelación y recuperación.
Compilación R2 aprobada y respaldada; cálculo integrado todavía pendiente.
La etapa 23 espera a cuantizador/AXI R2 y su respaldo; conserva el trabajo activo.
[Contrato y evidencias](../LLAMA32_ATTENTION_PROJECTION_MEMORY_R1.md).
Auditoría vigente R13; límite de esta tanda: dos CPU, 7.200 s, 5 ns OOC.
El enlace Q8_0 del usuario coincide por SHA-256 con el archivo local existente.

---

Actualización 20261013T214901Z: **escritor de activaciones aprobado en XSim y
síntesis**: 214 escenarios y 462.848 bytes publicados exactos; archivo
`c6cac10cf0c1f9913cca737cdb6ba1ef0cd9c3d1f41f7504b003805810bee15c` verificado en D: y NFS.
La composición cuantizador HLS → propietario → AXI está ejecutándose como
`20261013T214728Z-V7-llama32-attention-quantize-store-vivado-r2`. Coordinador R2 con lectura tolerante a escrituras parciales
y reemplazo atómico; cuatro pruebas de recuperación aprobadas.
[Resultados y recuperación](../LLAMA32_ATTENTION_STORE_XSIM_QUEUE_RESULTS_R2.md).
Auditoría vigente R12; cola de 22 etapas. La medición física del motor
[quedó revisada aparte](../LLAMA32_ATTENTION_ENGINE_ROUTED_RESULTS_R2.md).

---

Actualización 20261013T214349Z: **motor de atención enrutado con timing interno
OOC aprobado**: WNS +0,195 ns, WHS +0,015 ns; 33.574 redes completamente
enrutadas, cero errores. La revisión R2 distingue el reloj de un BUFGCE que
Vivado insertó para 2.048 habilitaciones; 23 controles negativos. Se conserva
el diagnóstico R1 y no se repite implementación.
[Mediciones, conectividad y límites](../LLAMA32_ATTENTION_ENGINE_ROUTED_RESULTS_R2.md).
El controlador O pasó 145 escenarios XSim/síntesis y quedó respaldado.
La tarjeta completa y simulación funcional posterior al ruteo siguen pendientes.

---

Actualización 20261013T213049Z: **cuantización HLS y propietario XSim/síntesis
aprobados y respaldados**. 27 casos C/RTL, cuatro vectores reales de atención
bit a bit; 101.571 ciclos por vector de 2.048 FP32. Propietario: 219 escenarios.
[Resultados y límites](../LLAMA32_ATTENTION_QUANTIZE_RESULTS_R1.md).

**Cuantizador → propiedad de memoria → escritor AXI encolado**: 72 escenarios
nativos, seis mutantes detectados, 90 corrupciones rechazadas. La tanda
formal usa el RTL HLS generado, sin sustituto aritmético, y depende de la
publicación AXI R1. [Contrato y evidencia](../LLAMA32_ATTENTION_QUANTIZE_STORE_R1.md).
Cola de 22 etapas; auditoría vigente `audit_attention_composition_active_r11.py`.
Se conserva el ruteo R3 en curso y sus sucesores proyección R3 y escritor R1.

---

Actualización 20261013T210631Z: **publicación de activaciones mediante el escritor
AXI real validada en RTL nativo y encolada**. Cuatro vectores reales, 214
escenarios, 8.132 transferencias W y 8.126 B (resets interrumpen seis
transacciones). Diez mutantes detectados y 123 controles negativos.
[Contrato, evidencias y límites](../LLAMA32_ATTENTION_ACTIVATION_STORE_R1.md).

La cola tiene 21 etapas: cuantización HLS R2 → propietario Vivado R3 →
ruteo R3 → proyección R3 → publicación AXI R1. La raíz R3 ya está
revisada y respaldada. Auditoría vigente R10. No se modificaron las
fuentes o pruebas en curso; la composición aritmética posterior sigue pendiente.

---

Actualización 20261013T210023Z: **wrapper completo de atención aprobado en
XSim/síntesis y respaldado**: 23 raíces, 426 consultas hijas, 24.576 resultados
públicos, 423 tiles numéricos y 0 ULP máximo. El verificador corrigió una
definición de elapsed de un ciclo; la tanda original no se repitió ni modificó.
La revisión archivada incluye 26 controles de traza y siete de identidad.
[Resultados, timing y procedencia](../LLAMA32_ATTENTION_ROOT_XSIM_SYNTH_RESULTS_R2.md).

20.709 LUT, 19.414 FF, 43 BRAM y 77 DSP; **WNS −0,443 ns a 5 ns**, todavía
sin cierre temporal. Coordinador raíz R3; cuantización HLS R2 reanudó la
cola, seguido de propietario/ruteo/proyección R3. Auditoría R9.

---

Actualización 20261013T204319Z: **fallo de liberación de fuente reproducido y
corregido en el propietario del cuantizador R2**. R1 podía reconocer una
respuesta ajena si cambiaba su identidad durante el cálculo. El nuevo RTL
compara secuencia y época completas antes del ACK; 219 escenarios nativos,
59.904 palabras, nueve mutantes y 78 controles de traza revisados.
[Contrato, reproducción y respaldos](../LLAMA32_ATTENTION_QUANTIZE_OWNER_RELEASE_R2.md).

Se sustituyeron únicamente tres coordinadores en espera. Cadena vigente:
raíz R2 → HLS R1 → propietario R2 → ruteo R2 → proyección R2. La auditoría R8
reemplaza a R7; los registros R1 retirados se conservan como evidencia histórica.
La composición aritmética con el cuantizador todavía requiere validación.

---

Actualización 2026-10-13, 20:27 UTC: **adaptador de identidad para
proyección O validado en RTL nativo y encolado**. Conserva la identidad
raíz de 64 bits y reserva secuencias internas por grupo del GEMV, sin
truncarla. Cuatro bancos: 145 escenarios, incluidos errores y rechazos
esperados; ocho mutantes RTL detectados y 83 corrupciones de traza
rechazadas. Todavía es un banco de protocolo, sin GEMV/AXI aritmético.
[Contrato, cobertura y respaldo](../LLAMA32_ATTENTION_PROJECTION_CONTROL_R1.md).

El wrapper completo conserva la tanda
`20261013T200719Z-V7-llama32-attention-root-vivado-r1`: XSim 88430 continúa
activo y avanzó hasta la operación hija 258. No tiene aprobación terminal
aún. La auditoría R7 verifica cinco propietarios y la cola ordenada:
wrapper → cuantización HLS → propietario del cuantizador → estudio de
ruteo → adaptador de proyección. Los archivos de cada etapa están congelados.

---

Actualización 2026-10-13, 20:11 UTC: **secuenciador R4 y guard raíz R3
aprobados en XSim/síntesis y respaldados**. El secuenciador acumuló 140
trabajos y 73.728 palabras en cuatro bancos; el guard pasó 4.896 comprobaciones
por capa. [Recursos, cobertura y hashes](../LLAMA32_ATTENTION_CONTROLLERS_XSIM_SYNTH_RESULTS_R1.md).

El wrapper completo R2 está ejecutándose en la tanda
`20261013T200719Z-V7-llama32-attention-root-vivado-r1`; XSim 88430 vivo,
avance observado hasta operación hija 54. Es progreso parcial, todavía sin
aprobación terminal. Las validaciones terminadas no se repiten.

**Estudio de ruteo del motor encolado** después del controlador de
cuantización. Reutiliza el checkpoint verificado, añade un BUFGCE explícito
y medirá timing/rutas del checkpoint reabierto, con dos CPU y límite de
1.800 s. No altera la tanda en curso ni relaja el reloj de 5 ns.
[Contexto físico, criterios y paquete](../LLAMA32_ATTENTION_ENGINE_ROUTE_PLAN_R1.md).
La auditoría R6 verifica cuatro propietarios vivos y una sola tanda activa
de esta cola.

---

Actualización 20261013T200013Z: **motor R7 aprobado y respaldado**.
155 consultas, 9.600 resultados públicos, 0 ULP máximo y 42 controles
negativos. Síntesis OOC: 19.410 LUT, 16.513 FF, 33,5 tiles BRAM y 77 DSP.
**Setup WNS −0,456 ns a 5 ns: timing todavía incumplido**; no se acredita
200 MHz ni timing físico. [Resultados, límites y hashes](../LLAMA32_ATTENTION_ENGINE_XSIM_SYNTH_RESULTS_R1.md).
El respaldo completo del motor permite continuar el secuenciador R4 y sus
dependientes sin repetir las validaciones aprobadas.

---

Actualización 2026-10-13, 19:56 UTC: **la traza completa del motor R7 pasó
revisión independiente**: 155 consultas, 9.600 resultados, 155 tiles
numéricos, error máximo de salida 0 ULP y 42 controles negativos rechazados.
Se ejercitaron las cuatro continuaciones que antes exponían la lectura
especulativa del estado. [Observación y límites](../LLAMA32_ATTENTION_ENGINE_R7_SIMULATION_OBSERVATION_R1.md).
El coordinador conserva la revisión terminal de síntesis/fuentes y el
respaldo de la tanda; sus dependientes permanecen en espera.

El respaldo de trabajo R31, de 1.677 miembros, quedó verificado en D:/NFS:
`llama32-model-working-r31-20261013T195451Z.zip`, SHA-256
`022729167728ee4e9dbb3f3c7113ddb0af96eef1cdc98051e5fb8e5bb6c5ebb1`.
La observación anterior se generó después de ese archivo y requiere
incorporación al respaldo siguiente.

---

Actualización 2026-10-13, 19:53 UTC: **controlador de propiedad de
cuantización validado en simulación nativa y encolado para XSim/OOC**.
Tres semillas: 183 trabajos, 50.688 palabras y 18 resets; seis mutantes RTL
detectados y 20 corrupciones de traza rechazadas por semilla. El banco
valida protocolo con un emulador HLS, sin calificar aún su aritmética ni
integración con la raíz. [Contrato y respaldo](../LLAMA32_ATTENTION_QUANTIZE_OWNER_R1.md).

El paquete nuevo está verificado en D:/NFS y espera al cuantizador HLS.
La auditoría `attention-composition-r5-active-audit-20261013T195135Z.json`
verifica seis propietarios locales y una sola tanda de esta cola: motor R7.
XSim terminó y Vivado avanzó a síntesis; revisión terminal y respaldo siguen
pendientes. No se reinició ni canceló esa validación.

El GGUF local `Llama-3.2-1B-Instruct-Q8_0.gguf` conserva SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`;
se reutiliza la descarga ya verificada.

---

Actualización 2026-10-13, 19:31 UTC: **añadida a la cola la cuantización de
atención FP32 → ABI1**, detrás del wrapper R2. La implementación C++ pasó
27 casos y 47.872 bytes exactos, con cuatro vectores reales completos de
atención. Los revisores rechazaron diez corrupciones numéricas, diez de
transacciones y cuatro de entradas; C/RTL sigue pendiente. Paquete de 36
miembros verificado en D:/NFS. [Contrato y siguiente enlace RTL](../LLAMA32_ATTENTION_QUANTIZE_R1.md).

Motor R7 continúa en la misma tanda XSim: proceso 71218 vivo y avance hasta
consulta 133. Cinco propietarios verificados; la tanda actual conserva su
slot y los cuatro dependientes esperan. No se atribuye PASS al motor aún.

---

Actualización 2026-10-13, 19:20 UTC: **motor R7 ejecutándose; secuenciador R4,
guard raíz R3 y wrapper completo R2 en espera ordenada**. El motor R6 encontró
una lectura especulativa adicional del estado en la consulta 135; diagnóstico
y fuentes quedaron respaldados. El guard nuevo conserva la última palabra
válida y pasó 2.304 comprobaciones nativas, reset y recuperación.
[Causa, revisiones y archivos](../LLAMA32_ATTENTION_PREFETCH_RECOVERY_R1.md).

**El banco del wrapper de 32 cabezas ya está encolado**: 23 escenarios, 426
consultas hijas y 24.576 resultados finales por comparar. Compilación de
interfaz y unión de referencias verificadas; XSim, revisión numérica y
síntesis siguen pendientes. Cada padre debe pasar y quedar respaldado antes
de ejecutar su dependiente. La auditoría verifica cuatro propietarios locales
y una sola tanda Vivado de esta cola.
[Cobertura, límites y paquete](../LLAMA32_ATTENTION_ROOT_R1.md).

---

Actualización 2026-10-13, 18:48 UTC: **contexto y memoria pasaron XSim y
síntesis OOC, con respaldo D:/NFS**. Memoria completa: 6.159 LUT, 7.256 FF,
23,5 tiles BRAM y 3 DSP; WNS +0,662 ns a 5 ns, sin timing ruteado.
[Evidencia](../LLAMA32_ATTENTION_MEMORY_XSIM_SYNTH_RESULTS_R1.md).

La integración real detectó dos lecturas especulativas del HLS: estado
inicial y un elemento adicional al terminar K/V. Se preservaron ambos
diagnósticos; dos adaptadores aíslan esas lecturas y conservan los controles
de memoria. El guard K/V pasó 811.206 comprobaciones nativas. **Motor R6
está ejecutándose**, seguido por secuenciador R3 y guard raíz R2. No se
repiten los padres completos. [Recuperación y hashes](../LLAMA32_ATTENTION_PREFETCH_RECOVERY_R1.md).

**Wrapper raíz implementado y compilado**. Su guard de mantenimiento pasó
4.896 comprobaciones por capa y tres mutantes RTL; queda pendiente la
simulación del wrapper completo con las 32 cabezas y aritmética real.
[Contrato y criterios siguientes](../LLAMA32_ATTENTION_ROOT_R1.md).

---

Actualización 2026-10-13, 18:20 UTC: **append, caché, RAM privada y camino KV
pasaron XSim y síntesis OOC** y quedaron respaldados. El camino conjunto
usa 3.971 LUT, 4.474 FF, 23,5 tiles BRAM y 3 DSP; WNS +0,394 ns al reloj
de 5 ns, sin calificación física. Se verificaron 70 consultas y 67.364
palabras K/V por suite. [Resultados y archivos](../LLAMA32_KV_PATH_XSIM_SYNTH_RESULTS_R1.md).
Contexto R5 está ejecutándose; memoria R4 y motor generado R4 esperan.

**Implementado el controlador de 32 cabezas y encolado su banco Vivado**
detrás del motor. La prueba aislada Icarus pasó 140 escenarios, 20 resets
y 73.728 palabras; cinco mutantes RTL fueron rechazados. Se verificaron
la carga Q completa antes de emitir cabezas, leases sucesivas, publicación
atómica, cancelación/revocación y recuperación. El peer no calcula atención;
falta el wrapper que aplique el bloqueo de bind/append al motor real.
[Contrato y evidencia](../LLAMA32_ATTENTION_HEAD_SEQUENCER_R1.md) ·
[Paquete y dependencias](../LLAMA32_ATTENTION_HEAD_SEQUENCER_QUEUE_R1.md).

---

Actualización 2026-10-13, 17:57 UTC: **cargador KV pasó 340 casos XSim
y síntesis OOC**, con 1.300 LUT, 2.013 FF, 2 DSP y cero cajas negras/latches.
WNS +2,572 ns y WHS +0,066 ns al reloj de 5 ns son estimaciones de síntesis,
sin cierre físico. El revisor confundía el nombre del objeto `design_1` con
el módulo superior: se corrigió usando tres informes y el comando real,
se rechazaron siete corrupciones y se adoptó la evidencia sin repetir herramientas.
[Resultado y archivo original preservado](../LLAMA32_KV_LOADER_XSIM_SYNTH_RESULTS_R1.md).

Append R9 está lanzado; le siguen caché R7, RAM R6, camino KV R6, contexto
R5, memoria R4 y motor R4. Todos mantienen fuentes/fixtures y dependencias,
con la comprobación de identidad corregida. La cápsula de recuperación
D:/NFS tiene SHA `fc64279a1e4d491a584436d768ed9f2cd1c434c8c4e6a5c855b38dfeacc29f91`.
Los 33 archivos del núcleo generado también se compararon con el archivo
terminal HLS; revisión `712f94b047ce7cd9ebb8c845ccafe6b6cdcd7d9f1987075f250ae138ce292b0e`.

---

Actualización 2026-10-13, 17:48 UTC: **atención HLS pasó las 291
transacciones C/RTL**, con 9.152 salidas exactas respecto de la referencia
independiente y 30.264 palabras de estado verificadas. Latencia máxima
58.266 ciclos; estimación HLS de 20 BRAM18K, 74 DSP, 7.265 FF y 15.227 LUT.
No se califica timing físico ni inferencia completa.
[Resultado y cierre del supervisor](../LLAMA32_ATTENTION_C_RTL_RESULTS_R1.md).

**La composición nativa atención→O→residual→entrada FFN también pasó**:
cuatro muestras, 524.288 bloques MAC y 20 archivos de salida exactos.
La referencia usa los pesos reales Q8_0 de O y las 32 cabezas de capa 0;
rechazó 13 corrupciones. Fuente, fixture y resultados tienen respaldo
D:/NFS. [Alcance y hashes](../LLAMA32_ATTENTION_OUTPUT_NATIVE_RESULTS_R1.md).
Se especificó el [secuenciador de 32 cabezas](../LLAMA32_ATTENTION_HEAD_SEQUENCER_R1.md),
pendiente de implementación y de la validación del motor generado.

La cola descubrió dos incompatibilidades de Vivado: sintaxis de `-testplusarg`
y tipos net implícitos bajo `default_nettype none`. Se preservaron los dos
fallos; el segundo ya había pasado 340 casos XSim. Tras probar sintaxis y
elaboración se lanzaron ocho revisiones nuevas. El cargador KV R7 está
en síntesis y los siete descendientes esperan revisión y respaldo.
[Recuperación y orden vigente](VIVADO_KV_PORTABILITY_RECOVERY_R1.md).
La auditoría de las 17:47 verifica ocho propietarios y un único trabajo
Vivado de esta cola; no se reiniciaron los bancos HLS completos.

---

Actualización 2026-10-13, 17:17 UTC: **encolada la integración real de
atención HLS con AXI y memoria privada**, detrás de la composición de
memoria R1 y del HLS de atención. El paquete de 81 archivos tiene respaldo
D:/NFS; se verificaron los 33 archivos generados y el adaptador. Los
revisores rechazaron 27 corrupciones de trazas y ocho controles de padres.
Estas comprobaciones previas no califican la integración: el banco real
de 155 consultas y la síntesis quedan pendientes de sus dependencias.
[Contrato y respaldo de la cola](../LLAMA32_ATTENTION_ENGINE_QUEUE_R1.md).

Atención continúa en la misma tanda y con el mismo XSIM. Tras 234/291
transacciones quedaban 612/1.063 tokens en tiles largos; el plazo efectivo
se amplió de 40 a 90 minutos totales, hasta 18:25:33 UTC, con supervisión
acotada y respaldo previo. [Identidades y mecanismo](ATTENTION_HLS_RUNTIME_EXTENSION_R1.md).
La auditoría R2 de las 17:17 verifica los nueve propietarios locales
activos y un solo grupo de herramientas dentro de esta cola.

---

Actualización 2026-10-13, 16:57 UTC: **normalización→Q8 pasó C/RTL**.
Se verificaron las 61 transacciones, 2.860 registros ABI1 y 183.040 bytes
exactos; los 12 rechazos conservaron sus 512 palabras de salida. Máximo
167.188 ciclos / 835,94 µs al reloj simulado de 5 ns, sin incluir AXI/DDR.
La tanda original terminó en 3.579,49 segundos, sin reiniciar herramientas;
el guardián y el resultado real están dentro del respaldo terminal.
Archivo D:/NFS: `411d90a513f6d68acb7735e617cd0451a5fc531fd38529b5fe585ae1b1433be4`.
[Métricas y revisión](../LLAMA32_NORM_Q8_C_RTL_RESULTS_R1.md).

La cola inició `20261013T165530Z-V7-llama32-attention-hls-r1` tras verificar
ese respaldo. Sus 35 puertos generados ya coinciden con el adaptador
estricto; la cosimulación continúa. El nuevo shell/banco de atención
compila y tiene respaldo; se corrigió la secuencia de reset y se preparó
el revisor AXI/RAM/Decimal90. **La integración completa todavía no se ha
simulado**. [Alcance y dependencias](../LLAMA32_ATTENTION_ENGINE_R1_PREFLIGHT.md).
Los restantes siete propietarios de la cola mantienen sus dependencias.

---

Actualización 2026-10-13, 16:26 UTC: **preparados y revisados los datos
para conectar atención HLS con KV por AXI**. Se conservaron los 143 casos
numéricos, incluidos 128 con proyecciones reales de capa 0. La partición
de producción de 64 tokens requiere 147 llamadas. El banco nativo pasó
con MSVC y con bibliotecas matemáticas AMD: 9.152 salidas FP32 con 0 ULP
de diferencia y 15.288 palabras de estado revisadas. Las 16 imágenes
contienen 747 appends / 3.059.712 bytes; el revisor reconstruyó 136.064
palabras causales K/V y rechazó 12 corrupciones semánticas. Se rechazaron
además 12 corrupciones numéricas y una referencia alterada ejecutada.
[Contrato, reproducción y alcance](../LLAMA32_ATTENTION_AXI_FIXTURE_R1.md).
Tanda `20261013T162456Z-V7-attention-axi-native-amd-r1`, archivo D:/NFS
`c0e02eae126c68c8649e2c4ebe49b0ddab3a703f0ad00732bd9ddc8b3966e089`;
revisión independiente
`6cf4dbcfde78538d15f31e0af3aae3646c2beca3ae3b07442313d500477ed6e5`.
Es validación nativa/de imágenes; el núcleo de atención generado aún no
está conectado a la composición RTL.

Norma→Q8 sigue en su **misma ejecución**, a 28/61 en la inspección viva.
Se conservó el banco completo ampliando el plazo total de 40 a 90 minutos
con un guardián acotado; vencimiento 17:23:37 UTC. Solo está detenido el
wrapper Python que esperaba; Vitis y XSIM mantienen sus procesos originales.
El plan, prueba de la espera y respaldo verifican la transferencia de
supervisión. [Detalles y seguimiento](NORM_Q8_RUNTIME_EXTENSION_R1.md).
La cola de dependencias posterior permanece intacta.

---

Actualización 2026-10-13, 16:03 UTC: **RMSNorm, RoPE y residual pasaron
C/RTL y revisión de transacciones reales**. Se compararon respectivamente
78.208 / 61.760 / 51.556 valores en 60 / 48 / 43 transacciones.
RMSNorm N=2.048 tarda 65.615 ciclos; RoPE y residual alcanzan máximos
de 27.511 y 4.110 ciclos en sus propios fixtures. Son simulaciones con
memoria local y estimaciones HLS, sin validación de DDR física o timing
ruteado. [Métricas, formato binario y evidencia](../LLAMA32_HLS_BINARY_TRANSACTIONS_R1.md).

La lectura UTF-8 del archivo binario de RMSNorm había detenido al observador.
Se recuperaron las transacciones desde el archivo original, sin repetir HLS.
La cola `hls-binary-tv-recovery-r1-preparation.json` conserva el orden y
añade decodificación estricta, hashes de bytes originales y cotejo de las
RAM de entrada. RMSNorm transacciones R8, RoPE R7 y residual R7 están
terminados y respaldados. Norma→Q8 R7 corre en
`20261013T155333Z-V7-llama32-norm-q8-hls-r1`; la inspección SSH confirmó
XSIM activo en el directorio propio. No se reiniciaron las validaciones.

**Contexto R1 queda descartado para integración.** La composición con KV
encontró realimentación combinacional de cancelación, aun en una consulta
válida. [Contexto R2](../LLAMA32_ATTENTION_CONTEXT_R2.md) registra esa
cancelación y pasó 207 casos independientes. La
[composición KV/contexto](../LLAMA32_ATTENTION_MEMORY_R1.md) pasó cuatro
suites: 244 consultas, 16 resets y 321.536 palabras K/V cotejadas desde AXI.
Se rechazaron siete mutaciones RTL y 25 corrupciones de evidencia por suite.
Revisión independiente:
`35dc139c1c232d0e251ba59aec0ae0a965fb278e73b48356a1df1711f10a3e42`.
El cálculo del banco es un checksum; aún no está conectado el HLS de atención.

Orden vigente después de norma→Q8: atención HLS R5 → KV loader Vivado R5 →
KV append R6 → KV cache R4 → KV RAM R3 → KV path R3 → **contexto R2 →
composición memory R1**. Los dos últimos ya tienen supervisores activos,
fuentes congeladas, paquetes verificados D:/NFS y pruebas de sus condiciones
de entrada. SHA-256 de paquetes:
`2e3df6a229163ce23a6326bcaf322a81d3270b4515519ad3046d5a40037c4621`
y `6b3c2d26cbe41359eed0ba3c0001a120096dba4e7bf21d05c0a4f83d02d51a20`.
No se reencoló contexto R1. Auditoría de los nueve supervisores vivos:
`hls-binary-tv-recovery-r1-active-audit-20261013T160336Z.json`, SHA-256
`43bf368618bf50246b73bdba857c5e3aef53bf88864f5b421c26014175f9bf9e`.

El GGUF enlazado por el usuario ya está en
`D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/`.
Se volvió a verificar su SHA-256 completo:
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
Se conserva fuera del repositorio; `AGENTS.md` existente permanece intacto.

---

Actualizaci?n 2026-10-13, 15:14 UTC: el replay funcional del **netlist propio
mask rows pas?**: 174.436 filas, 447 bits p?blicos, 16 bancos, seis variantes
de entrada y 34 corrupciones rechazadas. Tanda
`20261013T142543Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-mask-rows-binary-r1`,
archivo terminal SHA-256
`904bba10023b6fc16939d6b5aae80e06855224b09224e06142ac48ed76f3ecc3`,
revisi?n `3d32cdb12241a8bfe95549df28f6cd6ea32bcdcf6898711f0b248b820726ee8c`.
Respaldo D:/NFS verificado. La s?ntesis correspondiente sigue fallando
setup/hold: WNS ?2,768 ns y WHS ?0,075 ns a objetivo 4 ns. Este PASS
funcional sin SDF no acredita timing f?sico ni PCIe nativo.

RMSNorm intent? continuar en la tanda `20261013T145738Z-V7-llama32-rmsnorm-hls-r1`,
pero Vitis 2026.1 rechaz? `config_compile -unsafe_math_optimizations false`
antes de compilar. Se conserv? su diagn?stico completo: archivo SHA-256
`cd411ef77e642b6ffcf799bf886acbd3c1c1534d365468f91447b372ddbf6cab`.
La prueba real `20261013T150146Z-hls-config-bool-probe-r1` acept?
`config_compile -unsafe_math_optimizations=false`, sintaxis documentada por
[AMD](https://docs.amd.com/r/en-US/ug1399-vitis-hls/config_compile).
Se a?adieron `run_hls_r2.tcl` y runners Python R2 para RMSNorm, RoPE,
residual, norma?Q8 y atenci?n, sin cambiar C++, cabeceras o fixtures.
Los archivos y snapshots anteriores permanecen intactos.

La recuperaci?n `hls-bool-recovery-r1-preparation.json` conserva las
once dependencias desde RMSNorm HLS R7 hasta atenci?n KV path Vivado R2.
Los once procesos arrancaron con sus comprobaciones reales de entrada.
La c?psula previa de 390 miembros est? verificada en D:/NFS, SHA-256
`720c88650e66c4b1a2cb87ca1a7e893a540172c14ae7f3b7dea70461ad8be8bc`.
La tanda nueva `20261013T150621Z-V7-llama32-rmsnorm-hls-r1` ya super? C y
s?ntesis y lleva **27/60 transacciones RTL observadas**, sin resultado
terminal todav?a. Se verificaron Vitis 4158750/4158810, HLS 4158856 y
XSim 4161246 dentro de esa misma tanda; no se reinici? el trabajo.

El [contexto privado de atenci?n R1](../LLAMA32_ATTENTION_CONTEXT_R1.md)
pas? tres suites nativas: **207 casos, 231 tiles y 12 resets**. Q y el
estado permanecen privados entre tiles; start se mantiene hasta ready;
cancelar requiere drenar HLS; la salida necesita el certificado global
y autorizaci?n de ?poca vigente. La revisi?n independiente cotej?
26.664 palabras le?das entre las tres suites, rechaz? 15 corrupciones
por suite y seis defectos RTL compilados. Tanda nativa
`20261013T145956Z-V7-attention-context-native-r1`, archivo SHA-256
`989cc4d8f929598c7172891e6e571deaf7b401499482f0b9b2de4027acc232cc`.
El banco usa un par de protocolo, no aritm?tica HLS ni el camino KV real.

Queda encolado `llama32-attention-context-vivado-r1-coordinator.json`,
PID 80272, detr?s de KV path R2: tres suites XSim y s?ntesis OOC VU5P,
5 ns, dos CPU, 1200 s. Paquete SHA-256
`89850b2f343f1e13d20462f0dfdd60490777098d853cfc7b2f065552b244e37e`,
snapshot de 34 miembros `eb5a3c89751ef42bba796f803096dd7f84cb445d68e46c2b4f58a605c339f4e0`,
ambos respaldados. El preflight comprob? arranque, tres fragmentos Python,
sintaxis Tcl y seis rechazos de dependencia incompleta. Falta unir y probar
KV + contexto + puertos generados HLS, adem?s de DDR f?sico, timing de
la tarjeta e inferencia completa. AGENTS.md, README.md y CMakeLists.txt
conservan sus hashes protegidos.


Actualización 2026-10-13, 06:04 UTC: la simulación funcional del netlist
**ruteado de RQ bitmap pasó** su revisión independiente: 174.436 filas,
447 bits públicos, seis variantes de entrada rechazadas y 34 controles
de corrupción de evidencia. Tanda
`20261013T053805Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-route-binary-r1`,
archivo terminal verificado con SHA-256
`c8f791f8ec53d121d5a56ecbe71f22ec64891badc998d136676080c31e425536`.
La revisión tiene SHA-256
`928504ce233bfb3d1d2f4d1757dfdf2402a1fe1468bb3647d38dcb78405e2607`.
El PASS funcional no elimina las violaciones de timing descritas abajo.

Ya comenzó el replay ruteado de row enable, tanda
`20261013T060404Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-route-binary-r1`;
se verificaron procesos Vivado 4073484/4073535 y su invocación en curso.
El resto de la cola conserva sus dependencias y respaldos por tanda.

La [integración de caché KV](../LLAMA32_KV_CACHE_R1.md) dejó de ser un
borrador sin probar: aprobó cuatro suites Icarus con **1008 operaciones y
24 resets**, incluidos AXI de lectura/escritura concurrentes, prefijos
publicados, GQA, reintentos y reset con tráfico o concesiones retenidos.
Un revisor independiente cotejó los bytes escritos con los leídos y
rechazó 19 corrupciones por suite. Seis variantes RTL compiladas con
defectos también fueron rechazadas. La tanda nativa canónica es
`20261013T055226Z-V7-kv-cache-native-r1`.

Por suite: 307.200 bytes publicables de escritura y 53.760 bytes de
lecturas exitosas. El tile K/V de 64 posiciones necesitó 1261–1304 ciclos
entre solicitud y registro de finalización, incluidos cinco ciclos de
retención de respuesta; no es una medición de DDR físico.

El coordinador `llama32-kv-cache-vivado-r1-coordinator.json` está activo
(PID 43036) detrás del escritor KV R3. Su paquete de 40 miembros está
verificado en D: y NFS, SHA-256
`bf7aa1e21d7b1bdca6c7dae39cca4082be4efd10ff2e5fec4aff4f64dd7e5cae`.
El preflight ejecutó los controles reales de arranque, comprobó tres
fragmentos Python remotos y sintaxis Tcl, y rechazó siete dependencias
incompletas. **XSim/síntesis de la integración KV siguen pendientes**;
el adaptador hacia atención HLS, DDR real y la inferencia completa no
están calificados.

Actualización 2026-10-13, 05:36 UTC: se recuperó la cola detenida en la
revisión física. La detención del 12 de octubre a las 07:01 UTC fue un
rechazo real del comprobador R2: placement eliminó siete primitivas en
RQ bitmap y 25 en row enable. Se conservaron ese diagnóstico y todas las
tandas completadas. No se repitió la FFN, síntesis, placement ni readback.

La [revisión estructural nueva](PCIE_SYMBOLIC_MONITOR_PREDECODE_PRIMITIVE_CONGRUENCE_R1.md)
releyó los cuatro EDIF archivados, justificó las 32 eliminaciones y
comprobó las 32 réplicas añadidas, cada primitiva conservada y las 447
salidas por variante. Rechazó 21 alteraciones deliberadas. El comprobador
físico R3 conserva los 22 controles negativos anteriores. Ambas revisiones
físicas R2 de los candidatos pasaron y quedaron comparadas y respaldadas.
Esto permite ejecutar sus simulaciones ruteadas; no las sustituye.

A las 05:38 UTC comenzó la simulación ruteada de RQ bitmap en la tanda
`20261013T053805Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-route-binary-r1`.
La invocación remota conserva el netlist de la implementación del 12 de
octubre, el banco original de 174.436 filas y el límite de 3.000 segundos.
Todavía no tiene resultado terminal. Las demás etapas esperan sus
dependencias; los trabajos externos a esta cola se conservan.

Los recursos y slacks de estas tandas Quick/OOC son:

| Variante | LUT | FDRE | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: |
| RQ bitmap | 93374 | 27148 | -11.140 | -2.667 |
| Row enable | 97232 | 27701 | -8.728 | -3.683 |

**Ambas incumplen timing** y mantienen las limitaciones de interfaz OOC.
La mejora de setup de row enable viene acompañada de más recursos y peor
hold; no acredita frecuencia de operación física.

Se activaron 13 coordinadores nuevos, desde revisión física R6 y replay
ruteado R5 hasta atención HLS R2, lector KV Vivado R2 y escritor KV Vivado
R3. Las fuentes de hardware, vectores, límites y orden anterior se
conservan. El manifiesto es
`pcie-symbolic-monitor-predecode-physical-recovery-r1-preparation.json`;
el registro de inicio, `pcie-symbolic-monitor-predecode-physical-recovery-r1-launch.json`,
contiene sus PID y comprobaciones de arranque.

Respaldo completo adicional verificado en D: y NFS:
`physical-recovery-r1-capsule-20261013T053458Z.zip`, 375 miembros,
SHA-256 `44c31f00973bb417250c366a8521763fa87faf0e2369dc2f0c830054f7becaf6`.
Incluye los cuatro grafos, herramientas nuevas y toda la orquestación
cambiada. El borrador `ntrpu_kv_cache_r1.sv` y su banco incompleto se
preservaron expresamente como **no probados**. AGENTS.md, README.md y
CMakeLists.txt mantienen sus hashes protegidos.

También se volvió a verificar el GGUF Q8_0 de Unsloth en
`D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/`: 1.321.082.528
bytes, SHA-256 `3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`,
idéntico al publicado. No fue necesario descargarlo de nuevo.

Actualización 2026-10-12, 04:47 UTC: la [FFN completa con pesos reales](../LLAMA32_FULL_FFN_METRICS_RESULTS_R1.md)
pasó la revisión numérica y quedó respaldada. La tanda original
`20261012T010512Z-V7-llama32-full-ffn-r3` terminó sin reiniciar sus procesos
tras el corte anterior de SSH. Archivo verificado, SHA-256
`818a2bce67b8021dab959e820ec253d92dc1886a671f7598fb43391c34a97179`.

Resultado: **20.332.664 ciclos**, 56.754.176 bytes R y 155.648 bytes W
exactos, incluidos 2.048 resultados FP32. Gate/up/down consumen
31,74/31,75/31,69% de los ciclos; SwiGLU/Q8, 4,82%. La revisión de métricas
R4 reabrió el archivo, cotejó 49 entradas compiladas y once unidades
generadas, repitió la revisión numérica y rechazó catorce corrupciones.
Corrige únicamente la comparación entre claves enteras de Python y sus
claves de texto persistidas en JSON; se conserva el diagnóstico R3.
Estos datos corresponden a la FFN de capa 0 y entrada sintética 2,
no a inferencia completa ni a rendimiento físico.

La cola siguiente ya está ejecutando el tratamiento común de reloj
para la variante RQ bitmap: tanda
`20261012T044326Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-clock-r1`.
Se verificaron sus procesos Vivado 3983964/3984015 a las 04:45 UTC.
Las dependencias restantes conservan el orden establecido.

El [escritor/publicador KV](../LLAMA32_KV_APPEND_R1.md) aprobó 116
vinculaciones, 312 comandos, 20 resets y 6.848 lecturas de prefijos en
Icarus. Rechazó trece corrupciones por suite y seis variantes RTL
compiladas. El coordinador `llama32-kv-append-vivado-r2-coordinator.json`
está activo (PID 88304), esperando al lector KV revisado y respaldado.
Su paquete de 36 miembros está verificado en D: y NFS, SHA-256
`6d7e02a00e31c706be6ae021179dcd54cfbbee3619e1f2e9562ea01c577f9ad4`.
R1 quedó sin lanzar: el preflight detectó que el conteo de casos del
lector debía obtenerse de una lista. R2 corrigió ese control y rechazó
cinco dependencias incompletas, conservando el RTL del escritor.
XSim, síntesis y conexión del escritor con atención siguen pendientes.

Actualización 2026-10-12, 04:22 UTC: el [cargador KV FP32](../LLAMA32_KV_TILE_LOADER_R1.md)
tiene 340 casos RTL aprobados en Icarus, incluidos 20 resets y el watchdog
de producción. La revisión independiente comprobó cada dirección, respuesta
AXI y byte entregado; rechazó doce corrupciones por suite y cinco variantes
RTL compiladas con fallos deliberados. La reserva máxima del layout FP32
es 8 GiB para 131.072 posiciones; todavía no incluye el publicador KV.

El coordinador `llama32-kv-loader-vivado-r1-coordinator.json` está activo
(PID 175036), esperando atención HLS revisada y respaldada. Ejecutará las
mismas cuatro suites en XSIM y síntesis OOC del cargador VU5P a 5 ns,
con límite de 1.200 segundos y dos CPU/hilos. No hay aún resultado XSIM,
síntesis ni timing físico de este módulo. Paquete de 33 miembros verificado
en D: y NFS, SHA-256
`59ef46c55152687740e57a03841e2291f8f29d716126d365842d328ea2aacfe8`.
El respaldo previo al lanzamiento es
`llama32-model-working-r13-20261012T042056Z.zip`, 486 miembros, SHA-256
`4e5c09575f5f70559517190ef981ddb29709fe80f3ed82d44b5e3c60ebeca471`.

La FFN original seguía en `down` a las 04:20 UTC, offset 13.156.416,
sin errores ni advertencias de límites. Las 14 dependencias recuperadas
y atención continúan en espera; se conserva el orden de las tandas.

Actualizaci?n 2026-10-12, 03:47 UTC: la [atenci?n por tiles](../LLAMA32_ATTENTION_TILE_R1.md)
pas? sus dos bancos nativos. El banco de estado ejercit? 4.368 llamadas,
43 rechazos y 131.072 posiciones completas. El banco num?rico revis?
143 casos/291 tiles, 9.152 salidas y 30.264 palabras de estado; incluye
los 32 cabezales en cuatro prefijos de embeddings reales de Llama.
La referencia independiente Decimal90 observ? 0 ULP finales y rechaz?
doce corrupciones. Se corrigi? un falso desbordamiento al promediar
valores FP32 m?ximos; el diagn?stico inicial se conserva.

El coordinador `llama32-attention-hls-r1-coordinator.json` est? activo
(PID 193600), esperando norma?Q8 revisada y respaldada. Su paquete de
37 miembros se verific? en D: y NFS, SHA-256
`8c3e632e23ea4343a1b72bd9c2de7a48d7bcc810ac5381681736d571b2357be8`.
La futura tanda cubre 291 tiles correctos C/RTL; rechazos y contexto
completo contin?an siendo pruebas nativas. No hay todav?a inferencia
completa, adaptador KV/AXI ni resultado de timing f?sico de atenci?n.

La FFN mantiene los procesos originales; a las 03:46:17 UTC estaba
en `down`, offset 4.210.752, sin errores ni advertencias de l?mites.
Las 14 dependencias recuperadas conservan su orden y siguen en espera.


Actualizaci?n 2026-10-12, 03:32 UTC: se recuper? la supervisi?n de la
[FFN tras el corte de SSH](../LLAMA32_FFN_SSH_RECOVERY_R1.md). El
observador R6 est? activo (PID 39088), conserva el mismo runner remoto
3926126 y la tanda original; no se reinici? Vivado/XSIM. La lectura en
vivo ya alcanz? la fase 3 (`down`). No hay todav?a cierre num?rico de la FFN.

Las 14 dependencias volvieron a quedar activas en espera, seg?n
`pcie-symbolic-monitor-predecode-ssh-recovery-r2-preparation.json`:
las ocho etapas f?sicas/m?scaras preceden a RMSNorm, su auditor?a de
transacciones, RoPE, residual y norma?Q8. La auditor?a FFN espera su
cierre y respaldo, sin ocupar Vivado. Los coordinadores detenidos y
la preparaci?n R1 nunca lanzada se conservan como evidencia.

C?psula previa `llama32-model-working-r11-20261012T033011Z.zip`,
416 miembros verificados en D: y NFS, SHA-256
`1e88bc358286eb6e8acd72dfab35acbc1c7f428e00f459da1bc84146be8c6aed`.
Las fuentes y los criterios de aceptaci?n de las tandas no cambiaron.


Actualización 2026-10-12, 03:00 UTC: la [integración RMSNorm→Q8 ABI1](../LLAMA32_NORM_Q8_R1.md)
pasó 61 transacciones nativas: 49 correctas y 2.860 registros completos,
183.040 bytes exactos. Incluye cuatro recorridos embedding→norma de
atención de capa 0 con datos reales y los 33 vectores originales de norma
con activaciones sintéticas. Diez corrupciones numéricas fueron rechazadas;
las fuentes originales de RMSNorm y cuantización permanecen intactas.

El controlador `llama32-norm-q8-hls-r1-coordinator.json` está activo
(PID 110276), esperando la suma residual aprobada y respaldada. Su
paquete de 35 miembros se verificó en D: y NFS, SHA-256
`d5d6ffa4c1825cb19add41c7a738146fc6ed8b043bd34243f154d45249e753c6`.
El revisor de puertos pasó diez alteraciones con trazas construidas;
los resultados RTL, recursos y latencias del bloque siguen pendientes.
No se conecta todavía por AXI al GEMV ni constituye inferencia completa.

La FFN sigue activa en `up`, offset 12.427.328 a las 03:00:25 UTC,
sin errores ni advertencias de límites observados. Se mantienen la
tanda actual y el orden de las validaciones físicas y de cómputo.

Actualización 2026-10-12, 02:47 UTC: la [suma residual FP32](../LLAMA32_RESIDUAL_R1.md)
pasó 43 casos nativos, con 51.556 palabras exactas y ocho corrupciones
rechazadas. El banco CPU adicional ejercitó 2.097.152 parejas: comparó
1.937.408 resultados y comprobó salida intacta en 78 transacciones con
desbordamiento. Son pruebas nativas, todavía sin calificación RTL.

Su controlador HLS está activo en espera de RoPE revisado y respaldado
(`llama32-residual-hls-r1-coordinator.json`, PID 174432). Conserva la FFN,
las ocho tandas físicas/máscaras y RMSNorm/RoPE en su orden actual.
Paquete verificado en D: y NFS: SHA-256
`465e142950abf5f7cde55176eb644b9d5440f03361b73f5c04277f15d8cce6b8`.
El revisor de puertos pasó ocho alteraciones sobre trazas construidas;
exigirá las 43 transacciones reales de cosimulación antes de acreditar RTL.

La FFN real sigue en `up`, offset 8.908.864 a las 02:47:15 UTC, con
Vivado/XSim activos y sin errores ni advertencias de límites observados.
La revisión documental DDR conserva el fallo de reset en caliente: los
ensayos de ancho/fase/CKE anteriores no lo resolvieron. No se relanzaron
variaciones ni se modificó el modelo del proveedor.

Actualizaci?n 2026-10-12, 02:29 UTC: la [auditor?a de fases FFN](../LLAMA32_FULL_FFN_METRICS_R1.md)
qued? activa en espera del PASS y respaldo de la tanda real. Su preparaci?n
se verific? en D: y NFS antes del lanzamiento. Se ejecutar? localmente,
sin tomar el turno Vivado de las ocho validaciones f?sicas/m?scaras.


Actualizaci?n 2026-10-12, 02:27 UTC: la FFN real pas? de gate a up y
contin?a leyendo la segunda matriz (offset 3.584.064). Los procesos
Vivado/XSim siguen activos, sin errores ni advertencias de l?mites en
esta observaci?n. No hay a?n PASS de la ra?z completa.

La [revisi?n suplementaria de ciclos y tr?fico](../LLAMA32_FULL_FFN_METRICS_R1.md)
est? preparada para actuar despu?s del cierre num?rico y respaldo. Pas?
sobre el banco sint?tico hist?rico y rechaz? catorce corrupciones;
comprob? adem?s las 49 entradas RTL y once unidades generadas del banco
actual, con cuatro alteraciones de identidad rechazadas.
La auditor?a de prelanzamiento revis? 121 scripts y 140 enlaces: no detect?
el patr?n de sustituci?n global que se corrigi? en RMSNorm. Es revisi?n
Python/de hashes, no validaci?n RTL ni garant?a completa de ejecuci?n.


Actualizaci?n 2026-10-12, 02:10 UTC: el [n?cleo RoPE con tabla de fases](../LLAMA32_ROPE_TABLE_R1.md)
pas? en CPU 48 casos y 61.760 FP32 exactos. Usa los factores del GGUF,
parejas adyacentes y seis casos derivados de Q/K reales. C/s?ntesis/C-RTL
qued? encolado detr?s de la auditor?a RMSNorm R2, con paquete verificado
en D: y NFS. Sus recursos, transacciones RTL y timing siguen pendientes.

La [correcci?n de prelanzamiento RMSNorm R2](../LLAMA32_RMSNORM_PRELAUNCH_R2.md)
est? activa: la sustituci?n global de un placeholder habr?a roto el Python
remoto antes de Vitis. Se retiraron solo dos coordinadores R1 en espera,
sin reescribir sus registros ni alterar la FFN. Sus reemplazos conservaron
el paquete de hardware y los criterios num?ricos. Los registros R1 de espera
son hist?ricos; consultar el recibo de retiro y los coordinadores R2.
La FFN a las 02:08 UTC segu?a en gate, offset 16.465.984, sin errores ni
advertencias de l?mites y con Vivado/XSim activos.


Actualizaci?n 2026-10-12, 01:45 UTC: la auditor?a de transacciones RMSNorm
qued? activa, esperando el PASS HLS y su archivo verificado. Su preparaci?n
tiene respaldo local/NFS y 25 enlaces de archivos comprobados; no ocupa
Vivado/Vitis. [Resultados nativos y criterios RTL](../LLAMA32_RMSNORM_NATIVE_RESULTS_R1.md).
La FFN sigue en gate, offset 10.162.240, sin errores ni advertencias de
l?mites en esta observaci?n; todav?a no hay un resultado final de la FFN.


Actualizaci?n 2026-10-12, 01:44 UTC: RMSNorm pas? en MSVC y en GCC con la biblioteca
matem?tica de AMD: 60 casos, 44 correctos, 78.208 valores a cero ULP,
33 vectores reales y siete controles negativos del transcript. La tanda
nativa AMD est? respaldada en D: y NFS; [evidencia y l?mites](../LLAMA32_RMSNORM_NATIVE_RESULTS_R1.md).
Vitis C/s?ntesis/C-RTL contin?a en espera detr?s del replay de m?scaras.
La auditor?a suplementaria de las transacciones est? preparada y rechaz?
22 corrupciones; no se ha calificado RMSNorm RTL.

La observaci?n FFN de las 01:38 UTC conserva los tres hashes de im?genes,
registra gate en offset 8.179.776 y no muestra errores ni advertencias de
l?mites. Vivado/XSim siguen activos. Los ocho coordinadores dependientes
y el coordinador RMSNorm permanecen vivos en su orden de espera.
No se alteraron entradas ni se reinici? ninguna validaci?n en curso.

Actualización 2026-10-12, 01:10 UTC: la FFN R4 verificó las tres imágenes
completas cargadas, con hashes FNV `d22214a430288927`, `2f357a92b56df048`
y `0ad507f50ff8e421`. Continúa la fase gate más allá del fallo anterior;
todavía no hay PASS de la FFN. Los ocho coordinadores recuperados se
activaron a las 01:08 UTC y esperan en el orden documentado. La preparación
y el estado del modelo tienen respaldo local/NFS verificado.

Actualización 2026-10-12, 01:08 UTC: los [84 casos GEMV con pesos Llama
reales](../LLAMA32_Q8_GEMV_RESULTS_R1.md) pasaron, con 2.688 FP64 exactos
y archivo verificado en D:. La auditoría de contadores R2 también pasó:
93,59–93,61% de ciclos con algún MAC ocupado y 85,03–85,04% de espera Q8.
Son ciclos de simulación, sin calificación de frecuencia física o tokens/s.

La [FFN completa](../LLAMA32_FULL_FFN_RETRY_R1.md) conserva dos diagnósticos
respaldados: sintaxis Tcl de plusargs y carga de submatrices rechazada por
XSim. El coordinador R4 ejecuta `20261012T010512Z-V7-llama32-full-ffn-r3`
con arreglos planos y comprobación completa de cada imagen antes del DUT;
la primera imagen ya pasó, las demás y el resultado siguen pendientes.
La [cola dependiente recuperada](PCIE_SYMBOLIC_MONITOR_PREDECODE_MODEL_RETRY_TAIL_R1.md)
está preparada y auditada: ocho etapas, dieciséis helpers y cinco errores
de enlace deliberados rechazados. Espera respaldo y activación; conserva
criterios, orden y resultados históricos.

Actualización 2026-10-12, 00:43 UTC: la primera semilla Llama terminó
sus 28 trabajos y la suite continúa. El comprobador suplementario de
[ciclos y ocupación GEMV](../LLAMA32_Q8_GEMV_METRICS_R1.md) pasó una prueba
preliminar sobre ese log cerrado y rechazó catorce corrupciones;
la aceptación de los 84 trabajos exigirá repetir desde el archivo verificado.
El coordinador de métricas quedó activo a las 00:44 UTC, esperando ese
respaldo; su revisión local puede acompañar a la FFN sin ocupar Vivado.
El replay propio de máscaras quedó activo en espera a las 00:39 UTC,
después de revisar sus dependencias y respaldar la preparación.

Actualización 2026-10-12, 00:36 UTC: está preparado el [replay propio de la
netlist de máscaras fijas](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_MAPPED_R1.md),
condicionado al PASS/revisión/respaldo de su síntesis R2. Conserva
174.436 filas, 447 bits públicos y los comprobadores de RQ985.
La tanda Llama Q8 sigue ejecutándose; sus resultados parciales no son
todavía un PASS de los 84 trabajos previstos.

Actualización 2026-10-12, 00:30 UTC: el [selector aislado de máscaras fijas](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_RESULTS_R1.md)
pasó y está respaldado: 1.186.064 comprobaciones de propiedad/ancestros,
89.632 vectores de selección, tres mutantes y trece corrupciones de evidencia.
La integración completa y su síntesis R2 están en espera, conservando
la cola actual. La tanda GEMV con pesos reales Llama Q8_0 arrancó como
`20261012T002948Z-V7-llama32-q8-gemv-slices-r1`; aún no tiene resultado.

Actualización 2026-10-12, 00:24 UTC: el [diagnóstico E/D por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_STAGE_RESULTS_R1.md)
está revisado y respaldado. E→D y D→D mejoran 0,294 ns;
C→E empeora 0,245 ns y C→C 0,015 ns. B→R no cambia.
El WNS global sigue en −2,773 ns con 4.137 LUT/588 FF adicionales:
no se promueve E/D. Continúan el banco aislado de máscaras fijas,
las pruebas Llama y la comparación física en su orden de dependencias.
También quedó preparada la [síntesis propia de máscaras fijas R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_SYNTH_R2.md),
condicionada a su PASS completo. Comparará recursos y timing estimado
contra RQ984, con idéntica caracterización; aún no fue ejecutada.

Actualización 2026-10-12, 00:22 UTC: quedó preparada la
[integración completa de máscaras RC fijas](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_FULL_R1.md)
sobre RQ. Mantiene 26 unidades compiladas y los 16 bancos/183.058 filas
de referencia. Exige PASS aislado del selector y espera el cierre de la
cola actual hasta ambos replays ruteados. La revisión estática rechazó
cinco cambios erróneos de identidad; aún no hay simulación de esta integración.
Su coordinador quedó activo a las 00:22 UTC, esperando las dependencias,
tras verificar el respaldo de la preparación en D: y NFS.

Actualización 2026-10-12, 00:14 UTC: el [replay propio E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_BINARY_RESULTS_R1.md)
pasó la revisión independiente: 174.436 filas, 447 bits públicos,
seis mutantes y 34 controles de evidencia. Resultado y fuentes están
respaldados; sigue el diagnóstico por etapas frente a RQ984.
La nueva cola de replays ruteados está activa en espera detrás de la
revisión física R2. Llama GEMV/FFN RTL siguen pendientes en su orden
de dependencias. No hay cierre de timing ni promoción de E/D.

Actualización 2026-10-11, 23:49 UTC: la [síntesis E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_SYNTH_RESULTS_R1.md)
está revisada y respaldada: 97.050 LUT, 28.148 FF y WNS −2,773 ns.
Añade 4.137 LUT/588 FF y empeora WNS 0,005 ns frente a RQ984; no se promueve
por estos resultados. Continúan su replay propio y diagnóstico por etapas.
Después de los bancos Llama están encolados los
[relojes RQ/E/D comparables](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_CLOCK_R1.md)
y los [replays de ambas netlists optimizadas](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_CLOCK_BINARY_R1.md),
con respaldo por tanda. La [colocación/ruteo y reapertura de ambos DCP](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_PHYSICAL_R1.md)
también quedó encolada a las 23:55 UTC, detrás del PASS y respaldo de los
dos replays optimizados. Mantiene dos hilos, Quick/1800 s por implementación
y 900 s por reapertura; todavía no aporta resultados físicos nuevos.
Después quedó encolada, el 2026-10-12 a las 00:01 UTC, la [revisión y comparación independiente R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_PHYSICAL_REVIEW_R2.md),
que conserva diez grupos de timing y exige restricciones equivalentes.
Los [replays propios de las netlists ruteadas](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_ROUTE_BINARY_R1.md)
quedaron preparados el 2026-10-12 a las 00:10 UTC, detrás de la revisión
física R2: 174.436 filas por miembro, dos hilos y 3.000 s por tanda,
con revisión y respaldo antes de publicar éxito conjunto. El coordinador
quedó activo a las 00:12 UTC, esperando las dependencias; aún no ejecutó los bancos.

Actualización anterior, 23:36 UTC: E/B/C cerraron sus comparaciones
por etapas y quedaron respaldadas. [E](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_STAGE_RESULTS_R1.md)
mejora C→E en 2,523 ns; [B](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_STAGE_RESULTS_R1.md)
mejora B→R en 1,612 ns y empeora A→B en 0,836 ns.
[C](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_STAGE_RESULTS_R1.md)
mejora C→C en 0,228 ns, pero conserva la regresión global de entrada: WNS −3,857 ns.
El [banco aislado E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_ISOLATED_RESULTS_R1.md)
pasó 40.958 muestras. La [integración completa E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_FULL_RESULTS_R2.md)
pasó 183.058 filas nuevas; R2 corrige solamente el nombre de una advertencia
esperada. Resultado y revisión están respaldados. R6 reanuda síntesis propia,
replay, diagnóstico por etapas y `mask_rows`, conservando las tandas terminadas.
La [FFN Llama real nativa](../LLAMA32_FULL_FFN_REAL_R1.md) pasó 86.016 valores
bit exactos con NumPy/stdlib/AMD math. La [cola de pesos reales R2](../LLAMA32_REAL_WEIGHTS_QUEUE_R2.md)
mantiene GEMV y FFN RTL completos después de PCIe, con sus fixtures sellados
en D: y NFS. Todavía no hay PASS RTL de pesos reales ni timing físico aprobado.

Estado actual, 2026-10-11 UTC: el monitor R1 ruteado pasó 174.436 filas
binarias y está respaldado localmente y en NFS; el timing sigue fallando.
La variante de predecodificación E/D aprobó 183.058 filas RTL, incluidos
X/Z y los seis bancos drenados sin cancelar tráfico original. Su síntesis
propia está revisada y respaldada: 104.980 LUT, 27.542 FF y WNS −3,584 ns,
sin mejora frente a la referencia sintetizada. Su netlist propia aprobó
174.436 filas binarias y seis mutantes. La optimización con BUFGCE pasó
la auditoría estructural: 103.954 LUT, 27.086 FF y WNS estimado −3,723 ns;
su propio replay optimizado aprobó 174.436 filas, seis mutantes y 34 controles
de evidencia. Está respaldado y habilita colocación/ruteo; aún no hay timing
físico aprobado del candidato.
La tanda física `20261011T093541Z` terminó y quedó respaldada: 104.500 LUT,
27.106 FF, WNS −13,498 ns y WHS −2,972 ns con ruteo Quick. El setup empeora
respecto a la referencia; E→D sigue dominada por interconexión. La reapertura
`20261011T100051Z` y la revisión independiente pasaron y quedaron respaldadas.
El replay ruteado propio `20261011T101255Z` aprobó 174.436 filas binarias,
seis controles negativos y 34 corrupciones de evidencia; está respaldado
en D: y NFS. [Resultado](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROUTED_MAPPED_RESULTS_R1.md).
El ensayo aislado de fallos por lane aprobó 37.494 muestras y seis mutantes,
con revisión R4 respaldada. Los dos intentos Default, tandas 959/960,
agotaron 1.800 s y quedaron respaldados y revisados como diagnósticos,
sin exportar ruteo ni timing finales. [Resultados de los timeouts](PCIE_SYMBOLIC_MONITOR_PREDECODE_DEFAULT_TIMEOUT_R1.md).
La integración completa por hojas, tanda 961 `20261011T115750Z`, aprobó
183.058 filas nuevas, seis controles negativos y 34 corrupciones de evidencia,
con respaldo en D: y NFS. [Resultado integrado](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_RESULTS_R1.md).
Su síntesis propia, tanda 962 `20261011T121546Z`, terminó y quedó revisada y
respaldada: 105.147 LUT, 27.542 FF y WNS estimado −3,746 ns. Añade 167 LUT y
empeora el peor setup 0,162 ns; la ruta crítica mantiene 31 niveles.
La comparación confirmó iguales dispositivo, reloj, E/S, estrategia y builds.
[Métricas y decisión](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_SYNTHESIS_RESULTS_R1.md).
El replay propio, tanda 963, aprobó 174.436 filas binarias, seis controles
negativos y 34 corrupciones de evidencia; quedó respaldado en D: y NFS.
[Resultado de la netlist por hojas](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_MAPPED_RESULTS_R1.md).
No se inicia otro ruteo de esta variante.
El experimento aislado de contexto por prefijo, tanda 964, aprobó 47.734
muestras, incluidos 10.240 casos de frontera y las comparaciones por lane
antes del OR. Los seis mutantes divergieron donde se había previsto y el
revisor rechazó quince corrupciones. [Resultado aislado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_RESULTS_R1.md).
Su integración completa, tanda 965 `20261011T130303Z`, aprobó las
183.058 filas nuevas, seis controles negativos y 34 corrupciones de
evidencia. Quedó revisada y respaldada; los seis bancos drenados conservan
el tráfico original. [Resultado integrado del prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_RESULTS_R1.md).
La síntesis propia, tanda 966 `20261011T132012Z`, quedó revisada y
respaldada: 105.840 LUT, 27.534 FF y WNS estimado −2,546 ns. Mejora 1,038 ns
frente a la referencia comparable, pero siguen incumpliendo 1.415 endpoints
de setup. La peor ruta baja de 31 a 24 niveles y atraviesa dos CARRY8.
[Medidas y decisión del prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_SYNTHESIS_RESULTS_R1.md).
Su replay propio, tanda 967 `20261011T133146Z`, aprobó 174.436 filas binarias,
seis controles negativos y 34 corrupciones de evidencia, con respaldo en
D: y NFS. [Resultado del replay por prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_MAPPED_RESULTS_R1.md).
No se inicia otro ruteo. La [bandera del último evento](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_R1.md)
aprobó 47.734 muestras aisladas, seis mutantes y quince controles de evidencia
en la tanda 968; quedó respaldada en D: y NFS.
[Resultado aislado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_RESULTS_R1.md).
La integración 969 `20261011T141336Z` aprobó las 183.058 filas RTL nuevas,
seis controles negativos y 34 corrupciones; quedó respaldada en D: y NFS.
[Resultado integrado](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_RESULTS_R1.md).
La síntesis propia 970 `20261011T143058Z` quedó revisada y respaldada:
105.518 LUT, 27.534 FF y WNS estimado −2,768 ns. Reduce 322 LUT frente al
prefijo 966, pero empeora el peor setup 0,222 ns. La nueva ruta tiene
27 niveles y tres CARRY8; se mantiene 966 como referencia comparable.
[Resultado y decisión](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_SYNTHESIS_RESULTS_R1.md).
La [cola R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_QUEUE_R2.md)
conserva el resultado R1 y corrige el enlace al nombre histórico de su
revisión. Síntesis y comparación con el padre 966 están completas;
el replay propio 971 `20261011T144252Z` aprobó 174.436 filas binarias,
seis controles negativos y 34 corrupciones de evidencia. Está revisado y
respaldado en D: y NFS. [Resultado](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_MAPPED_RESULTS_R1.md).
La [lectura comparativa por fronteras](PCIE_SYMBOLIC_MONITOR_PREDECODE_STAGE_TIMING_RESULTS_R2.md)
de los dos DCP, tanda 972, terminó revisada y respaldada. Solo cambia
B→R; ambas variantes incumplen también C→C (−2,407 ns) y C→E (−1,184 ns).
R2 corrige 110 descripciones de rutas sin lógica y conserva todas las cifras.
El [análisis de los tres conos de 970](PCIE_SYMBOLIC_MONITOR_PREDECODE_WINNER_CONES_R1.md)
demostró dos salidas binariamente constantes y atribuyó la tercera al
límite superior del byte 3 del lane 6. El [borrador de distancia constante](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_R1.md)
aprobó el [banco aislado 973](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_RESULTS_R1.md):
47.734 muestras, seis mutantes y quince controles de evidencia. Está
respaldado; aún no se integra ni se atribuye una mejora de timing.
Se documentaron las [restricciones de acceso y orden de la tabla D](PCIE_SYMBOLIC_MONITOR_PREDECODE_D_TABLE_CONSTRAINTS_R1.md)
para evaluar una partición o etapa nueva sin asumir ausencia de conflictos.
La [comparación con `opt_design` sobre los DCP 966 y 970](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAIRED_OPT_RESULTS_R3.md),
tanda 974, terminó respaldada y revisada con R3, sin repetir Vivado.
El candidato usa 309 LUT ajustadas menos; ambos tienen WNS estimado
−2,546 ns, ahora desde la entrada RC hacia C. La revisión corrigió la
confusión entre LUT ajustadas y primitivas, y rechazó 16 corrupciones.
Se selecciona el candidato optimizado para replay propio completo y se
prioriza analizar C; todavía no aprueba funcionalidad ni timing físico.
El [replay 975](PCIE_SYMBOLIC_MONITOR_PREDECODE_OPT_WINNER_MAPPED_RESULTS_R1.md)
`20261011T155336Z` aprobó 174.436 filas, seis controles negativos y 34
corrupciones de evidencia, conservando el banco 971. Está revisado y
respaldado en D: y NFS. Se habilitó después de su cierre el
[experimento aislado de prioridad RC](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_PLAN_R1.md).
Su modelo de selección pasó 1.336.336 comparaciones. La tanda aislada
976 `20261011T162337Z` falló antes de simular por un argumento de
`set_property`; quedó respaldada. La [recuperación R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_QUEUE_R2.md)
corrige únicamente ese Tcl y vuelve a encolar los mismos bancos y tres
mutantes. La [integración completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_FULL_DRAFT_R1.md)
queda en cola condicional R2 detrás del PASS aislado y su respaldo, con
nombres separados para conservar el selector y C originales en las
referencias. La tanda aislada 977 `20261011T162925Z` ya está
[revisada y respaldada](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_RESULTS_R2.md):
1.186.064 comparaciones de ancestry, 89.632 vectores adicionales y tres
mutantes. La [integración 978 `20261011T163442Z`](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_FULL_RESULTS_R1.md)
aprobó 183.058 filas nuevas, seis controles negativos y 34 corrupciones.
Está respaldada; Full R3 conserva su revisión tras corregir una búsqueda
del owner equivocado en el coordinador R2, sin repetir Vivado.
Su [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_SYNTH_R1.md)
se lanzó como tanda 979 `20261011T165516Z` mediante la cola R2,
después de respaldar ese cierre y el snapshot nuevo;
la comparación será con la síntesis 970 bajo iguales restricciones.
El replay propio de 174.436 filas queda condicionado a esa síntesis.
La [síntesis 979 revisada](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_SYNTHESIS_RESULTS_R1.md)
midió 91.660 LUT, 27.532 FF y 568 CARRY8: ahorra 13.858 LUT frente a 970,
pero el WNS estimado empeora a −3,704 ns. Está comparada y respaldada;
la ruta peor atraviesa la realimentación del bitmap de tags de C.
El [replay propio 980](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_MAPPED_RESULTS_R1.md)
aprobo 174.436 filas binarias, seis controles negativos y 34 corrupciones.
Quedo revisado y respaldado en D: y NFS; los reportes comparativos
por etapas terminaron revisados y respaldados como tanda 981.
La [comparacion por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_STAGE_RESULTS_R1.md)
localiza el empeoramiento principal en C-C: -1,297 ns. B-R conserva
-2,768 ns; los reportes no califican timing fisico.
El [banco C real del bitmap RQ](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_FOCUS_RESULTS_R1.md),
tanda 982, aprobo 42.910 filas, cuatro fuentes defectuosas y 16 controles
de evidencia. Quedo revisado y respaldado en D: y NFS.
La [regresión completa RQ 983](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_FULL_RESULTS_R1.md)
aprobó 183.058 filas nuevas, seis controles negativos y 34 corrupciones
de evidencia. Quedó revisada y respaldada en D: y NFS a las 18:04 UTC.
La [síntesis RQ 984 revisada](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_SYNTHESIS_RESULTS_R1.md)
midió 92.913 LUT, 27.560 FF y WNS estimado −2,768 ns. Mejora 0,936 ns
frente a 979 por 1.253 LUT adicionales; B→R vuelve a ser la peor ruta.
Quedó comparada y respaldada en D: y NFS. El [replay propio 985](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_MAPPED_RESULTS_R1.md)
aprobó 174.436 filas binarias, seis controles negativos y 34 corrupciones
de evidencia. Quedó revisado y respaldado a las 18:43 UTC; no hay timing
físico aprobado. La cola continuó con el banco independiente de E.
Cada etapa exige el cierre revisado y respaldado de su predecesor.
Se encoló después la [prueba independiente de suma en E](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_RTL_R1.md):
283.148 filas nuevas sobre el módulo real y seis fuentes defectuosas.
La tanda 986 `20261011T184437Z` ya está [revisada y respaldada](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_RESULTS_R1.md):
aprobó 283.148 filas sobre 4.554 bits, los seis mutantes y 16 controles
de evidencia en 157,025 s. El árbol de suma de 14 bits conserva saturación,
primer word y X/Z; integración y efecto sobre timing siguen pendientes.
Después queda la [comparación RQ/979 por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_STAGE_TIMING_R1.md),
de solo lectura y condicionada a los dos replays propios. Reutiliza el
Tcl y las funciones de medición calificadas de 981; E solo fija el orden
de la cola y no forma parte de esos DCP.
La tanda 987 `20261011T184938Z` [terminó revisada y respaldada](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_STAGE_RESULTS_R1.md):
C→C mejora 0,946 ns hasta −2,758 ns; C→E sigue en −1,183 ns y B→R
en −2,768 ns. E→D y D→D pierden 0,294 ns, conservando margen positivo
estimado. Ninguna de estas cifras califica timing físico.
Después se encoló el [banco del productor B con límites registrados](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_FOCUS_R1.md):
211.606 filas, 2.967 bits públicos y seis fuentes defectuosas. Reutiliza
el borrador B existente sin integrar todavía su consumidor tardío.
La tanda 988 `20261011T185835Z` [aprobó con revisión R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_RESULTS_R2.md):
211.606 filas, seis mutantes y 20 controles de evidencia, en 144,233 s.
R2 corrige el primer testigo del reset omitido (fila 0, no 4), sin repetir
Vivado. Quedó revisada y respaldada a las 19:07 UTC; timing sigue pendiente.
Su PASS habilita el [banco del consumidor tardío](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_LATE_FOCUS_R1.md),
también encolado: 309.884 muestras nuevas, comparación del agregado y
las 32 contribuciones por lane, seis fuentes defectuosas y 21 controles
de evidencia. La tanda 989 `20261011T190901Z` [aprobó ese banco](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_LATE_RESULTS_R1.md)
en 166,648 s; el archivo está respaldado. Productor y consumidor quedaron
calificados por separado; su conexión al monitor todavía debe probarse.
Se añadió detrás la [integración completa de E](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_FULL_R1.md),
con 183.058 filas nuevas y seis controles de entrada. Deriva de RQ 983/985
y sustituye únicamente E; conserva B y el retiro winner. Espera los cierres
anteriores para medir por separado el efecto de cada cambio.
La [recuperación R2 de la cola](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_QUEUE_RECOVERY_R2.md)
conserva los coordinadores R1 detenidos y reanuda sus sucesores todavía
sin ejecutar. Quedaron encolados también la [síntesis propia de E](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_SYNTH_R1.md)
y su [replay de netlist](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_MAPPED_R1.md),
cada uno condicionado al resultado revisado y respaldado anterior.
La [cola R3](PCIE_SYMBOLIC_MONITOR_PREDECODE_PIPELINE_RECOVERY_R3.md)
resuelve después una búsqueda incorrecta del owner en el coordinador R2,
sin repetir 989. Conserva los owners R1 y coordina explícitamente E
completo, síntesis y replay. Añade luego el [consumidor sobre winner](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FOCUS_R1.md),
para conservar la arquitectura del monitor RQ al integrar los límites B.
E completo, tanda 990 `20261011T191854Z`, [aprobó la revisión independiente](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_FULL_RESULTS_R1.md):
183.058 filas nuevas, seis controles negativos y 34 corrupciones rechazadas,
en 777,077 s. Archivo y fuentes quedaron verificados en D: y NFS. La cola R3
pasó a la síntesis propia. La tanda 991 `20261011T193631Z` quedó
[revisada, comparada y respaldada](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_SYNTHESIS_RESULTS_R1.md):
92.644 LUT, 27.562 FF y WNS estimado −2,768 ns. Reduce 269 LUT y 88
endpoints de setup incumplidos frente a RQ 984; B→R conserva la peor ruta.
Su [replay propio 992 `20261011T194744Z`](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_MAPPED_RESULTS_R1.md)
aprobó 174.436 filas binarias, seis controles negativos y 34 corrupciones
de evidencia. Quedó revisado y respaldado en D: y NFS a las 20:13 UTC;
el ensayo sin SDF no califica timing físico. El [consumidor winner 993](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FOCUS_RESULTS_R1.md)
aprobó 309.884 muestras, seis mutantes y 21 controles de evidencia en
167,849 s. Quedó revisado y respaldado a las 20:18 UTC. Su integración
completa 994 `20261011T202003Z` [aprobó 183.058 filas nuevas](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FULL_RESULTS_R1.md),
los seis controles de entrada y las comprobaciones de 75 bits de límites
B. Quedó revisada y respaldada a las 20:36 UTC. Su síntesis propia,
tanda 995 `20261011T203802Z`, quedó [revisada y comparada](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_SYNTHESIS_RESULTS_R1.md):
93.179 LUT, 27.566 FF, 573 CARRY8 y WNS estimado −2,758 ns.
La revisión R2 corrigió cinco referencias al nombre anterior de B en la
lista de avisos; conservó el fallo R1 y no repitió la síntesis. La
[recuperación R4](PCIE_SYMBOLIC_MONITOR_PREDECODE_PIPELINE_RECOVERY_R4.md)
arrancó a las 21:00 UTC para reanudar las diez etapas pendientes,
comenzando por el replay propio de B995.
El runner se detuvo antes de Vivado por el límite de longitud de comandos
de Windows. La [recuperación R5](PCIE_SYMBOLIC_MONITOR_PREDECODE_PIPELINE_RECOVERY_R5.md)
conservó la configuración y el fallo, cambió el transporte SSH a stdin
en bytes UTF-8 y reanudó el owner previo al lanzamiento. La tanda B
996 `20261011T210603Z` [aprobó su replay propio](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_MAPPED_RESULTS_R1.md):
174.436 filas nuevas, 447 bits, seis controles negativos y 34 alteraciones
de evidencia rechazadas en 1.343,466 s. Quedó revisada y respaldada a las
21:31 UTC; la cola pasó a preparar el conteo C aislado. No califica timing
físico ni PCIe nativo.
La cadena vigente usa coordinadores R5 y mantiene las diez dependencias.
La [integración de límites B con winner](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FULL_R1.md)
conserva E original para medir el cambio B por separado: 183.058 filas,
447 bits públicos y 75 bits adicionales de alineación antes/después del
flanco. La auditoría estática R2 comprobó las 27 fuentes, los enlaces
owner/configuración y el revisor: 16 resúmenes válidos aceptados y 128
alteraciones rechazadas; sus resultados RTL y de síntesis constan arriba.
También quedaron encolados su [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_SYNTH_R1.md)
y [replay binario](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_MAPPED_R1.md),
condicionados a los respectivos cierres anteriores. La comparación usa
RQ 984 y mantiene E original. La auditoría comprobó las recetas, nombres
de owner/configuración y la alineación del preflight; dos resúmenes válidos
aceptados y dieciséis alteraciones rechazadas. R14 incluye los snapshots
pendientes de estas etapas en el respaldo local/NFS.
Detrás del replay B quedó encolado el [conteo temprano de admisiones C](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FOCUS_R1.md):
una asignación cambia respecto al C RQ calificado. El banco compara
51.110 filas con registro completo, bitmap, pending y un guard adicional
de 13 bits para conteos; incluye cuatro fuentes defectuosas. La auditoría
estática y los 21 controles sintéticos del revisor pasaron. La tanda 997
`20261011T213308Z` [aprobó el RTL aislado](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FOCUS_RESULTS_R1.md):
51.110 filas pre/post, cuatro mutantes y 21 alteraciones de evidencia
rechazadas en 117,582 s. Quedó revisada y respaldada a las 21:35 UTC;
la cola R5 pasó a preparar la integración completa. Aún no hay síntesis
ni timing propio del cambio C.
Quedaron encolados también su [monitor completo](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FULL_R1.md),
[síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_SYNTH_R1.md) y
[replay propio](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_MAPPED_R1.md).
Cada etapa exige el cierre revisado y respaldado anterior. El monitor
añade comparación interna del contador en los 16 bancos; se mantienen
los testigos públicos de los seis controles de entrada. Las auditorías
estáticas comprobaron fuentes y enlaces; los nuevos revisores rechazaron
128 alteraciones de cobertura completa y 16 del preflight. Estos controles
sintéticos no equivalen a una ejecución RTL ni a una mejora de timing.
La integración completa C, tanda 998 `20261011T213737Z`,
[aprobó las 183.058 filas](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FULL_RESULTS_R1.md)
y la comparación interna de 13 bits en todos los bancos y sondas. Los seis
controles negativos, 34 corrupciones de evidencia y ocho de cobertura
pasaron en 778,667 s; quedó revisada y respaldada a las 21:54 UTC.
Su [síntesis propia 999](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_SYNTHESIS_RESULTS_R1.md)
`20261011T215539Z` terminó, quedó revisada y respaldada: 93.025 LUT,
27.531 FF y WNS estimado −3,857 ns. Frente a RQ suma 112 LUT y 36
endpoints setup incumplidos; el peor margen empeora 1,089 ns aunque TNS
mejora 510,333 ns. La nueva ruta crítica entra por `rc_user[78]` hacia
la máscara EOP registrada en C, con 32 niveles. Se mantiene RQ como
referencia; sigue el replay propio y después la comparación por etapas,
leyendo también E/S porque esa ruta no pertenece a la matriz entre registros.
El [replay propio C1000](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_MAPPED_RESULTS_R1.md)
`20261011T220721Z` aprobó 174.436 filas binarias nuevas, diez sondas quiet,
seis controles de entrada y 34 corrupciones de evidencia. Terminó en
1.344,574 s y quedó respaldado y revisado a las 22:33 UTC. Confirma el
comportamiento cubierto de su netlist; no revierte la regresión temporal.
La cola pasa a los diagnósticos por etapas E/B/C, manteniendo RQ como referencia.
La [auditoría de rutas de entrada](PCIE_SYMBOLIC_MONITOR_PREDECODE_INPUT_PATHS_R1.md)
contrastó los cuatro reportes archivados y rechazó veinte corrupciones.
En C, 19 de las veinte rutas setup publicadas parten de puertos; en RQ
son siete. Los conjuntos no cubren todos los endpoints. E conserva los
veinte pares de RQ sin cambiar slack, B conserva diecisiete y C comparte
solo tres; no se interpreta la ausencia de una ruta como cierre temporal.
La [distribución de recursos de C](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_RESOURCES_R1.md)
confirmó −18 FF en C, −28 en E y +17 en D, con cambios de asignación y
réplicas en los tres grupos. Las primitivas LUT bajan 432, pero las CLB LUT
ajustadas por combinación suben 112. No se atribuye el descenso global de
29 FF al contador ni se interpreta el inventario como correspondencia de estados.
Quedó encolado el [experimento de máscaras RC con índices fijos](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_R1.md)
sobre el selector de RQ. Su modelo finito aprobó 640.000 ensamblajes,
600.000 con condiciones X/Z; el auditor verificó 31 enlaces de fuentes y
recetas, tres testigos sintéticos y trece corrupciones de evidencia.
El banco RTL heredado espera el cierre de los diagnósticos E/D: no cambia
la cola en curso. Conserva 1.186.064 comprobaciones de ancestros y 89.632
vectores de selección, más tres mutantes propios; R18 respalda el borrador
y las fuentes pendientes. Aún no hay PASS RTL ni mejora de timing.
Después del replay C quedó encolada la [comparación temporal por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMPONENT_STAGE_QUEUE_R1.md):
tres tandas de solo lectura, E/B/C frente a RQ 984, con los DCP propios
y sus replays calificados. Reutilizan Tcl y parser de 987: siete grupos,
49 pares setup/hold y diez controles de evidencia por tanda. La auditoría
estática pasó; la ejecución espera toda la cadena previa. R16 conserva
el plan y los snapshots pendientes, incluidos los de esta comparación.
Detrás quedó encolado el [banco de habilitaciones por fila E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_R1.md):
40.958 ciclos nuevos, comparación de todos los 14.336 bits de tabla,
contexto retenido y salidas, seis mutantes y 39 controles de evidencia.
El candidato captura las decisiones de escritura en E sin añadir etapas.
La auditoría estática y 12.288 transiciones del modelo pasaron; la prueba
RTL todavía espera la cola previa. R17 respalda el borrador y su orquestación.
Su [integración completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_FULL_R1.md)
quedó encolada detrás del PASS aislado: mantiene las 183.058 filas y añade
una pareja original E/D independiente desde el mismo registro C. Compara
21.951 bits internos antes/después de cada flanco y en las sondas quiet,
además de los 447 bits públicos. La auditoría comprobó las 28 unidades y
rechazó 128 alteraciones sintéticas de cobertura; aún no se ha ejecutado.
Quedaron encolados después su [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_SYNTH_R1.md)
y [replay binario](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_MAPPED_R1.md).
Las auditorías estáticas comprobaron las recetas y fuentes, las restricciones
sin cambios frente a RQ 984 y los revisores: dos resúmenes de preflight
aceptados y dieciséis alteraciones rechazadas. El replay conserva las
funciones originales y 174.436 filas binarias nuevas. Ambos esperan las
dependencias, con transporte SSH por stdin y respaldo R17 por tanda.
Después del replay E/D quedó encolada su [comparación por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_STAGE_R1.md),
de solo lectura frente al DCP RQ 984. Conserva el Tcl/parser calificado,
siete grupos de registros y 49 pares setup/hold; la auditoría estática
verificó las cuatro derivaciones y el enlace al replay propio. Esta tanda
medirá E→D y las demás rutas antes de decidir una implementación física.
La [atribución de recursos de RQ/E/B](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMPONENT_RESOURCES_R2.md)
verificó los tres inventarios archivados y dieciocho alteraciones rechazadas.
D conserva 58.504 primitivas LUT, 15.928 FF y 33.997 MUXF7/F8. El ahorro
del experimento E aparece en C tras síntesis; B reparte su incremento entre
A/B/R. Las primitivas LUT se distinguen explícitamente del total CLB LUT
ajustado por combinación: esta lectura no asigna ocupación física a etapas.
La [próxima comparación física E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_COMPARISON_REQUIREMENTS_R1.md)
deberá rutear también RQ 984 bajo el mismo flujo. La tanda 955 utiliza
un control anterior y queda como antecedente; no permite atribuir por sí
sola una mejora al cambio E/D. Se documentaron los requisitos de reloj,
replay y auditorías; aún no hay un nuevo ejecutor físico encolado.
El [segundo cono del padre](PCIE_SYMBOLIC_MONITOR_PREDECODE_PREFIX_PAYLOAD_CONE_R2.md)
también quedó atribuido formalmente al límite superior del primer byte
del lane 13. Esto no predice la ruta crítica de la nueva síntesis.
El [borrador de límites registrados en B](PCIE_SYMBOLIC_MONITOR_PREDECODE_REGISTERED_BOUND_DRAFT_R1.md)
tiene productor y consumidor calificados por separado. Su conexión con
winner, integración completa, síntesis y replay propio también pasaron
en las tandas 993–996; véanse los resultados B anteriores. Sigue pendiente
su comparación temporal por etapas y no tiene cierre físico.
En paralelo se verificó un [modelo binario del intervalo de payload](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAYLOAD_INTERVAL_MODEL_R1.md)
sobre 67.108.864 combinaciones y se midió el [coste de preparación XSIM](PCIE_SYMBOLIC_MONITOR_PREDECODE_SIMULATION_COST_R1.md).
Ambos son análisis locales; no cambian el RTL ni califican timing.
[Resultado ruteado](PCIE_SYMBOLIC_MONITOR_R1_ROUTED_MAPPED_RESULTS.md) ·
[Filtro predecode y corrección del revisor](PCIE_SYMBOLIC_MONITOR_PREDECODE_GATE_R1_RESULTS.md) ·
[Regresión predecode completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_FULL_R1_RESULTS.md) ·
[Síntesis predecode](PCIE_SYMBOLIC_MONITOR_PREDECODE_SYNTHESIS_R1.md) ·
[Netlist predecode](PCIE_SYMBOLIC_MONITOR_PREDECODE_MAPPED_RESULTS_R1.md) ·
[Reloj predecode](PCIE_SYMBOLIC_MONITOR_PREDECODE_CLOCK_RESULTS_R1.md) ·
[Replay optimizado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CLOCK_MAPPED_RESULTS_R1.md) ·
[Diagnóstico físico](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_DIAGNOSTIC_R1.md) ·
[Cola física predecode](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_QUEUE_R1.md) ·
[Plan de la variante](PCIE_SYMBOLIC_MONITOR_PREDECODE_PLAN_R1.md).

Actualizado el 2026-10-09 UTC. Requisitos: [VIVADO_QUEUE.md](VIVADO_QUEUE.md). Sesión persistente: `/lambda/nfs/vivado/ntrpu/board-vivado/20261002T085705Z/`. Cada tanda conserva una copia de fuentes con Git HEAD, diff, archivos nuevos y hashes; los proyectos anteriores permanecen separados.

## Estado por dependencia

Replay propio C/E/D aprobado: 205.493 filas binarias, 2.499 bits y cuatro controles negativos. La temporización y el monitor completo siguen abiertos. [Resultado C/E/D](PCIE_SYMBOLIC_CED_R1_MAPPED_RESULTS.md).

C/E/D tiene síntesis propia de sus tres etapas y del conjunto, con DCP y restricciones auditados. WNS conjunto -3.212 ns a 4 ns; el cierre físico sigue pendiente. [Caracterización C/E/D](PCIE_SYMBOLIC_CED_R1_SYNTHESIS.md).

La partición simbólica C/E/D pasó 80.120 filas originales, 88.808 filas con drenaje completo y 217 filas dirigidas; cinco mutantes reales fueron rechazados. El monitor completo sigue pendiente. [Resultados C/E/D](PCIE_SYMBOLIC_CED_R1_RESULTS.md).

Control R3 pasó las pruebas RTL y el replay de su propia netlist: 205.493
filas binarias y cuatro controles negativos. WNS OOC -2.546 ns a 4 ns;
el cierre físico sigue pendiente. A/B cerró 178.185 filas crudas y 266.681 flancos focalizados, incluidos
los siete controles negativos. A+B mide 3.942 LUT, 5.616 FF y WNS +1,730 ns
a 4 ns en síntesis OOC. Su netlist propia pasó 169.601 filas binarias
y cinco controles de frontera. Validación tardía, retiro y cierre físico
siguen pendientes. [Replay A/B](PCIE_SYMBOLIC_NUMERIC_AB_R1_MAPPED_RESULTS.md).
[Replay R3](PCIE_SYMBOLIC_CED_CONTROL_R3_MAPPED_RESULTS.md).

El guard de cabecera de 15 bits pasó 1.085.445 casos y cinco mutantes.
La integración completa de validación tardía está preparada y pendiente
de prueba. [Guard de cabecera](PCIE_SYMBOLIC_HEADER_GUARD_R1_RESULTS.md).

La validación tardía desde B pasó 178.185 filas y cinco mutantes reales.
El siguiente paso es integrar el retiro de bytes, control y fallos del
mismo origen. [Resultados](PCIE_SYMBOLIC_LATE_VALIDATION_R1_RESULTS.md).

R pasó una integración corta de 23243 filas contra el estado posterior
al flanco de R7. Falta el caso específico de persistencia del fallo de rango,
once bancos y mutantes de R. [Resultados](PCIE_SYMBOLIC_RETIREMENT_R1_SHORT_RESULTS.md).

R completó 182529 filas originales y 277 dirigidas, con nueve mutantes
reales detectados. El fallo de bytes persiste al volver al rango permitido
y se borra con frío. Pendientes síntesis y netlist propias, quiet e integración.
[Resultados completos](PCIE_SYMBOLIC_RETIREMENT_R1_FULL_RESULTS.md).

R y su validación tardía miden 3199 LUT, 465 FF y
WNS -3.096 ns a 4 ns en síntesis OOC. Pendientes netlist propia,
quiet e integración y cierre físico. [Medidas R](PCIE_SYMBOLIC_RETIREMENT_R1_SYNTHESIS.md).

La netlist propia de R pasó 174184 filas binarias y seis controles
negativos, comparando 463 bits públicos. El monitor completo tiene
un prototipo pendiente de prueba; el WNS de R sigue en −3,096 ns.
[Resultados de netlist R](PCIE_SYMBOLIC_RETIREMENT_R1_MAPPED_RESULTS.md).

El monitor completo C/E/D/A/B/R pasó 183058 filas RTL, con quiet
comparado antes y después del flanco y dieciséis controles reales.
La síntesis conjunta y su netlist quedan pendientes; los WNS aislados
negativos siguen abiertos. [Resultados](PCIE_SYMBOLIC_MONITOR_R1_FULL_RESULTS.md).

El monitor conjunto mide 104259 LUT, 26976 FF y WNS
-3.584 ns a 4 ns en síntesis OOC. Su netlist propia y el cierre
físico siguen pendientes. [Medidas conjuntas](PCIE_SYMBOLIC_MONITOR_R1_SYNTHESIS.md).

La netlist propia del monitor completo pasó 174436 filas binarias,
447 bits públicos en ambas fases, diez sondas de quiet y seis controles
reales. Sigue pendiente el reloj interno y la caracterización ruteada;
WNS OOC −3,584 ns a 4 ns. [Resultados](PCIE_SYMBOLIC_MONITOR_R1_MAPPED_RESULTS.md).

La optimización del monitor con BUFGCE conserva el grafo lógico de su
referencia optimizada y conecta sus 26521 registros al reloj global.
WNS estimado −3,723 ns a 4 ns; quedan la simulación de esta netlist y
las medidas colocadas/ruteadas. [Evidencia](PCIE_SYMBOLIC_MONITOR_R1_CLOCK_RESULTS.md).

La netlist optimizada con BUFGCE pasó sus 174436 filas binarias y seis
controles reales sobre los 447 bits públicos. La siguiente dependencia
es colocación/ruteo y reproducción de los artefactos resultantes.
[Resultados](PCIE_SYMBOLIC_MONITOR_R1_CLOCK_MAPPED_RESULTS.md).

El monitor completo terminó colocación/ruteo Quick y reapertura independiente: 103747 CLB LUT, 26540 FF; WNS −11,835 ns y hold interno −0,104 ns. Timing permanece abierto. Siguiente: su propia netlist ruteada, sin transferir el PASS optimizado. [Medidas y límites](PCIE_SYMBOLIC_MONITOR_R1_PHYSICAL_RESULTS.md).

La netlist ruteada del monitor pasó los dieciséis bancos y seis mutantes sobre 174436 filas y 447 bits públicos. Los fallos de timing físico permanecen; sigue el experimento de selección anticipada E/D. [Reproducción ruteada](PCIE_SYMBOLIC_MONITOR_R1_ROUTED_MAPPED_RESULTS.md).

### Antecedentes y decisiones anteriores

S1 simbólico R1 pasó 80.120 ciclos RTL y 79.709 ciclos de su propia netlist binaria. La síntesis aislada a 4 ns midió 74191 LUT, 18312 FF, WNS -6.566 ns y WHS -0.075 ns. S2/S3 y todas las condiciones físicas/nativas siguen abiertas. [Resultados y respaldos](PCIE_SYMBOLIC_FRONTEND_R1_RESULTS.md).

El modelo de metadatos por valor pasó todo el banco heredado y un suplemento
de 1.047 ciclos/18 escenarios con 1.048 observaciones directas de consultas,
deltas y contextos. Tres mutantes reales detectados; dos tandas respaldadas.
No añade etapas ni medidas físicas. Sigue implementar y medir S1 simbólico,
S2 de aplicación y S3 de validación/retiro. [Resultados](PCIE_SYMBOLIC_VALUE_MODEL_RESULTS.md)
y [contrato de la primera partición](PCIE_SYMBOLIC_PIPELINE_R1_CONTRACT.md).

R15 integra los efectos de metadatos en las consultas crudas y pasó el
banco RTL completo más 78.712 filas binarias de su propia netlist. Empeora
timing OOC: 52.880 LUT, 10.729 FF, WNS −9,544 ns frente a −7,068 ns en R12.
R12 sigue como referencia; no se promovió R15 ni se repitió ruteo. Cuatro
tandas respaldadas. Sigue modelar contexto simbólico antes de introducir
etapas reales. [Resultados](PCIE_INVENTORY_EFFECT_PREFIX_RESULTS.md) y
[diseño siguiente](PCIE_SYMBOLIC_CONTEXT_DESIGN.md).

La unidad de composición de metadatos de 57 bits pasó 17.114 casos RTL
y 12.805 filas binarias en su propia netlist. Síntesis aislada: 220 LUT,
88 FF, WNS +0,377 ns y WHS −0,042 ns; hold y calificación física abiertos.
Cuatro tandas respaldadas, incluido un fallo del lanzador previo a simular.
No sustituye el monitor R12: sigue integrar selección cruda, generaciones
y payload retenido. [Resultados](PCIE_METADATA_EFFECT_RESULTS.md) y
[siguiente integración](PCIE_METADATA_EFFECT_INTEGRATION_PLAN.md).

R14 completó RTL, síntesis, replay propio y lectura por destinos:
**78.712 filas binarias/30 controles**, 20.265 pruebas secuenciales y
10.758 registros clasificados. **WNS mejora a −6,442 ns; TNS empeora a
−71.772,922 ns**. 52.371 LUT/+137 y 10.758 FF/+29 frente a R12.
Los 4 ns siguen incumplidos; cinco tandas revisadas y respaldadas.
Se conserva R12 como referencia y R14 como comparación experimental.
[Resultados R14](PCIE_INVENTORY_REGISTERED_SUM_RESULTS.md) y
[siguiente corte de consulta/transición](PCIE_INVENTORY_REGISTERED_SUM_FOLLOWUP.md).

El censo original de R12 sigue disponible: 10.729 registros, 56 consultas,
WNS −7,068 ns y setup de tabla −5,795 ns, sin cambiar restricciones.
[Medidas R12](PCIE_INVENTORY_PATH_CENSUS_RESULTS.md).

R13 completó RTL y replay de su netlist: **78.712 filas binarias y
30 controles**, además del banco completo y 12.832 pruebas de guardas de
tag (7.680 X/Z). **No mejora:** WNS OOC −7,268 ns y 52.308 LUT, frente a
−7,068 ns y 52.234 LUT de R12. Cuatro tandas revisadas y respaldadas;
R12 sigue preferida; su lectura posterior por destinos está documentada arriba.
[Resultados R13](PCIE_INVENTORY_RC_RELEASE_TAG_RESULTS.md) y
[tanda siguiente](PCIE_INVENTORY_RC_RELEASE_TAG_FOLLOWUP.md).

La implementación Quick de R12 con BUFGCE interno pasó la reapertura
del DCP, auditoría del circuito y replay de sus cinco bancos: **78.712 filas,
30 controles**. Todas las rutas internas están conservadas, pero **setup y
hold fallan**: WNS −17,680 ns, WHS global −3,043 ns y hold interno −0,035 ns.
El intento Default agotó 3.600 s; ambos intentos y sus diagnósticos están
respaldados. La frontera OOC sigue incompleta. El ensayo posterior de tags RC
directos es R13, cuyos resultados figuran arriba; conserva capacidad, orden
y latencia, pero empeora la estimación de timing.
[Resultados](PCIE_INVENTORY_CLOCK_QUICK_ROUTE_RESULTS.md) y
[puertas del ensayo](PCIE_INVENTORY_CLOCK_QUICK_ROUTE_FOLLOWUP.md).

Antes de colocar, el BUFGCE interno pasó auditoría y los cinco bancos
completos: 78.712 filas y 30 controles. Ese replay pertenece a la netlist
optimizada con 10.723 registros; no valida los cambios posteriores del
placer. [Resultados previos de reloj](PCIE_INVENTORY_CLOCK_RESULTS.md).

La implementación diagnóstica R1 de R12 terminó, fue auditada desde
su DCP guardado y pasó el replay de su netlist en los cinco bancos
(78.712 filas de entrada). **Timing fallido**: WNS −18,945 ns y
WHS −1,773 ns; peor ruta interna de 22,926 ns. Un gap del reloj padre,
retardos de reloj estimados y frontera OOC incompleta impiden calificación
física. Cuatro tandas respaldadas, incluido el fallo inicial de ayuda.
[Resultados](PCIE_INVENTORY_PLACEMENT_RESULTS.md) y
[siguiente ensayo de reloj](PCIE_INVENTORY_PLACEMENT_FOLLOWUP.md).

La reducción RC por carril, **R12**, pasó todos los bancos RTL
y el replay de su propia netlist. Se conservan R8/selector R3 salvo la
acumulación de bytes retirados. Nuevos bancos: 70.464 comparaciones de
suma, incluidas 832 X/Z; 984 filas/92 escenarios integrados, de los que
632 filas/60 escenarios binarios también se reprodujeron en la netlist.
WNS OOC **-7.068 ns** (+0.374 ns frente a R8),
**52234 LUT**, **10729 FF** y
**1902 CARRY8**. R12 conserva el mejor
WNS medido. Cuatro tandas revisadas y respaldadas; los criterios nativos,
DDR/512 GiB y físicos permanecen abiertos.
[Resultados R12](PCIE_INVENTORY_RC_LANE_REDUCTION_RESULTS.md) y
[seguimiento](PCIE_INVENTORY_RC_LANE_REDUCTION_FOLLOWUP.md).

En la tanda anterior, la selección tardía de aritmética RC, **R11**, pasó todos los
bancos RTL y el replay de su propia netlist. Se añaden 2.048 casos de
máscaras de cuatro estados; se conserva el cuerpo aritmético de R8,
el selector R3 y `newest` de 32 bits. WNS OOC
**-8.414 ns** (-0.972 ns frente a R8), con
**56863 LUT**, **10729 FF** y
**2168 CARRY8**. R8 conserva el mejor
WNS medido. Cuatro tandas revisadas y respaldadas; la calificación nativa
y física sigue abierta. La siguiente acción está ligada a esta síntesis.
[Resultados R11](PCIE_INVENTORY_RC_LATE_ARITHMETIC_RESULTS.md) y
[seguimiento](PCIE_INVENTORY_RC_LATE_ARITHMETIC_FOLLOWUP.md).

En la tanda anterior, los destinos constantes de la máscara RC, **R10**, pasaron todos los
bancos RTL y el replay de su propia netlist. Sólo cambia la escritura de
la máscara; `newest` permanece en 32 bits. WNS OOC
**-7.789 ns** (-0.347 ns frente a R8), con
**53671 LUT**, **10729 FF** y
**1890 CARRY8**. R8 conserva el mejor
WNS medido. Cuatro tandas revisadas y respaldadas; la calificación nativa
y física sigue abierta. La siguiente acción está ligada a esta síntesis.
[Resultados R10](PCIE_INVENTORY_RC_STATIC_MASK_RESULTS.md) y
[seguimiento](PCIE_INVENTORY_RC_STATIC_MASK_FOLLOWUP.md).

En la tanda anterior, el selector RC de cinco bits, **R9**, pasó todos los bancos RTL y
el replay de su propia netlist. La revisión del texto confirma que sólo
cambia la anchura de `newest`. WNS OOC **-8.652 ns**
(-1.210 ns frente a R8), con **45541 LUT**,
**10729 FF** y **354 CARRY8**.
R8 queda como referencia de mejor WNS medido; se conservan todas
las variantes y sus diferencias de área. Setup/hold y diseño físico siguen
abiertos. Cuatro tandas revisadas y respaldadas. El próximo experimento
aislará los destinos constantes de la máscara de prefijos.
[Resultados R9](PCIE_INVENTORY_RC_SELECTOR_WIDTH_RESULTS.md) y
[siguiente experimento](PCIE_INVENTORY_RC_SELECTOR_WIDTH_FOLLOWUP.md).

En la tanda anterior, la ascendencia RC por EOP efectivos, **R8**, pasó el banco completo,
1.153.296 comparaciones heredadas y 32.768 adicionales, y todos los
suplementos RC/RQ. Su propia netlist coincide en las 22.330 filas principales
y los tres suplementos binarios. WNS OOC **−7,442 ns**, mejora **0,859 ns**
frente a R7, pero aumenta a **51.937 LUT/1.890 CARRY8**, con 10.729 FF.
R8 queda como referencia de mejor WNS medido; R7 se conserva por su menor
área. Setup/hold siguen fallando. DCP acíclico y cuatro tandas respaldadas.
El banco RTL tardó 305,695 s y su unidad ampliada 41 s; la comparación con
R7 no aísla el efecto de desactivar ondas porque también cambió el diseño.
El siguiente experimento reduce únicamente la anchura de `newest` de
32 a cinco bits con signo, antes de estudiar máscaras o aritmética.
[Resultados R8](PCIE_INVENTORY_RC_EVENT_ANCESTRY_RESULTS.md) y
[siguiente experimento](PCIE_INVENTORY_RC_EVENT_ANCESTRY_FOLLOWUP.md).

En la tanda anterior, la propiedad RC predecodificada, **R7**, pasó **1.153.296 comparaciones**,
incluidas **994.288 X/Z**, y **46.356 filas/3.872 escenarios**, además del
banco completo y todos los suplementos R6. Su propia netlist coincide
con RTL en 22.330 filas principales y los tres suplementos. WNS OOC
**−8,301 ns**, mejora **0,602 ns frente a R6** y **0,505 ns frente a R5**;
TNS **−73934,203 ns**, **42.362 LUT/10.729 FF**. R7 pasa a ser la referencia
de mejor WNS medido, conservando todas las variantes anteriores. Setup y
hold siguen fallando; no hay cierre físico. Las cuatro tandas están
revisadas y respaldadas, con DCP acíclico. El próximo experimento deriva
ascendencia desde EOP efectivos sin propagar máscaras entre escrituras.
[Resultados R7](PCIE_INVENTORY_RC_OWNERSHIP_RESULTS.md),
[siguiente experimento](PCIE_INVENTORY_RC_OWNERSHIP_FOLLOWUP.md) y
[coste de simulación](XSIM_OWNERSHIP_RUNTIME_NOTE.md).

La admisión RQ por dos eventos, **R6**, pasó **615.536 comparaciones**,
incluidas 5.712 X/Z, y **8.850 filas nuevas/240 escenarios**, además de
todos los bancos de R5 y del monitor completo. Su propia netlist coincide
con RTL en las 22.330 filas principales y ambos suplementos, RC y RQ.
Frente a R5 reduce LUT **44.857 → 42.403** y mejora TNS
**−127995,078 → −90778,977 ns**, pero empeora WNS **−8,806 → −8,903 ns**;
hold sigue fallando. **R5 conserva el mejor WNS**. Las cuatro tandas R6
están revisadas y respaldadas; el DCP es acíclico. El siguiente experimento
parte de R6 para reducir la cadena de propiedad/liberación RC que llega a
la selección de prefijos, conservando R5 como referencia de WNS. No hay
cierre de timing ni nueva calificación nativa o física.
[Resultados R6](PCIE_INVENTORY_RQ_ADMISSION_RESULTS.md) y
[plan de propiedad RC](PCIE_INVENTORY_RQ_ADMISSION_FOLLOWUP.md).

Los prefijos RC efectivos, **R5**, pasaron los bancos completos RTL y de su
propia netlist, más **89.632 casos de selección**, **45.056 comparaciones
aritméticas (6.624 X/Z)** y **544 filas adicionales/60 escenarios**.
El WNS OOC mejora de **−9,877 a −8,806 ns**, con **44.857 LUT/10.729 FF**
frente a 48.189 LUT del original. R5 queda como preferido por WNS estimado;
el TNS empeora y setup/hold siguen incumpliendo. DCP acíclico y cuatro tandas
revisadas y respaldadas. La peor ruta ahora recorre propiedad RC y admisión
RQ hasta el contador de bytes: **12,789 ns/51 niveles**. Sigue reducir la
admisión a los dos SOP RQ efectivos, preservando punteros malformados,
capacidad y estados posteriores a fallos. Conexión nativa y diseño físico
siguen pendientes. [Resultados R5](PCIE_INVENTORY_RC_PREFIX_RESULTS.md) y
[siguiente experimento](PCIE_INVENTORY_RC_PREFIX_FOLLOWUP.md).

El selector RC separado, R4, pasó **89.632 casos de control**, el banco
integrado de **22.339 filas/68 escenarios** y **22.330 filas/67 escenarios**
en RTL y su propia netlist. Reduce el monitor completo de **48.189 a
42.176 LUT**, con los mismos 10.729 FF, pero empeora WNS OOC de **−9,877
a −10,047 ns**. El DCP no contiene ciclos combinacionales; setup y hold
siguen fallando. Se conserva la referencia original como preferida de timing
y R4 como una alternativa con menos LUT. Las cuatro tandas están revisadas
y respaldadas. Sigue eliminar aritmética encadenada mediante prefijos de
escrituras efectivas; el modelo binario está probado, el cambio RTL pendiente.
[Resultados R4](PCIE_INVENTORY_RC_SELECTION_RESULTS.md) y
[plan de prefijos](PCIE_INVENTORY_RC_PREFIX_PLAN.md).

El monitor de inventario integrado ya pasó el banco completo con estado propio:
**22.339 filas/68 escenarios RTL** y **22.330 filas/67 escenarios binarios**
en RTL y en su netlist. El experimento posterior de deltas de bytes registrados
también pasó esos bancos y 20.414 muestras del acumulador, pero empeora el
WNS OOC de **−9,877 a −9,959 ns**, con **50.537 LUT/10.795 FF**. Se conserva
el productor original como preferido. La peor ruta del nuevo experimento
termina en la tabla de bytes restantes; sigue separar las decisiones RC de
su aritmética encadenada. Las cuatro tandas están revisadas y respaldadas.
Los resultados aislados RC de abajo no acreditan el timing del monitor completo.
Conexión nativa, setup/hold, DIMM exacto, cuatro DDR/512 GiB y diseño físico
siguen abiertos. [Referencia integrada](PCIE_INVENTORY_JOIN_RESULTS.md),
[experimento del contador](PCIE_INVENTORY_COUNTER_RETIMING_RESULTS.md) y
[siguiente plan](PCIE_INVENTORY_RC_FORWARDING_PLAN.md).

La revisión RC R7 registra el límite de payload junto con su contexto.
Pasó **22.339 filas de cuatro estados** y **304.604 filas binarias en RTL
y su propio netlist**, incluidos los controles de corrupción y una mutación
real del nuevo registro. Reduce **6.161 → 6.020 LUT**, usa **1.318 FF** y
mantiene 99 CARRY8. El peor camino baja a **3,760 ns** y el **WNS OOC es
+0,222 ns**, con cero fallos de setup; **hold sigue fallando** (−0,075 ns).
Las cuatro tandas están revisadas y respaldadas. Sigue la integración del
inventario, con sus propios metadatos y tratamiento explícito de las
diferencias en tráfico inválido. No hay cierre físico ni monitor completo
calificado. [Resultados R7](PCIE_RC_EVENT_ENDCACHE.md) y
[obligaciones de integración](PCIE_INVENTORY_INTEGRATION_GAPS.md).

La revisión RC R6 combina las máscaras de R5 con el framing registrado
de R4. Pasó **22.339 filas de cuatro estados** y **304.604 filas binarias
en RTL y netlist**, con las mutaciones y controles intactos. Frente a R5,
reduce **6.295 → 6.161 LUT**, aumenta **1.219 → 1.302 FF** y conserva
99 CARRY8. El peor camino baja de **4,171 a 4,075 ns**, con **WNS −0,093 ns**.
Fue la mejor revisión temporal antes de R7; falla setup y hold.
Las cuatro tandas están revisadas y respaldadas. Su siguiente experimento,
el registro del extremo exclusivo del payload retenido, ya se midió en R7
sin adelantar contexto ni cambiar latencia, capacidad o caudal.
[Resultados, evidencia y siguiente ajuste](PCIE_RC_EVENT_COMBINED.md).

La revisión RC R5 calcula límites de payload antes de seleccionar metadatos.
Pasó **22.339 filas/68 escenarios de cuatro estados** y **304.604 filas
binarias en RTL y netlist**, con mutaciones y verificadores revisados.
Reduce de R3 a R5 **7.333 → 6.295 LUT**, **1.223 → 1.219 FF** y
**186 → 99 CARRY8**. El peor camino baja de **5,245 a 4,171 ns**, con
**WNS −0,189 ns**: mejora respecto a R3, pero aún falla setup y hold.
Las cuatro tandas están respaldadas. Sigue evaluar la combinación con los
prefijos de framing de R4; integración global y cierre físico permanecen abiertos.
[Resultados y siguiente experimento](PCIE_RC_EVENT_PAYLOAD.md).
También se verificó el [traslado reversible de cachés de fuentes](LOCAL_SOURCE_CACHE_RELOCATION.md).

La revisión RC R4 registra la selección de fragmentos durante la captura.
Pasó los **22.339 ciclos/68 escenarios de cuatro estados**, las dos
mutaciones y **304.604 filas binarias en RTL y netlist**, sin diferencias.
Sin embargo, el peor camino empeora de **5,245 a 5,445 ns**, con
**WNS −1,463 ns** y **7.417 LUT/1.305 FF**: 84 LUT y 82 FF más que R3.
Las cuatro tandas están revisadas y respaldadas; se conserva el diagnóstico
de tres avisos de bits constantes que exigieron ampliar de forma precisa
el revisor. R4 fue una regresión frente a R3. El intento posterior R5,
descrito arriba, separó el cálculo de byte enables de la selección de
metadatos; setup, hold e integración global siguen abiertos.
[Experimento, regresión y siguiente paso](PCIE_RC_EVENT_PREFIX.md).

La revisión RC R3 paraleliza las comprobaciones de payload por DWORD.
Conserva los **22.339 ciclos/68 escenarios de cuatro estados** y las dos
mutaciones reales. Su netlist pasó **304.604 filas binarias**: 22.330 del
shadow, 370 de casos límite y 281.904 de una nueva matriz de fragmentación;
todas coinciden con RTL y modelo independiente. Esa matriz detectó antes
un fallo de arranque X en R2, conservado y corregido con `always_comb`.
El peor camino baja de **6,606 a 5,245 ns**, con **WNS −1,263 ns**, y
los recursos bajan a **7.333 LUT/1.223 FF**; usa 186 CARRY8. Quedan siete
endpoints con fallo de setup y sigue fallando hold. Las seis tandas están
revisadas y respaldadas. El experimento posterior R4 se detalla arriba;
no mejoró el timing de R3.
[Evidencia, límites y siguiente revisión](PCIE_RC_EVENT_FRAGMENTS.md).

La revisión RC R1 adelanta la decodificación SOP/EOP al registro de captura.
Conserva todas las salidas de los **22.339 ciclos RTL/68 escenarios** y
las dos mutaciones. Su propio netlist pasó **22.330 ciclos/67 escenarios
binarios más 370 ciclos/15 casos límite**, idénticos al RTL y al modelo.
El peor camino baja de **7,336 a 6,606 ns**, con WNS **−2,624 ns**:
mejora 0,730 ns a coste de 1.794 LUT y 105 FF adicionales, para un total
de **9.639 LUT/1.266 FF**. Setup y hold siguen fallando. Las cinco tandas
están revisadas y respaldadas; sigue pendiente paralelizar la validación
del payload e integrar metadatos, asignación y obligaciones globales.
[Evidencia, coste y siguiente revisión](PCIE_RC_EVENT_PREDECODE.md).

La etapa RC original implementa captura, contexto retenido y validación contra
estado esperado. Pasó **22.339 ciclos/68 escenarios RTL**, dos mutaciones
reales y **370 ciclos/15 fases en RTL y netlist**, incluidos reset con trabajo
pendiente, wrap y RC4 sostenido. Su síntesis usa **7.845 LUT/1.161 FF**;
el peor camino mide **7,336 ns**, con **WNS −3,354 ns y WHS −0,075 ns**
a 4 ns: todavía no cumple timing. Las cinco tandas están revisadas y
respaldadas; falta el shadow R7 completo sobre este netlist y la integración
del asignador, banco de metadatos y obligaciones globales.
[Contrato, métricas y siguiente revisión](PCIE_RC_EVENT_STAGE.md).

La unidad de propuestas RC de cuatro posiciones pasó **69.876 ciclos RTL y de netlist**
y la comparación de los **68 escenarios R7** sin diferencias de eventos
en tráfico válido. Conserva tres diferencias en tráfico inválido que exigen
validación contra metadatos esperados. La síntesis usa **449 LUT/552 FF**,
con setup OOC **+1,616 ns** a 4 ns y hold **−0,075 ns** todavía fallido.
El contexto de ese experimento era de simulación; la nueva etapa anterior
implementa parser y validación, aún sin la contabilidad global de obligaciones
en vuelo. [Evidencia de la unidad aritmética](PCIE_RC_PROPOSAL.md).

El banco aislado de metadatos **256×30, cuatro consultas/seis escrituras**
pasó 28.912 ciclos RTL y de netlist, con todas las filas idénticas.
Su síntesis usa **33.763 LUT/7.808 FF**: setup OOC **+1,924 ns** a 4 ns,
pero hold **−0,042 ns** todavía falla. El shadow de los 68 escenarios R7
comprobó 5.718.784 palabras y detectó dos mutaciones reales de orden/omisión.
Las cinco tandas, incluido un fallo inicial de lanzamiento Tcl, están
revisadas y respaldadas. El banco recibe escrituras ya calculadas; todavía
falta resolver cómo producirlas sin dependencias de lecturas en vuelo.
No es el monitor segmentado completo. [Mediciones y límites](PCIE_TAG_METADATA.md).

R6 terminó síntesis y pasó **22.330 ciclos/67 escenarios de netlist**,
con estados idénticos al RTL para ese banco. Sus métricas son las mismas
que R5: **46.359 LUT, 9.524 FF, WNS −11,266 ns y WHS −0,075 ns**.
El EDIF final es acíclico y no hay excepciones temporales. R7 mueve la
guarda de rango después de los registros: pasó **22.339 ciclos/68 escenarios**
y coincide con todos los estados emitidos por R6. Su síntesis
`20261007T162508Z` terminó: **46.617 LUT/9.531 FF, WNS −11,267 ns**.
Su netlist pasó 67 escenarios con estados idénticos a RTL. El peor camino
no mejora; ahora termina en el control RC de longitud/bytes, con 62 niveles.
Se requiere separar consulta, validación y contabilidad sin reducir RQ2/RC4,
ni publicar vacío mientras queden eventos en vuelo.
[Contrato de etapas, dependencias y pruebas](PCIE_INVENTORY_PIPELINE_CONTRACT.md).

La inspección XDMA revisó **1.357 bits/152.749 extremos** sin cambiar
celdas ni conexiones superiores. READY[1:3] y el número de bus superior
son constantes; READY[0] tiene un driver activo. La inspección de los
**7.457 pines de PCIE40E4** localizó sus ocho `CFGBUSNUMBER`. El ensayo
`20261007T164306Z` conectó ocho salidas de observación sin cambiar las
136.537 celdas ni los extremos de transporte. La reapertura `165658Z`
confirmó las conexiones en el DCP guardado; ambas revisiones rechazaron
seis corrupciones cada una y tienen respaldo D:/NFS. Quedan pendientes
muestreo funcional, monitor e identidad completa del requester. Los
fallos de hold/pulso, CDC y DRC OOC siguen abiertos.
[Frontera y evidencia](XDMA_OBSERVATION_BOUNDARY.md).

Medidas históricas de referencia:

El monitor PCIe R4 mejora WNS de **−17,501 a −11,964 ns** a 4 ns;
reduce LUT de 63.050 a **58.225**, conserva 9.524 registros y no infiere
latches. El camino crítico baja de 88 a 65 niveles, pero setup/hold
siguen fallando. Las restricciones I/O están completas para caracterización
OOC. Su netlist pasó **22.151 ciclos/45 escenarios** y revisión independiente.

R5 limita anchos temporales manteniendo la actualización única de tags
de R4. Ambas pasaron **22.219 ciclos/53 escenarios**, incluidos
31 negativos y 3.778 ciclos capturados, con máximos de **256 tags,
257 lecturas y 1.052.672 bytes** pendientes. Todas estas tandas terminadas
están revisadas y respaldadas en D:/NFS. La comparación directa conserva
todos los estados emitidos, incluidos los posteriores a fallo.
La síntesis R5 `20261007T145153Z` terminó y está respaldada: **46.359 LUT,
9.524 FF y 258 CARRY8**, sin latches. Reporta WNS −11,266 ns y WHS
−0,075 ns. Se investigaron dos avisos intermedios de bucles: el EDIF final
de 66.238 primitivas es acíclico y los XDC guardados no contienen
excepciones de camino. Su netlist pasó los 52 escenarios binarios y
coincide con los 22.210 estados emitidos por RTL. El DCP idéntico reabierto
confirma cero excepciones válidas o ignoradas y los mismos márgenes fallidos.
La sonda XDMA `20261007T151433Z` terminó y está respaldada:
buses RQ/RC conectados; READY[1:3] y origen del BDF siguen pendientes.
[Frontera y límites](XDMA_OBSERVATION_BOUNDARY.md).
R13 (R4), R14 (R6) y R15 (R7) pasaron 68 escenarios, 46 negativos y nueve
controles de corrupción por revisión. Sus comparaciones conservaron
todos los estados emitidos, incluidos los posteriores a fallo.
[Informes y límites](PCIE_REQUEST_INVENTORY_SYNTHESIS.md).
[Resultados y límites](PCIE_REQUEST_INVENTORY_RTL.md).

La captura PCIe `20261006T230834Z` pasó revisión independiente: tres fetches
de 32/64/32 bytes en **336/344/336 ns** simulados y C2H de 64 bytes recibido
por el host **172 ns después de la aceptación PCIe**. Se conserva una
obligación de payload durante **4 ns tras liberar el tag**. Se rechazaron
79 corrupciones (65 heredadas y 14 PCIe); el modelo pasa 63 pruebas locales.
La captura está respaldada D:/NFS. El máximo nativo fue un tag y no hubo
stalls ni múltiples SOP por beat: RQ2/RC4 y 256 tags tienen validación RTL
sintética, pero no evidencia de esa carga en el endpoint nativo.
Siguiente dependencia: monitor sintetizable y conexión
de producción. [Evidencia y criterios](XDMA_PCIE_INVENTORY_PLAN.md).

La composición bidireccional `20261006T224150Z` pasó revisión independiente:
64 bytes H2C, 64 C2H y 64 recibidos por el host; 29 lecturas/13 escrituras
locales y 65 corrupciones rechazadas. El ledger se completa **103,999 ns
antes de la recepción host**, y BAR permanece pendiente **36 ns después
del snapshot CSR**. Ambos monitores mantienen ACK=0 sin cierre del
productor. Se conservan el diagnóstico de captura a tiempo cero
`20261006T223045Z` y la corrección de dos nombres en Tcl; no cambió RTL
de producción ni código AMD. Ambas tandas están respaldadas D:/NFS.
La captura PCIe posterior respeta los tags del cliente y el framing
straddle del core generado; su conexión sintetizable permanece pendiente.
Detalles: [DMA/BAR](XDMA_DMA_BAR_INVENTORY.md) y
[plan PCIe](XDMA_PCIE_INVENTORY_PLAN.md).

La tanda `20261006T220923Z` compone dos monitores AXI4 de 32 escrituras
con los puertos DMA/BAR nativos. BAR conserva una AW y una W pendientes
durante **36 ns después del snapshot local**; solo B real retira esa
obligación. Sin cierre del productor, ambos ACK siguen en cero aun con
inventarios vacíos. Se verifican 128 bytes H2C y 8 bytes de escritura BAR,
35 lecturas/9 escrituras locales y 65 corrupciones rechazadas. El primer
intento `20261006T215941Z` es un diagnóstico de elaboración por conexiones
RLAST implícitas; R1 usa los mismos puertos explícitos sin cambiar AMD ni
RTL de producción. Ambos están respaldados en D:/NFS.
Detalles: [inventario DMA/BAR](XDMA_DMA_BAR_INVENTORY.md).

La tanda `20261006T212408Z` integra la cadena H2C precargada con el
secuenciador y ledger reales: dos descriptores, 128 bytes, 35 lecturas y
9 escrituras locales, cuatro rondas con conteos **0→1→1→2** y snapshot
solo en la última. Inicio→snapshot: **1,448 µs** simulados. Cada Done sigue
a B real; retirar el cierre revoca la aprobación de forma retenida.
La revisión rechazó 47 corrupciones de evidencia y conservó los 139 avisos
AMD identificados. El intento `20261006T212100Z` es un diagnóstico de
sintaxis del banco, corregido en R1 sin cambiar el RTL del diseño.
Ambos están respaldados en D:/NFS. La composición DMA/BAR posterior conserva
independientes recepción host y drenaje global; C2H está revisado arriba.
Detalles: [secuenciador XDMA](XDMA_STOP_REGISTERS.md).

**Estado vigente — watchdog real, secciones AXI, propietario y receptor con dos MIG:**
los perfiles R2 host/fuente de 7/5 y 7/37 ns pasaron la revisión independiente
con dos pausas de 96 ciclos: cierre observado en 680/640 ns, readback final
completo, CPL retenida y diez alteraciones de evidencia rechazadas por perfil. Se compararon
288 escrituras y 12 lecturas MMIO, 17 escrituras y 29 lecturas DDR.
**747 tandas terminales respaldadas localmente/NFS**, incluidos diagnósticos.
Los perfiles anteriores con pausas breves de 7/5 y 7/37 ns siguen aprobados
en su alcance propio. La cola de seguimiento incorpora las síntesis revisadas del host
y supervisor y sus auditorías. El host ocupa 7.899 LUT/14.848 FF; el supervisor,
102 LUT/149 FF. Dos perfiles del netlist pasan 13 escenarios; el ensayo
de restricciones en 45 CLR elimina `TIMING-6/7` y conserva 30 rutas D.
La síntesis R2 integra esas restricciones con recursos, celdas e INIT
idénticos; la reapertura del DCP conserva XDC y las 30 rutas D. Queda
resolver un aviso CDC en el reset externo. Ambos perfiles DATA R2 pasaron
la revisión completa y 14 controles de evidencia por perfil: observan el
fallo a 660/672 ns desde la parada (650/670 ns desde la primera muestra
de gestión), completan DATA y lectura DDR, y mantienen la CPL retenida.
El primer intento se conserva como diagnóstico de una espera del banco
sobre un reloj detenido. CMD de 5 ns pasó la revisión R2 y 18 controles:
cierre a 665 ns de la parada, ningún CMD/DATA aceptado ni escritura del
trabajo perdido, y lectura de los 64 bytes anteriores en 135 ns.
Se corrigió una expectativa heredada del revisor: ocho escrituras físicas
por controlador, sin cambiar la traza o el banco nativo. CMD de 37 ns pasó
la revisión R2 y 18 controles: cierre observado a 599 ns, lectura de los
bytes anteriores en 560 ns y ningún CMD/DATA aceptado para el trabajo 18.
CPL visible de 5 ns pasó revisión y 14 controles R2: cierre a 675 ns,
64 bytes escritos recuperados y CPL retenida sin DONE/publicación. El
guion R2 corrige solo el mensaje esperado por un control negativo; no
cambia el banco ni el verificador positivo. CPL de 37 ns también pasó
revisión y 14 controles: cierre a 673 ns y CPL retenida con datos intactos.
El banco de resultado ya publicado, con ACK de identidad correcta
precargado, pasó a 5 ns la revisión completa y 16 controles: cierre de
retirada a 690 ns, identidad/estado retenidos y readback de los 64 bytes.
El perfil de 37 ns también pasó la revisión completa y 16 controles:
cierre a 607 ns, resultado retenido y datos intactos. La variante XDMA
con AXI-Lite esclavo y puertos de estado ya está generada, revisada y
respaldada; solo añade las dos opciones previstas. El ejemplo AMD trae
un bucle maestro/esclavo que la copia de prueba separa. La tanda nativa
`20261006T170709Z` terminó en el timeout de descubrimiento: BAR0 quedó
sin respuesta antes de cualquier prueba local. El diagnóstico está
revisado y respaldado. R1 (`20261006T172223Z`) añade una única lectura
de descubrimiento BAR0 con respuesta cero y completó consultas y tráfico,
pero dejó sin cubrir la retención de B/R locales. Está respaldada como
diagnóstico; R2 (`20261006T174559Z`) pasó la revisión estricta y 12 controles:
31 lecturas, 13 escrituras, 39/93 ciclos B/R retenidos y 64 bytes completos
H2C/C2H. El maestro local RTL pasó diez casos unitarios y 14 controles;
su síntesis OOC revisada usa 42 LUT/279 FF con límite 32 y 44 LUT/290 FF
con 65.536. Faltan retardos externos y sitio de reloj: el WNS estimado de
2,795 ns no acredita cierre físico. La tanda `20261006T180930Z` pasó la
integración directa del maestro RTL con XDMA: 44 accesos con identidad,
respuestas reales y reportes retenidos; máximo solicitud→B/R de 48 ns.
Se rechazaron 18 corrupciones de evidencia. La comparación RTL/netlist
`20261006T183936Z` pasó diez casos en cada una de cuatro variantes, con
límites 32/65.536. Coinciden handshakes y valores por ciclo tras contabilizar
los 100 ps de asentamiento de celdas AMD; se rechazaron 22 alteraciones.
Los dos intentos previos se conservan como diagnósticos de preparación.
El registro de descriptores `20261006T190004Z` pasó 31 escenarios unitarios
y 22 controles de evidencia: completaciones tardías, identidades y cierre
revocables, saturación y pendientes conservados con Run=Busy=0. Su síntesis
revisada `20261006T190536Z` usa 563 LUT/273 FF con contadores de 32 bits
(WNS interno estimado 1,749 ns); la variante reducida usa 329 LUT/161 FF.
La comparación RTL/netlist `20261006T192532Z` pasa los mismos 31 escenarios
en ambas representaciones: 333/385 muestras coincidentes y 62 controles
negativos. La integración nativa `20261006T193312Z` correlaciona dos
descriptores publicados, sus bytes leídos por PCIe, Done y ocho lecturas
CSR; pasan 29 controles y se conserva la revocación del cierre. El retorno
de 64 bytes se completa 103,999 ns después de `accounted`: el inventario
de recepción del host sigue siendo necesario. El primer control negativo
del revisor se corrigió para alterar memoria publicada, conservando el
diagnóstico y sin repetir Vivado. El secuenciador W1C/CSR corregido
`20261006T201404Z` pasa 89 casos con el maestro real y un modelo de registros:
589 transacciones contrastadas y 19 controles de evidencia. La revisión
detectó una primera dirección indefinida que el banco anterior dejó pasar;
los dos intentos previos permanecen como diagnósticos. Su síntesis
`20261006T201715Z` usa 806 LUT/414 FF, WNS estimado +1,880 ns a 250 MHz,
sin restricciones externas ni cierre físico. La comparación `20261006T204759Z`
pasa los 89 casos con ambos bloques sintetizados: 15.536 observaciones
coincidentes tras contabilizar 100 ps de celdas y 40 controles rechazados.
Dos intentos mixtos se conservan como diagnósticos de LUT sin inicializar.
La integración nativa `20261006T205127Z` pasa revisión y 44 controles:
42 accesos locales, incluidos 20 autónomos; tres rondas CSR y snapshot a
1,108 µs del inicio. Se verificaron los 64 bytes y la revocación retenida.
Quedan cadenas del ledger, driver, inventarios PCIe/BAR y certificados globales.
Los contratos host/XDMA, reset operativo,
DIMM exacto y cierre físico siguen abiertos.
[Diseño, métricas, evidencia y límites](RETAINED_PATHS_REAL_MIG.md).
[Retención ante watchdog real](RETAINED_RECEIVER_EPOCH.md).
[Síntesis y auditoría de restricciones](RETAINED_SYNTHESIS_AUDIT.md).
[Resultados del supervisor](RETAINED_RESET_TIMING_RESULTS.md).
[Banco DATA pendiente](RETAINED_DATA_PENDING.md).
[Banco CMD pendiente](RETAINED_CMD_PENDING.md).
[Banco CPL visible](RETAINED_CPL_VISIBLE.md).
[Resultado publicado](RETAINED_RESULT_PUBLISHED.md).
[Gestión local XDMA](XDMA_LOCAL_MANAGEMENT.md).
[Maestro local y proveedor pendiente](XDMA_LOCAL_PROVIDER_DESIGN.md).
[Registro de obligaciones de descriptores](XDMA_DESCRIPTOR_LEDGER.md).
[Secuenciador W1C/CSR, diagnósticos y métricas](XDMA_STOP_REGISTERS.md).
[Investigación SPD de alternativas B1/B2, aún sin calificar](MEMORY_SPD_B1_B2.md).

**Hito anterior — control NTRD y propietario coordinado:** ocho casos de
pérdida de sesión revisados desde VCD, 176 escrituras y 86 lecturas MMIO.
Regresión original/coordinada idéntica con tres semillas. Síntesis OOC:
**3.054 LUT / 4.334 FF**, WNS interno **1,941 ns**, tres CDC informativos.
**689 tandas terminales respaldadas localmente/NFS**, incluidos diagnósticos.
La cola R34 continúa con las secciones AXI en el mismo datapath y el receptor
de memoria; contratos host/XDMA, reset y cierre físico siguen pendientes.
[Diseño, evidencia y límites](OWNED_MANAGED_COORDINATOR.md).

**Hito anterior — coordinador de ownership en RTL sintetizable:** ocho
perfiles reproducidos sin diferencias en 553.884 estados públicos.
Síntesis OOC: **2.164 LUT / 3.410 FF**, WNS interno **1,941 ns** y tres
cruces CDC informativos; sin cierre físico de E/S ni ruteo.
**685 tandas terminales respaldadas localmente/NFS**, incluidos diagnósticos.
La cola R33 integra control NTRD, propietario y secciones en el mismo camino;
proveedores host/XDMA, reset completo y placa siguen pendientes.
[Diseño, métricas y respaldo](OWNED_DRAIN_COORDINATOR.md).

**Hito anterior — propietario y secciones integrados en el banco:** ocho
perfiles revisados, dos controles adicionales y tres defectos RTL detectados.
**682 tandas terminales respaldadas localmente/NFS**, incluidos diagnósticos.
La cola R32 traslada la conexión al RTL sintetizable y mantiene pendientes
los proveedores reales host/XDMA y el datapath común.
[Evidencia y límites](OWNED_DRAIN_COMPOSITION.md).

**Hito anterior — proveedor de ownership real:** dos bancos positivos de
12 escenarios, tres defectos RTL detectados y revisión independiente de
payloads/épocas/VCD. El propietario con su certificado ocupa **1.604 LUT /
3.257 FF** en síntesis OOC; sus tres cruces CDC son informativos.
**679 tandas terminales respaldadas localmente/NFS**, incluidos diagnósticos.
La cola R31 conecta este proveedor al coordinador y a las secciones reales
ya revisadas. Host/XDMA protegidos, reset global y cierre físico siguen abiertos.
[Evidencia y siguiente integración](OWNERSHIP_CERTIFICATE.md).

**Hito anterior — cierre BAR del modelo:** dos generaciones revisadas,
incluida una escritura admitida que llega a AXI después del cierre.
Dos readbacks completos, cuatro solicitudes rechazadas y diez certificados
inválidos rechazados; 94 alteraciones de evidencia detectadas.
**662 tandas terminales respaldadas localmente/NFS.** La cola R26 mantiene
pendientes driver, inventario protegido e integración global; no autoriza
reset. [Evidencia](HOST_BAR_CLOSURE.md). La [limpieza de D:](D_BACKUP_CLEANUP.md)
retiró 12 extracciones redundantes conservando archivos y fuentes.

**Hito anterior — retorno C2H:** 64 bytes cargados por H2C y devueltos
por C2H; respuesta R estable durante Run=0/Busy=1 y memoria host reconstruida
desde cada byte CQ aceptado. 74 alteraciones de evidencia rechazadas.
**661 tandas terminales respaldadas localmente/NFS.** La cola R25 continúa
con cierre del productor BAR e integración de certificados; inventario
global, reset y placa completa siguen sin calificar. [Evidencia](XDMA_C2H_STOP.md).

**Hito anterior — cadena H2C:** dos descriptores adyacentes entregados
por PCIe y 2 completados en el ensayo de parada con B retenida. Se
compararon 128 bytes y se rechazaron 52 alteraciones de evidencia.
**660 tandas terminales respaldadas localmente/NFS.** La cola R24 continúa
con C2H y cierre del productor host; reset global y placa completa siguen
sin calificar. [Evidencia y tiempos](XDMA_CHAIN_STOP.md).

**Hito anterior — inventario nativo XDMA:** una transferencia H2C de
64 bytes y una escritura/lectura BAR de cuatro bytes revisadas sobre el
endpoint real simulado. Run=0 no retira la B pendiente; el BAR conserva
obligaciones independientes de SGDMA. 52 alteraciones de evidencia
rechazadas. **659 tandas terminales respaldadas localmente/NFS.** Se
conservan cinco diagnósticos de preparación del banco. La cola R23
continúa con cadenas H2C, C2H y cierre del productor host antes de integrar
reset global. [Resultados y alcance](XDMA_STOP_INVENTORY.md).

**Hito anterior — escrituras AXI4 durante reenvío:** ocho casos R2
revisados, 210 alteraciones de evidencia
rechazadas y un defecto deliberado de BVALID detectado. Un diagnóstico
conservado mostró que detener H podía impedir la entrega del último beat;
R2 separa las pausas y comprueba su entrega. **653 tandas terminales
respaldadas localmente/NFS.** La cola R22 pasa al contrato de parada e
inventario protegido XDMA; capacidad defensiva formal, CDC-15 y cierre
físico siguen abiertos. [Detalle](AXI4_FORWARDED_FAULTS.md).

Las entradas siguientes conservan el historial de hitos y sus conteos.

**Estado vigente — solicitudes AXI4 y síntesis R2:** cinco causas externas
en dos perfiles revisadas, con la geometría de 256 beats corregida y dos
controles RTL negativos detectados. Síntesis candidata: **10.603 LUT y
13.771 FF**; CDC-15 y cierre físico pendientes. **650 tandas terminales
respaldadas localmente/NFS.** La cola R21 continúa con fallos W reenviados,
alcanzabilidad de capacidad y contrato de inventario del productor XDMA.
[Pruebas y límites](AXI4_SECTION_REQUEST_FAULTS.md) ·
[Métricas de síntesis](AXI4_SECTION_SYNTHESIS_R2.md).

Las entradas siguientes conservan los hitos históricos y sus conteos.

**Estado vigente — fallos AXI4 y RTL R2:** 18 casos aprobados y
224 alteraciones de evidencia rechazadas; regresión de 65 casos
aislados y dos perfiles completos de cancelación/reset aprobada. Se conservan
tres diagnósticos: fallo inicial, sonda y espera prematura del banco.
**645 tandas terminales respaldadas localmente/NFS.** La cola R20 pasa a
síntesis del candidato R2 y causas restantes de fallo de solicitudes.
[Detalle y límites](AXI4_SECTION_FAULT_LIFETIMES.md).

Las entradas siguientes son el historial de hitos y sus conteos a esa fecha.

**Cancelaciones AXI4 en tránsito comprobadas:** `20261006T005042Z` pasa
cuatro casos en H/D=7/5 y 7/37 ns, conservando tráfico, respuestas locales,
reset y contención posterior de R huérfana. La revisión rechaza 92
alteraciones de evidencia. `005623Z` detecta dos defectos RTL privados de
retirada/sobrescritura de mensajes pendientes. **640 tandas terminales
respaldadas localmente/NFS.** La cola R19 continúa con fallos externos antes
y durante los certificados y generación nueva con mailbox ocupado.
[Detalle](AXI4_SECTION_CANCELLATION.md).

**Síntesis integrada AXI4 revisada:** `20261006T000729Z` mide **10.631 LUT
y 13.771 FF** incluyendo Clock Converter AMD. WNS OOC +1,523 ns; faltan
retardos de E/S y ruteo. Las inspecciones R2–R4 confirman caminos propios
acotados y un control negativo de exclusión temporal sin modificar el DCP.
Se conservan un aviso CDC-15 nativo y el bit 6 de fallo optimizado a cero.
**638 tandas terminales respaldadas localmente/NFS.** La cola R18 pasa a
cancelación/fallos anteriores y durante los certificados; XDMA/reset global
siguen abiertos. [Detalle y límites](AXI4_SECTION_SYNTHESIS.md).

**Sección AXI4 con CDC AMD real y retención:** R2 `232715Z` pasa ambos
perfiles H/D=7/5 y 7/37 ns, tres identidades frescas, cierre con W parcial,
reset con al menos 64 ciclos por dominio y B/R locales retenidas, y cuarentena
R real posterior al certificado. Rechaza 54 alteraciones de evidencia;
`232256Z` detecta dos defectos RTL privados, uno mediante revisión externa.
La limitación del pulso inicial `231510Z` queda conservada y documentada.
**633 tandas terminales respaldadas localmente/NFS.** R17 mantiene
`complete_drain` abierto para otras vidas de fallo/cancelación, síntesis/CDC
integrada y colas protegidas XDMA. [Detalle](AXI4_RETAINED_SECTION.md).

**Admisión AXI4 retenida con protector incorporado:** R2 `225132Z` pasa
13 escenarios y rechaza 35 alteraciones de evidencia: grupo finito 33/129
sin exceder capacidad 32/128, W parcial con/sin VALID al cerrar, ofertas
retenidas durante fallos y SLVERR locales de hasta 256 transferencias.
`225532Z` detecta cuatro mutaciones RTL: tres por aserciones y RLAST local
por revisión independiente. `224808Z` mide 4.113 LUT/3.496 FF; WNS OOC
+3,675 ns a 7 ns, sin aprobación física. **630 tandas terminales respaldadas
localmente/NFS.** R16 mantiene `complete_drain` abierto: composición con CDC
real, certificados y transporte de fallos, después colas protegidas XDMA.
Detalle: [AXI4_RETAINED_ADMISSION](AXI4_RETAINED_ADMISSION.md).

**Protector causal AXI4 aislado validado:** `221338Z` pasa 19 escenarios,
32 alteraciones de evidencia rechazadas y preservación de respuestas sanas
ante cuarentena/fallo de inventario. `221604Z` detecta tres mutaciones RTL.
`222016Z` mide 2.213 LUT/2.588 FF, WNS OOC +3,768 ns a 7 ns, sin aprobación
física. **626 tandas terminales respaldadas localmente/NFS.** La cola R15
mantiene `complete_drain` abierto: admisión activa, rechazos locales por
ráfaga, composición real con el protector y colas internas XDMA.
Detalle: [AXI4_RESPONSE_GUARD](AXI4_RESPONSE_GUARD.md) y
[contrato de frontera activa](AXI4_ACTIVE_BOUNDARY_PLAN.md).

**Capacidad AXI4 y composición CDC pasiva revisadas:** `212032Z` pasa once
casos con 32 escrituras/128 lecturas; `212342Z` mide 2.482 LUT/2.824 FF.
R2 `214547Z` pasa tráfico completo con H/D a 7/5 y 7/37 ns, tres pruebas
frescas y fallo R real con D detenido; 26 corrupciones de evidencia rechazadas.
`214938Z` detecta dos mutaciones RTL. `215303Z` mide 5.320 LUT/6.648 FF,
16 cruces CDC-3 sin avisos CDC y WNS OOC +1,773 ns. El modelo FIFO AMD
no representa retardos de sincronización; no hay aprobación física.
La primera composición `213844Z` queda como diagnóstico conservado.
**623 tandas terminales respaldadas localmente/NFS.** La cola R14 mantiene
`complete_drain` abierto: frontera activa, contención de respuestas,
colas internas XDMA y coordinación global. El par actual detecta la R
huérfana, pero la deja atravesar. Detalle: [AXI4_CDC_DRAIN](AXI4_CDC_DRAIN.md).

**Primer monitor AXI4 aislado revisado y respaldado:** `210303Z` pasa
22 escenarios con referencia independiente de transacciones/payload,
los 16 IDs y ráfagas de hasta 256 transferencias. `210557Z` detecta cuatro
mutaciones RTL. `210852Z` sintetiza en 2.438 LUT/2.771 FF, con margen
setup OOC +3,961 ns a 7 ns; faltan retardos de E/S y colocación física.
617 tandas terminales respaldadas localmente/NFS. La cola R13 mantiene
`complete_drain` abierto: ampliar cobertura de capacidad y luego probar
ambos extremos del CDC AMD antes de admisión activa y colas internas XDMA.
Detalle: [AXI4_DRAIN_MONITOR](AXI4_DRAIN_MONITOR.md).

**Criterios de fallos de la sección AXI-Lite cubiertos; siguiente AXI4:**
`203519Z-V4-ffn-supervised-guard-faults` añade 24 casos: alarma exclusiva
B/R antes/después del certificado y ambas respuestas por CDC real. Reporte
de 20,8–29,8 ns, con cero flancos host adicionales en la variante detenida.
`203858Z` detecta bypass y pérdida de cuarentena, pero termina parcial al
fallar antes de lo previsto; la adenda aprueba solo esos dos controles.
`204343Z` ejecuta únicamente el tercero y detecta el corte de R sana.
Tres tandas más respaldadas localmente/NFS; 614 tandas terminales en la
sesión. La cola R12 pasa al monitor retenido AXI4 con LEN/LAST e IDs, sin
dar por cerrado el inventario interno XDMA ni reset global.
Detalle: [AXIL_GUARD_FAULTS](AXIL_GUARD_FAULTS.md) y
[AXI4_DRAIN_PLAN](AXI4_DRAIN_PLAN.md).

**Contención de respuestas incorporada y validada en simulación:**
sección R3 con protector de B/R y certificado R5. `200930Z` pasa los
16 escenarios de vida útil con manager/CDC AMD reales; `201350Z` pasa
12 casos de regresión y `201810Z` otros cinco aislados. La B huérfana
queda retenida antes del manager y las respuestas válidas del otro canal
pueden completarse. Se preserva el fallo original `194744Z` como antecedente.
`201943Z` sintetiza 802 LUT/1.762 FF y 14 CDC-3, sin avisos CDC;
`202301Z` detecta dos mutaciones de alarma/mailbox. Cinco tandas y dos
adendas respaldadas localmente/NFS; 611 tandas terminales respaldadas en
la sesión. Falta probar alarma exclusiva del protector con host detenido,
R huérfana por CDC real y controles específicos de cuarentena. La cola R11
mantiene esas tareas antes del drenaje completo y reset global.
Detalle y límites: [AXIL_RESPONSE_GUARD](AXIL_RESPONSE_GUARD.md).

**Revocación de fallos posterior al certificado, R3:**
`20261005T191739Z-V4-ffn-supervised-shutdown-fault-r3` pasa doce casos con IP AMD real:
admisión/H/D, dispositivo a 5/37 ns y origen activo/detenido. Revocación
simulada de 20,8–28,8 ns, alarma retenida y nuevas identidades bloqueadas;
dos mutaciones RTL detectadas. Síntesis: 544 LUT/1.605 FF. R3 elimina
el CDC crítico de retorno; permanece el aviso CDC de admisión para analizar
en composición real. Sin aprobación física ni drenaje completo de XDMA.
Nueve tandas y adendas respaldadas localmente/NFS. Siguiente: esa composición
y fallos previos a prueba/buzón pendiente/con error local retenido.
Detalle: [AXIL_FAULT_REVOCATION](AXIL_FAULT_REVOCATION.md).

**Frontera retenida y productor AXI-Lite implementados:**
`182439Z-V4-ffn-supervised-shutdown-fence-r3` pasa tres escenarios en ambos
perfiles: 387 escrituras y 131 lecturas por perfil, créditos limitados a 32,
mitades AW/W completadas y rechazo local permanente después del cierre.
El certificado espera al productor RTL detenido; una respuesta local tardía
permanece estable durante reset del CDC. Síntesis y dos controles negativos
también revisados y respaldados; R1/R2 fallidas se conservan. Admisión:
226 LUT/117 FF; certificación: 540 LUT/1.589 FF. Sin cierre físico.
Sigue la revocación de pruebas ante fallos posteriores y la extensión a
AXI4/colas protegidas: no se aprueba drenaje completo de XDMA.
Detalle: [AXIL_RETAINED_FENCE](AXIL_RETAINED_FENCE.md).

**Saturación AXI-Lite medida y respaldada:**
`175658Z-V4-ffn-supervised-shutdown-saturation` pasa cinco escenarios en
ambos perfiles (5/37 ns): 33 AW sin W, 18 W sin AW y 51 solicitudes por
canal con respuestas bloqueadas. Cada perfil completa 517 escrituras y
261 lecturas, incluidos los rechazos, sin reset operativo ni repetición al
reabrir. El inventario U de ocho bits coincide con la reconstrucción de la
traza. Respaldo y adenda verificados; seis avisos revisados, 74 alteraciones
de evidencia detectadas. Siguiente: wrapper con inventario retenido,
detención real del productor, límite explícito de admisión y colas AXI4/XDMA.
Detalle: [AXIL_SHUTDOWN_SATURATION](AXIL_SHUTDOWN_SATURATION.md).


**Aislamiento AXI-Lite real y recursos revisados:**
`173146Z-V4-ffn-supervised-shutdown-pair` pasa seis casos en dos perfiles,
con Shutdown Manager AMD, convertidor, inventarios y reset RTL.
Se comprueba rechazo de accesos nuevos y tr?fico posterior al reset.
Tambi?n se demuestra que cierre H/D puede coexistir con W o respuestas
pendientes arriba: no basta para resetear XDMA completo.
`173553Z` mide la IP aislada: 518 LUT (196 LUTRAM) y 895 FF.
Las tres tandas, fuentes y adendas tienen respaldo local/NFS verificado.
La saturación posterior se registra arriba; AXI con ráfagas sigue pendiente.
Detalle: [AXIL_SHUTDOWN_MANAGER](AXIL_SHUTDOWN_MANAGER.md).



**Actuador de reset local RTL aprobado y respaldado:**
`170319Z-V4-ffn-supervised-local-reset-r2` pasa ocho casos adversos y dos
perfiles del convertidor AMD real; ahora el RTL dirige hold/conteo/release.
S?ntesis aislada: 91 LUT y 182 FF, sin RAM ni DSP. Ocho cruces CDC-9
reconocidos; no calificaci?n f?sica. La adenda revisa el aviso por una XDC de
IP ausente del top aislado y conserva el rechazo gen?rico original.
R1, R2 y revisi?n tienen respaldo local/NFS verificado. La integraci?n con
el guard, el aislamiento host y el drenaje completo de XDMA siguen pendientes.
Dos controles negativos RTL tambi?n pasan en `171232Z`: el banco rechaza
generaci?n ignorada y liberaci?n anticipada. Los intentos fallidos se conservan.
Detalle: [LOCAL_RESET_ACTUATOR](LOCAL_RESET_ACTUATOR.md).



**Drenaje AXI-Lite en ambos extremos revisado y respaldado:**
`164228Z-V4-ffn-supervised-axil-pair-r2` pasa seis casos en dos perfiles
con la IP AMD real, incluidas lecturas/escrituras despu?s de reset coordinado.
En el perfil lento los inventarios externos vac?os coexist?an con colas internas
de handshake; el reset solapado esper? el reloj detenido y 64 flancos reales
por dominio. La secuencia todav?a pertenece al banco y el aislamiento host
es una entrada contractual. R1 fallido, R2 y adenda revisada se conservan
localmente y en NFS. Siguiente: actuador RTL de reset local, sin aprobar
el drenaje completo de XDMA. Detalle: [AXIL_PAIR_DRAIN](AXIL_PAIR_DRAIN.md).


**Buzón de certificados entre relojes aprobado y respaldado, 16:18 UTC:**
`161235Z-V4-ffn-supervised-tagged-cdc` pasa doce casos usando XPM real,
incluido monitor AXI-Lite → buzón → secuenciador a 7 ns/10 ns. Trece
peticiones producen diez respuestas vigentes y tres descartes obsoletos.
Retorno del certificado: 31,2–38,7 ns simulados con relojes activos;
síntesis OOC: 212 LUT y 608 FF. Se revisaron los siete avisos y la
exención CDC-15 de AMD sobre el bus, conservando su máximo de 7 ns.
No se aprueban CDC físico ni drenaje de todo el endpoint.
Respaldo local/NFS: 72 archivos, SHA-256
`260bf16cca3f846cd61f57c472cf026f0b975c34e5fd6e4f75336ee881d334c6`;
adenda `98ea11bcae07583f5749af50969857c777cedfeef3211f27f95e5ca1603b55f9`.
Siguiente: inventario y aislamiento a ambos extremos de los CDC de datos,
antes de integrar el nuevo coordinador en el guard nativo.
Detalle: [TAGGED_RESET_MAILBOX](TAGGED_RESET_MAILBOX.md).

**Monitor de drenaje AXI-Lite aprobado y respaldado, 15:57 UTC:**
`155453Z-V4-ffn-supervised-axil-drain` pasa trece casos en 82,358 s,
incluida síntesis: 77 LUT y 80 FF. La conexión al secuenciador en un
mismo reloj no solicita reset ante AW incompleta ni B bloqueada;
espera el inventario vacío y un ACK fresco. No incluye cruces de reloj
ni prueba drenaje interno de XDMA o AXI con ráfagas. Los avisos y el
timing OOC están revisados con sus límites. Respaldo local/NFS:
65 archivos, SHA-256 `359ceb6f6f1b2cc2e95ba203f3d40f0adf00def484e88f1e2b360a44e4e866c7`.
Adenda `3b2ff1c535b18bcb62f2b6fb7028e17785b598273cdd663ac94acd6577d95a49`.
Siguiente: solicitudes/ACK atómicos entre dominios y aislamiento probado,
antes de integrar el guard y permitir reset del endpoint real.
Detalle: [AXIL_DRAIN_MONITOR](AXIL_DRAIN_MONITOR.md).

**Secuenciador aislado aprobado y controles negativos respaldados, 15:50 UTC:**
`153531Z-V4-ffn-supervised-endpoint-sequencer` pasa nueve casos,
incluido el contador completo de 50,00001 ms; síntesis OOC: 114 LUT y
198 FF. Los 84 avisos están clasificados y el timing conserva sus
limitaciones de E/S y reloj. `154758Z-V4-ffn-supervised-endpoint-sequence-mutants`
rechaza cuatro errores introducidos en copias RTL con el banco original.
Ambas tandas y la adenda tienen respaldo local/NFS verificado.
Detalles, hashes y límites en [ENDPOINT_RESET_SEQUENCER](ENDPOINT_RESET_SEQUENCER.md).
El [monitor AXI-Lite](AXIL_DRAIN_MONITOR.md) también pasa su banco aislado,
separado del endpoint nativo. El intervalo completo sobre XDMA y su liberación
siguen pendientes; pasar el contador aislado no cierra esa prueba.

**Reset opcional: observación nativa aprobada y respaldada, 15:14 UTC:**
`150122Z-V4-ffn-supervised-xdma-reset-assert-r2` comprueba aserción de
`axi_aresetn`, ambos enlaces arriba y 639 flancos por reloj nativo durante
2,555 µs con el reset activo. Termina en 503,036 s. El revisor R2 corrige
la lectura de intervalos sin cambios del VCD, conserva el fallo anterior
y rechaza 14 alteraciones; los 139 avisos coinciden con la referencia.
Respaldo: 842 archivos, SHA-256
`c8d0bd055ce3ea61570b2373f0f84388f1d09207325fda2353f3da81d2b02d5d`;
adenda aprobada `040f0747ca872512eb02ceecbe4bcd0f25dc5903f8c16f73556af3af00fde9bc`.
No se ha probado el intervalo completo de 50 ms ni la liberación del reset.
Siguiente: contrato del secuenciador, separación endpoint/CDC y ACK frescos.

**PERST aprobado y respaldado; diagnóstico de reset opcional, 15:02 UTC:**
La tanda `134131Z` termina con código 0 en 2.476,610 s. El verificador
funcional comprueba datos y reserva retenidos ante PERST, y el retorno
del enlace 62,110 µs después de liberarlo. La procedencia de 221 entradas
compiladas y los 170 avisos pasan la revisión R3, que rechaza 41
alteraciones. Respaldo verificado a las 14:58:19 UTC: 1.474 archivos,
258.110.567 bytes, SHA-256
`46c609d750b928e5f48d9251ba7b6933749ba31ccd1b148f9c9e138bacaf95e2`.
No se ha repetido la simulación ni aprobado recuperación de sesión.

El ejemplo con reset opcional `141836Z` ya tiene respaldo local/NFS:
634 archivos, SHA-256
`cd2f158bbe39f8e5bf8b39c0b20035731e3d769a788568bf6c8b869e4a20cc0e`.
El diagnóstico observará únicamente la aserción de ese reset en
XDMA nativo sin MIG, con presupuesto de 1.800 s. Su primer intento
`145900Z` falló antes de compilar por la sintaxis de `set_property` y
está respaldado. R2 usa `-dict`, conservando banco y verificador.
Su ejecución no habilita recuperación ni coordinación.
Detalles y límites en [ENDPOINT_RESET_COORDINATION](ENDPOINT_RESET_COORDINATION.md).

**Auditoría del reset opcional respaldada, 14:04:46 UTC:**
`20261005T140115Z-V4-ffn-supervised-xdma-reset-ports-r3` comprueba
145→146 puertos al habilitar `soft_reset_en` en copias aisladas del XDMA
DMA: solo aparece `dma_bridge_resetn`. Termina en 50,407 s sin avisos;
458 archivos del ejemplo original siguen intactos. No acredita semántica
de reset DMA ni coordinación. El [contrato y siguiente diseño](ENDPOINT_RESET_COORDINATION.md)
explican por qué no conectar directamente esa entrada al guard actual:
su espera de enlace/reset liberado podría impedir que retire su petición.
Respaldo: 58 archivos / 975.694 bytes, SHA-256
`6a48f123c2add662fa796150e3483fb6065a9cbbe48266eda22cc90c490e6a62`.
R1 y R2 se conservan como fallo de script y procedencia incompleta.

**Registro del lanzamiento de PERST nativo, 13:41:31 UTC:**
`20261005T134131Z-V7-ffn-supervised-guarded-native-reset` conserva los
18 trabajos y añade una escritura DDR con B retenida durante el reset
real del ejemplo PCIe. Exige lectura correcta y reserva conservada;
retira los ACK de coordinación aún no implementados y debe mantener
el transporte bloqueado incluso cuando vuelva el enlace. Presupuesto:
3.600 s; recolección y revisión por la misma cola, sin relanzamiento.
No hay aprobación hasta el resultado terminal y su respaldo.

**Endpoint nativo con guard aprobado y respaldado, 13:41:12 UTC.**
`20261005T124640Z-V7-ffn-supervised-guarded-native` termina en 1.935,889 s
y 175,635 µs simulados. Comprueba 18 trabajos, 35 frames, 294 escrituras
de control, nueve lecturas de control y 16 escrituras / 28 lecturas DDR.
Las 84 señales VCD corroboran el arranque: 17.564 muestras de gestión,
4.988 flancos host y 3.991 fuente en reset después del inicio de vida útil.
Se rechazan 30 alteraciones de evidencia. La revisión R4 conserva las
correcciones de frontera temporal y contabilidad de fuentes inactivas;
el RTL y la simulación no se repiten ni se modifican.
Latencia observada RQ→B por frame: 292–304 ns, mediana 292 ns; mide
aceptación, no persistencia DDR ni ancho de banda sostenido.
Respaldo: 1.476 archivos / 248.338.857 bytes; SHA-256
`af60cb1f1523c03d1a116ed194b162f6fda2775ab66fbb816a490bfc31ed1b5c`.
La coordinación del endpoint, recuperación y hardware siguen pendientes.

**Guard con dos MIG aprobado y respaldado, 12:30:06 UTC:** `121701Z`
verifica 18 trabajos sanos y una pérdida host con B retenida. Conserva
la reserva y un descriptor de completion bloqueado; lee correctamente
los 64 bytes escritos después de perder el enlace. Total: 17 escrituras,
29 lecturas, 37 frames; 61/57 lecturas periódicas por controlador y 18
alteraciones de evidencia rechazadas. Respaldo de 709 archivos, 115.118.458 bytes.
`120124Z` conserva el fallo previo de la comprobación del buffer de staging.
Host AXI de banco; endpoint nativo, timing físico y hardware siguen pendientes.
[Contrato y evidencia](HOST_LOSS_GUARD.md).

**Restricciones del guard: diagnóstico respaldado, equivalencia pendiente.**
La auditoría `113249Z` verifica 30 FDCE y 24 CLR. La prueba `113525Z`
añade únicamente esas excepciones y estima WNS +1,940 ns / WHS +0,013 ns,
sin cierre físico. `114041Z` pierde la procedencia de auxiliares EDIF al
exportar en carpeta común; `114859Z` los separa y conserva los 63 por lado.
Coincide el EDIF principal, pero difieren todos los auxiliares protegidos.
Ambos intentos están respaldados **sin aprobación de equivalencia completa**.
No se promueve ese checkpoint a implementación. [Detalle](HOST_LOSS_GUARD.md).

**Síntesis de la composición NTRD respaldada:** `112456Z` mide
4.083 LUT / 8.720 registros, sin latches/BRAM/URAM/DSP. WNS −0,513 ns,
12 extremos fallidos y ocho avisos críticos entre relojes; 979 entradas y
3.779 salidas carecen de retardos externos. No aprobada para implementar.
La auditoría posterior y sus límites se registran arriba. [Detalle](HOST_LOSS_GUARD.md).

**Propietario NTRD conectado al guard, revisado y respaldado:** `111635Z`
pasa nueve escenarios con los CDC AMD y el ACK del estado retenido real.
Nueve ARM, siete líneas/CMD/DATA, dos completions y un retiro; 17 alteraciones
rechazadas. CMD/DATA, época, cookie y reserva sobreviven al reset host.
El resultado previo no habilita reutilización. Pool/MIG y endpoint siguen
externos. Se conserva el primer verificador y su corrección R2 explícita.
Síntesis de esta composición conservada como diagnóstico. [Evidencia](HOST_LOSS_GUARD.md).

**Guard y frontera con CDC AMD revisados y respaldados, 11:09:56 UTC:**
`105941Z` pasa 17 escenarios con dos AXI Clock Converters reales: 80 capturas,
66 entregas y 14 retiros locales. Se rechazan 15 alteraciones de evidencia.
Incluye relojes detenidos, ACK demorado y reset repetido. La recepción host
puede preceder al ACK fuente; aquí el ACK vuelve antes de la detección.
Siguiente: sustituir el ACK controlado por la invalidación retenida NTRD.
Sin calificación de endpoint/DDR ni cierre físico. [Evidencia](HOST_LOSS_GUARD.md).

**Restricciones del guard revisadas y respaldadas, 10:49:18 UTC:**
`103904Z` atribuye las 48 rutas entre relojes a 24 CLR de sincronizadores.
`104435Z` aplica excepciones solo a esos pines después de verificar las
cadenas; conserva netlist y recursos. Desaparecen los ocho avisos críticos
de relación entre relojes y permanecen 92 retardos externos sin definir.
WNS estimado +4,092 ns; sin cierre físico. Próximo: conectar el guard a
los ocho canales/CDC y al ACK real NTRD. [Evidencia](HOST_LOSS_GUARD.md).

**Guard automático de pérdida, banco aislado revisado y respaldado:**
`103302Z` pasa diez escenarios con XPM y dos canales retenidos, incluyendo
reloj detenido, pulsos cortos, ACK tardío y reset repetido; rechaza 15
alteraciones. El intento `102730Z`, con estímulo inválido, también se conserva.
La síntesis `103345Z` mide 94 LUT / 239 registros, pero queda **sin aprobar**:
WNS −0,513 ns, 12 extremos setup fallidos y ocho avisos críticos de relojes.
Auditar los CLR y cruces antes de modificar restricciones; aún faltan CDC
AXI/NTRD/PCIe/MIG en este banco. [Contrato y evidencia](HOST_LOSS_GUARD.md).

**Diagnósticos y síntesis del adaptador retenido:** siete escenarios negativos
revisados en `100539Z`; la tanda conjunta falla por el nombre del puerto de
reloj en el Tcl y queda conservada. La síntesis corregida `101001Z` mide
1.132 LUT / 2.863 registros, sin BRAM/URAM; es una estimación OOC sin cierre
físico. `101054Z` rechaza dos variantes RTL deliberadamente defectuosas.
La variante que retira respuestas antes de reset pasa los conteos finales
del banco, pero falla la revisión por ciclo: ambos controles son obligatorios.
Faltan detector, secuencia de invalidación/ACK y reconciliación del inventario
host antes de integrar NTRD. [Evidencias y respaldos](PCIE_DUAL_LINK_LOSS.md).

**Frontera retenida con CDC AMD revisada y respaldada, 09:50:05 UTC:**
`094448Z` pasa 15 escenarios y rechaza 12 alteraciones de evidencia.
Conserva peticiones durante reset y distingue respuestas retiradas localmente
de recepciones host. Los casos 13/14 prueban las dos ventanas de ACK AXI-Lite.
El intento `093552Z`, fallido por una expectativa de ACK, también está respaldado.
Falta integración del detector y propiedad NTRD;
este banco no simula PERST nativo ni DDR. [Contrato y evidencia](PCIE_DUAL_LINK_LOSS.md).

**Ruteo OOC y auditoría estructural respaldados:** `083408Z` termina en
418 s con 9.684 LUT, 23.231 registros y 15 bloques RAM; WNS +0,380 ns
y WHS +0,010 ns bajo cinco orígenes de reloj supuestos. `084534Z`
atribuye las cinco discontinuidades a esos relojes padre y comprueba
que las 1.434 redes RTSTAT-10 carecen de cargas internas. Conserva los
78 sincronizadores propios; las nueve entradas CDC eliminadas corresponden
a cadenas sin cargas internas en su salida. Sin nuevas excepciones ni waivers.
El primer replay `085427Z` termina antes de la simulación principal por
sintaxis Tcl al pasar las bibliotecas a xelab; se conserva como fallo y
R2 `090132Z`, con un único cambio de configuración, pasa la revisión
funcional: 18 trabajos, 16 escrituras y 28 lecturas DDR comparadas;
54 barreras y 12 corrupciones rechazadas. Duración: 703,340 s.
Respaldo local/NFS verificado a las 09:17:21 UTC: 991 archivos / 98.722.233 bytes.
El fallo anterior también está respaldado.
El replay no usa SDF ni endpoint PCIe nativo y no cierra V6/V8.
[Contexto y criterios de revisión](PCIE_DUAL_HOST_SYNTH.md).

**RAM del pool caracterizada y respaldada, 08:23:03 UTC:** `081846Z`
confirma 14 RAMB36E2 + dos RAMB18E2 y 72 FDRE para contadores de ráfaga.
Los 40 caminos RAM→FIFO exportados tienen margen de síntesis positivo;
SYNTH-6 sigue presente porque los registros adicionales de salida están
desactivados. No se cambia la latencia ni se acredita timing físico.
Respaldo: 22 archivos / 6.262.348 bytes. Queda indexada la revisión de
596 avisos de puertos FIFO/CDC antes del ruteo exploratorio.
[Resultados y corrección de atribución del aviso de carga](PCIE_DUAL_HOST_SYNTH.md).

**Comparación EDIF respaldada; alcance corregido a las 12:07 UTC:**
`080626Z` compara los EDIF principales: conexiones y parámetros visibles idénticos,
excepto fecha de exportación y orden local de propiedades en seis RAM.
Las 38.557 identidades de primitivas coinciden; 14 corrupciones rechazadas.
La revisión retrospectiva encuentra 189 auxiliares protegidos exportados
en carpeta común. No se compararon por separado: se retira la afirmación
de preservación lógica completa y la dependencia de `074710Z` sigue abierta.
Respaldo: 215 archivos / 20.067.933 bytes. La auditoría posterior de
las 16 RAM con aviso SYNTH-6 también está revisada y respaldada.
[Evidencia y límites de la síntesis](PCIE_DUAL_HOST_SYNTH.md).

**Diagnóstico de restricciones revisado y respaldado, 07:54:14 UTC:**
`074710Z` demuestra que los objetos restantes de la consulta de timing
son 122 False Path, sin slack. Las excepciones sobre 45 CLR auditados
eliminan los 12 avisos críticos de relación entre relojes, conservando
recursos. WNS estimado +1,700 ns / WHS +0,013 ns. La comparación inicial
de las netlists visibles no cubría un bloque cifrado; la evidencia EDIF
posterior `080626Z` tampoco cubre sus auxiliares protegidos por separado.
El ruteo sigue sin habilitarse. Se conserva respaldado el intento previo
`074225Z`, detenido por esperar una lista vacía de rutas.
[Evidencia, límites y siguientes comprobaciones](PCIE_DUAL_HOST_SYNTH.md).

**Síntesis NTRD conservada sin aprobación, 07:33:30 UTC:** `072212Z`
terminó con 9.939 LUT / 23.279 registros / 15 bloques RAM. WNS estimado
+0,178 ns, WHS +0,013 ns; metodología informa 12 avisos críticos de
relaciones entre gestión y fuente/UI. Respaldo local/NFS verificado.
[Hallazgos y siguiente auditoría de reset](PCIE_DUAL_HOST_SYNTH.md).
Se conservan restricciones y severidades; este resultado no habilita
automáticamente ruteo ni sustituye la validación de la FPGA completa.

**PCIe nativo → NTRD → dos MIG aprobado y respaldado, 07:21:59 UTC:**
`063837Z` correlaciona TLP, AXI host/fuente, reserva, cookie y DDR:
18 trabajos, 35 tramas, 16 escrituras (1.024 bytes) y 28 lecturas comparadas
(1.792 bytes). Incluye 294 escrituras de control, nueve lecturas y rechazo
de 18 ACK sin barrera. Se verificaron 70 señales VCD, 54 barreras y drenaje
final; 20 alteraciones de evidencia rechazadas. Vivado: 1.909,028 s.
Respaldo: 1.453 archivos, 246.322.457 bytes, SHA-256
`64419a87c22bb35b69c862d21e20010095047f682259d18bb67639bc17e658a4`.
[Métricas y alcance](PCIE_DUAL_NATIVE_PLAN.md). Queda habilitada la síntesis
aislada del control/pool/supervisor y sus cuatro CDC; la extensión funcional
pendiente es [pérdida del host y respuestas abandonadas](PCIE_DUAL_LINK_LOSS.md).
Esta tanda acredita arranque frío con modelos genéricos y barreras externas
del banco; no cierra V7 completo, PERST, cuatro canales ni placa.

**Control NTRD y retiro conjunto aprobados y respaldados, 04:40 UTC:**
`043738Z` verifica el perfil nuevo de canal explícito y ACK MMIO sobre
recolector/puente. Seis simulaciones RTL/netlist; por ejecución, 257 escrituras
y 69 lecturas de control, 51 rechazos y ocho retiros conjuntos. Se conserva
la síntesis original de `042455Z`: 2.182 LUT / 3.405 registros. Aquella tanda
detectó nueve mutantes RTL y el revisor rechazó 236 corrupciones, pero su
replay falló por enviar AR durante el GSR inicial. La repetición observa
GSR y difiere solamente en espera de arranque/observación del banco.
También está respaldado el fallo previo de compilación `042341Z`.
Falta conectar este nuevo conjunto con supervisor/pool/CDC/ambos MIG.
[ABI, evidencia y límites](PCIE_DUAL_CONTROL.md).

**Recolector y retiro conjunto aprobados y respaldados, 04:13 UTC:**
`040503Z` conecta el recolector F01 al puente de dos canales mediante
descriptores confiables. Once casos × tres semillas × RTL/netlist:
23 tramas y ocho retiros por ejecución; ocho mutantes RTL detectados y
184 alteraciones de evidencia rechazadas. ARM y ACK son conjuntos; errores
locales permiten una nueva época y perder disponibilidad conserva propiedad.
Síntesis: 1.272 LUT / 2.481 registros; timing interno OOC sin colocar.
El destino sigue siendo un par de protocolo. El control MMIO ya se validó
en `043738Z`; resta repetir el conjunto sobre pool/CDC/ambos MIG.
[Evidencia y límites](PCIE_FRAGMENT_POOL_INTEGRATION.md).

**Puente con ambos MIG aprobado y respaldado, 03:46 UTC:** `032917Z`
compara 960 bytes escritos y 1.792 bytes leídos en 17 reservas. Verifica
cuatro ranks por canal, STOP, error local y conservación de identidades;
228 alteraciones de evidencia rechazadas. Los 35 modelos conservan su
procedencia y el tráfico periódico drena sin filtrarse a las lecturas.
`031705Z` valida por separado que el reset operativo del pool conserve
la reserva del puente: seis casos, un mutante RTL detectado y 123
alteraciones de evidencia rechazadas. Ambos respaldos están verificados.
[Resultados y siguiente conexión PCIe](PCIE_FRAGMENT_POOL_INTEGRATION.md).

**Puente fragmento → pool aislado aprobado y respaldado, 02:50 UTC:**
`022818Z` valida 30 escenarios y agotamiento de cookies en RTL/netlist;
`024535Z` añade el cruce de 32 a 64 bits. En conjunto: 18 simulaciones
positivas, ocho mutantes RTL detectados y 200 alteraciones de evidencia
rechazadas. Síntesis del puente: 324 LUT / 881 registros; sin timing físico
calificado. [Evidencia y conexión al pool](PCIE_FRAGMENT_POOL_INTEGRATION.md).
La composición con pool RTL y ambos CDC AMD pasó en `025843Z` R2:
dos perfiles asíncronos, trece reservas y once escrituras de 64 bytes por
perfil; tres mutantes detectados y 137 alteraciones de evidencia rechazadas.
Memoria y supervisor siguen siendo drivers de protocolo. El primer
lanzamiento falló por sintaxis Tcl antes de simular y se conserva respaldado.
La dependencia de simulación fría con ambos MIG quedó satisfecha en
`032917Z`. Sigue la conexión del recolector y el control del host; no se
adopta en estas tandas el patch ruteado de arranque del trabajo lateral.

**Integración PCIe nativa NTRF aprobada, 02:08:57 UTC del 2026-10-05:**
`014817Z` conecta BAR0 y BAR2 reales de XDMA mediante ambos CDC con MIG frío.
Se verifican 12 tramas, ocho aceptaciones, cuatro rechazos, tres escrituras
y ocho lecturas DDR; 82 escrituras y 73 lecturas de control incluyendo
descubrimiento. El revisor coteja 184 señales y rechaza 309 alteraciones de
evidencia. Vivado: 1.077,333 s. Respaldo local/NFS verificado a las
02:12:55 UTC, 1.404 archivos y 204.531.064 bytes.
[Evidencia y límites](PCIE_FRAGMENT_NATIVE.md).
La [siguiente conexión con multibuffer](PCIE_FRAGMENT_POOL_INTEGRATION.md)
está en desarrollo, separada del trabajo lateral dual MIG.

**Netlist ruteada NTRF aprobada, 01:32 UTC del 2026-10-05:** `012347Z` R2
reproduce los 7.283 eventos y tiempos del RTL y detecta 343 alteraciones
de evidencia. Mismo diseño y estímulo; la única corrección es la ruta
pasiva de observación del bit RRESP constante. Se verifican 15 reservas,
10 escrituras y 75 lecturas DDR, además de 695 escrituras y 147 lecturas
de control. Respaldo local/NFS verificado a las 01:34:48 UTC.
Simulación funcional sin SDF; esta tanda aislada no incluye XDMA nativo ni
restricciones físicas de placa. [Resultados](PCIE_FRAGMENT_HOST.md),
[integración nativa posterior](PCIE_FRAGMENT_NATIVE.md).

**Replay ruteado con observador corregido, 01:26 UTC:** `011556Z` se conserva
sin calificar y con respaldo verificado: el monitor leía un bit RRESP que
el ruteo dejó desconectado. R2 `012347Z` observa el registro constante
retenido, cuya estructura se comprobó en la netlist; no cambia el DUT ni
el estímulo. Ya aprobó la comparación completa, según el resultado anterior.
[Diagnóstico y respaldo](PCIE_FRAGMENT_HOST.md).

**Auditoría interna del ruteo, 01:15 UTC del 2026-10-05:** `011154Z`
clasifica los 268 alias PARTIAL como los dos relojes parentales OOC.
La reducción CDC 23→20 corresponde a tres cadenas sin cargas finales;
las 235 celdas ASYNC_REG retenidas conservan sus redes D/Q. Respaldo
local/NFS verificado. Replay funcional contra MIG descrito arriba,
sin SDF. [Inventario y límites](PCIE_FRAGMENT_HOST.md).

**Netlist de fragmentos aprobada; ruteo exploratorio en revisión, 01:11 UTC:**
`005916Z` reproduce 7.283 eventos y tiempos del RTL y detecta 343 alteraciones
de evidencia. `010456Z` completa 8.949 redes enrutables, con WNS interno
0,647 ns y cero errores de ruteo. La auditoría anterior clasifica las dos
discontinuidades de reloj y la diferencia CDC 23→20. Los dos respaldos
local/NFS están verificados. Puertos, clocks parentales y timing de placa
siguen pendientes. [Métricas y procedencia](PCIE_FRAGMENT_HOST.md).

**Síntesis de fragmentos con doble CDC, 00:58 UTC del 2026-10-05:**
`005058Z` revisada y respaldada local/NFS: 3.251 LUT, 6.277 FF, sin
BRAM/URAM/DSP ni latches. WNS interno sin colocar 1,972 ns; 844 entradas
y 1.289 salidas aún sin presupuesto temporal. CDC: 23 cruces de un bit
y diez resets sincronizados, Info con restricciones AMD activas. Los 870
avisos de puertos opcionales se inventariaron completos; no se añadieron
excepciones temporales globales. La tanda `005916Z` está ejecutando la
netlist contra MIG frío, ya aprobada arriba. [Alcance y respaldo](PCIE_FRAGMENT_HOST.md).

**Fragmentos con doble CDC y MIG frío, 00:49 UTC del 2026-10-05:**
`003714Z` y `003749Z` R2 aprueban host de 4 y 7 ns con UI de 5 ns.
Cada banco valida 15 reservas, 35 tramas (28 aceptadas, siete rechazadas),
10 escrituras de memoria, 75 lecturas de retorno y 695 escrituras/147
lecturas de control. El revisor detecta 343 alteraciones de evidencia por
perfil; los 123 inputs AMD compilados y los 35 archivos del modelo siguen
intactos. Todos los avisos DDR tienen correspondencia causal con tráfico
periódico y retornos drenados. Ambos respaldos local/NFS están verificados.
Los dos intentos iniciales conservan `NOT_QUALIFIED_WRONG_TCL`: habían
seleccionado el script DWORD. [Evidencia, métricas y límites](PCIE_FRAGMENT_HOST.md).
Síntesis y replay revisados arriba; quedan integración XDMA y cierre físico.

**Control de fragmentos NTRF, 00:16 UTC del 2026-10-05:** `001053Z`
aprueba 12 escenarios y tres semillas en RTL/netlist: 128 escrituras,
52 lecturas y tres ARM/retiros por ejecución. Nueve mutaciones RTL y
58 alteraciones de evidencia detectadas. Identifica el perfil y congela
los contadores de rechazo mediante una captura atómica. Síntesis: 839 LUT,
892 FF; WNS interno estimado 2,807 ns a 5 ns, sin cierre físico. Respaldo
local/NFS verificado.
[Registros, evidencia y alcance](PCIE_FRAGMENT_CONTROL.md).

**Composición de fragmentos, 00:24 UTC del 2026-10-05:** `001706Z` aprueba
control NTRF, ensamblador, receptor y guardián de capacidad, con tres semillas
RTL/netlist idénticas. Por ejecución: 21 reservas/resultados, 46 tramas
(39 aceptadas, siete rechazadas), 15 escrituras de 64 bytes, 976 operaciones
de escritura y 197 de lectura del control. Tres mutaciones RTL y 265
alteraciones de evidencia detectadas. Síntesis: 2.591 LUT, 4.067 FF; WNS
interno estimado 2,012 ns a 5 ns. Respaldo local/NFS verificado. Memoria y
barrera siguen siendo BFMs; quedan CDC AMD, MIG frío y XDMA en esta variante.

**Captura y ensamblado de fragmentos, 23:45 UTC:** `233833Z` aprueba seis
ejecuciones RTL/netlist de 20 escenarios, diez mutaciones RTL y 52 alteraciones
de evidencia. Por ejecución: 58 AW/B, 320 W, 23 rechazos, 16 líneas y 20
resultados. Síntesis: 897 LUT y 1.600 FF; WNS interno estimado 1,960 ns a 5 ns,
con entradas/salidas sin presupuesto y sin cierre físico. Respaldo local/NFS
verificado. [Contrato y límites](PCIE_FRAGMENT_INGRESS.md).

**Integración nativa XDMA, 23:56 UTC:** `234908Z` también aprueba el mismo RTL:
cuatro reservas, 16 escrituras, nueve aceptaciones, siete rechazos, cuatro
líneas/resultados y 12 lecturas. Se revisan 95 señales, 76 inputs compilados
y 173 alteraciones de evidencia. Hay contrapresión en AR, LINE y RESULT.
Conserva como BFMs el control, la barrera y el destino; no cierra BAR0/CDC/MIG.
No modifica el ingreso DWORD anterior.
Ambas tandas tienen respaldo local/NFS verificado; el de integración termina
de verificarse el 2026-10-05 a las 00:00:05 UTC, SHA-256
`362eeb54e9950b425c52718598c2e4d1d61439cbbfaf49895d3388f6bd83e9ec`.

**Mapeo nativo NTRPUF01, 23:26 UTC:** `231900Z` aprueba 16 escrituras de
64 B por XDMA y el decodificador de producción: nueve aceptadas, siete
rechazadas y 12 lecturas correctas. Se comprueban 61 señales, 75 inputs,
134 archivos AMD intactos y 128 alteraciones de evidencia. Las cuatro líneas
ensambladas pertenecen a memoria y gestión de reservas del banco; no cierran
el ensamblador de producción, DDR ni el drenaje anterior a ARM. Hay stalls
AW/AR, ninguno W/RQ/RC/B/R. [Evidencia y límites](PCIE_INGRESS_FRAGMENTS.md).

El primer intento `230609Z` falló antes del decodificador porque el generador
público y el registrador heredados no admitían este paquete de varios beats.
Queda respaldado como fallo; R2 introduce un productor de prueba de dos beats
de 512 bits y registro atómico, sin cambiar la IP ni el decodificador. El
primer staging de R2 se rechazó antes de lanzar por cambios concurrentes
del trabajo dual-MIG; se conservaron sus fuentes y procesos. Ambos respaldos
local/NFS están verificados. R2: 844 archivos, 107.367.835 bytes, SHA-256
`c0af15aee4e9d57fd960f5a87ee90af85698ab586aea5840725a4be797b945e7`.

**Control PCIe nativo, 22:57 UTC:** `223702Z` aprueba BAR0 de control y
BAR2 de datos a través de los dos CDC AMD hasta MIG frío. Cuatro reservas,
78 escrituras de control (16 rechazadas), 52 lecturas más una consulta
de enumeración, 83 DWORDs, tres escrituras DDR/192 B y ocho lecturas/512 B.
Se revisan 180 señales, 206 inputs compilados y 200 corrupciones de evidencia;
6.624 avisos DDR quedan correlacionados y 79 lecturas periódicas drenadas.
Duración: 902,600 s; final simulado: 113,118498 µs.
[Detalles y límites](PCIE_INGRESS_CONTROL.md#control-pcie-nativo-y-mig-frío).

El primer intento `221559Z` se conserva como fallo: su aserción final esperaba
52 AR/R y omitía la consulta de BAR0 que hace AMD al enumerar. R2 cambia ese
total a 53 y adelanta una declaración; DUT, estímulos, IP y modelos intactos.
Los 1.630 eventos y 11.959 estados UI muestreados coinciden exactamente.
Ambos tienen respaldo local/NFS verificado: diagnóstico
`1b53854d58933338e00e44a4d98bf8c9ec8138fd45929da1bab3a8d6affbbd69`;
R2 `0249c94facf247fc957f544bc892baffb6e1e4f34559e3db0830f9d13dbf4ebd`.

**Fragmentos con identidad:** `224226Z` aprueba el decodificador experimental
[NTRPUF01](PCIE_INGRESS_FRAGMENTS.md), con 1.176 casos por RTL/netlist y seis
defectos RTL detectados. Usa 678 LUT, cero FF; retardo combinacional máximo
sin colocar 1,206 ns bajo presupuesto de 5 ns. Duración: 169,150 s. Respaldo
verificado: `0da03262982919bc60b891c8c8ef21e2b1760c2bf9b8a769b46eae4a6e6e5c89`.
Es un decodificador aislado: falta comprobar TLP de 64 bytes contra XDMA,
captura/ensamblado y rechazo de fragmentos antiguos sin alterar la reserva
nueva. El ingreso DWORD sigue exigiendo la barrera confiable del banco.
Las tres tandas están terminadas y respaldadas; **504 respaldos verificados**
en esta sesión. No cierran drenaje real, reset con tráfico, DIMM exacto,
MSI, V4/V7/V8 completos ni la validación física.

**Control y datos con doble CDC, 21:42 UTC:** `212852Z` y `213121Z`
aprueban host/UI a 4/5 y 7/5 ns contra el MIG genérico frío. Cada tanda
completa 13 trabajos, 194 escrituras AXI4, nueve escrituras DDR/576 B y
63 lecturas/4.032 B; 588 escrituras y 70 lecturas de control, 78 comandos
rechazados. Se explican 432/720 avisos del modelo y se drenan 80/106
lecturas periódicas; 179 corrupciones de evidencia rechazadas por tanda.
Ambas tienen respaldo local/NFS verificado. La revisión de AXI-Lite se
corrigió contra el RTL AMD: sus handshakes miden confirmación de recepción,
no llegada de payload como en un FIFO. DUT, estímulos y logs intactos.
[Resultados, hashes y límites](PCIE_INGRESS_CONTROL.md#integración-host-y-control-cdc-contra-mig).
Síntesis integrada `214501Z` revisada y respaldada: 2.946 LUT, 6.117 FF,
682 LUTRAM; setup/hold interno sin colocar 1,747/0,013 ns. Los 15 grupos
de skew conservan margen mínimo 3,851 ns. Faltan retardos de 844 entradas
y 1.215 salidas. `215325Z` amplía solo el listado de avisos: 870 puertos
opcionales AMD sin uso, manteniendo inputs, recursos, timing y lógica visible.
Ambas tandas tienen respaldo local/NFS verificado; el bloque cifrado cambia
de hash y no se declara equivalencia binaria completa.

La netlist **original** `215351Z` aprueba contra MIG: 7.401 eventos idénticos
a RTL, 123 inputs compilados sin sustitución por RTL de producción y 179
corrupciones de evidencia rechazadas. Conserva 432 avisos DDR explicados y
80 lecturas periódicas drenadas; dura 415,786 s. Respaldo local/NFS verificado
a las 22:03:58 UTC: `521dea3b02dabef9b57ddcbac1585b671db17431863c7b508998cf23099948c6`.
Las cinco tandas nuevas están terminadas y respaldadas; 501 respaldos de
ejecuciones en esta sesión. El intento previo del replay se detuvo antes
de crear una ejecución por discrepancia del inventario de staging; se
conserva su diagnóstico. R2 vuelve a copiar sin cambiar DUT, banco ni netlist.
Siguen pendientes control PCIe nativo, drenaje real anterior al rearme y
calificación física; no se cierran V4/V7/V8 por estas pruebas aisladas.

**Control de ingreso, 21:07 UTC:** `20261004T210107Z-V7-ffn-supervised-ingress-control`
aprueba el bloque AXI-Lite aislado: tres semillas RTL y tres de netlist con
eventos/ciclos idénticos; 116 escrituras, 25 lecturas y tres reservas por replay.
Detecta cinco defectos RTL deliberados y 22 corrupciones de evidencia. Síntesis
VU5P: 821 LUT, 816 FF, sin RAM/DSP; margen interno sin colocar 2,759 ns setup
y 0,066 ns hold con reloj de 5 ns. Faltan retardos para 443 entradas y 302
salidas; los avisos de jerarquía plana, clock OOC y DRC parcial están revisados.
Respaldo local/NFS verificado: SHA-256
`4324d3e44808b0ff81ecaae63efe9d7e39fd95f5fee8aa5993721e0dd046bcae`.
El [contrato de registros](PCIE_INGRESS_CONTROL.md) conserva la barrera externa
como requisito: esta prueba no acredita drenaje PCIe/CDC. La composición con
el camino protegido se ensayó en tandas nuevas, preservando el intento fallido.

**Composición de control e ingreso:** `20261004T211253Z-V7-ffn-supervised-ingress-managed-r2`
aprueba tres semillas RTL y tres de netlist con coincidencia exacta de eventos
y ciclos. Por replay: 21 reservas, 309 escrituras AXI de entrada, 14 escrituras
de memoria (896 B), 954 escrituras y 105 lecturas de control; 128 comandos
incorrectos rechazados. Tres defectos RTL y 33 corrupciones de evidencia
detectados. Síntesis: 2.317 LUT, 3.907 FF, 298 LUTRAM; setup/hold internos
sin colocar 1,757/0,066 ns. Faltan retardos de 843 entradas y 1.215 salidas.
El respaldo verificado tiene SHA-256
`f5b135a22c0360e959eb5e60f61b0a92b163366e3efe45035dd55789cbde3647`.
La tanda previa `210950Z` se conserva como fallo de expectativa del script:
el banco detectó retirada prematura mediante `MONITOR_RESULT_STABLE`, antes
del mensaje esperado. R2 cambia solo esa expectativa, sin modificar RTL ni
estímulos. Respaldo previo: `5521cf00872394b6395636ba7c5bed6a2216a82845725a35a646b6a3a4c4d49f`.
Son pruebas con memoria y barrera modeladas, en un reloj. Las tandas posteriores
indicadas arriba añaden CDC y MIG; siguen pendientes control PCIe nativo y
demostración del drenaje real antes de reutilizar ventanas. No cierran
V4/V7/V8 ni el timing de placa.

**Corrección de alcance, 20:03 UTC:** los resúmenes `191958Z`, `193215Z` y
`193725Z` omitieron avisos `Model:WARNING:` al contar solo líneas `WARNING:`.
Son 144/144/432 avisos de lectura sin escritura previa. Sus comparaciones de
datos y respaldos siguen siendo válidos; la calificación DDR completa queda
pendiente de correlación interna y drenaje, conforme a [PCIE_DDR_FFN.md](PCIE_DDR_FFN.md).
Los archivos originales se conservan y la corrección se registra por separado.

**Actualización, 20:26 UTC:** `200944Z`, `201023Z` y `201100Z` pasan las
repeticiones MIG/CDC con correlación de los 144/144/432 avisos y drenaje de
8/17/29 solicitudes periódicas. Respaldos local/NFS verificados; 110/137/137
alteraciones de evidencia detectadas. `200726Z` pasa la cadena **PCIe nativo →
CDC → ingreso/filtro → MIG**: 83 MemWr, tres escrituras/192 bytes y ocho
lecturas/512 bytes correctos. Sus 6.048 avisos corresponden a lecturas internas;
43 solicitudes drenadas y 122 alteraciones detectadas. Respaldo local/NFS
verificado a las 20:26:20 UTC. [Evidencia y límites](PCIE_INGRESS_PLAN.md#repeticiones-con-observación-periódica-y-drenaje).

`195116Z` se conserva respaldado como comparación funcional parcial, sin
calificación causal DDR. Los resultados nuevos no reescriben ese diagnóstico.
Síntesis `202525Z`, ruteo `203200Z` y auditoría `204009Z` concluidos y respaldados:
**2.055 LUT/4.911 registros tras ruteo OOC**, WNS interno 0,856 ns, WHS 0,011 ns,
bus skew mínimo 3,174 ns. Los 229 alias PARTIAL corresponden a los dos relojes
externos; puertos y timing de la tarjeta siguen pendientes. La netlist `203709Z`
pasa contra MIG con 2.420 eventos idénticos a RTL y 137 alteraciones detectadas;
su respaldo quedó verificado a las 20:47:08 UTC. El fallo Tcl previo `203417Z` se conserva
respaldado. [Implementación y replay](PCIE_INGRESS_PLAN.md#implementación-exploratoria-del-ingreso-asíncrono).
Los trabajos externos permanecen separados e intactos.

Ingreso con filtro `190452Z`: **PASS RTL y netlist**, 21 escenarios y tres
semillas; cinco defectos RTL y 60 alteraciones de evidencia detectados.
1.531 LUT/3.091 FF, sin cierre físico de timing. MIG frío `191958Z`:
**PASS**, 13 trabajos, 194 escrituras fuente, nueve escrituras de memoria y
63 lecturas/4.032 bytes comparados en seis ubicaciones hasta la última línea.
Rechazos por capacidad, stop, duplicado conflictivo y recuperación verificados.
Respaldo de ambas tandas y del intento de compilación fallido `191519Z`
verificado local/NFS. [Evidencia y límites](PCIE_INGRESS_PLAN.md#mig-frío-y-lectura-de-retorno).
`193215Z` y `193725Z`: **PASS funcional del transporte asíncrono**,
host/UI a 4/5 y 7/5 ns. Cada tanda conserva las 194 transacciones, 13 trabajos
y 63 lecturas; 85 alteraciones de evidencia detectadas. Ambas respaldadas.
Los FIFO AMD no modelan retardos de sincronización: no hay cierre físico CDC.
[Resultados y siguiente integración nativa](PCIE_INGRESS_PLAN.md#transporte-asíncrono-dos-perfiles).

Puente ingreso→receptor `183931Z`: **PASS RTL y netlist**, 17 escenarios por
semilla, tres semillas, 245 escrituras fuente y 13 líneas destino por ejecución;
seis defectos RTL y 60 alteraciones de evidencia detectados. Stop y CPL temprana
retienen la reserva hasta B ID 0; rearme conjunto y ausencia de B ejercitados.
1.409 LUT/3.041 FF, tiempos internos sin colocar y E/S sin constraints completos.
Respaldo D/NFS verificado a las 18:46 UTC, SHA-256
`402f059d1a7beb45b6c16bfb370ef6074eb00821f6de73a49e5f8b82fd1ca5f6`.
[Contrato y límites](PCIE_INGRESS_PLAN.md#puente-al-receptor-rtl-y-netlist-validados).
La integración PCIe nativa `184808Z` pasa: 105 señales, 83 MemWr y tres
escrituras AXI completas, conflicto rechazado y recuperación sin reset.
48 alteraciones de evidencia detectadas; 78 entradas compiladas y 134 archivos
AMD intactos. Conserva presión AW/W de destino y drenaje antes de done/rearme.
Duración 424,154 s; respaldo local/NFS verificado: 846 archivos,
106.416.857 bytes, SHA-256
`534ad487c391f7f8bcb0c22eb6818a963e8310eef87c7d572eb2598b753c64e7`.
El destino AXI y el aislamiento son BFMs.

**Revisión vigente, 19:44 UTC:** el filtro AXI 64→37 bits ya pasó banco RTL,
síntesis, replay funcional de netlist e integración fría FFN+MIG en ambos
relojes host. Hay **483 tandas con respaldo local/NFS verificado**. Las ocho
tandas anteriores incluyen dos bancos preliminares y un fallo de Tcl conservados
con su alcance original. [Métricas y contratos del filtro](DDR_AXI_GUARD.md).

`144107Z` terminó con salida 0 en 2.074,383 s y pasa la revisión estricta:
siete casos DMA, dos DECERR sin acceder a MIG, lecturas de control exactas
y recuperación posterior. Se rechazaron 163 alteraciones y se verificaron
266 entradas compiladas; conserva la FFN y los 8.208 avisos correlacionados.
El cierre drena 233 lecturas periódicas y 68 de usuario. Su respaldo de
1.652 archivos/281.227.255 bytes quedó verificado a las 15:27:44 UTC,
SHA-256 `aeb1f4f0c84b9765655275c9dd197179c3b72c83a95bfd406fdfe56573d8f67a`.
El siguiente
[banco de cruce de capacidad](PCIE_DDR_ERROR_PLAN.md#siguiente-dependencia-descriptor-que-cruza-128-gib)
ya tiene estímulos, observadores y revisión independiente preparados.
Sus once descriptores pasan en la tanda `152800Z`: 1.293,109 s, 193
alteraciones rechazadas y drenaje de 117 retornos periódicos. El cruce H2C
escribe 64 bytes antes del DECERR; el cruce C2H publica 64 bytes antes del
error. Cero conserva su patrón, sin alias, y ambos motores se recuperan.
No hay atomicidad de descriptor. Respaldo verificado a las 15:55:15 UTC:
1.644 archivos, 280.862.653 bytes, SHA-256
`9f65b38480a1d0044b6b1e82bfb8d16ee092f24436b5d65a12c4760411fbea00`.
No modifica el RTL ni las IP.

La [variante con BAR bypass](PCIE_BYPASS.md) `154440Z` genera y verifica
BAR2/3 de 64 bits/1 MiB, conservando BAR0/1; dirección AXI de 64 bits y
traducción por encima de 4 GiB. Son 90,572 s y 14 alteraciones rechazadas.
Está respaldada; su alcance es generación de configuración.
El banco directo conserva cuatro intentos fallidos de preparación,
compilación y elaboración. `161130Z` terminó por la referencia a la señal
implícita `m_axib_rlast` del ejemplo. Está respaldado; `161947Z` recuperó
las tres palabras y también está respaldado. La repetición pasiva
`164020Z` pasa la revisión nativa de RQ/RC y AXI en 185,566 s, con 94
alteraciones rechazadas y 375 archivos del ejecutable idénticos. Comprueba
12 bytes por dirección, traducción 36 GiB→4 GiB, etiquetas asignadas por
el núcleo y `Byte Count=4`. Respaldo local/NFS verificado a las 16:50:22 UTC,
398 archivos/64.240.708 bytes, SHA-256
`15791566ca2a7fa7db1fb74c02303134dadbef6bd91c1cb23c756fd3a032c06f`.
Es un banco de tres DWORDs con memoria comportamental; no califica DDR,
P2P, reset, caudal ni ECRC físico. [Alcance y próxima tanda](PCIE_BYPASS.md).
`165525Z` terminó por una expectativa incorrecta del banco: para máscara
`0xa`, XDMA dirige AW al primer byte habilitado (`…003d`), no al inicio
del DWORD (`…003c`). Salida Vivado 0 con fatal XSIM; se conserva como fallo.
El respaldo ya está verificado. La revisión `170643Z` pasa las nueve máscaras
y lecturas: 24 bytes escritos y 36 leídos, centinelas intactos; 447,805 s.
La revisión nativa rechaza 61 alteraciones de evidencia y conserva 139 avisos
de compilación más uno de tiempo cero. Respaldo verificado a las 17:20:55 UTC:
841 archivos, 107.508.791 bytes, SHA-256
`d409ddafa67e85cc3afd5bea11bb5b281f1741abb3e4f269801c59139e30df1a`.
`172307Z` pasa el primer decodificador combinacional de DWORDs para
[ingreso por bytes](PCIE_INGRESS_PLAN.md): 9.609 casos en RTL y otros tantos
en netlist, cinco defectos introducidos detectados y 159 LUT/9 CARRY8/0 FF.
No implementa todavía captura AW/W ni ensamblado. Tanda de 156,818 s;
respaldo verificado a las 17:28:06 UTC, 242 archivos/1.463.665 bytes,
SHA-256 `fa46b3f86acc7e2f782f947f59a0827b711dfb2bb03e1de573d15091e75dab8d`.

El [ingreso de una línea](PCIE_INGRESS_PLAN.md#ensamblador-y-captura-axi-de-una-línea)
ya captura AW/W, ensambla bytes y retiene reserva/finalización. `174126Z`
conserva un fallo de estímulo; `174359Z` pasa RTL/síntesis pero falla al
elaborar netlist sin `glbl`. `175152Z` repite la netlist exacta con esa raíz:
tres semillas coinciden ciclo por ciclo con RTL, 19 escenarios cada una,
seis mutaciones RTL y 40 alteraciones de evidencia rechazadas. Son 658 LUT
y 1.515 FF; falta timing de interfaz y ruteo. Respaldo final 17:55:28 UTC,
57 archivos/1.985.399 bytes, SHA-256
`3732934ef678003472b27dc585a274f238fed7852d2e7ae55521aacebd835d5e`.
`180851Z` conecta ese RTL al bypass XDMA y pasa la revisión nativa de
81 señales: 83 MemWr, 83 AW/W/B y tres líneas completas, incluidas máscaras
y último DWORD del BAR. El conflicto devuelve SLVERR y hay recuperación
posterior sin reset. Se rechazan 37 alteraciones; RQ→B mide 192–196 ns en
el modelo. Tanda de 422,185 s, sin cambios en el RTL previamente calificado
ni en 134 archivos AMD. Respaldo verificado 18:19:56 UTC, 843 archivos,
106.322.837 bytes, SHA-256
`a91f3c8ca3f7cfc37a093bc59ab884a742d974c48cd2cbe3e0d6a9cfd93b670f`.
El destino y la barrera de aislamiento todavía los proporciona el banco.
El puente al receptor ya pasa su banco aislado; sigue la conexión a MIG,
con épocas y finalización coordinadas;
estas pruebas no equivalen a DDR ni P2P físico.

Los bancos completos RTL/netlist prueban 60 casos cada uno. La integración
MIG añade seis FFN D32/H32, 192 FP32 exactos, 108 transacciones y cuatro
ranks activos, con accesos válidos; los rechazos se califican por separado.
El filtro usa 1.380 LUT y 164 FF. Su timing de interfaz/ruteo y la propiedad
de buffers siguen pendientes.

La tanda posterior de [errores desde el host sobre MIG](DDR_AXI_GUARD_ERRORS.md)
`093814Z`/`093824Z` pasó ambos relojes: 24 descriptores rechazados con DECERR,
42 lecturas de control sin corrupción y seis FFN posteriores con 192 FP32
exactos. Son 180 transacciones host en total; 124 alteraciones de evidencia
y seis de conexiones rechazadas. El control negativo `094552Z` detecta la
eliminación deliberada de los comparadores en el primer acceso inválido.
Las tres tandas están revisadas y respaldadas; el control negativo conserva
su salida 1 como fallo esperado. No cambió el RTL normal.

La conexión en el [top sintetizable](FFN_GUARDED_TOP.md), tandas `100229Z`
y `100239Z`, también pasa y está respaldada. Conserva exactamente 30.497
eventos del banco anterior: seis FFN, 192 FP32, 24 DECERR. Se rechazan 124
alteraciones de trazas y 14 de cableado. La síntesis OOC `101702Z` y auditoría
`103636Z` pasan y están respaldadas: 56.488 LUT/51.484 FF/153 DSP;
WNS +0,173/WHS +0,013 ns estimados, 128 bits de dirección registrados,
51 CLR y 131 INIT de IRQ revisados. Los tres diagnósticos de jerarquía se
conservan, incluidos dos intentos fallidos. El ruteo `104259Z` terminó con
**WNS −0,176 ns, TNS −5,375 ns y 144 extremos de setup**; hold +0,010 ns.
La auditoría `110541Z` confirma 128 bits de dirección, 51 CLR, 131 INIT y
100.722 redes ruteadas, sin errores; el revisor rechaza 20 alteraciones.
Ambas tandas están respaldadas. La optimización `111431Z` y auditoría
`112432Z` pasan: **WNS +0,010/WHS +0,010 ns**, cero extremos con fallo,
56.321 LUT/51.153 FF/153 DSP. Están respaldadas; se rechazan 24 alteraciones.
Los controles RTL públicos `112246Z`/`113749Z` están certificados: 14.498 y
15.599 eventos idénticos a sus referencias. El netlist optimizado `113319Z`
pasa revisión funcional y está respaldado: tres FFN/96 FP32, 36 pruebas,
12 DECERR y 78 alteraciones de evidencia rechazadas. Difieren eventos de
inicialización/reset; no se afirma equivalencia temporal completa ni SDF.
El replay con host lento `120039Z` también pasa y está respaldado: tres
FFN/96 FP32, 36 pruebas, 12 DECERR y 78 alteraciones rechazadas;
sin equivalencia temporal completa. El banco [PCIe + DDR + FFN](PCIE_DDR_FFN.md)
conserva dos fallos respaldados de preparación/monitor. `123643Z` terminó
y está respaldada: transferencias exactas, 32 FP32, 2.429 ciclos UI y
41 alteraciones rechazadas. Sus avisos de modelo impiden la aprobación
completa. `130150Z` está respaldada: los 8.208 avisos quedan correlacionados
con lecturas periódicas en el tramo completado, con 43 alteraciones del
testigo rechazadas. El último retorno queda 19.998 ps después de `$finish`;
`133315Z` pasó la revisión estricta: 185 retornos periódicos, 64 lecturas
normales y 94 alteraciones rechazadas. Su respaldo local/NFS está verificado;
`140242Z` está respaldada como banco incompleto: interpretaba `idle_stopped`
como otro error al recibir `0x4040`. Se observa DECERR H2C sin acceso MIG,
pero la recuperación sigue pendiente. `144107Z` ejecuta la corrección del
banco, conservando la FFN completa y el RTL/IP original.
Permanecen tres huecos de relojes padre y 1.402 entradas/1.291 salidas sin
retardos externos; no hay cierre físico ni integración de placa acreditados.

Ambos perfiles FFN `060223Z` pasan y están
respaldados: 144 casos, 18 resets RELEASE, 24 pérdidas y 2.496 FP32 exactos.
La síntesis `061341Z` y auditoría `062607Z` de la corrección CDC están
revisadas y respaldadas: 51 CLR, 55.362 LUT/51.313 FF/153 DSP, estimación
WNS +0,173/WHS +0,013 ns y 12 alteraciones de evidencia rechazadas. El
ruteo `064105Z` terminó y falla setup: **WNS −0,235 ns, 103 extremos**,
WHS +0,010 ns. El ruteo y sus cuatro auditorías están respaldados; las
51 etapas conservan polaridad, INIT y conexiones. La optimización posterior
`072337Z` y sus cuatro auditorías pasan la revisión independiente y están
respaldadas: **WNS +0,023/WHS +0,010 ns**, sin extremos fallidos, con
54.795 LUT/50.985 FF/153 DSP. Conserva todos los comandos XDC y rechaza
22 alteraciones de evidencia conjunta. Es timing interno OOC; fuentes de
pares, relojes padre y retardos de placa siguen pendientes.
**El ruteo histórico +0,062 ns no califica el RTL actual.**

Los bancos DMA `062528Z` y `064115Z` pasan en ambos perfiles de reloj:
60 fallos, 2.100 transacciones, 60 FFN sintéticas, 1.920 FP32 exactos y
198 alteraciones rechazadas. Los fallos ocurren con cómputo en reposo;
cada recuperación incluye recarga AXI, FFN completa y lectura del resultado.
Ambos respaldos están verificados. Los bancos concurrentes `065918Z` y
`070826Z` añaden **96 interrupciones DMA+FFN, 96 FFN completas posteriores,
3.072 FP32 exactos y 186 alteraciones rechazadas**, también respaldados.
Son lecturas DMA con backpressure legal durante las cuatro fases de cómputo.
La cola de optimización y sus auditorías terminó. Los fallos anteriores
conservan sus diagnósticos.
Detalle y límites: [FFN supervisada](../SUPERVISED_FFN.md).

El diagnóstico DDR `075538Z` también está revisado y respaldado: ocho
combinaciones de fase/CKE conservan ocho errores por caso con fila abierta,
y ninguno sin fila abierta; 96 alteraciones de evidencia rechazadas.
La fase y el orden temporal de CKE no resuelven el fallo. V3 sigue abierto;
ver [reproducción y límites](MEMORY_QUALIFICATION.md#fase-de-reset-y-separación-de-cke).
El diagnóstico `142430Z` invierte el orden: RESET antes de CKE, con retrasos
de 12,5 ns y 2 µs. Conserva ocho errores por caso con fila abierta y cero
en los controles. Las cuatro ventanas, 50 alteraciones rechazadas y los
35 archivos originales del modelo están revisados y respaldados; tampoco
se observa una segunda inicialización. El reset completo sigue FAIL.

La integración fría de FFN supervisada con MIG `082658Z` pasa y está
respaldada: host de 4 ns, tres FFN D32/H32, **96 FP32 exactos**, 54
transacciones, 9.984 B de entrada y 384 B recuperados desde DDR. Las
activaciones físicas cubren cuatro ranks; el revisor rechaza 36 alteraciones.
El perfil de host de 6,2 ns `083424Z` también pasa y está respaldado:
entre ambos, **seis FFN, 192 FP32, 108 transacciones y 72 alteraciones
rechazadas**. Los dos intentos anteriores están respaldados como fallos
del banco. Este avance no cierra V2/V3 ni el reset DDR con tráfico:
ver [conexión, métricas y límites](DDR_FFN_COLD.md).

| ID | Estado | Evidencia y siguiente paso |
| --- | --- | --- |
| V0 | aprobado | Volumen persistente confirmado; Vivado 2026.1 SW 6511674/IP 6504888; Enterprise válida hasta 2026-12-01. Síntesis mínima VU5P correcta. Cuatro hilos y 600 s por exploración; replay PCIe/capacidad DDR con 1800 s y ECC con 900 s explícitos. |
| V1 | aprobado | `xcvu5p-flva2104-2-i` disponible; 2104 bolas y 832 E/S HP coinciden con AMD. DDR0/2 en SLR0, DDR1/3 en SLR1. Inventario, no aprobación de placa. |
| V2 | bloqueado | Parte Micron ausente del catálogo; faltan SPD/modelo y parámetros exactos. Perfil genérico de 128 GiB+ECC generado a 1600 MT/s. [Alternativa 3DS](MEMORY_3DS_ALTERNATIVE.md): catálogo y geometría 128 GiB revisados, pero el modelo Samsung no admite XSIM. |
| V3 | bloqueado | Depende de V2 para el DIMM exacto. Capacidad dispersa/ECC genéricos PASS; reset con tráfico FAIL. Canal genérico ruteado en 64–67 con DCI y cumple restricciones actuales; faltan contratos/retardos de placa. |
| V4 | en_curso | DMA 768 bytes PASS serial/PIPE; 1536 bytes PASS con host64/cadenas/páginas/pausas en AW/W/B/AR/R. MSI-X y BAR ABI2 pasan la integración pequeña V7. DDR+PCIe: 221 bolas, BIVC/DCI=0 y ruteo completo; WNS +0,006 ns, WHS +0,004 ns. TIMING-16 acotado a multicycle heredado; faltan revisión CDC/constraints y circuito PERST. [Evidencia](INTERFACE_COEXISTENCE.md). |
| V5–V6 | pendiente | Mantener dependencias; resolver memoria y reset antes de implementar el conjunto. |
| V7 | en_curso | FFN supervisada, filtros de capacidad y caminos PCIe/DDR tienen evidencia por bloque: [FFN](../SUPERVISED_FFN.md), [integración PCIe/DDR](PCIE_DDR_FFN.md). El transporte NTRD retenido pasa el banco [PCIe nativo a dos MIG](PCIE_DUAL_NATIVE_PLAN.md). Su [netlist OOC ruteada](PCIE_DUAL_HOST_SYNTH.md) pasa el replay frío funcional `090132Z`: 18 trabajos, 16 escrituras y 28 lecturas comparadas, sin SDF. Pendientes: pérdida del host/respuestas abandonadas, reset DDR, integración de cuatro canales con cómputo, restricciones físicas y pesos Meta/inferencia completa. Los resultados anteriores siguen conservados en el historial y sus respaldos. |
| V8 | pendiente | Requiere V6 y V7 cerrados; no hay diseño integrado ni entrega ECAD aprobada. |
| V9 | pendiente | Requiere hardware. |

La parte `-2-i` es provisional: velocidad intermedia y variante industrial. No fija BOM, coste, disponibilidad ni margen térmico/timing. `io_plan.json` conserva su carácter preliminar.

## Historial: FFN supervisada, ruteo y pérdida de reloj

La [FFN supervisada](../SUPERVISED_FFN.md) pasa 126 casos a dos relaciones
de reloj, con 30 recuperaciones, 24 cancelaciones y 1.344 valores FP32
exactos. El revisor reconstruye los handshakes B/R y rechaza 174
alteraciones de evidencia. Se preservó el primer fallo del monitor en
el perfil lento y se corrigió el registro atómico de sus respuestas.
Los seis intentos, incluidos regresión y negativo, están respaldados.

El ruteo exploratorio `20261004T034418Z` y sus auditorías están revisados
y respaldados: **54.820 LUT/50.985 FF/153 DSP**, WNS **+0,062 ns** y WHS
**+0,010 ns**. Sus 98.664 redes ruteables están completas; los tres huecos
padre corresponden a relojes OOC. RTSTAT-10 clasifica 1.332 salidas sin carga
externa y 45 señales AMD sin uso; conserva el aviso. Se rechazaron 18
alteraciones de evidencia de ruteo y ocho de conectividad. Los cinco
CDC-26 externos, CDC-15 y restricciones de placa siguen abiertos.

Los bancos `20261004T035245Z` y `20261004T040110Z` añaden **126 casos con
24 pérdidas automáticas de reloj**, seis arranques y 216 alteraciones
rechazadas. Comparan 1.344 FP32 exactos y 33.826 handshakes B/R. El bloqueo
se observa a 640,699–689,999 ns y la recuperación a 594,999–704,999 ns;
son intervalos RTL. Cada perfil prueba una FFN correcta tras la última
pérdida. Falta exigirla tras cada pérdida individual, reset durante RELEASE
y DMA interrumpido. Los pares de memoria/host son modelos; V2/V3 y la
recuperación con DDR siguen abiertos. Las siete ejecuciones nuevas,
incluido el primer diagnóstico Tcl fallido, tienen respaldo verificado.

La ampliación `20261004T044322Z/044332Z` pasa **144 casos**, con FFN completa
tras cada una de las **24 pérdidas**, **2.496 FP32 exactos** y **240
alteraciones rechazadas**. Ambos perfiles están respaldados. Se conserva
el primer intento rápido como no calificado: aceptaba W antes de AW y
detenía el reloj antes de la escritura física exigida. El disparador del
banco fue corregido sin modificar RTL ni relajar el revisor. La cola
arrancó `20261004T045439Z-V7-ffn-supervised-release-retry` y su perfil lento
tras verificar esas dependencias; ambos fallaron por estabilidad CDC durante
RELEASE. El resultado y la corrección actual se distinguen arriba.

## Tanda anterior: supervisor conectado al transporte IRQ

La [isla IRQ supervisada](../SUPERVISED_IRQ.md) pasa 63 casos con
confirmaciones reales de drenaje/disponibilidad, 42 recuperaciones y
18 pérdidas automáticas de reloj. El verificador reconstruye 105 eventos,
66 retiros y 39 abortos; rechaza 63 alteraciones de logs. La prueba negativa
detecta una retirada prematura de IRQ. La regresión anterior conserva
exactamente sus trazas en 45 casos.

Síntesis y auditoría: 149 LUT/597 FF, cero DSP/RAM; WNS estimado
+2,467/WHS +0,077 ns. Se conserva CDC-26 externo y faltan los retardos de
279 fronteras. Los nueve intentos de `20261004T021425Z` a `20261004T023621Z`
están respaldados local/NFS, incluidos compilación fallida y cuatro errores
XPM S-1 previos a la corrección. No cambian los bloqueos V2/V3 ni el margen
OOC de 3 ps de la FFN. La tanda posterior integra AXI/FFN con pares
conductuales; esta isla IRQ no cierra la recuperación del sistema.

## Tandas y respaldos

Los archivos se guardan primero en el volumen persistente y después en `artifacts/ntrpu-board-vivado/<run_id>/`. Cada carpeta local incluye el comprimido, fuentes, `VERIFIED.json` con SHA-256 y verificación de todos los miembros, y el directorio extraído con `result.md`, `FILES.json`, comandos, tiempos y logs.

**Almacenamiento local:** C: se llenó durante el respaldo. Tres copias extraídas
ya verificadas se trasladaron a `D:\AI\ntransformer\artifacts\ntrpu-board-vivado\extracted\`;
sus rutas originales siguen funcionando mediante junctions. Los respaldos nuevos
se alojan en `D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`, también
enlazados desde `artifacts/`. Se conservan archivos originales y hashes. Las copias
interrumpidas se reanudan solo después de verificar el hash del prefijo recibido.
El sellado ahora rechaza documentos `result.md` vacíos o truncados.

| Run ID | Salida Vivado | Alcance |
| --- | ---: | --- |
| `20261002T085802Z-V0-V1-discover` | 0; 7,757 s | Siete partes VU5P y siete VU9P instaladas. |
| `20261002T085948Z-V1-inventory` | 0; 28,779 s | Parte, bolas, bancos, SLR e IP reales. |
| `20261002T090258Z-V2-ip-probe` | 0; 61,737 s | Propiedades/XCI; licencias Included; síntesis de un FDRE. |
| `20261002T091135Z-V2-exact-dimm` | 1; 32,231 s | Rechazos de parte, selector DM y reloj; conservados. |
| `20261002T091508Z-V2-exact-dimm-r2` | 1; 32,807 s | Rechazo aislado de la parte exacta. |
| `20261002T091518Z-V4-pcie` | 1; 41,442 s | Targets generados; falló una opción del comando de ayuda. |
| `20261002T091826Z-V4-pcie-r2` | 0; 89,987 s | Targets y proyecto de ejemplo PCIe/DMA generados. |
| `20261002T092606Z-V4-bank65` | 1; 37,623 s | BIVC-1 detectado; falló después una opción de consulta DRC. |
| `20261002T093023Z-V4-bank65-r2` | 0; 39,525 s | BIVC-1: un conflicto a 1,8/1,2 V, ninguno a 1,2/1,2 V; mapeo GT y refclock. |
| `20261002T094116Z-V4-dma-sim` | 1; 19,861 s | Opción Tcl de elaboración mal interpretada; corregida en nueva tanda. |
| `20261002T094308Z-V4-dma-sim-r2` | 1; 55,064 s | HDL compila; XELAB rechaza duplicación de `--mt`. |
| `20261002T094531Z-V4-dma-sim-r3` | 124; 600,007 s | Elaboración completa; simulación inicia y agota presupuesto, sin comparación DMA. Snapshot respaldado y reutilizable. |
| `20261002T095945Z-V4-dma-replay` | 124; 600,005 s | Ejecutable reutilizado en copia nueva; no confirma enlace antes del límite. |
| `20261002T100234Z-V2-generic-geometry` | 1; 33,320 s | Dos RDIMM x4+ECC a 833 ps rechazados; requiere 1250–1600 ps para slot/ancho. Incluye fichas Micron y CSV AMD. |
| `20261002T100600Z-V2-generic-1600` | 0; 49,167 s | Targets genéricos a 1250 ps, 37/512 bits AXI, cuatro ranks y 128 GiB útiles. No califica F1. |
| `20261002T101447Z-V4-dma-progress` | 0; 1571,986 s | PASS: 16 bytes H2C y C2H, enlace Gen3 x16, respuestas AXI correctas y monitor independiente. |
| `20261002T102039Z-V2-generic-example` | 0; 82,506 s | Ejemplo DDR generado; banco conserva `_4G` y requiere adaptación antes de validar la geometría de chips de 16 Gb. |
| `20261002T104404Z-V3-generic-capacity` | 0; 215,211 s | PASS genérico: 36 ubicaciones, máscaras ECC, ráfagas, cuatro ranks y 5632 bytes comparados. Conserva avisos del modelo; no califica el Micron. |
| `20261002T110203Z-V3-generic-ecc` | 0; 207,339 s | PASS: corrección de un bit, detección de dos con SLVERR, IRQ, W1C, contador CE y reescrituras limpias. |
| `20261002T110846Z-V3-planner-api` | 1; 7,357 s | Diagnóstico: `redirect` no existe en este Tcl; no se abrió un diseño. |
| `20261002T111134Z-V3-planner-api-r2` | 1; 7,421 s | Diagnóstico: `get_port_interfaces` no tiene ayuda disponible; inventario parcial conservado. |
| `20261002T111254Z-V3-planner-api-r3` | 0; 7,404 s | Ayuda real exportada para `synth_design`, `place_ports`, bancos y byte groups. No es colocación. |
| `20261002T112130Z-V3-generic-pin-synth` | 0; 439,816 s | Síntesis: 30.556 LUT, 34.675 registros, 26,5 BRAM, 3 DSP. Ejemplo usa cuatro bancos; `place_ports` no admite memoria. |
| `20261002T112359Z-V3-generic-reset` | 0; 259,304 s | FAIL de revisión: 558 errores del modelo tras reset, pese a marcas PASS del banco. |
| `20261002T113227Z-V3-generic-reset-r2` | 0; 255,130 s | FAIL: persisten 558 errores con pulso externo de 1 µs; reset del RCD bajo durante 2,597 µs. |
| `20261002T113237Z-V3-bank64-remap` | 0; 49,227 s | 154 puertos a 64–67; AM17 libre, AP17 ocupado por DQ25; 48 DCIRST-1 y cuatro REQP-1877. No implementado. |
| `20261002T114559Z-V3-reset-trace` | 1; 198,145 s | Instrumentación fallida: nombre de top parametrizado; se terminó solo este XSIM inactivo tras comprobar PID/ruta. |
| `20261002T114946Z-V3-reset-trace-r2` | 1; 16,427 s | Consulta de scopes incorrecta; salida inmediata, sin ventana de simulación. |
| `20261002T115124Z-V3-reset-trace-r3` | 0; 115,786 s | Traza 0–55 µs de reset/reloj RCD; 735 flancos positivos por slot durante reset. Diagnóstico, no PASS funcional. |
| `20261002T115825Z-V3-reset-component-trace` | 0; 117,880 s | Reset, CK y CKE observados en las 72 interfaces de chips. Reset conectado y reloj presente; los errores persisten. |
| `20261002T121240Z-V4-dci-unit` | 0; 19,546 s | PASS del secuenciador con primitiva DCIRESET: un pulso de 256 ns y cuatro liberaciones controladas. |
| `20261002T121632Z-V3-dci-integrated-synth` | 1; 425,562 s | Síntesis completa y checkpoint guardado; falla consulta final DRC. Elimina 48 DCIRST-1. |
| `20261002T121909Z-V3-reset-command-trace` | 1; 16,517 s | Propiedad SCOPE no soportada en XSIM; diagnóstico conservado. |
| `20261002T122011Z-V3-reset-command-trace-r2` | 0; 117,916 s | Comandos públicos de un chip y reset/reloj de 72; ventana diagnóstica, sin PASS funcional. |
| `20261002T122611Z-V3-ddr-place` | 1; 274,735 s | opt_design completo; reloj GCIO64/MMCM66 rechazado por colocación 30-675/30-716. |
| `20261002T122810Z-V3-model-reset-isolation` | 0; 35,053 s | Reproduce ocho errores posreset en un chip con fila abierta; sin MIG/RCD/AXI. |
| `20261002T123042Z-V3-model-reset-widths` | 0; 58,836 s | Cuatro casos: ocho errores con fila abierta a 2,6 y 200 µs de reset; cero en los casos sin fila abierta. |
| `20261002T123715Z-V3-ddr-place-backbone` | 0; 491,669 s | Colocación completa con referencia global BACKBONE; WNS +0,553 ns, WHS −0,277 ns antes de ruteo. Timing sin cerrar. |
| `20261002T124830Z-V3-constraint-audit` | 0; 86,092 s | Auditoría sin cambios: 550 endpoints RIU cubiertos por las excepciones originales; clocks/CDC/bus skew y puertos pendientes identificados. |
| `20261002T125230Z-V3-ddr-route` | 0; 187,353 s | 70.578 nets ruteadas, cero errores; WNS +0,334 ns, WHS +0,011 ns. Cumple restricciones presentes, con cuatro puertos aún sin retardos. |
| `20261002T130332Z-V3-model-reset-full-delays` | 0; 76,202 s | Esperas RESET→CKE de 500 µs: persisten ocho errores con fila abierta. Diagnóstico, no PASS funcional. |
| `20261002T130647Z-V4-dci-reset-stress` | 0; 19,685 s | PASS: 588 cambios de fuentes de fallo y tres casos sin reloj, 591 liberaciones y un pulso DCI. |
| `20261002T131009Z-V4-reset-structure` | 0; 71,284 s | LUT4 alimenta solo dos FDPE ASYNC_REG colocados juntos. MTBF incompleto; LUTAR-1 conservado. |
| `20261002T131336Z-V4-dci-negative` | 1; 19,600 s | El banco rechaza la regresión; el lanzador esperaba el texto del otro monitor simultáneo. |
| `20261002T131532Z-V4-dci-negative-r2` | 0; 31,591 s | PASS del verificador: rechaza activación síncrona y liberación de una etapa en RTL copiado. |
| `20261002T133639Z-V4-cq-check-unit` | 0; 64,069 s | Verificador CQ: flujo válido de 768 bytes y siete rechazos negativos PASS. Respaldo local verificado. |
| `20261002T133826Z-V4-dma-multi` | 0; 1836,187 s | FAIL antes de DMA por MRRS sobrescrito, igual que PIPE inicial. Respaldo local reanudado y verificado tras disco lleno y corte SSH. |
| `20261002T134243Z-V4-pcie-pipe-example` | 0; 88,205 s | Variante PIPE generada, 16 lanes y reloj conectados; respaldo local verificado. Sin validación funcional por generación. |
| `20261002T134534Z-V4-dma-multi-pipe` | 0; 512,933 s | FAIL antes de DMA: MRRS borrado por habilitación posterior de etiquetas extendidas. Gen3 x16 enumerado; respaldo local verificado. |
| `20261002T135615Z-V4-dma-multi-pipe-r2` | 0; 565,855 s | PASS funcional: 768 bytes H2C/C2H, tres paquetes por sentido. Falso FAIL del primer revisor de consola corregido en auditoría separada; respaldo verificado. |
| `20261002T140844Z-V4-dma-multi-review` | 0; 7,689 s | Revisión numérica PASS del log intacto; result.md quedó vacío al llenarse C:. Intento conservado y respaldado. |
| `20261002T140854Z-V4-dma-multi-serial-r2` | 0; 2143,482 s | PASS serial: mismos conteos y datos que PIPE; diagnóstico manual revisado. Respaldo local verificado. |
| `20261002T141751Z-V4-dma-review-complete` | 0; 7,842 s | Auditoría PASS con documento completo, diagnósticos y límites; respaldo local verificado. |
| `20261002T144222Z-V4-ready-gate-unit` | 0; 44,118 s | PASS de la compuerta de pausas: 64 beats y dos regresiones rechazadas. Respaldo local verificado. |
| `20261002T144323Z-V4-dma-stalls-pipe` | 0; 532,662 s | PASS: 768 bytes por sentido con 17 pausas del productor W y 14 del productor R; revisión manual y respaldo verificados. |
| `20261002T145416Z-V4-stalls-wrapper-compile` | 0; 10,198 s | Declaración movida antes del uso: compilación sin avisos/errores, resto del wrapper idéntico. Sin nueva simulación; respaldo verificado. |
| `20261002T150840Z-V4-host-memory-unit` | 0; 64,588 s | Modelo de host de 64 bits: datos y cuatro descriptores, siete negativos; respaldo verificado. |
| `20261002T150850Z-V4-cq-host64-unit` | 0; 77,214 s | CQ: regiones altas, cruces de página y nueve negativos PASS; respaldo verificado. |
| `20261002T150919Z-V4-cq-baseline-regression` | 0; 63,446 s | Regresión del CQ anterior: positivo y siete negativos PASS; respaldo verificado. |
| `20261002T151608Z-V4-host64-dma-pipe` | 0; 511,878 s | FAIL: H2C 1536 bytes correcto; el logger interpreta mal la respuesta C2H y detiene el banco. Respaldo verificado. |
| `20261002T153855Z-V4-pcie-logger-unit` | 0; 51,881 s | Reproduce el defecto original; 42 comprobaciones y cinco rechazos negativos PASS. Respaldo verificado. |
| `20261002T154049Z-V4-host64-dma-pipe-r2` | 0; 493,774 s | FAIL: argumentos estáticos compartidos por dos llamadas concurrentes duplican el cierre del logger. Respaldo verificado. |
| `20261002T155149Z-V4-pcie-logger-concurrent-unit` | 0; 51,710 s | Reproduce dos cierres concurrentes; argumentos automáticos dan uno y 76 bytes. PASS y respaldo verificado. |
| `20261002T155357Z-V4-host64-dma-pipe-r3` | 0; 534,477 s | PASS: 1536 bytes por sentido, cuatro descriptores, regiones >4 GiB, límites de página y pausas W/R. |
| `20261002T155658Z-V4-pcie-lane-synth` | 1; 48,439 s | Elaboración falla por targets de BRAM no generados; diagnóstico conservado. |
| `20261002T160051Z-V4-pcie-lane-synth-r2` | 0; 481,806 s | PASS de conectividad: 16 lanes, 64 puertos/bolas; lane0→X1Y15. Checkpoint y recursos del endpoint serial. |
| `20261002T162036Z-V4-interface-checkpoint-inventory` | 0; 97,111 s | Inventario de ambos DCP: 154+67 puertos sin colisión; respaldo verificado. |
| `20261002T162725Z-V4-ddr-pcie-coexist` | 0; 204,624 s | PASS específico: 221 puertos conservados y BIVC/DCI=0 con PERST LVCMOS12; respaldo verificado. |
| `20261002T164022Z-V4-ddr-pcie-place` | 0; 864,280 s | Colocación completa; WNS −0,811 ns/WHS −0,581 ns. 221 puertos conservados; respaldo verificado. |
| `20261002T165523Z-V4-ddr-pcie-route` | 0; 654,465 s | 177.212 nets completas sin errores; WNS −0,288 ns/WHS +0,010 ns. Respaldo verificado. |
| `20261002T170751Z-V4-ddr-pcie-postroute-physopt` | 0; 425,872 s | 177.213 nets completas; WNS −0,215 ns/WHS +0,001 ns, 612 fallos de setup. Respaldo verificado. |
| `20261002T172152Z-V4-ddr-pcie-preroute-physopt` | 0; 846,522 s | Cumple resumen temporal actual: WNS +0,006 ns, WHS +0,004 ns; 177.381 nets completas, 221 E/S conservadas. Metodología pendiente. |
| `20261002T172439Z-V4-pcie-isolated-implementation` | 0; 802,651 s | PCIe aislado ruteado: WNS +0,031 ns, WHS +0,010 ns, 67 E/S y 106.680 nets completas; sin TIMING-16. |
| `20261002T173116Z-V2-catalog-3ds128` | 1; 39,475 s | Diagnóstico de selector DataMask deshabilitado; no rechaza el RDIMM de catálogo. |
| `20261002T173548Z-V2-catalog-3ds128-r2` | 0; 45,483 s | PASS de geometría 3DS: un slot, dos ranks físicos × cuatro dies, ECC, 37/512 AXI y 128 GiB. Sin simulación. |
| `20261002T174842Z-V4-joint-timing-audit` | 0; 206,293 s | Reproduce TIMING-16 tras reabrir; pareja #1 da +0,964 ns con multicycle y −1,036 ns en metodología. |
| `20261002T175303Z-V2-catalog-3ds-example` | 0; 77,457 s | Ejemplo generado; modelo Samsung no admite XSIM. Mensaje de éxito seguido de finish no es validación. |
| `20261002T175745Z-V4-multicycle-path-audit` | 0; 151,962 s | 55 parejas positivas con multicycle; todas difieren 2 ns del aviso. Diagnóstico de un ciclo reproduce #1; DCP original intacto. |
| `20261002T181538Z-V4-all-channel-gate-unit` | 0; 79,703 s | Tres positivos de pausas y tres regresiones rechazadas; respaldo verificado. |
| `20261002T181750Z-V4-host64-all-channel-dma` | 0; 535,208 s | PASS: 1536 bytes por sentido con pausas AW/W/B/AR/R y correlación BID/RID. 138 avisos iguales a r3; respaldo verificado. |
| `20261002T182015Z-V7-ffn-vu5p-synth` | 0; 135,011 s | C simulation y síntesis HLS VU5P; 38.726 LUT/86 DSP estimados; ALLOCATION ignorada por clonación. Respaldo verificado. |
| `20261002T182535Z-V7-ffn-vu5p-ooc-synth` | 0; 306,833 s | Síntesis Vivado: 27.918 LUT, 22.961 FF, 55 DSP y 9 BRAM equivalentes; cero cajas negras. Sin implementación física; respaldo verificado. |
| `20261002T193955Z-V7-ffn-vu5p-cosim-smoke` | 0; 535,836 s | PASS 21/21, 564.333 ciclos totales; excluye forma máxima y pesos reales. |
| `20261002T194824Z-V7-q8-block-mac-cosim` | 0; 161,230 s | PASS 317/317 con continuidad K=8192; bloque válido 193 ciclos; [contrato](../Q8_BLOCK_MAC.md). |
| `20261002T195413Z-V7-q8-block-mac-bitcast` | 0; 129,466 s | PASS 317/317; conversión FP16 exacta por bits reduce latencia a 21 ciclos e intervalo a 22; 2.908 LUT estimadas. |
| `20261003T013858Z-V7-gemv-stream-integration` | 0; 73,754 s | FAIL del banco: cancelación usa contador residual y agota timeout. Código 0 no acredita PASS; intento conservado. |
| `20261003T014314Z-V7-gemv-stream-integration-r2` | 1; 28,315 s | Rechazo Tcl del valor testplusarg; corregido con `-dict`. |
| `20261003T014535Z-V7-gemv-stream-integration-r3` | 0; 129,492 s | PASS 66/66 con MAC real, activaciones, acumulación, errores tardíos y cancelación; nueve corrupciones de logs rechazadas. |
| `20261003T015039Z-V7-gemv-stream-reset` | 0; 129,857 s | PASS 72/72: añade reset coordinado durante AXI y MAC, sin resultados antiguos. |
| `20261003T015049Z-V7-gemv-stream-synth` | 0; 207,876 s | Síntesis completa pero timing FAIL: WNS −1,322 ns, 200 fallos de setup. Motiva segmentar validación de comando. |
| `20261003T015649Z-V7-gemv-pipelined-validation` | 0; 120,919 s | PASS 72/72 con validación segmentada; once corrupciones de logs rechazadas. |
| `20261003T015929Z-V7-gemv-pipelined-synth` | 0; 195,762 s | 10.396 LUT/7.704 FF/26 DSP; WNS +1,785 ns/WHS +0,046 ns estimados, sin fallos internos. |
| `20261003T021016Z-V7-gemv-pipelined-route` | 1; 35,362 s | Checkpoint abierto; comprobación NAME demasiado estricta rechaza antes de implementar. |
| `20261003T021547Z-V7-gemv-brandon-projections` | 0; 359,584 s | PASS numérico 162 operaciones/5088 salidas; banco reutiliza dirección entre tiles pequeños. Errata y repetición posteriores; archivo intacto. |
| `20261003T021557Z-V7-gemv-pipelined-route-r2` | 0; 242,335 s | Ruteo OOC exploratorio, 10.293 LUT/7.655 FF/26 DSP; WNS +0,564 ns/WHS +0,010 ns. Una net con gaps de reloj; no cierre de tarjeta. |
| `20261003T021623Z-V7-gemv-fixture-bank-regression` | 0; 122,038 s | Regresión 72/72 PASS tras añadir fixtures; conserva pruebas negativas y reset. |
| `20261003T022306Z-V7-gemv-route-context-audit` | 0; 82,188 s | Auditoría sin modificar DCP: `clk` parcial, 203 segmentos jerárquicos con idéntica ruta. Fronteras sin PARTPIN_LOCS. |
| `20261003T023437Z-V7-gemv-brandon-distinct-tiles` | 0; 356,925 s | PASS 162 operaciones/5088 salidas con direcciones distintas por tile; mismos valores y ciclos. Diferencia Q8 cero; L2 FP32 0,717–0,792%. |
| `20261003T024712Z-V7-gemv-metric-regression` | 0; 130,968 s | PASS 72/72 escalar; contadores de fases, ocupación y bytes reconciliados. |
| `20261003T024712Z-V7-gemv-scalar-occupancy` | 0; 366,601 s | PASS 162 jobs; MAC ocupado ~91,5%, 418265–418348 ciclos de jobs. |
| `20261003T025308Z-V7-gemv-dual-regression` | 0; 124,202 s | PASS 75/75 con dos MAC, layout2 y rechazo previo de layout1. |
| `20261003T025308Z-V7-gemv-dual-brandon` | 0; 353,036 s | PASS 162 jobs/5088 valores; 218195–218328 ciclos, ~1,92× frente a escalar. |
| `20261003T025620Z-V7-gemv-dual-synth` | 0; 265,751 s | 18.126 LUT/52 DSP; caché inferida como LUTRAM, cuatro Synth8-6849; corregida después. |
| `20261003T030230Z-V7-gemv-dual-bram-synth` | 0; 238,868 s | 16.258 LUT/10.567 FF/37RAMB36+2RAMB18/52 DSP, cero LUTRAM; WNS estimado +1,163 ns. |
| `20261003T030230Z-V7-gemv-dual-stress` | 1; 45,045 s | FAIL de elaboración: llamada del banco mal escrita, sin validación funcional. |
| `20261003T030230Z-V7-gemv-dual-bram-brandon` | 1; 46,207 s | Mismo fallo de elaboración; intento conservado. |
| `20261003T030438Z-V7-gemv-dual-stress-r2` | 1; 67,989 s | Banco rechaza inyección en motor1 sin solapamiento requerido; corrige el estímulo. |
| `20261003T030438Z-V7-gemv-dual-bram-brandon-r2` | 0; 358,169 s | PASS 162/162 tras corregir BRAM; mismos ciclos y valores, error Q8 cero. Avisos de rama de inyección no ejecutada registrados. |
| `20261003T030747Z-V7-gemv-dual-stress-r3` | 1; 66,993 s | Caso24 de error dual PASS; referencia adversarial falla por representación del literal decimal, sin PASS de suite. |
| `20261003T031546Z-V7-fp64-reference-probe` | 0; 25,164 s | FAIL de revisión pese a exit0: reproduce literal de 2^53 dos unidades bajo en este XSIM. |
| `20261003T031546Z-V7-gemv-dual-route` | 0; 320,399 s | 16.318 LUT/10.489 FF/52 DSP; WNS +0,271 ns/WHS +0,010 ns; 26231 nets, gap de reloj conservado. |
| `20261003T031738Z-V7-fp64-reference-probe-r2` | 0; 25,267 s | PASS con bits IEEE exactos: acumulación ordenada da cero; literal decimal defectuoso aislado. |
| `20261003T031837Z-V7-gemv-dual-stress-r4` | 0; 127,232 s | PASS 87 casos, error/reset de ambos motores; su patrón K31/K32 no distingue subtotales de tiles. Límite documentado, repetición dirigida posterior. |
| `20261003T032242Z-V7-gemv-dual-route-audit` | 0; 177,141 s | 401 alias de la misma ruta parcial clk; fronteras sin restricción/localización, no cierre de placa. |
| `20261003T032531Z-V7-gemv-dual-order-boundary` | 0; 128,130 s | PASS 87/87; K32/K33 distingue acumulación ordenada=0 de subtotales=1; error/reset dual y 16 corrupciones de evidencia rechazadas. |
| `20261003T033430Z-V7-gemv-scalar-max` | 0; 277,028 s | PASS 12/12, formas 32×2048/8192, 1×8192 y 31×2048; 288 valores, diferencia Python cero. |
| `20261003T033440Z-V7-gemv-dual-max` | 0; 268,583 s | PASS 12/12, mismos valores; 32×8192 baja de ~194,8k a ~100,6k ciclos; una fila no acelera. |
| `20261003T034056Z-V7-gemv-host-packed` | 0; 544,588 s | PASS 174/174, 5376 valores, error cero; consume imagen/descriptores del host. Decodificación independiente de 407 archivos y 16 corrupciones de logs rechazadas. |
| `20261003T034748Z-V7-gemv-host-packed-negative` | 1; 61,303 s | Fallo numérico esperado al cambiar un solo byte de imagen con hashes consistentes; fuente/referencia intactas. No es detector de corrupción del hardware. |
| `20261003T035216Z-V7-gemv-packed-driver-regression` | 0; 129,490 s | Regresión del lanzador y banco ampliados: 87/87 casos de errores/reset PASS, mismas métricas que la referencia anterior. |
| `20261003T040812Z-V7-activation-loader-unit` | 0; 50,333 s | PASS 96/96: ABI1, rangos/épocas, escalas, transporte, cancelación, watchdog, drenaje y reset. |
| `20261003T041650Z-V7-activation-loader-synth` | 0; 97,659 s | 788 LUT/1429 FF, cero DSP/BRAM; WNS estimado +2,559 ns. |
| `20261003T042547Z-V7-gemv-memory-unit` | 0; 89,132 s | PASS 60/60: descriptor registrado, activaciones antes de pesos, valores exactos, fallos/cancelación/reset sin salida parcial. |
| `20261003T042802Z-V7-gemv-memory-synth` | 0; 246,621 s | Wrapper integrado: 16.993 LUT/12.438 FF/52 DSP/37 RAMB36+2 RAMB18, WNS estimado +0,980 ns. Sin ruteo ni fronteras calificadas. |
| `20261003T043551Z-V7-gemv-memory-regression` | 0; 97,487 s | Regresión del banco con modo imagen: mismos 60 casos y conteos, diez logs alterados rechazados. |
| `20261003T043551Z-V7-gemv-memory-fixture` | 0; 537,170 s | PASS 174 jobs/5376 valores, error cero; 1.092.480 bytes AXI por semilla de imágenes de pesos y activaciones. |
| `20261003T043854Z-V7-gemv-activation-negative` | 1 esperado; 54,600 s | Un byte de activación 254→255 provoca discrepancia en la primera fila, ciclo 3203; originales y pesos intactos. |
| `20261003T050240Z-V7-result-writer-unit` | 0; 49,268 s | PASS 126/126: staging privado, commit, páginas, máscaras, errores, cancelación, watchdog y resets durante escritura. |
| `20261003T050528Z-V7-result-writer-synth` | 0; 97,883 s | 1215 LUT/446 FF, cero BRAM/DSP; WNS estimado +2,521 ns. |
| `20261003T051344Z-V7-gemv-store-unit` | 0; 138,225 s | PASS 105/105 integrados; valores físicos y completion tras B. Avisos de declaraciones conservados. |
| `20261003T051540Z-V7-result-writer-netlist` | 1; 35,518 s | Exporta netlist; falla opción Tcl antes de simular. |
| `20261003T051836Z-V7-gemv-store-synth` | 0; 264,195 s | 18.382 LUT/13.744 FF/52 DSP/37 RAMB36+2 RAMB18; WNS estimado +0,979 ns. |
| `20261003T051836Z-V7-gemv-store-fixture` | 0; 544,532 s | PASS 174 jobs/5376 valores escritos exactos; 115 ráfagas de salida por semilla. |
| `20261003T051836Z-V7-result-writer-netlist-r2` | 1; 55,677 s | Falla estímulo: esperaba watchdog de 32 ciclos en netlist de producción con 65.536. |
| `20261003T052315Z-V7-result-writer-netlist-r3` | 0; 91,859 s | PASS 126/126 en netlist original, probando watchdog de 65.536; simulación funcional sin SDF. |
| `20261003T052315Z-V7-gemv-store-regression` | 0; 146,923 s | PASS 105/105 tras corregir declaraciones; sin esos avisos. |
| `20261003T053410Z-V7-gemv-store-route` | 1; 39,134 s | Guard del lanzador usaba NAME en lugar de TOP; no llegó a colocar. Checkpoint intacto. |
| `20261003T053852Z-V7-gemv-store-route-r2` | 0; 381,702 s | Ruteo OOC: 18.239 LUT/13.644 FF/52 DSP, 30.741 nets; WNS +0,395 ns/WHS +0,010 ns. Reloj/fronteras pendientes. |
| `20261003T054211Z-V7-gemv-store-write-reset` | 0; 158,114 s | PASS 111/111: cuatro fases de reset, incluidos W parcial y B pendiente, sin completion y con recuperación posterior. |
| `20261003T054620Z-V7-gemv-store-route-audit` | 0; 213,366 s | Revisión sin cambios: 446 aliases PARTIAL con ruta idéntica del reloj; puertos sin ubicación ni retardos. |

| `20261003T060606Z-V7-matrix-sequencer-unit` | 0; 77,339 s | PASS 135/135 de protocolo/descriptores; motor simulado, sin MAC. |
| `20261003T060606Z-V7-matrix-sequencer-synth` | 0; 95,359 s | FAIL temporal: WNS −1,310 ns, 60 endpoints; cálculo de dirección sin registrar. |
| `20261003T061228Z-V7-matrix-sequencer-unit-r2` | 0; 101,686 s | PASS 135/135 con etapas registradas; incluye 65.536 tiles de descriptores. |
| `20261003T061228Z-V7-matrix-sequencer-synth-r2` | 0; 95,098 s | 1165 LUT/1449 FF/2 DSP; WNS +1,770 ns/WHS +0,039 ns estimados. |
| `20261003T062214Z-V7-matrix-memory-unit` | 0; 240,608 s | PASS 30/30 con MAC real: errores en segundo grupo, cancelación/reset y recuperación. |
| `20261003T062214Z-V7-matrix-memory-synth` | 0; 302,375 s | 19.369 LUT/15.191 FF/54 DSP; WNS +1,450 ns/WHS +0,039 ns estimados. |
| `20261003T063235Z-V7-matrix-memory-brandon` | 0; 359,986 s | PASS 9 matrices completas/5088 valores físicos exactos; 219.005–219.394 ciclos por semilla. |
| `20261003T064137Z-V7-matrix-memory-route` | 0; 342,570 s | Ruteo OOC: 19.212 LUT/15.095 FF/54 DSP; WNS +0,614 ns/WHS +0,010 ns. 32.842 nets contadas completas, reloj padre con gaps. |
| `20261003T064736Z-V7-matrix-memory-route-audit` | 0; 244,169 s | Auditoría sin cambios: 450 aliases PARTIAL con ruta idéntica del reloj. Retardos/ubicaciones de puertos pendientes. |
| `20261003T064207Z-V7-matrix-llama-up` | 0; 2603,995 s | PASS 8192×2048 sintético: 8192 valores exactos, 6.463.892 ciclos, 65.536 bytes y 272 respuestas B. |
| `20261003T064207Z-V7-matrix-llama-down` | 0; 2582,270 s | PASS 2048×8192 sintético: 2048 valores exactos, 6.451.524 ciclos, 16.384 bytes y 68 respuestas B. |

| `20261003T065730Z-V7-swiglu-block-cosim` | 0; 243,270 s | PASS 26/26 C/RTL; 13 registros ABI1, 13 errores sin modificar salida. 2934–3794 ciclos válidos; operador local sin AXI. |
| `20261003T070338Z-V7-swiglu-block-synth` | 0; 188,858 s | 9372 LUT/6532 FF/37 DSP/2 RAMB36+1 RAMB18. WNS +1,747 ns/WHS +0,039 ns estimados. |
| `20261003T071037Z-V7-swiglu-block-route` | 0; 212,037 s | 9020 LUT/6428 FF/37 DSP, WNS +1,038 ns/WHS +0,011 ns internos. |
| `20261003T071520Z-V7-swiglu-block-route-audit` | 0; 104,097 s | 615 aliases PARTIAL idénticos del reloj; 162 entradas/87 salidas sin retardos. Sin excepciones nuevas. |

| `20261003T072154Z-V7-swiglu-block-raw-payload` | 0; 233,599 s | PASS 27/27; registro INT8 con palabras NaN si se leen como doubles. 30 archivos RTL idénticos al banco inicial. |
| `20261003T074923Z-V7-activation-writer-unit` | 0; 49,437 s | PASS 159/159; registros ABI1, cancelación durante AXI, errores, watchdog y cinco resets por semilla. |
| `20261003T074923Z-V7-activation-writer-synth` | 0; 87,148 s | 385 LUT/932 FF, cero DSP/BRAM; WNS estimado +3,422 ns/WHS +0,066 ns. |
| `20261003T075418Z-V7-activation-writer-netlist` | 0; 107,629 s | PASS 159/159 sobre netlist, watchdog real 65.536; sin SDF. |
| `20261003T080726Z-V7-swiglu-memory-unit` | 0; 270,515 s | PASS 105/105; gate/up AXI → SwiGLU/Q8 real → registros privados. 123 registros exactos; errores tardíos/reset. |
| `20261003T080726Z-V7-swiglu-memory-synth` | 0; 243,679 s | 17.424 LUT/14.060 FF/37 DSP; WNS estimado +1,747 ns/WHS +0,039 ns. Buffers gate/up en registros. |
| `20261003T081935Z-V7-swiglu-memory-varied` | 0; 271,495 s | PASS 105/105, gate variable por bloque; 123 registros W exactos. RTL idéntico al sintetizado. |
| `20261003T081935Z-V7-swiglu-memory-8192` | 0; 388,948 s | PASS 8192 parejas/256 registros completos, 980.075 ciclos; 131.072 bytes leídos y 16.384 escritos exactos. |
| `20261003T081935Z-V7-swiglu-memory-route` | 0; 267,076 s | 17.024 LUT/13.956 FF/37 DSP; WNS +0,567 ns/WHS +0,010 ns internos. |
| `20261003T082422Z-V7-swiglu-memory-route-audit` | 0; 117,673 s | 624 aliases PARTIAL del mismo reloj; 24.495 nets contadas completas, un gap explícito. Fronteras sin calificar. |
| `20261003T085310Z-V7-ffn-sequencer-unit` | 0; 51,071 s | PASS 498/498 de protocolo: ocho buffers, fases, completions corruptas, cancelación y reset. Sin MAC. |
| `20261003T085310Z-V7-ffn-sequencer-synth` | 0; 102,920 s | 2701 LUT/3165 FF/2 DSP; WNS estimado +1,438 ns/WHS +0,039 ns. |
| `20261003T085902Z-V7-ffn-memory-synth` | 0; 329,721 s | FFN integrada: 38.917 LUT/32.200 FF/93 DSP/39 RAMB36+3 RAMB18; WNS estimado +0,849 ns/WHS +0,039 ns. |
| `20261003T090406Z-V7-ffn-memory-unit` | 0; 371,413 s | PASS 63/63 con motores HLS reales y AXI compartido; 74.176 bytes físicos exactos, errores/cancel/reset en las cuatro fases. |
| `20261003T090959Z-V7-ffn-memory-route` | 0; 602,321 s | 38.627 LUT/31.997 FF/93 DSP; WNS +0,068 ns/WHS +0,010 ns internos. 61.685 nets, sin errores de ruteo; margen setup estrecho. |
| `20261003T092050Z-V7-ffn-memory-route-audit` | 0; 570,017 s | 1090 segmentos PARTIAL del mismo reloj, un gap explícito. Retardos/fronteras pendientes; pico de RAM 57.688,504 MB. |
| `20261003T092954Z-V7-ffn-memory-route-audit-stream` | 0; 75,681 s | Auditoría equivalente del mismo DCP: mismo conjunto de aliases/rutas/contexto. Pico 4099,055 MB al comparar rutas una a una. |
| `20261003T093914Z-V7-ffn-memory-brandon` | 0; 638,627 s | PASS tres FFN de capa 0 con pesos reales y x compartida sintética: 45.120 bytes físicos exactos, 768 salidas down; 307.289–307.377 ciclos. FP32 final comprobado fuera del hardware. |
| `20261003T100601Z-V7-fp32-converter-unit` | 1; 28,470 s | Fallo del banco al interpretar EOF; no ejecutó conversiones. Se preserva y corrige solo el lector en r2. |
| `20261003T100601Z-V7-fp32-converter-synth` | 0; 85,272 s | Conversor: 376 LUT/382 FF, cero DSP/BRAM; WNS estimado +2,755 ns/WHS +0,083 ns. |
| `20261003T100710Z-V7-fp32-converter-unit-r2` | 0; 77,335 s | PASS 155.784 conversiones IEEE/Brandon con metadatos, pausas y cuatro resets por semilla; tres ciclos sin pausas. |
| `20261003T101108Z-V7-fp32-converter-netlist` | 0; 93,216 s | PASS las mismas 155.784 conversiones en netlist funcional sin SDF. Conversor aún separado del escritor AXI. |
| `20261003T103033Z-V7-fp32-writer-unit` | 0; 49,573 s | PASS integración inicial: 171 casos mixtos FP32/binary64, 2608 bytes por semilla. |
| `20261003T103033Z-V7-result-writer-format-regression` | 0; 49,581 s | PASS 126 casos del escritor binary64, 3296 bytes por semilla. |
| `20261003T103033Z-V7-ffn-control-format-regression` | 0; 51,214 s | PASS 498 casos del controlador con formato binary64 explícito. |
| `20261003T103251Z-V7-fp32-writer-synth` | 0; 108,984 s | Primera integración: 4744 LUT/2867 FF; WNS +2,578 ns/WHS +0,066 ns. Buffer en registros. |
| `20261003T103251Z-V7-ffn-fp32-memory-unit` | 0; 375,108 s | PASS 63 casos de cadena FP32; 71.040 bytes físicos exactos; errores/cancelación/reset y recuperación. |
| `20261003T103251Z-V7-ffn-fp32-memory-synth` | 0; 338,016 s | Primera integración: 44.703 LUT/34.634 FF/93 DSP; WNS +0,843 ns/WHS +0,039 ns. |
| `20261003T103720Z-V7-ffn-fp32-control` | 0; 50,906 s | PASS 498 casos del controlador FP32; protocolo, no cálculo a dimensiones Llama. |
| `20261003T103804Z-V7-fp32-writer-netlist` | 0; 103,032 s | PASS 171 casos del escritor inicial sintetizado, watchdog 65.536. |
| `20261003T104133Z-V7-fp32-writer-ram-unit` | 1; 29,047 s | RECHAZADO: optimización con escritura en bloque combinacional; falla el primer resultado FP32. |
| `20261003T104133Z-V7-fp32-writer-ram-synth` | 0; 105,243 s | RECHAZADO pese al código 0: infiere 32 latches. No se acepta su timing. |
| `20261003T104133Z-V7-ffn-fp32-memory-ram-unit` | 1; 81,296 s | RECHAZADO: salida down incorrecta en primer caso, fase 3, ciclo 5347. |
| `20261003T104133Z-V7-ffn-fp32-memory-ram-synth` | −15; 166,403 s | Detenida exclusivamente esta síntesis de RTL inválido; PID/ruta verificados y motivo respaldado. |
| `20261003T104452Z-V7-fp32-writer-ram-r2-unit` | 0; 49,215 s | PASS 171 casos tras mover el único puerto de escritura a always_ff. |
| `20261003T104647Z-V7-fp32-writer-ram-r2-synth` | 0; 101,481 s | 1988 LUT/2861 FF, −58,1% LUT frente al escritor inicial; buffer aún en registros. WNS +2,578 ns/WHS +0,066 ns. |
| `20261003T104647Z-V7-ffn-fp32-memory-ram-r2-unit` | 0; 375,593 s | PASS 63 casos; mismos 71.040 bytes físicos y métricas de la integración inicial. |
| `20261003T104647Z-V7-ffn-fp32-memory-ram-r2-synth` | 0; 353,124 s | 40.407 LUT/34.617 FF/93 DSP, 39 RAMB36+3 RAMB18; WNS +0,959 ns/WHS +0,039 ns. Sin latches. |
| `20261003T104647Z-V7-ffn-binary64-format-regression` | 0; 374,554 s | PASS 63 casos con RTL corregido y salida binary64; 74.176 bytes físicos exactos. |
| `20261003T104647Z-V7-ffn-fp32-brandon` | 0; 652,020 s | PASS tres cadenas reales de capa 0: 42.048 bytes exactos; 768 FP32 almacenados coinciden bit a bit con Q8 nativo. 307.247/307.382/307.387 ciclos. |
| `20261003T104951Z-V7-fp32-writer-ram-r2-netlist` | 0; 102,590 s | PASS 171 casos sobre el DCP corregido, watchdog 65.536; todos los bytes/máscaras comprobados. |
| `20261003T105639Z-V7-ffn-fp32-memory-route` | 0; 590,806 s | Ruteo OOC: 40.080 LUT/34.427 FF/93 DSP; WNS +0,164 ns/WHS +0,010 ns. 65.492 nets enrutables completas; un gap de reloj padre. |
| `20261003T110717Z-V7-ffn-fp32-memory-route-audit` | 0; 76,982 s | 1051 segmentos PARTIAL con ruta de reloj idéntica; cero latches. 2470 entradas/845 salidas sin retardos; pico RAM 4113,297 MB. Sin cierre de placa. |
| `20261003T112205Z-V7-ffn-large-smoke` | 0; 206,900 s | PASS cadena 2048/96: 243.755 ciclos, 9920 bytes W exactos; plan de 376 AR y checksums de 665.088 bytes R. Diez alteraciones de evidencia rechazadas. |
| `20261003T113156Z-V7-ffn-large-bank-regression` | 0; 373,311 s | PASS 63 casos; ciclos/bytes/pausas idénticos al banco FP32 previo, 71.040 bytes físicos. |
| `20261003T113913Z-V7-ffn-large-fast-smoke` | 0; 206,987 s | PASS traza idéntica al smoke con debug off. Pico XSIM 324.280 frente a 468.136 KB; sin mejora de tiempo de pared. |
| `20261003T120619Z-V7-ffn-csr-unit` | 1; 12,572 s | Lanzador CSR rechazado por opción Tcl; sin simulación. Corregido en tanda nueva. |
| `20261003T120823Z-V7-ffn-csr-unit-r2` | 0; 49,058 s | PASS AXI-Lite FP32: 27 grupos, 894 lecturas/1.236 escrituras, descriptor de 64 bits, IRQ/cancelación/reset y siete alteraciones de evidencia rechazadas. Par de cómputo simulado a nivel de protocolo. |
| `20261003T121223Z-V7-ffn-csr-fp64-unit` | 0; 49,112 s | PASS del mismo banco CSR con salida binary64; conserva todas las comparaciones y cobertura. |
| `20261003T121223Z-V7-ffn-csr-synth` | 0; 208,932 s | Control AXI-Lite: 1.729 LUT/4.372 FF, sin BRAM/DSP/latches. WNS estimado +2,280 ns/WHS +0,066 ns; reloj padre y retardos de fronteras pendientes. |
| `20261003T122927Z-V7-ffn-host-memory` | 0; 392,839 s | PASS host AXI-Lite → FFN/HLS real: 63 casos, 71.232 bytes físicos, 8.052 escrituras MMIO; doce alteraciones de evidencia rechazadas. |
| `20261003T122927Z-V7-ffn-host-bank-regression` | 0; 373,494 s | Regresión directa 63/63, 71.040 bytes físicos; todas las trazas NTRPU idénticas al banco anterior, incluidos ciclos. |
| `20261003T123822Z-V7-ffn-host-synth` | 0; 448,517 s | Dispositivo host + FFN: 41.802 LUT/38.981 FF/93 DSP, 40,5 BRAM. WNS estimado +0,683 ns/WHS +0,039 ns; fronteras pendientes. |
| `20261003T123822Z-V7-ffn-host-brandon` | 0; 656,075 s | PASS tres FFN Brandon por MMIO: 42.048 bytes físicos, 768 FP32 idénticos a Q8; 307.134–307.219 ciclos de motor y 307.669–307.755 de servicio host simulado. |
| `20261003T124752Z-V7-ffn-host-route` | 0; 673,419 s | Dispositivo host + FFN ruteado: 41.536 LUT/38.775 FF/93 DSP, WNS +0,099 ns/WHS +0,010 ns. 70.799 nets enrutables completas, un hueco explícito del reloj padre. |
| `20261003T125921Z-V7-ffn-host-route-audit` | 0; 79,427 s | Auditoría: 1.053 alias PARTIAL con la misma ruta de reloj; cero latches y excepciones añadidas. 636 entradas/769 salidas y reloj padre siguen sin calificar. |
| `20261003T132136Z-V7-axi-arbiter-unit` | 0; 51,789 s | Banco inicial del árbitro: tres semillas. RTL idéntico y trazas conservadas idénticas al banco posterior con instrumentación completa. |
| `20261003T132256Z-V7-axi-arbiter-trace` | 0; 52,299 s | PASS aislado: 27 bancos, 12 resets con recuperación, contención y respuestas de error; handshakes reconstruidos por Python, doce corrupciones rechazadas. |
| `20261003T132256Z-V7-axi-arbiter-synth` | 0; 87,423 s | 419 LUT/1.300 FF, cero DSP/BRAM/latches. WNS estimado +4,046 ns/WHS +0,077 ns; reloj padre y fronteras pendientes. |
| `20261003T133841Z-V7-ffn-host-transfer-regression` | 0; 392,555 s | PASS 63 casos MMIO anteriores: 71.232 bytes, 8.052 escrituras de control; todas las líneas NTRPU idénticas a la referencia, incluidos ciclos. |
| `20261003T133841Z-V7-ffn-host-transfer` | 0; 714,259 s | PASS host AXI → RAM vacía → FFN real → lectura host: 1.777.152 bytes cargados, 42.048 calculados exactos, 3.072 recuperados; contención, MMIO y doce corrupciones revisados. |
| `20261003T135323Z-V7-ffn-fabric-synth` | 0; 448,253 s | Conjunto FFN/control/árbitro: 42.531 LUT/40.260 FF/93 DSP; WNS estimado +0,735 ns, WHS +0,039 ns. Los 22 RTL coinciden con la simulación; reloj padre, fronteras y ruteo pendientes. |
| `20261003T141646Z-V7-axi-arbiter-2deep-unit` | 0; 53,354 s | Dos entradas: 27 bancos/12 resets; traza independiente y doce corrupciones rechazadas. |
| `20261003T141646Z-V7-axi-arbiter-2deep-pipeline` | 0; 53,600 s | Otros 27 bancos/12 resets con solicitudes y W anticipadas. Doce corrupciones rechazadas. |
| `20261003T141646Z-V7-axi-arbiter-2deep-rate` | 0; 29,880 s | Cuatro casos, 64 beats en 64 ciclos por dirección; buffer lleno bajo pausa, nueve corrupciones rechazadas. |
| `20261003T141646Z-V7-axi-arbiter-1deep-rate-reference` | 0; 29,712 s | Mismo banco con RTL anterior fijado por hash: 64 beats en 127 ciclos. |
| `20261003T142059Z-V7-axi-arbiter-2deep-negative` | 1 esperado; 29,268 s | Mutación RTL aislada de la protección WLAST: assertion de W prematura en ciclo 4. No es PASS funcional. |
| `20261003T142109Z-V7-ffn-host-transfer-2deep` | 0; 705,216 s | Tres FFN Brandon: carga 1.777.152 bytes, 42.048 calculados exactos y 3.072 recuperados. 309.289–309.408 ciclos de programación/cómputo; doce corrupciones rechazadas. |
| `20261003T142216Z-V7-ffn-fabric-2deep-synth` | 0; 452,198 s | 43.146 LUT/41.364 FF/93 DSP, WNS estimado +0,735 ns/WHS +0,039 ns. Los 22 RTL coinciden con la integración; fronteras y ruteo pendientes. |
| `20261003T113023Z-V7-ffn-full-llama-shape` | 0; 11.160,111 s | FFN sintética completa 2048/8192: 20.332.664 ciclos, 56.754.176 bytes leídos y 155.648 escritos exactos, incluidos 2.048 FP32. Diez corrupciones rechazadas; respaldo 483.736.666 bytes verificado. |
| `20261003T143841Z-V7-ffn-fabric-2deep-route` | 0; 762,734 s | Fabric 43.148 LUT/41.216 FF/93 DSP, WNS +0,206 ns/WHS +0,010 ns. 74.978 nets ruteables completas; un hueco de reloj padre. No cierre de placa. |
| `20261003T145146Z-V7-ffn-fabric-2deep-audit` | 0; 82,971 s | 1.055 segmentos PARTIAL con idéntica ruta de reloj; cero latches. 1396/1333 entradas/salidas sin retardos. Hashes de checkpoints y 22 RTL cotejados; ambas tandas respaldadas. |
| `20261003T145923Z-V7-host-clock-probe` | 0; 18,832 s | Generación AXI4 512/64/4 y AXI4-Lite 32/32, asíncronos, tres etapas; revisión 36 Included. No prueba funcional. |
| `20261003T150817Z-V7-host-clock-sim` | 1; 18,878 s | Opciones XSIM interpretadas como opciones Tcl; no se ejecutó simulación. |
| `20261003T151033Z-V7-host-clock-sim-r2` | 1; 18,535 s | Repite fallo con snapshot anterior mientras subía la corrección. El lanzador ahora rechaza fuentes desactualizadas antes de crear la tanda. |
| `20261003T151147Z-V7-host-clock-sim-r3` | 1; 40,575 s | Banco rechaza W AXI-Lite por asumir orden entre READY de ambos dominios; corregido para seguir oferta y confirmaciones. |
| `20261003T151612Z-V7-host-clock-sim-r4` | 0; 70,713 s | PASS de herramienta rechazado por revisión: marcador ABORT antes del último flanco host. Evidencia no aprobada. |
| `20261003T152740Z-V7-host-clock-sim-r5` | 0; 71,759 s | PASS revisado: 54 casos/24 resets; tres relaciones de reloj, bytes/sidebands/orden exactos. 36 mutaciones de evidencia rechazadas. Modelo FIFO sin retardos físicos de sincronización. |
| `20261003T152812Z-V7-host-clock-stale` | 1 esperado; 38,587 s | Dato de época anterior inyectado en R detectado en caso 3, sc27/mc21. Prueba negativa real XSIM. |
| `20261003T153412Z-V7-host-clock-axi-synth` | 0; 101,438 s | Síntesis preliminar 998 LUT/3102 FF. Avisos de producto y módulo no instanciado conservados; no cierre CDC. |
| `20261003T153412Z-V7-host-clock-axil-synth` | 0; 92,567 s | Síntesis preliminar 49 LUT/377 FF; mismos avisos del lanzador conservados. |
| `20261003T153913Z-V7-host-clock-axi-synth-r2` | 0; 99,616 s | Flujo corregido, sin errores/avisos críticos. 998 LUT/3102 FF; WNS +3,216/WHS +0,013 ns estimados. Diez restricciones bus skew con margen positivo; un CDC-15 y fronteras pendientes. |
| `20261003T153913Z-V7-host-clock-axil-synth-r2` | 0; 91,142 s | 49 LUT/377 FF; WNS +3,345/WHS +0,077 ns estimados. Diez CDC-3 informativos, cinco restricciones bus skew positivas; waivers XPM y fronteras conservados. |

Las tandas de esta tabla se cotejan con sus respaldos locales antes de emitir el índice de entrega. El ruteo ocupa 47.182.925 bytes, SHA-256 `a6feb860c8eec48fcb13aa6232e2229b0f7a25f39405f525b8a9a561b89d665f`; su carpeta incluye checkpoint, XDC, reportes, fuentes y `review.json`. El índice vuelve a comprobar los archivos locales y NFS, sin modificar las tandas selladas.

| Tanda asíncrona | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T155432Z-V7-ffn-async-brandon` | 0; 820,797 s | Host 4 ns/motor 5 ns: tres FFN Brandon, carga AXI 1.777.152 B, 768 FP32 exactos recuperados, IRQ retenida y confirmada por MMIO. |
| `20261003T155637Z-V7-ffn-async-slow` | 0; 815,549 s | Host 6,2 ns/motor 5 ns: mismas tres muestras y comparación exacta; tiempo absoluto y flancos de ambos dominios revisados. |
| `20261003T155637Z-V7-ffn-async-synth` | 0; 487,065 s | Conjunto CDC/FFN: 44.193 LUT/44.819 FF/93 DSP, 39 RAMB36 + 3 RAMB18; WNS estimado +0,735 ns, WHS +0,013 ns. CDC-15 y TIMING-37 pendientes. |
| `20261003T161520Z-V7-ffn-async-route` | 0; 786,281 s | 43.857 LUT/44.596 FF/93 DSP, WNS +0,177 ns/WHS +0,010 ns; 79.143 nets ruteables completas. Route 35-470 ignora un grupo de bus skew: condición abierta. |
| `20261003T162839Z-V7-ffn-async-route-audit` | 0; 98,738 s | Dos huecos de reloj padre: 1.185/146 alias, ningún otro PARTIAL. Fanout RRESP acotado a dos bits del mismo handshake. Márgenes OOC, no cierre de placa. |


| Revisión RRESP | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T164300Z-V7-host-recode-sim` | 0; 72,928 s | 54 casos/24 resets; 27 casos/12 resets usan AXI-Lite recodificado. Trazas externas idénticas; 36 mutaciones rechazadas. |
| `20261003T164300Z-V7-host-recode-corrupt` | 1 esperado; 41,721 s | Detecta un bit RRESP alterado después de decodificar, caso 3 tras reset. |
| `20261003T164514Z-V7-ffn-recode-brandon` | 0; 797,301 s | Tres FFN/768 FP32 exactos; todas las trazas externas, ciclos e IRQ idénticos a la referencia 4/5 ns. 17 mutaciones rechazadas. |
| `20261003T164514Z-V7-ffn-recode-slow` | 0; 814,207 s | Misma comprobación en 6,2/5 ns; 17 mutaciones rechazadas. |
| `20261003T164514Z-V7-ffn-recode-synth` | 0; 492,116 s | 44.194 LUT/44.819 FF/93 DSP, una LUT adicional; TIMING-37 eliminado sin cambiar XDC AMD. CDC-15 pendiente. |
| `20261003T165611Z-V7-ffn-recode-route` | 0; 753,579 s | 44.038 LUT/44.607 FF/93 DSP; WNS +0,068/WHS +0,010 ns; 79.143 redes completas. Sin TIMING-37/Route 35-470; 15 grupos bus skew con margen. |
| `20261003T170857Z-V7-ffn-recode-audit` | 0; 99,958 s | 33 fuentes variables con destino único y captura bit32 a GND. Dos huecos de reloj, 1.186/147 alias; doce mutaciones de evidencia rechazadas. |

Las siete tandas terminaron, fueron revisadas y tienen respaldo local/NFS
verificado con todos los miembros y fuentes. Véase
[revisión RRESP](../FFN_ASYNC_INTEGRATION.md).

| Transporte IRQ aislado | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T172351Z-V7-irq-transport-sim` | 1; 12,575 s | Fallo Tcl antes de XSIM; conservado y corregido. |
| `20261003T172457Z-V7-irq-transport-sim-r2` | 1; 28,968 s | Ocho errores XPM S-2 al retirar envío durante reset; marcador del banco insuficiente, tanda rechazada. |
| `20261003T172721Z-V7-irq-transport-sim-r3` | 0; 49,901 s | 45 casos/24 resets; 69 eventos, 48 retiros, 21 abortos; 45 mutaciones de evidencia rechazadas. |
| `20261003T173040Z-V7-irq-transport-corrupt` | 1 esperado; 28,208 s | Detecta IRQ retirada después de ACK de transporte y antes de atención por MMIO. |
| `20261003T173051Z-V7-irq-transport-synth` | 0; 83,284 s | 78 LUT/486 FF, sin DSP/BRAM; WNS estimado +2,467/WHS +0,077 ns. Cinco CDC-3 visibles y dos waivers CDC-15 internos AMD. |
| `20261003T173352Z-V7-irq-transport-audit` | 0; 49,479 s | 131 FDRE INIT=0; constraints idénticos al reabrir checkpoint. Sin ruteo, relojes padre/fronteras pendientes. |

Las seis tandas tienen evidencia y respaldo local/NFS completos. El
[bloque IRQ](../IRQ_TRANSPORT.md) se conecta ahora a un CSR opcional ABI2;
La integración posterior con FFN y el banco pequeño XDMA se registran abajo.

| Control e IRQ ABI2 | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T175212Z-V7-ffn-irq-control-sim` | 1; 82,069 s | Watchdog por force/release del banco; conservado. Sustituido por pausa mediante ACK host. |
| `20261003T175212Z-V7-ffn-control-abi1-regression` | 0; 49,655 s | ABI1 FP32: 27 grupos, 2.220 líneas idénticas a la referencia; 21 mutaciones rechazadas. |
| `20261003T175212Z-V7-ffn-control-abi1-fp64-regression` | 0; 49,637 s | ABI1 binary64: misma cobertura y comparación exacta. |
| `20261003T175349Z-V7-ffn-irq-control-sim-r2` | 0; 54,688 s | 54 casos/27 resets, 57 retiros y 18 abortos; 798 escrituras/291 lecturas reconstruidas. Aviso de declaración implícita acotado al banco. |
| `20261003T175413Z-V7-ffn-host-abi1-regression` | 0; 391,349 s | FFN/HLS real con ABI1: 63 casos, 71.232 B numéricos y 9.298 líneas idénticas a la referencia. |
| `20261003T175525Z-V7-ffn-irq-control-corrupt` | 1 esperado; 30,727 s | Detecta retirada prematura de IRQ tras ACK PCIe y antes de atención del software. |
| `20261003T175525Z-V7-ffn-irq-control-synth` | 0; 268,517 s | 2.093 LUT/4.927 FF; WNS estimado +1,988/WHS +0,066 ns. Sin DSP/BRAM; fronteras/relojes padre pendientes. |
| `20261003T180052Z-V7-ffn-irq-control-audit` | 0; 51,070 s | 131 FDRE INIT=0; cinco CDC-3, dos waivers internos CDC-15 AMD y constraints idénticos. |
| `20261003T180226Z-V7-ffn-irq-control-sim-r3` | 0; 54,773 s | Banco final: 54 casos PASS, 45 alteraciones rechazadas, sin aviso de declaración implícita; 24.715 líneas idénticas a r2. |

Las nueve tandas nuevas están respaldadas y verificadas. La
[ABI 2.0](../FFN_IRQ_CONTROL.md) conserva finalización e identidad hasta
atención del software y retiro del transporte. La tanda siguiente propaga
la opción a la FFN asíncrona, manteniendo ABI1 por defecto.

| FFN completa con ABI2 | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T181642Z-V7-ffn-irq-async-errors` | 0; 476,914 s | 63 casos PASS, host 4 ns, errores/cancelación/reset; 57 alteraciones rechazadas. |
| `20261003T181652Z-V7-ffn-irq-wrapper-legacy` | 0; 397,386 s | 63 casos ABI1 PASS; 9.298 líneas NTRPU idénticas a la referencia. |
| `20261003T181921Z-V7-ffn-irq-async-errors-slow` | 0; 479,129 s | 63 casos PASS, host 6,2 ns; 57 alteraciones rechazadas. |
| `20261003T181921Z-V7-ffn-irq-brandon` | 0; 829,373 s | Tres FFN Brandon, 768 FP32 exactos; carga 1.777.152 B y lectura 3.072 B por host; 27 alteraciones rechazadas. |
| `20261003T181921Z-V7-ffn-irq-brandon-slow` | 0; 833,702 s | Misma prueba con host 6,2 ns: tres FFN/768 FP32 exactos y 27 alteraciones rechazadas. |
| `20261003T181949Z-V7-ffn-irq-async-corrupt` | 1; 86,789 s | Fallo esperado: IRQ retirada tras ACK transporte antes del servicio; detectado en primera FFN. |
| `20261003T181949Z-V7-ffn-irq-async-synth` | 0; 551,078 s | 44.585 LUT/45.370 FF/93 DSP; WNS estimado +0,768/WHS +0,013 ns; 17 grupos bus skew. |
| `20261003T183700Z-V7-ffn-irq-async-route` | 0; 803,280 s | 44.472 LUT/45.219 FF/93 DSP; 80.297 redes completas, WNS +0,104/WHS +0,010 ns; margen bus skew mínimo+3,389 ns. |
| `20261003T185027Z-V7-ffn-irq-async-audit` | 0; 100,720 s | 131 FDRE IRQ INIT=0; 1.192/155 alias de relojes padre, 33 fuentes RRESP únicas; 15 alteraciones rechazadas. |

Las nueve tandas tienen respaldo local/NFS con hashes y todos los miembros
verificados. La [integración ABI2](../FFN_IRQ_INTEGRATION.md) conserva el
motor, HLS y CDC AMD. Sus resets durante cómputo usan matrices pequeñas;
Brandon cubre carga, cálculo y lectura con reset inicial y par de protocolo
para ACK. El banco posterior incorpora XDMA/MSI-X con matrices pequeñas;
supervisor y recuperación DDR siguen pendientes. Ruteo OOC, sin cierre de placa.

| PCIe/XDMA, FFN y MSI-X | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T192048Z-V7-pcie-ffn-msix` | 1; 34,922 s | Fallo Tcl al asignar un valor iniciado en guion; corregido con `-dict`, sin simulación. |
| `20261003T192229Z-V7-pcie-ffn-msix-fixed-launch` | 1; 593,951 s | Enlace/BAR correctos; MSI-X deshabilitado en XCI de entrada. Sin DMA/cómputo. |
| `20261003T193438Z-V4-pcie-pipe-msix-example` | 0; 90,984 s | Nueva IP MSI-X habilitada, 32 vectores, tabla/PBA en BAR1; generación revisada, sin prueba funcional en esta tanda. |
| `20261003T193636Z-V7-pcie-ffn-msix-enabled` | 1; 486,104 s | MSI-X habilitado; primera AW rechazada por modo fijo activado por el controlador del banco. |
| `20261003T194619Z-V7-pcie-ffn-aw-diagnosis` | 0 diagnóstico; 208,567 s | Reproduce fatal esperado: AWBURST=0 a ambos lados del CDC. Binarios y diez transacciones idénticos; no PASS funcional. |
| `20261003T194753Z-V7-pcie-ffn-incrementing` | 1; 492,443 s | Modo INCR corregido; 640 B exactos. Solapamiento MMIO/DMA rompe buffers compartidos RQ/CC del controlador raíz. |
| `20261003T200010Z-V7-pcie-ffn-serialized` | 0; 961,264 s | PASS con fases separadas: H2C 3.328 B, cómputo 5.424 ciclos, C2H 128 B/32 FP32 exactos; dos ACK/MSI-X, tres retiros y 25 alteraciones de evidencia rechazadas. |

La [revisión PCIe/FFN](../PCIE_FFN_INTEGRATION.md) verifica además 204
escrituras/90 lecturas CSR, un rechazo por identidad incorrecta y los
offsets MSI-X anunciados. Los 134 archivos HDL/IP AMD y el RTL NTRPU
conservan hashes. Los seis avisos añadidos al PCIe previo quedan revisados:
cinco por FIFO conductual y uno por puerto de consulta MSI-X de tabla
interna deshabilitada. No califica máscaras PCIe de función/vector/PBA,
concurrencia MMIO/DMA, supervisor ni DDR.

Las siete tandas PCIe/FFN tienen respaldo local/NFS verificado. La aprobada
conserva 1.128 archivos, 230.159.372 bytes comprimidos y SHA-256
`dd64c548d99608cd4dc022de0bc8e85bea5ada07e246ffb0466d44a452a3c170`;
incluye revisores y evidencias bajo `review/`.

| Registro concurrente PCIe | Salida y tiempo | Evidencia |
| --- | --- | --- |
| `20261003T202838Z-V4-pcie-concurrent-log` | 0; 72,801 s | 16 paquetes/2.256 bytes exactos; ocho negativos y reproducción del estado compartido anterior. Etiqueta temporal local ns, conservada como diagnóstico. |
| `20261003T203150Z-V4-pcie-concurrent-log-time` | 0; 72,338 s | Misma cobertura con tiempo normalizado a ps; 14 productores separados RQ/CC sin cambiar AXIS/parser. |
| `20261003T203427Z-V7-pcie-ffn-concurrent` | 1; 507,017 s | H2C 3.328 B exactos y diez MMIO solapados; falla al exigir estado 0x6 frente a 0x4. Sin FFN iniciada; respaldo verificado. |
| `20261003T204611Z-V7-pcie-ffn-stop-poll` | 0; 1.154,231 s | Numérica correcta; diagnóstico sin aprobación por callback antiguo. Respaldo verificado. |
| `20261003T204611Z-V7-pcie-ffn-stop-poll-fast` | 0; 1.165,428 s | Mismos 932 eventos, sin aceleración observada; mismo límite del callback. Respaldo verificado. |
| `20261003T205059Z-V4-pcie-completion-callback` | 0; 89,658 s | Cuatro casos prueban callback RX únicamente; diez negativos y 16 paquetes/2.256 bytes exactos. Respaldo verificado. |
| `20261003T205405Z-V7-pcie-ffn-rx-only` | 0; 1.068,218 s | PASS revisado del sondeo concurrente: 32 FP32 exactos, 5.431 ciclos; MMIO observado durante ambos sentidos DMA. Respaldo verificado. |
| `20261003T210329Z-V4-pcie-rx-log` | 0; 76,261 s | PASS del registrador CQ/RC separado: 16 paquetes/2.304 bytes, ocho callbacks y ocho negativos. Reproduce la mezcla anterior; respaldo verificado. |
| `20261003T210622Z-V7-pcie-ffn-rx-split` | 0; 958,987 s | PASS de integración: 25 CQ/143 RC cotejados, 33 alteraciones rechazadas; datos/ciclos conservados. Respaldo verificado. |
| `20261003T212322Z-V7-pcie-brandon-host` | 0; 74,748 s | Host sirve datos correctos, pero el revisor rechaza el watchdog truncado por VRFC 10-8884. Intento respaldado. |
| `20261003T212632Z-V7-pcie-brandon-host-wide-time` | 0; 74,750 s | PASS: 592.544 bytes con descriptores, 2.320 peticiones, seis CRC32 y ocho negativos. Watchdog corregido; respaldo verificado. No ejecuta FFN/PCIe. |
| `20261003T212950Z-V7-pcie-ffn-brandon` | 0; 6.714,865 s | PASS revisado: 256 FP32 exactos, 307.291 ciclos, 592.384 B H2C y 1.024 B C2H; MMIO concurrente/MSI-X; 33 negativos. Respaldo verificado, 1.147 archivos. |
| `20261003T214142Z-V7-ffn-phase-profile` | 0; 837.369 s | PASS: 12 etapas; 768 FP32 y toda la traza temporal idénticos a la referencia. MAC ocupados 86–88%; SwiGLU HLS ≈99%; 12 alteraciones rechazadas. [Medición](../FFN_PERFORMANCE.md). |

Al corte 2026-10-04 04:54 UTC hay **384 tandas con respaldo local
verificado**, incluido un rechazo anterior al arranque de Vivado
(`20261004T003148Z`), que no constituye una simulación. PCIe Brandon dual,
los bancos FIFO de GEMV/FFN, Brandon y síntesis
ya están revisados y respaldados. El ruteo FFN con cola Q8 falla setup;
su revisión, auditoría y respaldos están completos.
SwiGLU con pipeline pasa memoria, FFN pequeña, Brandon y síntesis:
34,02–34,09% menos ciclos FFN, con los mismos resultados y ocho DSP más.
Su ruteo pasa timing interno OOC (WNS +0,046/WHS +0,009 ns) y la prueba
corta PCIe pasa con 32 FP32 exactos/1.809 ciclos; ambos están respaldados.
Los 126 casos ABI2 quad/FIFO a dos relojes también pasan y están respaldados.
Los dos bancos Brandon ABI2 pasan y están respaldados. Su ruta inicial
se conserva como FAIL: un endpoint de setup a −0,003 ns. La optimización
posterior y su auditoría pasan a +0,003 ns, sin relajar constraints.
Brandon quad por PCIe, el supervisor aislado y los nueve intentos de la
isla IRQ supervisada también están respaldados. Los seis intentos de la
FFN supervisada añaden sus manifiestos y revisión; el primer perfil lento
conserva su rechazo independiente. La síntesis supervisada posterior y su
auditoría están también revisadas y respaldadas. Se añaden el ruteo de
tres relojes y sus dos auditorías, ambos bancos de pérdida automática de
reloj y los dos intentos de extracción de conectividad RTSTAT-10. Los cuatro
intentos de recuperación con FFN tras cada pérdida también están respaldados;
el primero rápido mantiene su rechazo de cobertura. Los dos bancos RELEASE
posteriores siguen en ejecución y quedan fuera de este recuento.
Todas estas tandas son terminales; se conserva la dependencia de integración
con pares reales y placa. No se apagó ni reinició la VM.
El control dual Brandon y el ruteo quad
terminaron: la comparación numérica pasa; el ruteo completo falla setup.
Cualquier fallo exige revisión y no se aprueba automáticamente.
Los bancos cortos incluyen manifiesto, revisión
y negativos esperados; no califican por sí solos el DMA. El fixture Brandon
para PCIe tiene respaldo separado; su integración completa pasa con RAM conductual.
El histórico de tandas terminadas incluye las
medidas escalar/dual, BRAM/ruteo dual, formas máximas, empaquetador del host,
su prueba negativa y la regresión de errores/reset. Las tandas de activaciones
añaden cargador, wrapper GEMV, síntesis, imágenes y control negativo.
Las trece posteriores añaden escritor, conjunto de memoria a memoria,
netlist, regresión, cuatro fases de reset, ruteo y auditoría del reloj;
conservan todos los fallos y sus correcciones en tandas separadas.
Las dieciséis más recientes incorporan matrices completas, el fallo temporal
inicial y su corrección, ruteo con auditoría, las dos formas completas Llama
con datos sintéticos y cinco tandas del operador local SwiGLU/Q8.
Todas están terminadas y respaldadas; los fallos conservan su clasificación.
Las nueve posteriores validan el escritor ABI1 en RTL, síntesis y netlist,
el adaptador AXI inicial, la regresión con gate variable, el vector máximo
de 8192 elementos y su ruteo con auditoría. Todas están terminadas y
respaldadas; los márgenes OOC no equivalen a cierre de placa.
Las siete siguientes incorporan el controlador FFN, la cadena sintética con
memoria compartida, su síntesis/ruteo y dos auditorías de reloj equivalentes.
Se conserva la auditoría exhaustiva original y se reduce su coste de memoria
y tiempo en el script vigente. La octava añade tres FFN completas con pesos
reales de Brandon, comprobación independiente y respaldo verificado;
no se infiere de las formas sintéticas ni equivale a inferencia completa.
Las cuatro siguientes incorporan el [conversor FP32](../FP32_OUTPUT.md),
su síntesis y replay funcional, conservando el fallo inicial del banco y la
errata de finales de línea del fixture. Las diecinueve posteriores integran
la salida FP32 al escritor y la cadena FFN, con regresión binary64 y tres
muestras de pesos reales de Brandon. Conservan la primera integración,
cuatro intentos de optimización rechazados y la corrección validada en
RTL/síntesis/netlist. Las dos siguientes implementan y auditan el ruteo FP32,
con timing interno positivo y reloj/fronteras todavía sin calificar.
Ambas están respaldadas y cuentan en el total. El archivo del ruteo ocupa
79.106.555 bytes, SHA-256
`e2d248dde6aba3846fe33e1c6f8517a3ab864b1b5dba194ad10bd0fbff7ef01c`;
la auditoría, 33.736.517 bytes, SHA-256
`4e7b97c6853872c7488e42c012cc3cfae8ffa707afd89ecf74ebbc45db8bd2f5`.
La siguiente valida el banco virtual de memoria con una cadena corta antes
de ejecutar dimensiones completas. [FFN_LARGE](../FFN_LARGE.md) registra
los contratos numéricos y distingue expectativas de medidas.

[Control host del FFN](../FFN_HOST_CONTROL.md): ABI AXI-Lite experimental,
27 grupos de prueba por formato FP32/binary64 y síntesis de bloque aprobada
con fronteras pendientes. Preserva las direcciones de 64 bits, captura el
descriptor, retiene la finalización y ofrece cancelación e IRQ. El par de
cómputo del banco unitario es simulado a nivel de protocolo; esa prueba
conserva su alcance y se complementa con la integración indicada abajo.
Sus cuatro tandas, incluido el fallo inicial del lanzador, están respaldadas.
La [integración host → FFN/HLS](../FFN_HOST_INTEGRATION.md) ya pasa 63 casos
contra memoria simulada, con descriptor, cancelación y finalización por MMIO.
Conserva por separado la regresión directa y aún requiere PCIe/DMA/DDR.
Tres FFN con pesos Brandon y salida FP32 pasan también por MMIO; el
conjunto está sintetizado con 41.802 LUT/93 DSP, con timing estimado positivo.
Su implementación conserva 41.536 LUT/93 DSP, WNS +0,099 ns y WHS +0,010 ns;
la auditoría mantiene explícitos el hueco del reloj padre y las fronteras sin
restricciones. Las seis tandas de integración están respaldadas y no cierran V7
sobre DDR real ni el timing de placa.

Las once tandas de conversores añaden generación, simulación corregida,
negativo de época y cuatro síntesis. Todas tienen copia local verificada,
incluidos los intentos rechazados; los detalles están en
[HOST_CLOCK_INTEGRATION](../HOST_CLOCK_INTEGRATION.md).
El ruteo de dos entradas y su auditoría quedaron terminados y respaldados.
Las simulaciones PCIe posteriores se enumeran arriba con su estado vigente;
la VM no fue apagada.

La FFN larga terminó y se respaldó sin reiniciarla. Los conversores AMD
aislados ya pasan el banco de relojes/reset y tienen síntesis revisada.
Los conversores y la ABI2 ya están conectados al fabric y pasan carga, FFN y
lectura Brandon. XDMA/MSI-X pasa un fixture pequeño con RAM conductual;
siguen pendientes ampliar intercalados CQ/RC y Brandon, supervisor y un
CDC-15 del FIFO por revisar.

Los PID sirven para localizar procesos: verificar comando e instante de
creación antes de actuar. Un timeout de observación no es una terminación.
Incluyen DMA host64,
implementación PCIe aislada, ruteos conjuntos, auditoría temporal y alternativa
3DS. Explore antes de rutear cumple las restricciones actuales; los avisos y
contratos pendientes se detallan en [coexistencia](INTERFACE_COEXISTENCE.md).
Tres tandas anteriores tienen archivos verificados de 111.410.362 bytes
(DMA, SHA-256 `be46b64a04fc85d844265f59439cab345739188f230c661d45a7e13656dbaa3a`),
12.195.255 bytes (HLS FFN, `c23510f3a09fbcb26d3eaa30007977f0d5541c708e99e28739d80fa9b0f257db`)
y 67.883.564 bytes (síntesis Vivado FFN, `d684849a4cca1e53d8774feedae52b1e58828b5047a67ee30e9c84e0d5141816`).
Cada copia contiene fuentes y revisión; no se alteró la validación ZU7EV previa.
También están verificados los C/RTL VU5P: FFN corto, 85.314.787 bytes,
SHA-256 `cddbf74f63ec62490af7cc869bfa72cdffc30c01b4114f857aab079a49454c91`;
MAC inicial, 44.193.912 bytes, `812763f0699ca787bfafc50c1677c51b215d1773cfb4f76d5607e578bd28c66a`;
MAC optimizado, 37.691.700 bytes, `eecbd67609a2062db7055f7e6f8ce07e7af5bb33a47004c65f6923c6130b0c96`.
El GEMV Brandon corregido ocupa 37.586.153 bytes, SHA-256
`f8c8d27060f2724234d5b3719d1bab97b85b2083d36923f6191925911b46ed36`;
ruteo OOC, 20.447.038 bytes,
`4c7fff47052a49a627b1e5ab20a93e2f26c7c0326df08bf43462b80f1c9feb3f`;
auditoría del reloj, 11.949.876 bytes,
`503ff758f8de0a47b56407e1434f271a0423bd0cb1bb4e303073700bde5bfe2b`.
Los cinco RTL sintetizables del escalar permanecen intactos. Los cinco del
camino dual, incluido su nuevo controlador, coinciden byte a byte entre
síntesis BRAM, Brandon, banco ampliado y la entrada al ruteo; los cambios
posteriores fueron del banco y documentación. `gemv-rtl-identity.json`
registra ambas comparaciones en el índice de entrega.
El cargador y wrapper nuevos tienen hashes separados, idénticos entre su
síntesis y simulaciones; no se les atribuye el ruteo del núcleo dual previo.
El escritor y wrapper de escritura también coinciden entre síntesis,
bancos e imágenes. El fixture de resultados ocupa 43.445.564 bytes,
SHA-256 `0b497196ba04e9b6bc8857f44f8c55178ef8e023e02d6f1de3a002c539f11aa4`;
la netlist funcional 6.022.304 bytes,
`1de00fe825496e3ed0afe814d496bf0be086a4f3dae7cf5822458fa9939b9325`.
El ruteo del conjunto ocupa 37.822.266 bytes,
`09ea78f0022819c2e0a88fee408c54856704eeb61390a45796b7cee422dc73a5`;
su auditoría 25.974.872 bytes,
`bdb5bf3930717400489a226dda9913560a0495cf4389b4461a1b76c317ec1961`.
El seguimiento
automático aplica una copia fijada de `check_ntrpu_pcie_multi.py`, conserva fallos,
comprueba procesos activos antes de archivar y verifica comprimido, miembros y
fuentes tras copiarlo. Las descargas grandes usan bloques de 8 MiB y comprueban
el hash final para tolerar cortes de conexión. Los campos de revisión manual
pendiente del archivo automático serial y del ensayo con pausas quedan resueltos
en [PCIE_DMA_VALIDATION.md](PCIE_DMA_VALIDATION.md), sin modificar sus archivos.

Los fallos son diagnósticos conservados; generar un ejemplo no equivale a una prueba funcional. Los 15 cambios de representación BANK entre ZIP y Vivado corresponden a pines dedicados: las funciones coinciden; Vivado les asocia bancos donde el ZIP indica NA. El comparador revisado rechaza diferencias desconocidas y pasó el inventario real y tres comprobaciones negativas de función, banco y SLR.

La generación DDR fue contrastada con los parámetros del wrapper, no solo con el código de salida: `ROW_WIDTH=18`, `COL_WIDTH=10`, `RANKS=4`, `ECC="ON"`, dos slots y 137438953472 bytes útiles. El comprobador rechaza cambios de fila, ancho AXI, slot y ECC. La configuración exacta Micron sigue sin aprobarse. El comprobador PCIe exige enlace Gen3 x16, comparaciones H2C/C2H, monitor independiente, conteos y ausencia de errores; once mutaciones negativas probaron su rechazo, incluidas unidades de tiempo. El `dma_test0` original mueve solo 16 bytes: su PASS es una prueba básica, no caudal sostenido.

La tanda PCIe instrumentada terminó con **PASS**: el root port confirmó **8,0 GT/s y x16** a 77,69248 µs simulados. H2C y C2H compararon 16 bytes cada uno; hubo una transacción y un beat AXI por sentido, respuestas OKAY y un descriptor completado por canal. Primer beat de escritura a 118,332405 µs y de lectura a 125,932405 µs; reloj AXI de 250 MHz. Respaldo de 250.802.939 bytes, SHA-256 `fe593f40dbbcabf8e18a7a1824139796e7a3b97fcf5f7081907539299e6438f9`, verificado localmente. El aviso inicial de `unique case` y los avisos de elaboración AMD permanecen en la evidencia.

La prueba DDR genérica calibró a **28,051098 µs** y terminó a **46,788498 µs** simulados: 3747 ciclos de UI, 64 beats escritos, 88 leídos y 5632 bytes comparados; 100 ciclos de pausa W y 28 de pausa R. Hubo 90/4/2/10 activaciones en ranks 0/1/2/3. Es cobertura dispersa de bits de dirección 6–36, máscaras y dos ráfagas de 512 bytes que terminan en 4 GiB y 128 GiB; no un barrido exhaustivo ni caudal sostenido. El verificador rechazó 15 alteraciones negativas del log. Véanse [adaptación del modelo y límites](MEMORY_QUALIFICATION.md).

La capacidad DDR tiene respaldo local de 49.140.285 bytes, SHA-256 `a3cd65804c78c9093a4a15b1596305d427100cb28f60299466c56bb75f9f722c`; `VERIFIED.json` confirma los 524 archivos y la copia de fuentes. El ensayo ECC ocupa 49.179.603 bytes, SHA-256 `d67c81ef47d811040967e739ea2bab5260af8670c59bf52f6e36b9b4d493afd7`, con 526 archivos verificados. Terminó con 68 beats escritos, 92 leídos y 5824 bytes comparados a 48,408498 µs simulados; el dato no corregible queda excluido del conteo de comparaciones.

**Unidades del monitor:** el snapshot PCIe original usa `%t`; XSIM declara resolución de 1 fs, por lo que sus campos llamados `*_ps` contienen valores en fs. `check_ntrpu_pcie_sim.py --legacy-time-format-fs` conserva los valores originales y divide explícitamente por 1000. La revisión v2 del HDL imprime tiempos decimales en la unidad local de 1 ps y anuncia su formato; se aplicará al siguiente banco compilado. No modificar ni recompilar la simulación activa para corregir solo la presentación.

## Hallazgos PCIe que condicionan la placa

El ejemplo selecciona `X1Y0`, quad227, Gen3 x16, DMA AXI de 64 bits de dirección/512 de datos y una ventana AXI-Lite solicitada de 64 KiB. Los XDC fijan el bloque físico `PCIE40E4_X1Y0` y los GT `GTYE4_CHANNEL_X1Y0` a `X1Y15`. La netlist serial confirma orden lógico inverso: lane0→X1Y15 hasta lane15→X1Y0.

La enumeración simulada asigna **BAR0 y BAR1 MEM32 de 64 KiB** cada uno; BAR2–BAR5 desactivados. El identificador `0x1fc00006` y las transferencias sitúan los registros XDMA en **BAR1**, base simulada `0x00010000`; no asumir BAR2. En el ejemplo inicial BAR0 devuelve cero en offset 0. La integración PCIe/FFN posterior conecta esa ventana al CSR ABI2 y verifica su identificación, configuración y finalizaciones. Las bases las asigna este root port de simulación; el host físico puede elegir otras.

`xilinx_xdma_pcie_x1y0.xdc` usa AM17/PERSTN0 con LVCMOS18 y pull-up. La prueba sintética de banco65 combina ese reset con una salida SSTL12_DCI: **BIVC-1 confirma el conflicto VCCO 1,8 V frente a 1,2 V**. No es todavía un MIG ni un diseño PCIe implementado. El modelo generado tiene `DEDICATE_PERST=false`; eso no autoriza un cambio arbitrario de bola. El reloj de ejemplo usa REFCLK0 asociado a `GTYE4_CHANNEL_X1Y7`. Revisar las excepciones y waivers del ejemplo al integrar.

La segunda prueba mantiene AM17 y cambia solo su estándar a LVCMOS12: **cero violaciones BIVC-1** junto a la salida sintética SSTL12_DCI en BE20. Es evidencia limitada de compatibilidad VCCO, coherente con el [soporte LVCMOS12 en HP de AMD](https://docs.amd.com/r/en-US/ug861-ultrascale-selectio/Low-Voltage-CMOS-LVCMOS). No valida traducción de nivel del host, temporización de reset, disponibilidad de AM17 bajo MIG ni otros DRC. Conservar este candidato para el diseño eléctrico y la prueba conjunta.

Vivado confirma GT X1Y0–3 en quad224, X1Y4–7 en quad225, X1Y8–11 en quad226 y X1Y12–15 en quad227. El refclock del ejemplo resuelve a **AL9/AL8, REFCLK0 del quad225**. La selección de quad227 no implica que el refclock esté en ese quad. El orden lógico ya está trazado; el pinout definitivo requiere colocación/ruteo e integración conjunta.

La [traza sintetizada](PCIE_LANE_TRACE.md) confirma esa inversión y coteja las
64 bolas con sus funciones, P/N y RX/TX. Conserva 61.778 LUT, 58.653 registros
y 205 BRAM del endpoint de ejemplo. No incluye colocación/ruteo ni DDR conjunta.

Mantener Tandem desactivado. Definir traducción eléctrica de PERST, secuencia configuración/reloj/enumeración, rangos DMA, escritura de pesos, colas/épocas, interrupciones y recuperación. Una interfaz AXI de 64 bits no valida por sí sola los límites de los cuatro canales. Un canal genérico y PCIe ya están ruteados juntos; la conexión DMA→DDR, cuatro canales y validación de placa siguen pendientes.

## Próxima tanda de diseño

La síntesis incluye el generador de tráfico y calibración del ejemplo AMD; no mide recursos del motor NTRPU ni posrutado. [PIN_PLANNING.md](PIN_PLANNING.md) registra ocupación, DCI y diagnósticos pendientes. `io_plan.json` conserva las reservas originales de tres bancos, aún sin validar.

Ambos ensayos de reset son **FAIL de revisión**, aunque XSIM termina con código 0 y el banco compara correctamente sus datos. El modelo informa 558 errores después del reinicio. Alargar el pulso externo no lo resuelve. Los PASS previos de capacidad y ECC mantienen su alcance independiente.

Las trazas reutilizan el snapshot compilado sin alterar el modelo. Los 72 chips reciben el mismo pulso bajo de 2,597 µs y 735 flancos positivos durante ese intervalo. Hay una pausa de reloj de 1,678875 µs, pero el reloj vuelve antes de liberar reset. La hipótesis de reset desconectado o ausencia completa de reloj queda descartada en esas interfaces. El ensayo aislado reproduce errores con una fila abierta incluso con reloj continuo y reset de 200 µs; acota la secuencia/modelo, sin atribuir un fallo físico. La parada diagnóstica a 55 µs produce `INCOMPLETE` intencionalmente y no se usa como PASS funcional.

El [arranque DCI](DCI_STARTUP.md) está probado e integrado. El canal genérico termina ruteo con 28.805 LUT, 34.962 registros CLB, 26,5 BRAM y 3 DSP. El ajuste BACKBONE conserva reloj dedicado entre bancos 64 y 66. El ruteo resuelve las 499 violaciones de hold estimadas tras colocación: **WNS +0,334 ns, WHS +0,011 ns y ancho de pulso +0,123 ns** bajo las restricciones presentes. Las 70.578 nets enrutables están completas, sin errores de ruteo. No se añaden excepciones ni retardos supuestos.

Quedan un puerto de entrada y tres de salida sin retardos, siete LUTAR-1 y ocho avisos DRC de LUT/debug/buffers sin carga. El modelo sigue fallando en reset y el DIMM exacto no está calificado. Este resultado implementa un canal genérico con el ejemplo de tráfico; no cierra timing de placa, convivencia PCIe, cuatro canales ni V5/V6.

1. Completar el contrato eléctrico de las fuentes de reset y de los cuatro puertos sin retardos. Topología ruteada y 591 liberaciones digitales comprobadas; MTBF/glitches analógicos siguen sin calificar.
2. Contrastar la reproducción aislada de reset con el contrato del modelo y un componente soportado; mantener el ensayo completo FAIL mientras haya errores.
3. Completar V2 y validar coexistencia PCIe/PERST antes de V5/V6. Comparar tres bancos con la alternativa colocada de cuatro; `place_ports` no demuestra imposibilidad de la terna.
4. Ampliar el banco PCIe/FFN a concurrencia MMIO/DMA y Brandon. La prueba pequeña con MSI-X y fases separadas ya pasa; mantener las dependencias DDR para cerrar V7.

La ampliación DMA de **768 bytes por sentido pasó en serial y PIPE** con buffers separados,
completador de host corregido y scoreboards AXI/CQ. Hubo 12 beats W y 12 R,
tres AW/tres B, un AR, tres paquetes por sentido y dos descriptores completados.
El verificador CQ pasó una prueba positiva y siete negativas. La revisión Python
distingue código Tcl impreso de errores ejecutados y pasó siete pruebas.
El gate de pausas pasó su banco independiente y el DMA completo con pausas W/R
también pasó. Véanse [evidencia, tiempos,
correcciones y límites](PCIE_DMA_VALIDATION.md).

## Validación anterior conservada

La FFN terminó sin interrupción: `20261002T062547Z-cosim-sc8kn0wf`, **PASS 23/23**, incluida una muestra real de Brandon. Diferencia máxima frente a Q8: cero; error relativo L2 frente a FP32: 1,14920248439%. Archivo de 95.387.074 bytes, SHA-256 `a701147da9016fb8706d5851ae6e328dd4ed477e8370c54980ad0d0287aaa9a3`, verificado en NFS y `artifacts/ntrpu-rtl-20261002/`. Véanse [resultados y límites](../HLS_FFN.md). No es inferencia completa ni timing posrutado.

## Integración asíncrona: referencia inicial conservada

Las [dos simulaciones FFN, síntesis y ruteo exploratorio](../FFN_ASYNC_INTEGRATION.md)
están terminados y respaldados localmente con todos los miembros verificados.
Se conservan las validaciones previas. El ruteo informa
WNS +0,177 ns/WHS +0,010 ns y 79.143 redes enrutables completas. La auditoría
identifica dos huecos de reloj padre (1.185/146 alias), fronteras sin
restricciones y fanout de dos bits RRESP desde un mismo registro.
**Route 35-470: el router ignora el grupo de bus skew de respuesta R**,
aunque el análisis posterior informa margen. No se marca cierre temporal;
se conserva TIMING-37, CDC-15 y el alcance OOC. Corregir o justificar ese
cruce antes de integrar XDMA, supervisor y recuperación de trabajos.

## Integración asíncrona: corrección RRESP revisada

La nueva revisión conserva las trazas externas de ambos perfiles FFN y los
códigos de error del CSR. El ruteo y su auditoría independiente confirman
la eliminación de TIMING-37/Route 35-470, sin cambios en las restricciones
AMD. Las 33 fuentes variables del grupo R tienen un destino cada una;
la captura restante recibe cero. Los quince grupos bus skew cumplen,
con margen mínimo +3,281 ns. El revisor rechaza doce corrupciones deliberadas.

Persisten CDC-15, dos huecos de reloj padre y fronteras sin restricciones;
los márgenes internos no cierran timing de placa. La [FFN ABI2](../FFN_IRQ_INTEGRATION.md)
ya retiene ambas confirmaciones y asegura el rearme, con resets coordinados
durante cómputo. La integración PCIe pequeña añade XDMA/MSI-X; faltan
supervisor y reset del conjunto con DDR. Los ensayos originales con avisos permanecen conservados.
El transporte IRQ aislado ya cumple su banco y auditoría; su reset drena
los handshakes XPM en lugar de retirar envíos antes de la confirmación.

## Variante de cuatro MAC (2026-10-03)

Ver [arquitectura, casos y comparación de ciclos](../Q8_GEMV_QUAD.md).
El perfil justifica replicar los MAC; el dual sigue siendo predeterminado.
Las métricas de GEMV aislada no se extrapolan a tokens/s ni a DDR física.

| Run ID | Estado / evidencia |
| --- | --- |
| `20261003T220450Z-V7-gemv-quad-extended` | Numérica exacta; reset no alcanzaba el arranque HLS de los cuatro motores. Diagnóstico respaldado. |
| `20261003T220450Z-V7-gemv-dual-extended` | Mismo límite del estímulo de reset. Diagnóstico respaldado. |
| `20261003T220934Z-V7-gemv-quad-started-reset` | PASS: 123 casos, 474 binary64, 12 resets con arranques observados y 39 alteraciones rechazadas. Respaldo verificado. |
| `20261003T220934Z-V7-gemv-dual-started-reset` | PASS del control dual con los mismos criterios. Respaldo verificado. |
| `20261003T220808Z-V7-gemv-quad-synth` | Cuatro MAC reales, 104 DSP, dos cachés BRAM; WNS estimado +0,148/WHS +0,046 ns. Respaldo verificado. |
| `20261003T221203Z-V7-gemv-quad-brandon` | PASS: 174 trabajos, 5.376 binary64 y 33 negativos. Factor dual/quad 1,753–1,865× en ciclos de Brandon. Respaldo verificado. |
| `20261003T222310Z-V7-gemv-quad-route` | Revisado: 25.168 LUT/17.592 FF/104 DSP; WNS +0,263/WHS +0,010 ns a 5 ns. 43.552 nets enrutables completas, un gap de reloj padre. Respaldo verificado. |
| `20261003T222527Z-V7-ffn-quad-errors` | PASS: 63 cadenas/casos, 71.232 bytes exactos y 24 alteraciones rechazadas; 392,086 s. Respaldo verificado. |
| `20261003T222559Z-V7-ffn-dual-regression` | PASS: 63 casos con el wrapper predeterminado, 71.232 bytes exactos y 24 alteraciones; 380,384 s. Respaldo verificado. |
| `20261003T223324Z-V7-gemv-quad-route-audit` | Timeout 300,006 s en consulta masiva de rutas; no alteró el DCP. Diagnóstico respaldado. |
| `20261003T223550Z-V7-ffn-quad-brandon` | PASS revisado: tres FFN, 42.048 bytes exactos/768 FP32, ocho negativos; 207.366–207.430 ciclos. Respaldo verificado. |
| `20261003T223656Z-V7-ffn-quad-synth` | Revisado: 50.628 LUT/39.991 FF/145 DSP, 48 RAMB36+3 RAMB18, WNS estimado +0,057/WHS +0,039 ns; 419,843 s. Respaldo verificado. |
| `20261003T224026Z-V7-gemv-quad-route-audit-r2` | Revisado en 66,351 s: 824 segmentos PARTIAL de reloj con ruta idéntica; la consulta cubre rutas parciales y conserva el informe global. Respaldo verificado. |
| `20261003T224122Z-V7-ffn-dual-brandon-control` | PASS y respaldo verificado: tres FFN, 42.048 bytes exactos. Comparación pareada: quad reduce ciclos 32,49–32,54%, 1,481–1,482×. |
| `20261003T224908Z-V7-ffn-quad-route` | FAIL setup: WNS −0,215 ns, 610 endpoints; WHS +0,010 ns. 50.767 LUT/39.775 FF/145 DSP. Respaldo verificado. |
| `20261003T230708Z-V7-ffn-quad-route-audit` | Auditoría: 1.459 segmentos del reloj padre incompleto; cero errores de ruteo. No corrige el fallo setup. Respaldo verificado. |
| `20261003T225952Z-V7-gemv-quad-mac-faults` | Diagnóstico: la escala Q8 externa se rechaza antes de HLS; timeout del decoder con watchdog reducido. Respaldado. |
| `20261003T230321Z-V7-gemv-quad-mac-fault-trace` | Confirma timeout con cero MAC ocupados, sin error HLS. Respaldado. |
| `20261003T230417Z-V7-gemv-quad-mac-faults-r2` | Error HLS correcto, cobertura de solapamiento insuficiente con cinco filas. Diagnóstico respaldado. |
| `20261003T230624Z-V7-gemv-quad-mac-faults-r3` | PASS: 105 trabajos/216 binary64 exactos; 30 errores HLS reales, nueve timeouts, seis cancelaciones y seis resets con completions retenidas; 66 negativos. Respaldado. |
| `20261003T231448Z-V7-gemv-quad-fifo-extended` | PASS: 123 casos, 474 binary64, 54 negativos; scoreboard de orden/metadatos y descarte por reset. Respaldado. |
| `20261003T231448Z-V7-gemv-quad-fifo-off-control` | PASS: 123 casos/474 valores; toda la traza funcional coincide con la referencia quad, 39 negativos. Respaldado. |
| `20261003T232011Z-V7-gemv-quad-fifo-faults` | PASS y respaldo: 105 trabajos, 216 binary64, errores HLS/completions retenidas y 81 negativos; scoreboard FIFO sin pérdidas. |
| `20261003T232011Z-V7-ffn-quad-fifo-errors` | PASS y respaldo: 63 casos, 71.232 bytes exactos y 24 negativos; 376,952 s. |
| `20261003T232548Z-V7-gemv-quad-fifo-fixture` | PASS y respaldo: 174 trabajos, 5.376 binary64 exactos; Brandon y formas hasta 32 × 8192. Por semilla 29.888 entradas/salidas FIFO sin pérdidas; 555,030 s. |
| `20261003T232820Z-V7-ffn-quad-fifo-brandon` | PASS y respaldo: tres FFN, 42.048 bytes exactos, 768 FP32; 207.523–207.559 ciclos; 611,574 s. |
| `20261003T232820Z-V7-ffn-quad-fifo-synth` | Síntesis y respaldo: 50.605 LUT/40.014 FF/145 DSP, 48 RAMB36 + 3 RAMB18, WNS +0,378/WHS +0,039 ns estimados; 466,860 s. |
| `20261003T233132Z-V7-swiglu-pipeline-both` | PASS y respaldo: 27 casos C/RTL, 14 registros y estados idénticos; 598–722 ciclos válidos, ocho negativos; 231,734 s. |
| `20261003T234000Z-V7-ffn-quad-fifo-route` | FAIL setup: WNS −0,416/WHS +0,010 ns, 7.517 endpoints, 82.718 nets completas; 1087,764 s. Revisado y respaldado. |
| `20261003T234033Z-V7-swiglu-pipeline-memory` | PASS y respaldo: 105 casos, 123 registros y 13 negativos; 182,791 s. |
| `20261003T234033Z-V7-ffn-quad-pipeline-errors` | PASS y respaldo: 63 casos de FFN quad/FIFO y SwiGLU optimizado, 71.232 bytes exactos y 24 negativos; 289,220 s. |
| `20261003T234644Z-V7-swiglu-pipeline-memory-large` | PASS y respaldo: 8192 elementos, 256 registros Q8 exactos y nueve negativos; 222,302 s. |
| `20261003T234644Z-V7-ffn-quad-pipeline-brandon` | PASS y respaldo: tres FFN, 42.048 bytes exactos/768 FP32; 136.806–136.933 ciclos, mejora pareada 1,516–1,517×; 545,020 s. |
| `20261003T234644Z-V7-ffn-quad-pipeline-synth` | Revisada y respaldada: 51.086 LUT/40.426 FF/153 DSP y 48 RAMB36 + 3 RAMB18; WNS +0,378/WHS +0,039 ns estimados. |
| `20261003T235718Z-V7-ffn-quad-pipeline-route` | Timing interno OOC PASS y respaldo: WNS +0,046/WHS +0,009 ns; 50.880 LUT/40.214 FF/153 DSP, 83.702 nets completas; 1016,086 s. |
| `20261003T235734Z-V7-pcie-quad-pipeline-small` | PASS y respaldo: D32/H32, 32 FP32 exactos, 1.809 ciclos, MMIO concurrente, dos MSI-X y dos descriptores inválidos; 33 negativos; 953,372 s. |
| `20261003T235811Z-V7-ffn-quad-fifo-route-audit` | Revisado y respaldado: 1.482 aliases del reloj padre incompleto; cuatro avisos nuevos de contexto AXI conservados. |
| `20261004T000957Z-V7-ffn-quad-fifo-physopt` | FAIL setup conservado y respaldado: WNS −0,343/WHS +0,010 ns; 50.455 LUT/39.718 FF/145 DSP. Optimización física del DCP anterior, sin modificar RTL/restricciones. |
| `20261004T001636Z-V7-ffn-quad-fifo-physopt-audit` | Auditoría revisada y respaldada; reloj padre/fronteras OOC pendientes. |
| `20261004T001638Z-V7-ffn-quad-pipeline-route-audit` | Revisada y respaldada: 1.531 aliases del reloj padre; 2.470 entradas/845 salidas sin retardos, cero latches/endpoints internos sin restricciones. |
| `20261004T002842Z-V7-ffn-quad-irq-errors` | PASS y respaldo: 63 casos ABI2/57 negativos, host 4 ns, motor 5 ns; 12 resets y 12 cancelaciones; 381,767 s. |
| `20261004T002842Z-V7-ffn-quad-irq-errors-slow` | PASS y respaldo: mismos 63 casos/57 negativos, host 6,2 ns; 388,836 s. |
| `20261004T003148Z-V7-pcie-quad-pipeline-brandon` | Rechazo del presupuesto antes de Vivado; sin invocation/simulación. Diagnóstico respaldado; no es resultado funcional. |
| `20261004T003453Z-V7-pcie-quad-pipeline-brandon` | PASS y respaldo: 256 FP32 exactos/136.909 ciclos, H2C 592.384 B/C2H 1.024 B, dos inválidos/dos MSI-X; 33 negativos; 4.081,766 s. |
| `20261004T004355Z-V7-ffn-quad-irq-synth` | Revisada y respaldada: 55.254 LUT/51.190 FF/153 DSP, 48 RAMB36 + 3 RAMB18; WNS estimado +0,173/WHS +0,013 ns; 17 grupos de skew cumplen, CDC-15 visible; 13 negativos; 640,792 s. |
| `20261004T005107Z-V7-ffn-quad-irq-brandon` | PASS y respaldo: tres FFN/768 FP32 exactos, host 4 ns; CSR 138.493/138.524/138.650 ciclos; 744,147 s. |
| `20261004T005107Z-V7-ffn-quad-irq-brandon-slow` | PASS y respaldo: mismos resultados, host 6,2 ns; CSR 138.574/138.367/138.510 ciclos; 765,829 s. |
| `20261004T011231Z-V7-ffn-quad-irq-route` | FAIL y respaldo: WNS −0,003/WHS +0,010 ns, un fallo de setup; 98.530 nets completas, 54.969 LUT/50.946 FF; 1.324,211 s. |
| `20261004T013503Z-V7-ffn-quad-irq-route-audit` | Auditoría y respaldo: confirma timing fallido, 131 INIT=0, 33 destinos únicos AXI-Lite, 17 grupos skew y dos huecos padre; 15 negativos; 131,903 s. |
| `20261004T010834Z-V7-reset-supervisor-sim` | FAIL independiente y respaldo: aserciones tardías pasan, pero el reset se libera durante GSR con host ausente; 57,771 s. |
| `20261004T011754Z-V7-reset-supervisor-sim-r2` | PASS y respaldo: INIT=0 explícito, 36 casos/51 arranques/15 paradas, 21 negativos; 57,371 s. |
| `20261004T012414Z-V7-reset-supervisor-synth` | No calificada y respaldada: seis CDC críticos de requests combinacionales; 47 LUT/84 FF; 80,836 s. |
| `20261004T012801Z-V7-reset-supervisor-init-audit` | Diagnóstico respaldado: parser rechaza ASYNC_REG=1; solo revisa un registro, sin aprobación; 35,775 s. |
| `20261004T012916Z-V7-reset-supervisor-sim-r3` | PASS y respaldo: requests registrados, 36 casos/21 negativos, trazas públicas idénticas a r2; 57,541 s. |
| `20261004T013229Z-V7-reset-supervisor-synth-r2` | Revisada y respaldada: 48 LUT/87 FF, CDC sin críticos, WNS estimado +3,092/WHS +0,079 ns; LUTAR-1 abierto; 80,570 s. |
| `20261004T013548Z-V7-reset-supervisor-init-audit-r2` | Revisada y respaldada: 12 FDCE INIT=0/ASYNC_REG=1, solo cuatro excepciones propias a CLR, cero ignoradas; diez negativos; 50,206 s. |
| `20261004T014712Z-V7-ffn-quad-irq-physopt` | PASS OOC y respaldo: WNS +0,003/WHS +0,010 ns, sin cambiar celdas, períodos ni excepciones; 225,691 s. |
| `20261004T015319Z-V7-ffn-quad-irq-physopt-audit` | Auditoría y respaldo: clocks, INIT, CDC, 17 skew y 98.530 nets completas; 15 negativos; 121,140 s. |
| `20261004T015030Z-V7-reset-supervisor-sim-r4` | PASS y respaldo: 36 casos/21 negativos; pérdidas de reloj bloqueadas en 574,999–684,999 ns en operación; 57,400 s. |
| `20261004T015040Z-V7-reset-supervisor-synth-r3` | Revisada y respaldada: 51 LUT/87 FF; LUTAR-1 eliminado, CDC-26 externo retenido, estimaciones WNS +3,092/WHS +0,079 ns; 78,628 s. |
| `20261004T015259Z-V7-reset-supervisor-init-audit-r3` | Auditoría y respaldo: 12 FDCE INIT=0/ASYNC_REG=1, constraints conservadas; 49,833 s. |
| `20261004T015604Z-V7-reset-supervisor-external-audit` | Auditoría y respaldo: puerto externo directo a tres FDPE, sin LUT ni RAM, cadena D/Q correcta; aviso CDC-26 abierto y 13 negativos conjuntos; 49,672 s. |

Siguiente tanda V7: conectar el supervisor a host, motor, conversores y
memoria conductual reales del banco; demostrar reset solapado, inhibición
de admisión, épocas antiguas y recuperación tras pérdida de reloj. Las
confirmaciones `peers_held/ready` deben derivarse de los pares, no de
constantes. Mantener separados los avisos CDC externos, el margen OOC de
3 ps y los bloqueos V2/V3; ninguno queda cerrado por el banco aislado.

El ruteo no tiene retardos externos ni reloj padre implementado; no cierra
la placa. La auditoría r2 consulta ROUTE solo para las nets PARTIAL, tras
contar todos los estados; evita materializar las rutas de toda la jerarquía.
No se agregaron excepciones temporales ni se degradaron DRC.
La prueba PCIe Brandon `20261003T212950Z-V7-pcie-ffn-brandon` terminó
sin cambios, pasó la revisión y conserva la variante dual original.
El monitor del ruteo quad se detuvo al encontrar `Route 35-328`; la revisión
posterior registra `EXPLORATORY_OOC_TIMING_FAIL` y respalda ambas tandas.
No se modifica el resultado original ni se atenúan restricciones. La
cola Q8 debe pasar numérica/fallos y un nuevo ruteo antes de aceptarse.
El controlador local `quad-fifo-queue` fija hashes de fuentes y revisores.
Brandon y síntesis ya cumplieron los criterios y habilitaron ruteo/auditoría
a los mismos 5 ns. La cola `swiglu-pipeline-queue` conserva el RTL y cambia
solo el artefacto HLS aislado: primero memoria/FFN pequeña, después
Brandon, vector de 8192 y síntesis; finalmente ruteo/auditoría. Fija hashes
y exige respaldo antes de cada dependencia. No cambia las pruebas PCIe
ni los archivos de tandas anteriores.

El ruteo FIFO falla setup de nuevo. Su cola se detuvo al encontrar cuatro
avisos AXI `Timing 38-242` adicionales; una revisión separada, limitada a
esa tanda, conserva tanto esos pendientes de frontera como el fallo.
La ruta crítica pasa por la propagación global `error_now` hacia los
acumuladores. La [hipótesis de próxima corrección](../Q8_GEMV_QUAD.md)
debe conservar drenaje, prioridad de errores y publicación atómica.

`pcie-quad-pipeline-queue` aprobó la prueba D32/H32. Su copia local y
los respaldos de ruteo se recuperaron tras cortes SSH, sin volver a ejecutar
Vivado ni modificar archivos sellados. Los registros originales de error
se conservan junto a cierres separados `*-manual-review-result.json`.
`pcie-quad-pipeline-brandon-queue-r3` administra la prueba grande y fija
sus revisores por hash. Dos intentos previos fueron rechazados antes de
Vivado; los selectores de presupuesto local/remoto ya reconocen ese
workload específico. Las validaciones activas continúan sin reinicios.

`irq-quad-route-queue` espera revisión y respaldo de la síntesis integrada
y ambos bancos Brandon. Ejecutará colocación/ruteo una sola vez, con
presupuesto de 2.400 s y cuatro hilos, seguido de auditoría de solo lectura
de hasta 1.200 s. Revisa inicialización de los 131 registros de transporte,
excepciones por scope, 17 grupos de skew y contexto de ambos relojes.
Los revisores y scripts quedan fijados por hash mientras la cola está activa.


## NTRD y pool: vida retenida — 2026-10-05

`20261005T045125Z-V7-ffn-supervised-managed-dual-lifetime`: aprobado y respaldado.
Ocho escenarios con supervisor/pool/dos CDC reales; memoria AXI simulada por
drivers. Siete pérdidas de disponibilidad conservan sesión y transacciones
MMIO/F01 pendientes; tres mutantes RTL y 259 corrupciones detectados.
Respaldo SHA-256 `d1df17a5c33a86224de5c7410d6c6a30e2a899f5ac640d07b8c113fdcc77cc9c`.
Véanse [contrato, evidencia y límites](PCIE_DUAL_CONTROL.md#vida-retenida-con-supervisor-y-pool-reales-2026-10-05).
La tanda posterior con dos MIG se registra por separado; V7 integrado y
el resto de la cola conservan sus criterios abiertos.


## NTRD con dos MIG — 2026-10-05

`20261005T050945Z-V7-ffn-supervised-managed-dual-mig`: aprobado y respaldado. 18 retiros MMIO, 35 tramas, 16 escrituras (1.024 bytes) y 28 lecturas (1.792 bytes) comparadas. Cuatro ranks por canal, direcciones altas, STOP parcial y durante W/B retenidas. 209 corrupciones detectadas; modelos/tráfico periódico auditados. SHA-256 `d555e2efcb6c27c2f7da0b87c529ca60021b306cc3a6b709fbd3ed10334c5bc5`. Véanse [evidencia y límites](PCIE_DUAL_CONTROL.md#ntrd-sobre-dos-mig-readback-completo-2026-10-05) y [siguiente integración PCIe](PCIE_DUAL_NATIVE_PLAN.md). Host AXI de prueba y memoria genérica; V7 completo y los demás criterios abiertos de la cola siguen pendientes.

## Host CDC y cierre de transacciones — 2026-10-05

`20261005T053839Z-V7-ffn-supervised-managed-host-cold` queda rechazado y
respaldado: el banco terminaba antes del último retorno de confirmación
de control. Su revisión R2 `20261005T055601Z-V7-ffn-supervised-managed-host-cold-r2`
está aprobada y respaldada, sin cambiar el RTL: 18 trabajos, 35 tramas,
16 escrituras DDR y 28 lecturas comparadas. Conservación en ambos CDC,
54 barreras y drenaje final comprobados; 80 alteraciones rechazadas.

Respaldo R2: 716 archivos, 114.582.301 bytes; SHA-256
`73793b715a45fdc5be7adc3163196cb0ceab967fc5369ae736be5d9b4a78cbb9`.
Véanse [métricas y alcance](PCIE_DUAL_HOST_CDC.md). Siguiente: conexión
PCIe nativa BAR0/BAR2 del núcleo con dos MIG, primero con reset frío.
Se conservan trabajos concurrentes y todos los criterios abiertos de
V0–V9; no hay cierre de timing integrado, placa ni inferencia.

## Composición de drenaje y proveedor local R2 — 2026-10-06

Quedaron revisados y respaldados diez perfiles de composición/fallo compartido,
un control positivo adicional y tres defectos RTL detectados por sus aserciones
previstas. La revisión independiente compara transacciones con VCD y rechazó
29 alteraciones de evidencia. La síntesis eliminó 12 CDC-11 por duplicación;
la revocación local medida fue de 25,5 ns sin otro flanco host. La alarma
compartida también se retiene después de cancelar y cambiar de identidad.

Véanse [resultados, respaldos y límites](LOCAL_REPLY_R2_NEGATIVE_RESULTS.md).
Los contratos externos reales de host/XDMA/ownership, el CDC-15 nativo,
timing físico y reset completo del endpoint siguen pendientes. Los perfiles
con contratos de prueba no califican la placa ni autorizan reset nativo.

## Proveedor de ownership — 2026-10-06

Se añadió y revisó el [certificado de invalidación del propietario real](OWNERSHIP_CERTIFICATE.md):
dos bancos positivos de 12 escenarios, tres defectos RTL detectados,
13 alteraciones de evidencia rechazadas y síntesis OOC del propietario con
su proveedor (1.604 LUT, 3.257 FF; tres cruces CDC-3 informativos).
Las cuatro tandas, incluido el diagnóstico inicial de fuentes, tienen respaldo
local/NFS verificado. Sigue pendiente conectarlo al coordinador de drenaje;
host/XDMA, reset nativo y cierre físico conservan sus criterios abiertos.

## 2026-10-06 — Coordinador sintetizable de propietario y drenaje

Tandas `20261006T095925Z`, `20261006T100616Z` y `20261006T101300Z`:
revisión independiente de ocho perfiles, comparación exacta de señales
públicas y auditoría OOC. Los 685 respaldos incluyen los diagnósticos previos;
no significan 685 aprobaciones funcionales. La dependencia vigente es R33.
La [documentación del coordinador](OWNED_DRAIN_COORDINATOR.md) registra
los hashes, el alcance y la siguiente integración del datapath real.
