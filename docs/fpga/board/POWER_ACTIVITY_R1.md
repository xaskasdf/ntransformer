# Captura R1 de actividad del ejemplo DDR

> **Corte actual:** DDR R1 está revisada y respaldada. El análisis offline
> SAIF/DCP quedó documentado, pero su fase Vivado no se lanzó: la dependencia
> del controlador de token requiere inspección. El usuario reporta pérdida de
> acceso a la VM; no se volvió a conectar. No hay estimación nueva de consumo.
> Véase el [traspaso local](VIVADO_ACCESS_LOSS_HANDOFF_20261016.md).

Fecha: 2026-10-15. Estado actualizado 2026-10-16 12:46 UTC:
**captura ejecutada, dos SAIF revisados y respaldados; potencia pendiente**.
Véanse [resultados, diagnóstico original y revisiones suplementarias](POWER_ACTIVITY_RESULTS_R1.md).
Continúa el [diagnóstico de 3,015 W](POWER_PROBE_RESULTS_R1.md), cuya actividad
inferida y resets impiden dimensionar la tarjeta. Este paquete no reserva
el slot Vivado por sí mismo. La integración del hilo principal queda registrada abajo.

## Integración y turno asignado

Resultado 2026-10-16 12:46 UTC: la tanda terminó en 412,01 segundos con
código cero; ambas ventanas tienen 20,48 µs, 4.096 ciclos UI y 577.280
señales. Captura de 585 archivos restaurada; SHA-256
`203dd7b2de98042c6775b5e3f2a8e9b342d9caeb1f6c90070ae4f06bb93a28ff`.
Las revisiones nuevas resuelven lectura SAIF, ecos Tcl, finales de línea
y selección del modelo sin alterar la tanda ni su diagnóstico original.
La variante `glbl` y el wrapper RDIMM activo quedan registrados; el wrapper
SDRAM preparado no se compiló. Faltan refresh/ranks, X/Z, representatividad
y asociación al DCP. La cola avanzó a memoria Q/K/V. **No hay consumo nuevo.**

Inicio nativo: `20261016T122423Z-V8-ddr-activity-vivado-r3`. Invocación
iniciada a las 12:24:31 UTC; Vivado está generando la simulación de la IP.
Conserva el ZIP fuente R1 de 394 archivos, dos CPU, límite de 1.200 segundos
y ausencia de reintentos automáticos. BRAM terminó con revisión aprobada,
respaldo SHA-256 `af89519c5abd8258bcdbf289d41eac727194767ad12b035eac69c31f2a1164b6`
y restauración de 447 resultados y 165 fuentes. Los observadores de
actividad e identidad esperan la captura terminal respaldada.
Auditoría de treinta propietarios: `attention-composition-r77-active-audit-20261016T122525Z.json`.
**Elaboración y captura aún en curso; sin SAIF ni consumo nuevo.**

Comprobación 2026-10-16 12:02 UTC: BRAM continúa en implementación física;
DDR conserva el turno posterior a su revisión y respaldo. Los 394 archivos
coinciden en D:/NFS: `ddr-activity-handoff-r6-audit-20261016T120214Z.json`,
SHA-256 `3b3e7979c8e7921a331ef1f956ca174cba2d3c202a925f78393f58800da5930a`.
Treinta y un propietarios mantienen sus dependencias; el control de
vocabulario se añadió al final. **Este paquete sigue sin elaboración,
SAIF ni nueva estimación de consumo.**

Comprobación 2026-10-16 11:43 UTC: BRAM permanece en implementación física;
DDR conserva el turno posterior. Treinta propietarios verificados en
`attention-composition-r76-active-audit-20261016T114303Z.json`.
Los observadores de actividad, identidad y restauración siguen activos.
El trabajo numérico paralelo de Llama terminó y quedó respaldado; no
reservó Vivado ni cambió este paquete. **Sin elaboración R1, SAIF ni
una nueva estimación de consumo.**

Comprobación 2026-10-16 11:30 UTC: BRAM terminó XSim y el control aritmético;
continúa la implementación física. DDR conserva el turno posterior a su
revisión y respaldo. Treinta propietarios mantienen las dependencias.
Sus 394 archivos coinciden en D:/NFS: auditoría
`ddr-activity-handoff-r6-audit-20261016T113028Z.json`, SHA-256
`13fdd345e9faf35cc324550c829441485aa7041373eea1f6dd4d25f01bd7137b`.
**R1 sigue pendiente de elaboración; sin SAIF ni potencia nueva.**

Comprobación 2026-10-16 11:00 UTC: BRAM sigue ejecutando XSim y DDR mantiene
su turno posterior. Treinta propietarios verificados; los 394 archivos R1
coinciden en D:/NFS (`ddr-activity-handoff-r6-audit-20261016T110046Z.json`).
Se preparó un [revisor de comandos por rank](POWER_RANK_TRACE_R1.md) usando
un VCD histórico, sin lanzar Vivado ni cambiar R1. Su helper de captura
requiere una revisión nueva; no se ha integrado al banco pendiente.
**Todavía no hay SAIF ni una estimación nueva de consumo.**

Comprobación 2026-10-16 10:48 UTC: BRAM continúa en XSim, ciclo 1.710.051
observado. **DDR conserva el turno inmediatamente posterior**. Los 394
archivos volvieron a verificarse en D:/NFS: auditoría
`ddr-activity-handoff-r6-audit-20261016T104808Z.json`, SHA-256
`293bbc32b896651418a50f56ae4337869a919b1ec0a36f9bd3b17e85a0ec7e86`.
La comparación de cuatro MAC se añadió al final; treinta propietarios
mantienen sus dependencias. Los observadores DDR esperan la captura
respaldada. **R1 sigue sin elaborarse; no hay SAIF ni consumo nuevo.**

Comprobación 2026-10-16 10:31 UTC: BRAM ejecuta XSim, ciclo 541.774
observado; todavía no tiene resultado final. DDR conserva su turno.
La [revisión de identidad de fuentes](POWER_ACTIVITY_IDENTITY_R1.md)
quedó conectada a un observador separado, PID 3696, que espera la captura
respaldada. Pasaron 26 comprobaciones del revisor y 15 del observador en
Windows/Linux. No cambia R1 ni ocupa Vivado; no demuestra asociación
SAIF/DCP. **La captura DDR sigue sin elaborarse y no hay potencia nueva.**

Comprobación 2026-10-16 10:19 UTC: BRAM continúa en síntesis; DDR espera
su revisión y respaldo sin cambiar el paquete ni su turno. El observador
`restore_projected_root_bram_vivado_r3.py`, PID 23628, completará además
la extracción y comprobación de todos los archivos de BRAM y sus fuentes
en D:, con recibo separado en NFS. No reserva Vivado ni modifica la cola;
su espera máxima es de seis horas. Estado: `projected-root-bram-restore-r1-monitor.json`
en `build/cloud-setup/20261002-board/`. La restauración está pendiente,
igual que la elaboración DDR, los SAIF y una nueva estimación de consumo.

Comprobación 2026-10-16 10:12 UTC: atención/FFN/residual terminó con aprobación,
respaldo y restauración verificados. BRAM: `running`; **DDR mantiene
su turno inmediatamente posterior**. La prueba de cancelación se añadió
al final de la cola, sin modificar este paquete. El observador DDR continúa
esperando el respaldo de la captura. **R1 sigue sin elaboración, SAIF ni potencia nueva.**

Comprobación 2026-10-16 09:52 UTC: **DDR conserva su turno después de BRAM**.
El banco aritmético de capa completa se agregó al final, después de su
síntesis, sin modificar R1. Veintinueve propietarios verificados.
Sus 394 archivos siguen coincidiendo en D: y NFS: `ddr-activity-handoff-r6-audit-20261016T095059Z.json`,
SHA-256 `e8e7d239b584e4b393b551c02ff3881d546d4f0bb28957171a9842a099342aab`.
El observador complementario sigue esperando la captura respaldada.
**Todavía sin elaboración de R1, SAIF ni nueva estimación de consumo.**

Comprobación 2026-10-16 09:14 UTC: **DDR conserva su turno después de BRAM**.
La capa completa se agregó al final, después de preflight; hay 28 propietarios
verificados. Atención/FFN permanece activa, ciclo 19.775.825 a las 09:13 UTC.
Los 394 archivos R1 coinciden en D: y NFS: auditoría
`ddr-activity-handoff-r6-audit-20261016T091347Z.json`, SHA-256
`f514c34af0e80e1d4e3c3cb1348a479dfc0ee30cd6eff40212e44c8ae482d96d`.
El observador complementario sigue esperando el respaldo de la captura.
**Todavía sin elaboración de R1, SAIF ni potencia nueva.**

Comprobación 2026-10-16 08:38 UTC: **DDR conserva su turno después de BRAM**.
La validación de memoria de capa se añadió al final, tras entrada/atención
aritmética. Atención/FFN sigue activa, ciclo 18.033.464 a las 08:37 UTC.
El observador complementario permanece esperando la captura respaldada.
**Todavía sin elaboración de R1, SAIF ni potencia nueva.**

Actualización 2026-10-16 08:13 UTC: la revisión complementaria tiene un
[observador automático](POWER_ACTIVITY_DETAIL_R1.md), PID 16980, que espera
la captura terminada y respaldada. Verificará el archivo completo y la copia
restaurada antes de analizar ventanas y señales; guardará su informe en un
respaldo separado D:/NFS. Quince pruebas sintéticas pasaron en Windows/Linux.
No añade un propietario Vivado ni cambia dependencias. DDR sigue después
de BRAM; atención/FFN permanece activa, ciclo 16.830.230 a las 08:13 UTC.
**Captura aún sin elaboración, SAIF ni consumo nuevo.**

Comprobación 2026-10-16 07:56 UTC: **DDR mantiene su turno tras BRAM**.
El banco aritmético de entrada→atención se añadió al final, después de
su síntesis/protocolo. Veintiséis propietarios verificados; atención/FFN
sigue activa, con 15.940.191 ciclos observados a las 07:55 UTC.
Los 394 archivos DDR coinciden en D: y NFS: auditoría
`ddr-activity-handoff-r6-audit-20261016T075513Z.json`, SHA-256
`c3e6e1ac7c5ec60a1f7397c4ca84b23ecbefc06a49bf10a2cabf5dfd6e96d3d4`.
**Captura DDR todavía sin elaboración, SAIF ni consumo nuevo.**

Comprobación 2026-10-16 07:27 UTC: **DDR conserva su turno tras BRAM**.
Los 394 archivos R1 se revalidaron en D: y NFS a las 07:17 UTC;
veinticinco propietarios continúan verificados. Atención/FFN llegó al
ciclo 14.370.338 a las 07:24 UTC, todavía sin resultado final.
Se preparó una [revisión complementaria](POWER_ACTIVITY_DETAIL_R1.md)
de tiempos, resets, X/Z y diferencias por señal: 39 rechazos y 22
comprobaciones diagnósticas idénticas en Windows/Linux. Usa datos
sintéticos; no modifica el ejecutor ni el revisor sellados. El hilo
principal la aplicará después del respaldo de la captura. La ocupación
SAIF de CS/CKE/ODT no establece cobertura de comandos ni refresh.
**R1 sigue sin elaboración, SAIF ni estimación nueva de consumo.**

Comprobación 2026-10-16 07:06 UTC: **DDR sigue después de BRAM**.
La composición entrada→atención se añadió al final, después de consulta KV.
Veinticinco propietarios verificados; atención/FFN permanece activa, con
13.402.910 ciclos a las 07:05 UTC y sin resultado final. Los 394 archivos
DDR siguen íntegros en D: y NFS: `ddr-activity-handoff-r6-audit-20261016T070528Z.json`,
SHA-256 `64ce9a44d62c7fd0de19b81280afc3608c1cdf2e9007166bb6930fd999f1674c`.
R6 ejecuta la auditoría de propietarios más reciente antes de comprobar
el turno. **Esta captura sigue sin elaboración, SAIF ni potencia nueva.**

Comprobación 2026-10-16 06:34 UTC: **DDR conserva su turno después de BRAM**.
La consulta independiente KV se añadió al final, tras el controlador de
retención. Veinticuatro propietarios verificados; atención/FFN alcanzó
11.606.693 ciclos a las 06:29 UTC, todavía sin resultado final. Los 394
archivos DDR siguen íntegros en D: y NFS: auditoría
`ddr-activity-handoff-r5-audit-20261016T063355Z.json`, SHA-256
`042b84cf6d92bc257bf100f140b67c32db90b126ec4843e5fe0f60eb8a4d794b`.
R5 exige una observación reciente de los propietarios; R4 consultaba una
auditoría de cola anterior. No cambió el paquete ejecutable ni su turno.
**Todavía sin elaboración de esta captura, SAIF ni potencia nueva.**

Comprobación 2026-10-16 06:13 UTC: **DDR conserva su turno tras BRAM**.
El nuevo control de retención Q/K/V se añadió al final, después del banco
aritmético RoPE/atención. Veintitrés propietarios verificados. Atención/FFN
permanece activa: 10.651.945 ciclos a las 06:09 UTC, sin resultado final.
Los 394 archivos R1 siguen íntegros en D: y NFS:
`ddr-activity-handoff-r4-audit-20261016T061252Z.json`, SHA-256
`dabdd6f6165e5d1a7e0b6c9017540f52ba0aaaa00c8aa996eb83a337d757cd3c`.
**Captura aún pendiente de elaboración; sin SAIF ni potencia nueva.**

Comprobación 2026-10-16 05:40 UTC: **DDR mantiene su turno después de BRAM**.
El banco aritmético RoPE/atención se agregó al final, después de su
síntesis; no altera esa precedencia. Veintidós propietarios verificados.
Atención/FFN permanece activa, con 9.131.528 ciclos observados a las
05:38 UTC y sin resultado final. Auditoría de los 394 archivos en D: y NFS:
`ddr-activity-handoff-r4-audit-20261016T053950Z.json`, SHA-256
`51dcd333a55a73fdee1e5a70ac9080f98df16744bd0768bdc06f755ea5752357`.
**Todavía no se elaboró esta captura: no hay SAIF ni consumo nuevo.**

Comprobación 2026-10-16 05:13 UTC: **DDR conserva el turno después de BRAM**.
La síntesis RoPE/atención se añadió al final, tras reutilización KV, sin
interrumpir validaciones. Veintiún propietarios verificados; atención/FFN
alcanzó 7.863.026 ciclos a las 05:12 UTC, todavía sin resultado final.
Los 394 archivos DDR permanecen intactos en D: y NFS:
`ddr-activity-handoff-r4-audit-20261016T051253Z.json`, SHA-256
`994b38e007f8004543d48af03bc8d424b2bcf417e4a0d2318a0d0c40f4ccd9dd`.
**No hay SAIF ni nueva estimación de consumo.**

Comprobación 2026-10-16 04:52 UTC: **R1 sigue detrás de BRAM**.
Atención/FFN alcanzó 6.682.171 ciclos a las 04:49 UTC; permanece activa,
sin resultado final. La prueba de caché entre capas se agregó al final,
después de RoPE aritmético, sin alterar el turno DDR. Veinte propietarios
verificados. Auditoría de los 394 archivos R1 en D: y NFS:
`ddr-activity-handoff-r4-audit-20261016T044506Z.json`, SHA-256
`310dbb5ed3066a2add575de983e73a00f8d1dd90366606b038c743fc525509ef`.
**Todavía no hay SAIF ni una nueva estimación de consumo.**

Comprobación 2026-10-16 04:25 UTC: **DDR continúa después de BRAM**.
Atención/FFN avanza: 5.425.966 ciclos a las 04:23 UTC, aún sin resultado
final. La simulación de RoPE/captura/despacho se añadió al final, tras su
síntesis exacta; no cambia el turno DDR. Diecinueve coordinadores activos.
Nueva verificación de los 394 archivos R1 en D: y NFS:
`ddr-activity-handoff-r4-audit-20261016T042349Z.json`, SHA-256
`4a0f6471cc5d0de1bb0c699f43f7d9a8167b4d393a787f2e8dbb60eb800298fa`.
**No hay todavía SAIF ni nueva estimación de consumo.**

Comprobación 2026-10-16 04:04 UTC: **R1 conserva su turno tras BRAM**.
Atención/FFN sigue activa: ciclo 4.358.041 observado a las 04:02 UTC, sin
fatal registrado, todavía incompleta. La síntesis de RoPE/captura/despacho
se añadió al final, después de captura/despacho, sin cambiar la precedencia
de DDR. Dieciocho coordinadores verificados. Los 394 archivos R1 volvieron
a verificarse en D: y NFS: `ddr-activity-handoff-r4-audit-20261016T040343Z.json`,
SHA-256 `cd05e69d85d0574c4aeeb5f14698753f378e382e0d5a52a9d350d43be7eb62db`.
**Sigue sin SAIF ni estimación nueva de consumo.**

Comprobación 2026-10-16 03:33 UTC: **R1 sigue esperando después de BRAM**.
Atención/FFN permanece activa en XSim; la observación de las 03:20 UTC llegó
al ciclo 2.260.261, sin fatal registrado. La nueva composición captura/despacho
se añadió al final de la cola, después del despachador, sin adelantarla a DDR.
Los 394 archivos R1 se verificaron de nuevo en D: y NFS:
`ddr-activity-handoff-r4-audit-20261016T033303Z.json`, SHA-256
`3f1f82ba3c9d4d575e6fbf02f91364e69633da8c108d6f2d0e91641f1213d69b`.
Diecisiete coordinadores mantienen el orden. **No hay SAIF ni potencia nueva.**

Comprobación 2026-10-16 03:04 UTC: **R1 continúa esperando su turno**.
Atención/FFN R4 sigue activa; la observación de las 03:01 UTC llegó al ciclo
1.307.658, con 32 cabezas completadas y sin fatal registrado. Es progreso
parcial, no aprobación. BRAM R3 permanece entre esa prueba y DDR R3.
Se revalidaron los 394 archivos DDR sin cambiar entradas ni parámetros:
`ddr-activity-handoff-r4-audit-20261016T025550Z.json`, SHA-256
`bb5eac270e3065ee984f689fbbfd270195d7f19dd147f5b296ba24f524db9ab5`.
El nuevo despachador Q/K/V se añadió al final, detrás de captura Q/K/V;
conserva el turno DDR. Auditoría R61: dieciséis propietarios activos;
18 paquetes y 874 vínculos verificados. **Sigue sin SAIF ni potencia nueva.**

Actualización 2026-10-16 02:31 UTC: **cola reanudada; DDR sigue pendiente**.
Se aisló la ejecución incorrecta de una función de aserción condicional
en el banco XSim del predecesor. La corrección conserva sus 139 condiciones
y pasó doce casos mínimos XSim/Icarus. La simulación integrada está activa
sin repetir su síntesis. Los quince sucesores mantienen el orden original.
DDR usa ahora `ddr-activity-vivado-r3-coordinator.json`, PID 11608, y espera
BRAM R3 revisado y respaldado. Su paquete ejecutable sigue siendo el mismo
ZIP R1, con los 394 archivos revalidados en D: y NFS; no cambiaron estímulos,
límites ni criterios. Evidencia:
`build/cloud-setup/20261002-board/ddr-activity-handoff-r4-audit-20261016T023117Z.json`,
SHA-256 `f8ed8a09f77aeb10f8055e6c74bfeb7b16f05cc92b07634c887c6dc430804ebb`.
Las cinco tandas de diagnóstico y la recuperación de cola tienen respaldo
y restauración verificados. **Todavía no hay SAIF ni potencia nueva.**

Actualización 2026-10-16 02:03 UTC: **captura aún no ejecutada**.
La integración atención/FFN que precede a BRAM falló en el ciclo 6,
antes de admitir trabajo; su diagnóstico completo quedó respaldado y
restaurado. El coordinador DDR R2 se detuvo por dependencia sin lanzar
Vivado. Se conserva su paquete, su lugar después de BRAM y todos los
criterios de este documento. Una repetición corta de arranque está activa
para investigar el predecesor; no se omite ese requisito para adelantar DDR.
Véase [el diagnóstico de atención/FFN](../LLAMA32_ATTENTION_FFN_REAL_R1.md).
**Sigue sin SAIF ni nueva estimación de potencia.**

Comprobación 2026-10-16 01:20 UTC: el coordinador sucesor R2, PID 17120,
espera BRAM. La FFN/residual terminó; su revisión R2 acepta la última
escritura HLS y `done` en el mismo ciclo y conserva 64 controles negativos.
El diagnóstico y los coordinadores detenidos R1 permanecen intactos.
La cola se reanudó con quince coordinadores sucesores y el mismo orden;
el handoff post-O está en Vivado. Se verificaron nuevamente los 394
archivos DDR restaurados en D: y NFS. El paquete ejecutable, el banco y
sus límites no cambiaron. **Sigue sin SAIF ni nueva estimación de potencia.**

Evidencia de esta comprobación:
`build/cloud-setup/20261002-board/ddr-activity-handoff-r3-audit-20261016T012053Z.json`
(SHA-256 `03b4a6dad1b77d261bdc654ade4b5c02973c5e487853c49014d7218ac399d8b3`).
La recuperación de los coordinadores tiene respaldo y restauración en
`artifacts/ntrpu-board-vivado/ffn-review-queue-recovery-20261016-r2/`;
esta actualización y la auditoría se conservan en
`artifacts/ntrpu-board-vivado/ffn-review-handoff-20261016-r2/`.

El hilo principal agregó la captura después de la cadena existente:
FFN/residual real → handoff post-O → atención/FFN/residual real → BRAM →
actividad DDR. La FFN terminó y fue revisada sin repetir Vivado; no se
interrumpieron validaciones ni se cambiaron los paquetes anteriores.
El coordinador de actividad, PID 17120, espera la revisión y el respaldo
verificado de BRAM, además del slot
vacío. Un fallo previo detiene el lanzamiento para inspección.

Paquete ejecutable: `20261015T213741Z-ddr-activity-vivado-r1`, 394 archivos,
104.489.116 bytes comprimidos. Copias idénticas en D: y NFS; restauración
local completa y preparación de referencias verificadas. SHA-256 del ZIP:
`acb1a3923b940cc5d9d076c264c30787336ed188b7eaf97e837291e0fc2442a3`.
Configuración y estado:
`build/cloud-setup/20261002-board/ddr-activity-vivado-r2-package.json` y
`ddr-activity-vivado-r2-coordinator.json` en la misma carpeta. R2 modifica
únicamente la coordinación y sus dependencias; conserva el ZIP fuente R1.
El traspaso de cola y documentación tiene respaldo separado en
`artifacts/ntrpu-board-vivado/ddr-activity-queue-20261015-r1/`; su manifiesto
fija como dependencia el ZIP completo anterior, sin duplicarlo.

El [preparador](../../../tools/prepare_ntrpu_ddr_activity_project_r1.py)
copia el proyecto completo y resuelve cinco referencias externas del XPR
hacia startup, geometría y constraints privados. Conserva el XCI hasta
abrir Vivado; el [Tcl](../../../tools/ntrpu_board_ddr_activity_r1.tcl)
permite cambiar únicamente la ruta de `CustomParts` y registra los demás
parámetros antes/después. El TB original queda intacto. Una copia derivada
añade sólo la impresión y aserción de los parámetros `SIMULATION` del DUT
y del generador, sin modificar estímulos ni RTL de hardware.

El [ejecutor](../../../tools/run_ntrpu_ddr_activity_r1.py) limita toda la
invocación inicial de Vivado, incluida elaboración y captura, a 20 minutos
con CPU 0–1; no reintenta automáticamente. Captura parcial, fallo o timeout
tienen revisión diagnóstica, archivo en NFS, copia local y restauración
verificada antes de liberar el slot. La desconexión del observador no
reinicia la prueba.

El [revisor R2](../../../tools/review_ntrpu_ddr_activity_r2.py) comprueba
duración, contabilidad, continuidad y ámbito del SAIF, y registra X/Z y
diferencias entre ventanas. Pasó doce controles sintéticos de corrupción;
el error del parser R1 también quedó conservado. Estos controles **no son
una ejecución RTL**. Refresh/ranks, representatividad y correspondencia
con el DCP requieren revisión adicional: el coordinador no lanza potencia
ni declara calificada la captura automáticamente.

## Alcance concreto

Paquete: [`build/board-power-activity-20261015-r1/`](../../../build/board-power-activity-20261015-r1).
Su `MANIFEST.json` identifica entradas y archivos por SHA-256.
El banco instancia el `example_top` archivado, con buffers externos,
`ntrpu_startup`, controlador `No_Buffer` y generador AMD. Observa señales;
no fuerza resets, enables, control del generador ni estados de los modelos.

El generador arranca boot y luego PRBS automáticamente; sus entradas stop
y custom están fijadas en el top. Por ello esta revisión captura **PRBS del
ejemplo**, no reposo controlado, lectura sostenida ni inferencia NTRPU.
Conserva `SIMULATION="FALSE"`: el generador cambia cantidad y longitud de
instrucciones cuando ese parámetro es `TRUE`. Confirmar el parámetro de la
elaboración frente a la síntesis archivada antes de aceptar correspondencia.

## Archivos preparados

| Archivo | Función |
| --- | --- |
| `ntrpu_ddr_example_activity_tb.sv` | Top archivado, dos modelos RDIMM genéricos y observadores AXI |
| `capture_activity_r1.tcl` | Espera calibración/PRBS y registra dos SAIF separados |
| `ntrpu_dci_startup.sv` | Copia exacta del ZIP fuente usado en la síntesis |
| `ddr4_sdram_model_wrapper.sv` | Copia del envoltorio con selectores genéricos de 16 Gb, x4 y 1600 MT/s |
| `geometry.csv` | Copia de la geometría archivada; no califica el Micron F1 |

El proyecto original conserva `DDR4_4G_X8`, `DDR4_750_Timing` y `FIXED_2666`
en el envoltorio de simulación. La copia preparada aplica exactamente los
tres cambios ya usados en la [exploración genérica](MEMORY_QUALIFICATION.md):
`DDR4_16G_X4`, `DDR4_1250_Timing` y `FIXED_1600`. Los modelos cifrados y el
top implementado permanecen intactos. La población del banco es `_16G`,
dos ranks por slot, con CK/CS/CKE/ODT separados correctamente por slot.

## Integración pendiente en el hilo principal

1. Esperar el turno de la cola. Crear una tanda y una copia nueva del proyecto
   `20261002T121632Z-V3-dci-integrated-synth/output/example`; verificar las
   entradas contra el manifiesto. Resolver las referencias externas del XPR,
   incluida la fuente de startup, hacia copias privadas. Conservar originales.
2. Reemplazar solamente el envoltorio de simulación por la copia preparada,
   retirar `sim_tb_top.sv` de simulación y añadir el banco nuevo como top
   `ntrpu_ddr_example_activity_tb`. Usar el startup archivado y la misma IP;
   cualquier regeneración ocurre en la tanda. Documentar parámetros antes/después.
3. Elaborar primero con Vivado 2026.1, bibliotecas vendor y `glbl`, con debug
   `typical` o suficiente para observar el DUT. Verificar geometría, `No_Buffer`,
   ECC y jerarquía; resolver errores antes de iniciar captura. No sustituir IP
   por modelos de comportamiento ni usar el banco de capacidad como DUT.
4. Cargar XSim detenido al inicio y ejecutar `capture_activity_r1.tcl` como
   script de simulación personalizado. No dejar `runtime all` ejecutar antes
   del capturador. Usar dos CPU y un límite inicial de 20 minutos de ejecución;
   registrar fallo/timeout sin reintento automático.

El flujo SAIF usa `open_saif`, selección recursiva del DUT, `log_saif` y
`close_saif`, según [AMD UG900 2026.1](https://docs.amd.com/r/en-US/ug900-vivado-logic-simulation/Generating-SAIF-Dumping).
Los modelos de memoria externos y los contadores del banco quedan fuera del
ámbito de potencia FPGA.

## Ventanas y criterios de revisión

El banco espera calibración, resets liberados y PRBS activo, más 256 ciclos
UI de asentamiento inicial. El capturador espera como máximo 1 ms simulado y
solicita dos ventanas contiguas de **20,480 µs / 4.096 ciclos UI** cada una.
Estos tamaños son un primer muestreo propuesto; no demuestran estabilidad ni
cobertura de todos los estados. El reset externo de 20 ns es un estímulo
acelerado de simulación, no la especificación eléctrica de arranque.

Conservar `prbs_0.saif`, `prbs_1.saif`, `windows.tsv`, logs completos y
`CAPTURE_COMPLETE.txt`. El marcador significa sólo captura exportada. El
revisor debe comprobar independientemente:

- Ausencia de errores del modelo, aserciones, watchdog y terminación prematura;
  código cero o `TEST PASSED` aislado no bastan.
- Duración y unidad de cada SAIF, tiempo simulado y 4.096 ciclos por ventana;
  calibración y resets estables durante todo el intervalo.
- Lecturas y escrituras realmente aceptadas, bytes escritos por WSTRB,
  stalls y solicitudes pendientes en ambos bordes; respuestas OKAY y
  diagnóstico del comparador vendor sin errores.
- Ámbito del SAIF limitado al DUT, cobertura de señales y estados X/Z,
  cobertura de refresh y ranks, y diferencias entre ambas ventanas.

Refresh/ranks, comparación estadística entre ventanas y revisión independiente
de los archivos SAIF todavía requieren implementación/verificación nativa.
Los contadores preparados no certifican esos puntos. Si no aparece PRBS,
hay errores o la actividad no resulta estable, guardar el diagnóstico antes
de decidir una revisión del banco o una ventana mayor.

## Uso posterior en potencia

Sólo tras esa revisión, abrir el DCP original en una sesión nueva por ventana,
comprobar `TOP=example_top` y revisar la correspondencia jerárquica. El prefijo
esperado del SAIF es `ntrpu_ddr_example_activity_tb/dut`; confirmarlo en el
archivo real. Guardar redes no asociadas mediante `read_saif -out_file`:
[AMD UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/read_saif).

Comparar inicialmente con las condiciones térmicas del diagnóstico previo,
registrando anotación real, actividad inferida y advertencias de resets.
No suprimir avisos ni completar lógica ausente con actividad arbitraria.
La lógica de depuración añadida durante implementación también requiere
contabilizarse en la cobertura. La comparación sigue siendo del ejemplo;
condiciones térmicas físicas, DIMM y placa integrada continúan pendientes.

## Comprobaciones locales y respaldo

Se contrastaron 22 identificadores jerárquicos con las fuentes, la estructura
léxica Tcl y los hashes de siete entradas. Esto no equivale a compilar
SystemVerilog ni elaborar las IP: ambas comprobaciones siguen pendientes.
El respaldo documental y del paquete se guarda en
`artifacts/ntrpu-board-power-activity-20261015-r1/`, con manifiesto y
restauración verificada. No contiene una captura SAIF ni resultados de Vivado.
