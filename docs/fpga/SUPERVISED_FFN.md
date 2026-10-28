# FFN con reset coordinado y admisión supervisada

Actualizado el 2026-10-04 UTC. Trabajo V7 sobre VU5P, todavía sin cierre
de placa. `ntrpu_ffn_supervised_fabric.sv` conecta el supervisor a la FFN
quad/FIFO, ABI2, transporte IRQ y conversores AMD AXI4/AXI-Lite existentes.
La etapa anterior está en [SUPERVISED_IRQ.md](SUPERVISED_IRQ.md).

## Contrato de la isla

El wrapper genera los resets de host y motor, y bloquea VALID/READY en
las fronteras de control, DMA del host y memoria hasta admitir tráfico.
Las IRQ también quedan bloqueadas. Los dos extremos de cada conversor
comparten el ciclo de reset; el supervisor cuenta al menos 64 flancos
reales de ambos dominios antes de liberar sus resets.

Esto sigue el requisito de resets solapados de todos los pares AXI de
[AMD PG059](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).
La liberación de reset es local y síncrona; la habilitación del tráfico
espera además que todos los pares estén operativos. No se usa READY de
una transacción como indicador de disponibilidad general.

Las cuatro observaciones registradas del transporte IRQ se propagan por
control, dispositivo y fabric bajo `NTRPU_SUPERVISED_FFN`. La revisión actual
lleva ocho niveles a gestión mediante cadenas de tres etapas con borrado
asíncrono a cero. Un alto breve puede descartarse; un bajo cierra la
calificación aunque el reloj de destino esté detenido. Las fuentes deben
estar registradas. El transporte IRQ también registra y califica `host_armed`.
Los handshakes persistentes de eventos y servicio conservan sus XPM.

El iniciador y la memoria deben proporcionar `*_peer_held` cuando hayan
limpiado su estado de protocolo bajo reset, y `*_peer_ready` cuando puedan
operar tras la liberación. Son señales registradas en su dominio local;
READY debe mantenerse durante actividad normal. No conectar constantes
para sustituir confirmaciones reales. El reloj de gestión es continuo
e independiente. Los modelos del banco implementan estas confirmaciones
a partir de su reset y estado local; esto no califica aún un controlador DDR.

## Banco numérico y recuperación

El banco `tb_ntrpu_ffn_supervised.sv` conserva los núcleos HLS reales y las
referencias numéricas independientes. Ejecuta tres semillas, 21 casos por
semilla: cuatro FFN correctas, errores de lectura/escritura, cancelaciones,
resets durante las cuatro fases y descriptor inválido. Después del último
reset ejecuta otra FFN completa y exige sus bytes exactos.

El reloj del motor es de 5 ns; los perfiles del host son de 4 y 6,2 ns,
con fase de 700 ps. Gestión usa 10 ns. En cada arranque se retiene primero
la confirmación de memoria: el supervisor debe mantener ambos resets y
la admisión cerrados. Las trazas registran flancos, resets, admisión y
confirmaciones; un verificador independiente reconstruye los intervalos.

**Límites de esta tanda:** los relojes continúan funcionando durante estos
resets; las pérdidas de reloj se califican en el banco separado descrito abajo.
El puerto AXI4 DMA del host permanece inactivo en este banco, aunque su
conversor forme parte del wrapper. La memoria y el host son modelos de
protocolo. Los FIFO AMD conductuales no modelan retardo de sincronización.
No son pruebas de DDR, XDMA, inferencia completa ni rendimiento físico.

## Defecto encontrado en la instrumentación

En el primer perfil lento, semilla 7, un reset interrumpió `write_reg`
después de registrar una respuesta pero antes de incrementar su contador.
El verificador rechazó el orden AW/W del siguiente trabajo. La tanda
conserva su PASS procedimental y se etiqueta **no calificada**.

El BFM revisado registra y cuenta cada B/R de forma atómica en el flanco
real del handshake. Las nuevas trazas vinculan ID, dirección, datos,
respuesta, contador y tiempo al log funcional. La revisión comprueba
ambos registros sin omitir el fallo previo ni relajar el criterio.

## Registro de ejecución

| Tanda | Estado |
| --- | --- |
| `20261004T025817Z-V7-ffn-supervised-legacy-regression` | 63 casos revisados y respaldados; modo anterior |
| `20261004T025817Z-V7-ffn-supervised-errors` | 63 casos revisados y respaldados; instrumentación inicial |
| `20261004T025817Z-V7-ffn-supervised-errors-slow` | Rechazo independiente por contador del monitor; respaldado |
| `20261004T030544Z-V7-ffn-supervised-peer-negative` | Fallo esperado por confirmación falsa de memoria; respaldado |
| `20261004T031037Z-V7-ffn-supervised-errors-r2` | 63 casos revisados y respaldados con trazas B/R; 87 alteraciones rechazadas |
| `20261004T031037Z-V7-ffn-supervised-errors-slow-r2` | 63 casos revisados y respaldados con trazas B/R; 87 alteraciones rechazadas |
| `20261004T032759Z-V7-ffn-supervised-synth` | Síntesis revisada y respaldada; 606,520 s; advertencias abiertas |
| `20261004T033815Z-V7-ffn-supervised-audit` | Auditoría estructural revisada y respaldada; 92,479 s |
| `20261004T034418Z-V7-ffn-supervised-route` | Timing interno revisado y respaldado: +0,062/+0,010 ns; 1.161,650 s |
| `20261004T035245Z-V7-ffn-supervised-clock-loss` | 63 casos revisados y respaldados; 12 pérdidas automáticas, 108 alteraciones rechazadas |
| `20261004T040110Z-V7-ffn-supervised-clock-loss-slow` | Perfil lento invertido revisado y respaldado: 63 casos, 12 pérdidas, 108 alteraciones rechazadas |
| `20261004T040349Z-V7-ffn-supervised-route-audit` | Estructura tras ruteo revisada y respaldada; 99,424 s |
| `20261004T040526Z-V7-ffn-supervised-clock-audit` | Tres huecos padre y fanout AXI-Lite revisados y respaldados; 127,026 s |
| `20261004T041529Z-V7-ffn-supervised-boundary-audit` | Extracción RTSTAT-10 fallida por opción Tcl; diagnóstico respaldado |
| `20261004T041750Z-V7-ffn-supervised-boundary-audit-r2` | Conectividad completa revisada y respaldada; ocho alteraciones rechazadas; 57,240 s |
| `20261004T043230Z-V7-ffn-supervised-recovery-each` | No calificado y respaldado: semilla 7 detiene reloj tras W, antes de AW/escritura efectiva; 456,275 s |
| `20261004T043240Z-V7-ffn-supervised-recovery-each-slow` | Revisado y respaldado: 72 casos, 12 FFN posteriores a recuperación, 120 alteraciones rechazadas; 473,385 s |
| `20261004T044322Z-V7-ffn-supervised-recovery-each-r2` | Revisado y respaldado: 72 casos/120 alteraciones rechazadas; 458,530 s |
| `20261004T044332Z-V7-ffn-supervised-recovery-each-slow-r2` | Revisado y respaldado: 72 casos/120 alteraciones rechazadas; 470,758 s |
| `20261004T045439Z-V7-ffn-supervised-release-retry` | FAIL XPM S-1: nueve violaciones al truncar niveles durante RELEASE; respaldado |
| `20261004T045439Z-V7-ffn-supervised-release-retry-slow` | FAIL XPM S-1: nueve violaciones; respaldado |
| `20261004T051051Z-V7-ffn-supervised-level-irq-regression` | RTL corregido: 45 casos y 45 alteraciones rechazadas; respaldado |
| `20261004T051051Z-V7-ffn-supervised-level-reset-regression` | RTL corregido: 36 casos, 51 arranques y 15 pérdidas de reloj; respaldado |
| `20261004T051855Z-V7-ffn-supervised-level-sim-r3` | 288 pulsos/297 borrados en tres frecuencias; revisión independiente y 15 alteraciones rechazadas; respaldado |
| `20261004T051855Z-V7-ffn-supervised-level-early-r3` / `level-late-r3` | Seis ejecuciones con RTL modificado rechazadas: liberación temprana y borrado diferido; respaldadas |
| `20261004T053018Z-V7-ffn-supervised-release-level` / `release-level-slow` | Revisados y respaldados: 144 casos, 18 resets RELEASE, 24 pérdidas, 2.496 FP32 exactos y 276 alteraciones rechazadas; 471,530/477,875 s |
| `20261004T054240Z-V7-ffn-supervised-level-synth` / `055321Z-V7-ffn-supervised-level-audit` | Estructura revisada y respaldada: 45 CLR, 55.362 LUT/51.313 FF/153 DSP; estimación +0,173/+0,013 ns; 12 alteraciones rechazadas. Esta revisión falla el ensayo HOLD posterior; no se rutea |
| `20261004T055319Z-V7-ffn-supervised-hold-retry` | 16 violaciones XPM S-1 en contadores de retención; diagnóstico independiente y respaldo |
| `20261004T055952Z-V7-ffn-supervised-hold-retry-r2` | Corrección de contadores: 16 reintentos sin S-1; revisión y respaldo |
| `20261004T055952Z-V7-ffn-supervised-hold-regression` | 36 casos, 51 arranques y 15 pérdidas; 21 alteraciones rechazadas; respaldado |
| `20261004T060223Z-V7-ffn-supervised-release-hold` / `release-hold-slow` | En curso con contadores corregidos; síntesis de 51 CLR espera revisión y respaldo |

Cada tanda está aislada en NFS. Los respaldos verificados aparecen en
`artifacts/ntrpu-board-vivado/<ID>/VERIFIED.json`, con fuentes, compilación,
IP, modelos HLS, logs, revisión y hashes. La revisión compara los modelos
AMD y núcleos HLS con las tandas anteriores. Para el RTL previo comprueba
equivalencia del modo por defecto, eliminando solo los bloques condicionales
nuevos y normalizando finales de línea CRLF/LF.

El banco con relojes activos suma **126 casos, 30 recuperaciones** (24 resets durante
cómputo y seis arranques), 24 cancelaciones y 102 retiros. Comprueba 16.323
handshakes B y 17.490 R del host. Las 24 FFN completas comparan **1.344 FP32
exactos**, además de los bytes intermedios y parciales de los casos de error.
Son matrices sintéticas; no representan inferencia con pesos de un modelo.

| Dimensión / hidden | Ciclos de cómputo observados | Tiempo RTL a 5 ns |
| --- | ---: | ---: |
| 32 / 32 | 1.807–1.815 | 9,035–9,075 µs |
| 64 / 96 | 7.047–7.092 | 35,235–35,460 µs |
| 64 / 80 | 6.536–6.558 | 32,680–32,790 µs |

Los ciclos se miden entre aceptación y finalización del trabajo en el motor;
no incluyen todo el control MMIO ni transferencias DMA.

## Síntesis y auditoría del conjunto

La revisión conserva los mismos RTL/HLS, conversores AMD y XPM que ambos
bancos corregidos. El checkpoint tiene SHA-256
`089b546e9dd47082fcfe8f1e71f0fa53cfa9bec9adf837b54de8753cf58d7fe1`.
Ocupa **55.352 LUT, 51.313 FF, 153 DSP, 48 RAMB36 y tres RAMB18**:
98 LUT y 123 FF adicionales respecto a la síntesis ABI2 anterior.

Las estimaciones de síntesis son **WNS +0,173 ns, WHS +0,013 ns**.
Los 17 grupos de skew cumplen, con margen mínimo estimado de +3,851 ns.
Esto todavía no mide timing ruteado ni rendimiento físico. Faltan fuentes
padre para los tres relojes y retardos de 1.408 entradas/1.344 salidas.

Se verificaron 12 FDCE de liberación, tres FDPE del reset externo,
131 registros persistentes IRQ y 24 etapas de estado, todos INIT=0.
El reset externo llega directamente a PRE, sin LUT. Cuatro estados vienen
de registros IRQ; los cuatro estados de pares externos aún requieren
conductores registrados en el top integrador. Trece alteraciones de
evidencia fueron rechazadas.

CDC conserva 40 CDC-3 y 16 CDC-9 informativos, un CDC-15 en el conversor
AXI y **cinco CDC-26 abiertos**: reset externo y las cuatro entradas de
estado de pares. La auditoría de sus destinos no califica los orígenes
externos. Hay 59 false paths de proveedor y cuatro propios, limitados a
los 12 CLR del supervisor; se conservan 21 waivers de proveedor y ninguno
nuevo. Las advertencias de síntesis coinciden con la referencia, salvo
el aviso del nuevo reloj de gestión sin contexto padre. Los listados
de dos avisos de síntesis y TIMING-18 están truncados por el límite de
mensajes de Vivado; los logs completos permanecen en los respaldos.

## Ruteo exploratorio con tres relojes

La implementación ocupa **54.820 LUT, 50.985 FF y 153 DSP**, con las mismas
48 RAMB36 y tres RAMB18. Setup/hold internos cumplen a **+0,062/+0,010 ns**;
98.664 redes ruteables están completas, sin errores. Los tres huecos
explícitos corresponden a los relojes padre provisionales, comprobando
1.862 alias PARTIAL contra sus rutas completas: motor 1.680, host 163 y
gestión 19. No son huecos de datos.

La estructura de reset/estado/IRQ coincide con la auditoría de síntesis.
Los 33 bits dinámicos del paquete AXI-Lite R conservan un único destino;
el bit 32 es tierra constante. Se mantienen 21 waivers de proveedor y las
excepciones propias a CLR. Solo desaparecen tres false paths internos de
reset-busy cuyas celdas elimina implementación; CDC-3 pasa de 40 a 37.
La revisión rechaza 18 alteraciones de evidencia.

Persisten CDC-15, cinco CDC-26 y **RTSTAT-10: 1.377 redes sin cargas
ruteables**. La auditoría completa identifica 1.332 puertos de salida sin
carga externa y 45 señales sin uso del FIFO AMD, nueve por cada canal AXI.
Todas conservan conductores colocados y ninguna tiene cargas internas
pendientes de ruteo. Se contrastaron las direcciones de los puertos con
la auditoría independiente y se rechazaron ocho alteraciones de evidencia.
Se conserva el intento Tcl fallido y no se oculta ni rebaja el aviso.
Faltan retardos de 1.402 entradas/1.344 salidas y contexto físico de placa;
el cumplimiento interno OOC no cierra V8 ni autoriza fabricación.

## Pérdida automática de reloj con cómputo real

`tb_ntrpu_ffn_clock_loss.sv` deriva del banco anterior y conserva el RTL,
HLS y BFM. Detiene el reloj efectivo del DUT en fase baja, mientras un
oscilador de referencia y gestión siguen funcionando. El motor se detiene
en fases 0/2 y el host en 1/3; el perfil lento invierte esa selección.
Cada pérdida se prolonga al menos 1 µs después del bloqueo, sin pulsar
reset externo. El banco exige reset/admisión cerrados hasta que vuelva
el reloj, y una FFN completa después de la última recuperación.

La memoria sigue atendiendo hasta el reset real y limpia entonces el
protocolo, conservando los bytes escritos. Durante el tiempo de detección
pueden completarse más de los 64 bytes permitidos por el banco de reset
inmediato. El modo explícito `delayed_reset` del revisor conserva la
referencia numérica, offsets consecutivos, límites, fases y contadores;
el verificador de pérdidas añade causalidad temporal y prohíbe escrituras
posteriores al reset. El criterio original sigue siendo el predeterminado;
sus tres semillas anteriores continúan pasando.

La revisión reconstruye todos los flancos omitidos a partir de las pausas,
comprueba que no haya handshakes con su reloj detenido y exige las cuatro
secuencias STOP/BLOCKED/RESUME/RECOVERED. Ambos perfiles suman **126 casos,
24 pérdidas, seis arranques y 216 alteraciones rechazadas**. Cada combinación
de fase y reloj detenido aparece tres veces. Las 24 FFN normales comparan
1.344 valores FP32 exactos; seis de esas FFN siguen a la última recuperación.
Se verifican 16.324 handshakes B y 17.502 R.

El bloqueo se observa a 640,699–689,999 ns de detener el reloj; la recuperación
se observa a 594,999–704,999 ns de restaurarlo. El observador muestrea en
flancos negativos de gestión: son intervalos RTL, no medidas eléctricas.
La primera prueba negativa del perfil lento se corrigió porque intentaba
alterar un enable ya activo; ahora selecciona el reloj realmente detenido.
Se repitieron las 36 alteraciones por semilla, sin cambiar el DUT ni los logs.
Estos bancos mantienen inactivo
el AXI4 DMA del host. Siguen pendientes una FFN completa después de cada
pérdida individual, resets durante RELEASE y recuperación con DMA/DDR.

### Ampliación validada: trabajo tras cada pérdida

El modo explícito `recovery_each_` conserva los 15 casos iniciales y añade
cuatro pares de pérdida/FFN correcta, seguidos del descriptor inválido:
24 casos por semilla. El revisor exige los IDs de fallo 16/18/20/22 y
trabajos completos 17/19/21/23. Comprueba sus bytes contra la referencia,
admisión posterior a la recuperación, retiro anterior al siguiente fallo
y contadores ABI2 reinicializados. Añade negativos para pérdida de la
evidencia de recuperación, propietario incorrecto, cierre prematuro y
resultado numérico corrupto. La cobertura anterior no satisface este modo.
No cambia el RTL sintetizable ni las restricciones del diseño ruteado.

La primera ejecución rápida conserva un rechazo independiente en la
semilla 7, caso 18. `phase_bytes` contaba W aceptados, pero AXI permite
recibir W antes de AW; la pérdida precedió a la primera escritura física
de la fase elegida. El disparador revisado espera una marca puesta después
de registrar la escritura con dirección y bytes comprobados. El criterio
temporal del revisor permanece intacto. El perfil lento anterior pasa,
pero se repiten ambos perfiles con el banco corregido para mantener una
misma revisión de prueba.

Los dos perfiles corregidos pasan **144 casos y 240 alteraciones negativas**,
con **42 FFN completas/2.496 FP32 exactos**. Las 24 FFN que siguen a cada
pérdida comparan 1.536 de esos valores. Se revisaron 18.706 handshakes B
y 22.873 R. El bloqueo se observa a 644,999–689,999 ns; la recuperación,
a 594,999–706,499 ns. Ambos resultados y fuentes están respaldados.
Los primeros 15 casos conservan trazas idénticas en las seis semillas;
solo cambia la traza completa de la semilla rápida que había disparado
antes de AW. El criterio temporal anterior sigue rechazando ese intento.

### Resets durante RELEASE: defecto CDC y corrección

`ffn-supervised-release-queue` verificó revisión y respaldo de ambos perfiles
corregidos y lanzó las dos tandas a las 04:54 UTC. Ambas fallaron por XPM S-1,
aunque el primer banco numérico alcanzó su marca procedimental de fin. El modo
`release_retry_recovery_each_` retiene la disponibilidad de memoria,
espera ambos resets locales liberados con admisión cerrada y aplica tres
pulsos externos de 78,7 ns. La cuarta liberación permite recuperar memoria
y comenzar los 24 casos. El revisor exige tres aserciones adicionales,
64 flancos reales mínimos en cada retención y ninguna admisión transitoria;
añade seis alteraciones negativas. Es cobertura de reinicio durante el
arranque: aún no demuestra reset durante recuperación de DMA activo.

Los pulsos truncaban los altos de `host_up`, `device_up` y, según el perfil,
`host_operational` o `host_armed`, violando el contrato de estabilidad del
XPM de nivel. Los logs completos de ambos fallos se conservan; no se
desactivaron aserciones. Se modificaron tres archivos RTL: supervisor,
transporte IRQ y wrapper supervisado. La diferencia exacta está registrada
en `ffn-supervised-level-source-change-review.json` dentro del respaldo local.

El banco aislado corregido separa los estímulos de los flancos de muestreo.
Se conserva como no calificado el PASS procedimental R2 que dependía del
orden de eventos, además de tres errores iniciales de compilación. R3 pasa
la referencia independiente de flancos y rechaza los seis mutantes RTL.
Las regresiones del supervisor y del transporte también pasan. Los dos
perfiles completos `053018Z` pasan y tienen respaldo verificado: 144 casos,
18 resets RELEASE, 24 pérdidas y 42 FFN completas/2.496 FP32 exactos. Las
24 FFN posteriores a cada pérdida comparan 1.536 de esos valores. Se
revisaron 18.706 handshakes B y 22.873 R; las 276 alteraciones fueron
rechazadas. El bloqueo se observa a 644,999–689,999 ns y la recuperación
a 594,999–706,699 ns. Son tiempos RTL con pares de memoria simulados.

**El ruteo anterior no califica este RTL.** La síntesis `054240Z` y su auditoría
comprobaron 45 CLR: seis cadenas del supervisor, ocho de estado y una de
armado. Verificaron INIT, ASYNC_REG, flancos de destino, conexiones entre
etapas y 11 inversiones LUT1 de una sola fuente. Los cuatro estados de
pares externos siguen siendo contratos pendientes, aunque ahora Vivado
los clasifique CDC-9. La estimación de síntesis es +0,173/+0,013 ns; no es
timing ruteado ni supera el fallo funcional descubierto después.

### Reset justo al completar la retención

El diagnóstico `tb_ntrpu_reset_hold_retry.sv` provoca reset después de la
primera muestra de gestión del nivel `done`, alternando los contadores
de host y motor. No fuerza estado interno, no solicita rearmado y conserva
cerrada la admisión. Con periodos 4/5 ns y 31/23 ns reproduce **16 XPM S-1**
en `host_count_cdc`/`device_count_cdc`. Por eso se retuvo el ruteo de la
revisión de 45 CLR, aunque su estructura y timing estimado pasaran.

Esos dos cruces usan ahora `ntrpu_reset_release`: bajo asíncrono, alto tras
tres flancos de gestión. La corrección exacta se registra en
`ffn-supervised-hold-source-change-review.json`. Los mismos 16 reintentos
pasan sin errores y la regresión de 36 casos también pasa; ambos están
respaldados. La FFN completa `060223Z` pasa en ambos perfiles y está
respaldada: 144 casos, 18 resets RELEASE, 24 pérdidas, 42 FFN completas,
2.496 FP32 exactos y 276 alteraciones rechazadas.

La síntesis `061341Z` y auditoría `062607Z` también están revisadas y
respaldadas. Verifican **51 CLR**: ocho cadenas del supervisor y las nueve
de estado/armado, 13 inversiones de polaridad y 131 registros IRQ
persistentes. Conservan los 17 grupos de skew y las excepciones AMD;
12 alteraciones de evidencia son rechazadas. Recursos: 55.362 LUT,
51.313 FF, 153 DSP, 48 RAMB36 y tres RAMB18. La estimación de síntesis
es WNS +0,173/WHS +0,013 ns; no es timing ruteado. Permanecen CDC-15,
CDC-26 y el contrato de cuatro fuentes externas registradas. El ruteo
`20261004T064105Z` terminó y **falla setup a 200 MHz**: WNS −0,235 ns,
TNS −6,019 ns y 103 extremos afectados; WHS +0,010 ns. Usa 54.777 LUT,
50.985 FF y 153 DSP. Las 98.682 redes ruteables están completas. El camino
crítico tiene 17 niveles y 4,913 ns de datos, de los cuales 3,534 ns son ruteo.

El ruteo y cuatro auditorías están revisados y respaldados. La optimización
absorbió 13 inversores LUT1 en 39 pines CLR invertidos; las otras 12 etapas
conservan CLR no invertido. La extracción física comprueba las 51 etapas,
INIT, reloj, conexiones entre etapas y el BUFGCE no inversor que distribuye
el reset del motor. Se rechazaron 18 alteraciones de evidencia de ruteo,
ocho de conectividad y nueve de polaridad. Los cuatro CLR de pares externos
pasan de CDC-9 a CDC-26 en el reporte físico; estos avisos quedan abiertos.
RTSTAT-10 conserva 1.332 salidas externas y 45 señales AMD sin carga.

La tanda `20261004T072337Z` aplica `phys_opt_design -directive Explore` al
checkpoint ruteado en 737,089 s. La revisión independiente confirma **WNS
+0,023 ns y WHS +0,010 ns**, sin extremos fallidos de setup, hold o pulsos.
Usa 54.795 LUT, 50.985 FF y 153 DSP: 18 LUT adicionales frente al ruteo
inicial. El nuevo peor camino tiene 16 niveles y 4,650 ns de datos.

Todos los comandos XDC son idénticos antes y después de optimizar, incluidos
periodos 5/4/10 ns, sitios OOC, excepciones y 21 waivers AMD. Los 17 grupos
de skew cumplen. Las auditorías de estructura `073604Z`, reloj `073751Z`,
conectividad `074012Z` y polaridad `074119Z` conservan los contratos revisados;
se rechazan 22 alteraciones de evidencia conjunta, ocho de conectividad y
nueve de polaridad. Los cinco resultados, fuentes y archivos de evidencia
tienen respaldo verificado localmente y en NFS.

**Es timing interno OOC con margen de 23 ps**, no cierre de placa. Persisten
los tres huecos de reloj padre, retardos externos sin definir, cuatro fuentes
de pares por integrar y advertencias CDC/RTSTAT-10. Las comparaciones numéricas
califican RTL/HLS; esta tanda no ejecuta simulación de netlist con SDF.

## DMA interrumpido y RAM persistente

`tb_ntrpu_ffn_dma_recovery.sv` conecta un iniciador AXI4 de 512 bits a los
conversores AMD, el árbitro y los núcleos HLS reales. El nuevo
`ntrpu_axi_dma_bfm.sv` abandona sus esperas mediante reset asíncrono, incluso
si su reloj se detiene. La confirmación del host incluye el estado de DMA,
AXI-Lite e IRQ. La memoria limpia protocolo al reset, conservando payload.

Los perfiles `062528Z` (host 4 ns) y `064115Z` (6,2 ns) pasan la revisión
independiente. Cada uno ejecuta tres semillas, con cinco cortes del reloj
host y cinco del motor por semilla: AW sin W, W sin AW, B retenida, AR
pendiente y R retenida. No se aplica otro reset externo para recuperarlos.
Tras cada fallo se recargan 3.328 bytes mediante AXI, se ejecuta una FFN
D32/H32, se leen 128 bytes y se completa el retiro ABI2.

Los seis perfiles/semillas suman **60 fallos, 2.100 transacciones, 60 FFN y
1.920 FP32 exactos**. El revisor reconstruye la RAM desde cada escritura
física, compara 60 hashes de retención y 42.240 bytes de cómputo, y rechaza
198 alteraciones de logs. En las pruebas B retenida, los bytes ya escritos
permanecen después del aborto; la recarga posterior los reemplaza por AXI.
Ambos respaldos están verificados en NFS y localmente, incluidos cada
archivo y su copia de fuentes. La detección observada tarda 645–690 ns y
la recuperación 643,699–805,999 ns de simulación.

**Alcance:** los fallos de este banco DMA ocurren con el motor en reposo.
La concurrencia se prueba en el banco siguiente; XDMA/PCIe integrado y DDR
siguen pendientes. Son pesos
sintéticos y memoria conductual. Los modelos FIFO AMD no representan los
retardos de sincronización; estos ciclos no miden rendimiento de placa.

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_dma_recovery.tcl \
  -tclargs MAC_RUN SWIGLU_RUN supervised_sim_irq_errors_quad_fp32_fifo build/DMA_RUN_NUEVO
python3 tools/check_ntrpu_ffn_dma_recovery.py build/DMA_RUN_NUEVO/seed-1.log \
  --seed 1 --host-period-ps 4000
# Repetir semillas 7/42; perfil lento: ...quad_slow_fp32_fifo y 6200 ps.
```

## DMA y cómputo activos durante la pérdida de reloj

`tb_ntrpu_ffn_dma_active.sv` conserva el RTL y HLS calificados. Inicia una
lectura DMA de pesos mientras una de las cuatro fases de FFN está activa.
Tras la primera escritura física de esa fase, retiene legalmente B para
mantener actividad; corta el reloj de host o motor con AR pendiente o R
retenida. No fuerza señales internas ni usa otro reset externo de rescate.

Los perfiles `20261004T065918Z` y `20261004T070826Z` están revisados y
respaldados. Tres semillas por perfil suman **96 abortos simultáneos,
3.360 transacciones y 96 FFN completas tras recuperación**, con **3.072 FP32
exactos**. El revisor reconstruye 105.984 bytes escritos por el motor:
38.400 parciales antes del aborto y 67.584 de trabajos completos. Comprueba
retención de RAM, épocas, retiro ABI2 y ausencia de completaciones antiguas;
rechaza 186 alteraciones de evidencia. La detección tarda 642,3–690 ns y
la recuperación 639,999–805,999 ns de simulación.

El tráfico concurrente probado son lecturas DMA; no califica una política
de escrituras concurrentes sobre buffers activos. Se mantienen los límites
de memoria conductual, backpressure inducida y relojes simulados. No mide
ancho de banda físico ni demuestra funcionamiento con DDR/XDMA.

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_dma_active.tcl \
  -tclargs MAC_RUN SWIGLU_RUN supervised_sim_irq_errors_quad_fp32_fifo build/DMA_ACTIVE_NUEVO
python3 tools/check_ntrpu_ffn_dma_active.py build/DMA_ACTIVE_NUEVO/seed-1.log \
  --seed 1 --host-period-ps 4000
# Repetir semillas 7/42 y perfil lento con 6200 ps.
```

```bash
# MAC_RUN y SWIGLU_RUN son exports HLS identificados por manifiesto y hashes.
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs MAC_RUN SWIGLU_RUN supervised_sim_irq_errors_quad_fp32_fifo build/RUN_NUEVO
python3 tools/check_ntrpu_ffn_supervised.py build/RUN_NUEVO/seed-1.log \
  --seed 1 --host-period-ps 4000
# Repetir semillas 7 y 42; perfil lento: ...quad_slow_fp32_fifo y 6200 ps.
# Negativo: supervised_bypass_sim_irq_errors_quad_fp32_fifo debe fallar.
# Pérdidas de reloj con FFN completa después de cada recuperación:
vivado -mode batch -source tools/ntrpu_board_ffn_clock_loss.tcl \
  -tclargs MAC_RUN SWIGLU_RUN recovery_each_supervised_sim_irq_errors_quad_fp32_fifo build/OTRO_RUN_NUEVO
python3 tools/check_ntrpu_ffn_clock_loss.py build/OTRO_RUN_NUEVO/seed-1.log \
  --seed 1 --host-period-ps 4000 --recovery-each
# RELEASE: prefijar el modo con release_retry_ y añadir --release-retry al revisor.
# Calificador aislado; early/late deben detectar el RTL deliberadamente defectuoso:
vivado -mode batch -source tools/ntrpu_board_level_qualifier.tcl \
  -tclargs sim build/LEVEL_RUN_NUEVO
```

## Dependencias siguientes

El [banco de arranque frío sobre MIG](board/DDR_FFN_COLD.md) conecta el
RTL vigente al controlador y modelo genéricos de 128 GiB. Los perfiles de
host de 4 y 6,2 ns están revisados y respaldados: seis FFN/192 FP32 exactos.
Esto añade lectura y escritura a través de MIG en simulación BFM;
los acuses de pares y la comprobación de rangos siguen siendo del banco.

Mantener el cierre interno y sus auditorías al integrar los pares reales.
La implementación usa posiciones padre provisionales
`BUFGCE_X0Y60/61/62` para motor, host y gestión. Esas posiciones no asignan
relojes de placa ni resuelven sus fronteras. Los resultados ruteados de
revisiones anteriores no califican el RTL actual.

Definir propiedad de buffers antes de permitir escrituras DMA concurrentes:
no modificar pesos o salidas de un trabajo activo. Conservar las lecturas
concurrentes, casos AW/W independientes y respuestas retenidas comprobados.
Integrar confirmaciones
registradas de pares reales, sin sustituirlas por constantes.
La memoria DDR y sus restricciones siguen las dependencias V2/V3 de
[la cola de diseño](board/VIVADO_QUEUE.md).
