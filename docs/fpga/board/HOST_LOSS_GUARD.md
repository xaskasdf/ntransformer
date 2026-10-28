# Guard de pérdida del host

Extensión V4/V7, 2026-10-05. `fpga/rtl/ntrpu_host_loss_guard.sv` implementa
detección y secuenciación. Su integración con el propietario NTRD y el
arranque del endpoint PCIe nativo están aprobados. El banco de PERST pasó
su revisión funcional y 41 alteraciones negativas; el respaldo local/NFS
quedó verificado a las 14:58:19 UTC. La coordinación del reset aún no está implementada.
Complementa el [adaptador retenido](PCIE_DUAL_LINK_LOSS.md) y el
[contrato de reset del endpoint](ENDPOINT_RESET_COORDINATION.md).
El RTL, bancos y scripts de cada intento quedan congelados con sus respaldos.

## Contrato de integración

- `mgmt_clk` debe seguir presente cuando se detengan host o fuente.
  `cold_rst_n` pertenece a la vida de configuración; **no conectar PERST
  ni un reset operativo a esa entrada**.
- `link_up_async` y `endpoint_reset_async` son observaciones. Las cadenas
  existentes `ntrpu_reset_release` prolongan su aserción y sincronizan la
  liberación en tres flancos. Los dos `ntrpu_clock_probe` usan ida/vuelta
  estable y timeout de 64 ciclos de gestión; no un toggle libre que pueda
  ocultarse por alias entre relojes.
- `arm`, `peers_held` y `peers_ready` son contratos del dominio de gestión.
  La sesión abre una vez, después de relojes/enlace válidos, reset compartido,
  ambos contadores de espera y reconocimiento de los peers. La ausencia de
  enlace durante arranque mantiene cerrado el sistema sin inventar pérdida.
- `source_fault` se registra de forma persistente en fuente.
  `source_invalidate` debe cerrar la propiedad NTRD y la frontera retenida.
  `source_invalid_ack` debe acreditar que el propietario **ya retuvo** esa
  invalidación y no permitirá reutilizar la reserva. No significa que todas
  las transferencias hayan drenado.
- `host_admit` y `source_admit` autorizan transacciones nuevas. No usarlos
  como un AND sobre VALID ya presentado: deben conservarse payloads y
  obligaciones previas. El adaptador retenido cumple esa función.

## Pérdida y restauración

La secuencia es: detectar y retener causa → cerrar admisión → invalidar en
fuente → registrar ACK del propietario → devolver ACK a gestión → pedir
reset de los peers y ambos extremos CDC. Sin ACK, el reset de transporte
queda pendiente; esto también aplica si el reloj fuente está detenido.

Los resets comparten aserción y cada liberación se sincroniza localmente.
No se liberan hasta observar al menos 64 flancos en **cada** dominio,
relojes/enlace restablecidos y `peers_held`. Restaurar las interfaces y
`peers_ready` lleva a `FAULTED`, con admisión cerrada e invalidación retenida.
Una pérdida posterior repite la espera de reset; ARM no reabre la sesión.

`causes[4:0]` retiene, de mayor a menor bit: fallo fuente, reloj fuente,
reloj host, reset endpoint y pérdida de enlace. `first_loss_tick` conserva
el primer ciclo observado de gestión; su contador de 64 bits satura.

El endpoint real puede resetearse antes de recibir nuestra petición.
La coordinación con sus interfaces adyacentes, señales eléctricas y PERST
sigue pendiente; este bloque no demuestra que podamos demorar el reset
interno del endpoint. Tampoco demuestra recuperación caliente DDR.

## Evidencia funcional

El intento `20261005T102730Z-V7-ffn-supervised-host-loss-guard` termina
con código 0 y diez marcadores PASS, pero la revisión lo rechaza:
el productor del test retiraba VALID de una petición bloqueada sin
handshake ni reset. Su respaldo conserva ese fallo; no aprueba el guard.

R2 `20261005T103302Z-V7-ffn-supervised-host-loss-guard-r2` mantiene
esa oferta hasta el siguiente reset de configuración, sin cambiar el RTL.
Pasa los diez casos en 33,543 s y 43,180 µs simulados. Se revisan 4.284
muestras de gestión, 8.366 de fuente y 5.289 de host; 15 alteraciones de
evidencia y el intento anterior son rechazados.

| Estímulo | Observación en simulación |
| --- | --- |
| Sin enlace/arm/peer listo durante arranque | Admisión cerrada, sin invalidación ficticia |
| Reset de 2 ns o pérdida de enlace de 1 ns | Capturados entre flancos de gestión |
| Reloj host detenido en fase baja/alta | Detección en 642,220 / 673,320 ns |
| Reloj fuente detenido | Detección en 684,482 ns; ningún reset CDC antes de volver y obtener ACK |
| ACK del núcleo demorado | Peticiones y respuestas retenidas, reset pendiente |
| Pulso de fallo fuente | Causa retenida aunque termine el pulso |
| `peers_held` demorado | Resets siguen activos tras completar contadores |
| Segunda pérdida tras restauración | Nuevo intervalo de reset, sin rearmar la sesión |

En los diez intervalos de reset operativo hay al menos 71 flancos host
y 120 de fuente dentro de la aserción compartida. Se conservan 20 peticiones
hasta entregarlas; las respuestas suman 29 capturas, dos entregas normales
y 27 retiros locales. Las pruebas usan XPM reales y dos canales retenidos,
pero el ACK de propiedad y los peers son estímulos controlados. **Faltan
los AXI Clock Converters, núcleo NTRD, endpoint y MIG en esta integración.**
Los pulsos y latencias son medidas comportamentales; no califican mínimos
eléctricos, metastabilidad ni timing de placa.

## Integración con los CDC AMD

`20261005T105941Z-V7-ffn-supervised-host-guarded-boundary` pasa 17 casos
con los ocho canales retenidos, el guard y AXI Clock Converter 2.1 reales:
AXI4 de 512 bits/dirección de 64 bits y AXI-Lite de 32 bits, ambos con tres
etapas de sincronización. Tiempo de herramienta: 45,621 s. Se verifican
80 capturas, 66 entregas y 14 retiros locales; 15 alteraciones son rechazadas.

Incluye AW/W parciales, respuestas bloqueadas, ACK demorado 205 ns,
relojes host/fuente detenidos y dos resets con respuestas tardías. Los 17
intervalos de reset contienen al menos 71 flancos host y 120 de fuente.
El reloj host detenido se detecta en 632,770–685,970 ns y el de fuente en
670,001 ns; sin reloj fuente, el reset espera la invalidación y su ACK.

En el caso 14 el host recibe B antes del handshake del productor, pero
este ACK llega durante la detección de pérdida, antes del reset: cuenta
como entrega local. No debe confundirse con la prueba de reset anticipado
del banco anterior. Se comprueba el orden con trazas de ambos dominios.

ACK de propiedad y peers siguen controlados en este banco. No incorpora
el propietario NTRD, endpoint PCIe ni DDR, y no califica timing físico.
Respaldo: 132 archivos / 33.454.142 bytes; SHA-256
`785b89561481c97452e655cac136b6471b0efe49c73dcdae1ec9d2ea8b958471`.

## ACK del propietario NTRD real

`20261005T111635Z-V7-ffn-supervised-guarded-owner` prueba el nuevo
`ntrpu_pcie_fragment_guarded_host.sv`: control NTRD, recolector y puente
retenidos, los ocho canales, guard y ambos CDC AMD. CMD/DATA/CPL del pool
y los peers adyacentes son contratos externos controlados; no incluye MIG
ni endpoint. `cold_rst_n` conserva la vida de configuración y nunca recibe
el reset de transporte. El ACK viene de `session_lost && session_invalid`,
no del pedido combinacional de invalidación.

Pasan nueve escenarios en 53,142 s / 54 µs simulados: operación sana,
fragmento parcial, CMD/DATA bloqueados, reloj fuente/host detenido,
resultado exitoso ya visible, AW de datos/control sin W y pérdida del pool.
Se reconstruyen nueve ARM, siete líneas/CMD/DATA, dos completions y un ACK
de retiro; se comparan 15 frames completos y todos los MMIO host/fuente.
La dirección y el cookie cruzan 32 bits; se conservan época y reservas.

- La invalidación del host queda registrada por NTRD un ciclo fuente
  después de llegar a la frontera (5 ns); el reset espera el retorno CDC.
- Detección de reloj host: 634,270 ns; fuente: 645,001 ns. Sin reloj fuente
  no hay ACK ficticio ni reset prematuro.
- Los ocho resets operativos mantienen al menos 69 flancos host y 118 fuente
  dentro de la ventana compartida medida. Restaurar enlace y ARM no reabre.
- CMD/DATA ya expuestos permanecen estables y se entregan una sola vez.
  Las respuestas tardías no producen un resultado exitoso nuevo. Un resultado
  visible anterior permanece inmutable, pero no autoriza retiro ni reutilización.
- Los AW incompletos quedan dentro del núcleo retenido. Este banco drena la
  frontera; los retiros de respuestas se cubren en el banco anterior de 17 casos.

El verificador inicial queda conservado: exigía datos definidos incluso
con VALID=0 en las FIFO AMD. La revisión R2 limita esa exigencia a ofertas
válidas y reconoce el orden inverso de pérdida originada en el pool:
primero NTRD retiene la pérdida, después la comunica al guard. No cambian
RTL ni estímulos; la revisión tiene hash propio. Rechaza 17 alteraciones.

Respaldo: 143 archivos / 36.802.339 bytes; SHA-256
`c1345f3c44eaf429f7894455a4e5b873b93fbd1a4886372f5442d9b76e2f06ee`.
Sigue pendiente conectar el pool/MIG real, coordinar reset independiente del
endpoint y calificar timing físico. Las barreras externas deben incluir
los CDC y la frontera; `transport_idle` solo describe el recolector.

## Síntesis de la composición NTRD

`20261005T112456Z-V7-ffn-supervised-guarded-owner-synth` conserva el RTL
del banco funcional y sintetiza ambos CDC con sus restricciones AMD.
En 208,705 s informa 4.083 LUT y 8.720 registros, sin latches/BRAM/URAM/DSP.
La cifra incluye guard, frontera y propietario; no pool, MIG, PCIe ni cómputo.

WNS estimado −0,513 ns, WHS +0,013 ns; 12 extremos setup fallidos,
ocho TIMING-6/7 y 979 entradas / 3.779 salidas sin retardos externos.
El reporte TIMING-18 limita su lista a 1.000; no representa el total.
CDC informa 30 cruces de datos y 20 de reset sincronizados. Los 15 grupos
de bus skew cumplen en esta estimación sin ruteo; no acredita timing físico.

Se conservan 873 avisos de puertos opcionales sin carga dentro de módulos
AMD, dos RAM del canal de lectura desusado eliminadas y un aviso sobre
un waiver jerárquico del fabricante. No se cambian severidades ni se añaden
excepciones. DRC informa cero hallazgos, con cobertura OOC incompleta.
La tanda queda como **diagnóstico, sin aprobación de implementación**;
se auditan las cadenas de reset dentro de esta netlist antes de restringirlas.

Respaldo: 94 archivos / 34.695.084 bytes; SHA-256
`e377a5d5b50f39952b1931fd411cd65e32f083ae247028f453a9c372de76e254`.

### Auditoría integrada y prueba de restricciones

La auditoría `20261005T113249Z-V7-ffn-supervised-guarded-owner-audit`
verifica las 30 FDCE, INIT=0, ASYNC_REG, relojes, D/Q y origen de CLR.
Las 48 rutas max/min examinadas, entre gestión y host/fuente, terminan
exactamente en 24 CLR; incluyen los 12 extremos negativos. No modifica
el checkpoint ni reclasifica los cruces AXI entre host y fuente.
Respaldo: 27 archivos / 2.940.303 bytes; SHA-256
`91e7bd58200fe3358d26b611951f30825d164c46ef3e29d607852d98dc1f0327`.

`20261005T113525Z-V7-ffn-supervised-guarded-owner-constraints` repite
la auditoría y verifica CE constante antes de exceptuar solo esos 24 CLR.
Mantiene todos los comandos XDC anteriores, incluidos los de AMD AXI/XPM.
Los ocho avisos críticos desaparecen; WNS estimado +1,940 ns, WHS +0,013 ns,
recursos y los 15 grupos de bus skew permanecen iguales. Los retardos
externos siguen sin definir: **no es cierre físico**.

La aprobación de esta prueba queda pendiente de comparar la lógica completa.
La parte visible de los Verilog coincide, pero su bloque protegido cambia
al exportarlo. No se descarta ese bloque para declarar equivalencia.
Respaldo: 46 archivos / 8.108.267 bytes; SHA-256
`98e20c4dcc5da786f724bd9406d055d91fdefe8aa3dc651b9e81bece212238c0`.

### Comparación EDIF: equivalencia completa pendiente

`20261005T114041Z-V7-ffn-supervised-guarded-owner-edif` exportó ambos
checkpoints en una carpeta común: la segunda exportación sustituyó los
auxiliares protegidos `.edn`. Se conserva como diagnóstico incompleto,
sin aprobar la preservación lógica. El primer revisor se detuvo al buscar
un parámetro INIT_A ausente; no emitió una aprobación.

La repetición `20261005T114859Z-V7-ffn-supervised-guarded-owner-edif-r2`
usa carpetas separadas. En 52,628 s obtiene 15.490 primitivas, 58.000 nets,
5.098 puertos y tres relojes por checkpoint. Coinciden las identidades de
primitivas y el EDIF principal salvo su fecha, pero **los 63 auxiliares
protegidos difieren**. No se eliminan ni normalizan para declarar igualdad.
La prueba de equivalencia completa sigue abierta; el checkpoint con
restricciones no queda aprobado como entrada de implementación.

Ambos intentos tienen respaldo local y NFS verificado:

| Tanda | Archivos / bytes comprimidos | SHA-256 |
| --- | --- | --- |
| `114041Z` | 85 / 8.256.625 | `66ea11fa4695d5b73f357e1f978867c460bc1190ecce5b94a81ea20354bff234` |
| `114859Z` | 147 / 8.489.879 | `5165cd3e756696c38db73012188a66def2d5647389124e76aa514120f2a72f9a` |

La revisión retrospectiva de `080626Z` encuentra 189 auxiliares protegidos
en carpeta común. Se retira su afirmación de preservación lógica completa,
conservando el archivo y la revisión originales. La corrección y el inventario
están en `guarded-owner-prior-edif-caveat-20261005T120700Z.json`, SHA-256
`c8c1182d6b112d6ea598e0343942c1a577c1763b08935b5cb6978aebb38c3aa0`.
Las simulaciones funcionales independientes conservan sus resultados, pero
no reparan esta falta de evidencia estructural ni acreditan timing físico.

### Integración con dos MIG: primera tanda y corrección del banco

`ntrpu_pcie_fragment_dual_host_guarded` conecta el guard y el propietario
NTRD al pool y los dos caminos MIG existentes. La memoria conserva su
vida de configuración; los resets de transporte no la reinician.
`tb_ntrpu_pcie_guarded_mig` repite 18 trabajos sanos y añade un trabajo
con B de memoria retenida, pérdida de enlace, completion tardía y lectura
de los 64 bytes escritos. Exige conservar cookie, época y reserva, y
bloquear publicación y reutilización después de la pérdida.

`20261005T120124Z-V7-ffn-supervised-guarded-mig` terminó en 616,530 s
con un fallo del banco, `RESERVATION_RETAINED`. Los 18 trabajos previos
pasan la comparación independiente: 16 escrituras, 28 lecturas, 35 frames,
276 escrituras MMIO y diez lecturas MMIO. El caso adicional lee correctamente
los 64 bytes después de perder el enlace, pero no alcanza el cierre global.

La comprobación fallida exigía que siguiera ocupado el buffer de datos.
El contrato existente devuelve ese buffer después del último W interno
y descuenta la operación pendiente al aceptar B. La completion bloqueada
conserva su descriptor y la reserva del puente; no necesita conservar el
buffer de staging. R2 observará explícitamente `occupied[0] == 1`, con
`banks_used == 0` y `pending_counts == 0`, sin cambiar el RTL del diseño.

También se adapta explícitamente la cobertura del verificador: el corte
retenido acepta B/R del núcleo de inmediato, por lo que sus pausas se
comprueban en el lado host. Se conservan todas las comparaciones semánticas
ciclo a ciclo del núcleo. La primera tanda queda **no aprobada en conjunto**:
faltan los controles finales, incluido el drenaje de lecturas periódicas.

Respaldo de `120124Z`: 709 archivos / 115.098.809 bytes, SHA-256
`c9c206eccdb9c848c300d7428b86bf32eaee113d6d0254da95c4e5775f67791e`,
verificado localmente y en NFS a las 12:16:06 UTC.
R2, `20261005T121701Z-V7-ffn-supervised-guarded-mig-r2`, queda **aprobada
para este banco funcional** tras 621,273 s de ejecución y revisión independiente.
Son 19 ARM, 17 escrituras, 29 lecturas y 37 frames. Las primeras 16
completions se consumen; la última queda bloqueada, con `occupied[0] == 1`,
`occupied[1] == 0`, buffer de staging libre y ninguna B pendiente.
Se conservan cookie `0x100000010` y época `0x100000013` durante 391 muestras
consecutivas del reloj fuente; no se publica resultado ni se permite retirar
o reutilizar la reserva después de la pérdida.

En esta simulación, gestión observa el ACK del propietario 34 ns después
del pulso de pérdida y solicita reset a los 74 ns. La invalidez precede al
ACK registrado una arista fuente; el verificador R3 comprueba ese orden
explícito, sin cambiar DUT ni estímulos. Ambos resets abarcan al menos
64 aristas locales, mientras memoria conserva su vida de configuración.
La lectura posterior compara los 64 bytes en el último bloque de 128 GiB.
Estos tiempos describen el banco, no una medida física de la tarjeta.

El observador DDR verifica 61/57 lecturas periódicas completas y 15/14
lecturas de usuario, con nueve/ocho escrituras físicas en los modelos.
Se conservan los 32 avisos de compilación/generación, 15 avisos FIFO y
2.304 avisos de modelo atribuidos mediante el observador; no se ocultan.
El verificador rechaza 18 alteraciones de identidad, memoria, reset,
descriptor y continuidad de la traza.

Respaldo R2: 709 archivos / 115.118.458 bytes, verificado localmente y
en NFS a las 12:30:06 UTC. SHA-256
`5fe9ab7226661f2afa136a45deeae5633305c8d9e9a85ba5161426dca740dc94`.
La fuente ejecutada tiene SHA-256
`b7252f3e544213d8fdcfaadf8b72246e1b09aa026f6a8944c423e2f729c52354`.

La ejecución es una prueba RTL nueva, independiente del checkpoint de
restricciones pendiente. Usa MIG y modelos DDR4 genéricos, host AXI de
banco y confirmaciones de pares controladas. Todavía no califica el
endpoint PCIe nativo, el DIMM comercial ni el reset eléctrico de placa.

## Síntesis del guard aislado

`20261005T103345Z-V7-ffn-supervised-host-loss-guard-synth` termina en
88,274 s: 94 LUT, 239 registros, sin latches/BRAM/URAM/DSP. La estimación
con gestión a 10 ns, fuente a 5 ns y host a 7,3 ns informa WNS **−0,513 ns**,
12 extremos setup fallidos y ocho avisos críticos TIMING-6/7. CDC reconoce
siete cruces de datos sincronizados y diez de reset; eso no elimina los
avisos de timing.

Hay ocho entradas y 84 salidas sin retardos externos, 92 TIMING-18,
orígenes físicos de reloj sin definir y cobertura DRC OOC incompleta.
La tanda se conserva como diagnóstico, **no aprobada para implementar**.
Revisar primero las 30 FDCE de liberación, su INIT, ASYNC_REG, encadenamiento
y origen de CLR, además de todos los extremos entre relojes; no introducir
excepciones generales para obtener un reporte verde.

### Auditoría y revisión de restricciones

La auditoría de solo lectura `20261005T103904Z-V7-ffn-supervised-host-loss-guard-audit`
comprueba 30 FDCE en diez cadenas de tres etapas: INIT=0, ASYNC_REG,
reloj, D/Q y origen de CLR. Las 48 rutas max/min corresponden exactamente
a 24 CLR de ocho cadenas; entre ellas están los 12 extremos negativos.
Las seis etapas conectadas a reset/enlace externos se mantienen separadas.

`20261005T104435Z-V7-ffn-supervised-host-loss-guard-constraints` repite
la auditoría, comprueba CE constante y añade únicamente excepciones a esos
24 CLR en una copia del checkpoint. Conserva las restricciones XPM y las
rutas D/Q; no agrupa relojes ni cambia severidades. Los 74 objetos restantes
son False Path explícitos, sin slack: 24 CLR propios más 13 extremos XPM,
cada uno consultado en max/min.

La netlist antes/después coincide exactamente salvo un comentario con
el nombre del archivo. Permanecen 94 LUT / 239 registros. Desaparecen los
ocho TIMING-6/7 y quedan los 92 TIMING-18. WNS estimado +4,092 ns y WHS
+0,068 ns bajo los relojes supuestos; el cambio refleja la clasificación
correcta de entradas asíncronas, **no una mejora física demostrada**.
Faltan colocación/ruteo, orígenes físicos, retardos externos y pulso mínimo.

El criterio general de sincronización se contrasta con
[UG906: sincronizadores de reset](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Asynchronous-Reset-Synchronizer)
y [UG903: cruces asíncronos](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Asynchronous-Clock-Domain-Crossings).
La selección concreta de estos 24 pines procede de la netlist auditada;
no debe copiarse a otro diseño sin repetir esa comprobación.

## Respaldos

### Integración nativa aprobada y respaldada

La tanda `20261005T124640Z-V7-ffn-supervised-guarded-native` inicia el
arranque en frío con el root/endpoint XDMA del ejemplo AMD y el top
`ntrpu_pcie_fragment_dual_host_guarded`. Usa señales reales de enlace,
reset del endpoint y PERST del banco. Observa solicitudes nativas inactivas
antes de liberar el transporte y disponibilidad de ambos dominios antes
de admitir trabajo. Esa observación de arranque **no implementa un ACK
de reset del endpoint durante ejecución**.

Se mantienen las 18 operaciones del banco nativo anterior: 35 frames,
294 escrituras de control y nueve lecturas contando identificación.
Los verificadores conservan comparación de bytes, orden y semántica NTRD;
registran las pausas B/R del núcleo sin exigirlas, porque la nueva frontera
retenida puede aceptar esas respuestas inmediatamente. Siguen exigiendo
pausas AW/W/AR del host y al menos 64 flancos reales de reset por dominio.

Fuentes: `source-20261005T124627Z.zip`, SHA-256
`3ce0c203a5da84e9edeb17138dbe1df1f1ee3cee48ea8be4ecea9badc1fa6495`.
La revisión terminal R4 aprueba el arranque y los 18 trabajos en
175,635 µs simulados; XSIM tarda 1.935,889 s. Coinciden 17.564 muestras
de gestión, incluidos los valores tomados exactamente al finalizar.
Se corrige esa exclusión del primer verificador en un archivo separado.
También se documentan las dos fuentes inactivas que Vivado omite: el
allocator de referencia bajo un `ifdef` deshabilitado y el wrapper
`ntrpu_pcie_fragment_dual_mig`, sustituido por el top con guard.
Los 221 archivos realmente compilados mantienen los hashes esperados.

Se rechazan 30 alteraciones de evidencia. El diagnóstico periódico
contabiliza 141 y 134 transacciones por canal, 14 lecturas de usuario y
ocho escrituras físicas en cada uno. Los 13.680 avisos del modelo DDR
se atribuyen mediante los comandos y retornos observados; no se suprimen.
Permanecen 169 avisos de generación/compilación y 16 avisos de ejecución,
contrastados con la referencia nativa anterior.

El respaldo local en C: y el remoto contienen 1.476 archivos,
248.338.857 bytes comprimidos, verificados a las 13:41:12 UTC. SHA-256:
`af60cb1f1523c03d1a116ed194b162f6fda2775ab66fbb816a490bfc31ed1b5c`.
Se conservan los revisores iniciales y sus diagnósticos; no se repite ni
modifica la simulación. No se usan checkpoints cuya equivalencia
protegida sigue pendiente.

### Tandas terminadas

Fuentes, compilación real, logs, verificadores y revisión están conservados
localmente y en NFS con comprobación de hashes y miembros:

| Tanda | Archivos / bytes comprimidos | SHA-256 |
| --- | --- | --- |
| `102730Z` | 64 / 1.148.388 | `f00f13a9421e95189d39a6330d03c969e8c9966ce8290a54f676becfbb002753` |
| `103302Z` | 64 / 1.149.573 | `ac2532b790f0882a76d58c92038e1821f7f1bd4c55973a50c5497e90a766c8df` |
| `103345Z` | 37 / 194.559 | `bf4b3f7e7be2695871474c070e1d2f6ca39d98d9f3232a5f6acf2da89ff403ef` |
| `103904Z` | 27 / 147.630 | `ba07d138e500d46ea0aa2760c9c0951c7dfde9baaf82d6b99ab88dbcf6ed721f` |
| `104435Z` | 44 / 309.886 | `be825b451f62e799c120f3e60346462b34d28386b8f45a66499278c352b69c5f` |

Siguientes dependencias: probar pérdida/reset del endpoint nativo,
resolver su coordinación y completar la comparación protegida. La [cola V0–V9](VIVADO_QUEUE.md)
conserva pendientes memoria exacta, cuatro canales, cómputo, ECAD y hardware.

## Prueba nativa de PERST en ejecución

`ntrpu_pcie_guarded_native_reset.sv` conserva las 18 operaciones del
arranque nativo y añade una escritura al último bloque de 64 bytes de
128 GiB. El banco retiene la respuesta DDR B, drena las solicitudes host
y acciona `sys_rst_n` mediante una tarea del ejemplo público AMD. No
fuerza las salidas de XDMA ni modifica su implementación protegida.

Durante PERST debe conservar época, cookie y descriptor; la respuesta
tardía permanece bloqueada y los bytes se comprueban directamente en
DDR. El pulso dura al menos 500 flancos del reloj de referencia, igual
que el estímulo inicial del ejemplo; no es una especificación eléctrica
para la placa. Se observarán 86 señales VCD, incluidos ambos relojes de
referencia, y se exigirá que el enlace nativo vuelva después del pulso.

El wrapper todavía carece de un coordinador de reset del endpoint. Por
eso este banco retira explícitamente los ACK `guard_peers_held/ready`
durante la pérdida. Aunque vuelva el enlace y se repita ARM, el guard
debe permanecer en `FAULT_HOLD`, con transporte en reset y admisión
cerrada. Un eventual PASS acreditará retención en ese estado bloqueado;
no autorizará reutilización, reenumeración ni recuperación física.

El diseño siguiente debe resolver ese contrato: observar solicitudes
AW/W/AR inactivas no equivale a confirmar que el endpoint obedeció un
reset. También deberá comprobar la secuencia de reloj, invalidación,
reset de los dos extremos y liberación; si se detiene el reloj host,
no se pueden sustituir sus 64 flancos por tiempo transcurrido.

El generador y el parche público pasaron comprobaciones locales de
reproducción. Las herramientas que extraen el prefijo saludable se
compararon contra la evidencia nativa `063837Z`: mantienen las muestras
VCD, paquetes decodificados y diagnóstico DDR; rechazan tres ausencias
de evidencia. **El nuevo banco y su verificador de reset aún no están
validados en XSIM.** Tras aprobar y respaldar
`20261005T124640Z-V7-ffn-supervised-guarded-native`, la cola inicia
`20261005T134131Z-V7-ffn-supervised-guarded-native-reset` a las
13:41:31 UTC, con límite de 3.600 s. Fuentes e instalación anteriores
se conservan; la nueva tanda tendrá su propio resultado y respaldo.
