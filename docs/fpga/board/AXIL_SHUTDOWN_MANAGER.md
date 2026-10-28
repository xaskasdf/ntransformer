# Aislamiento AXI-Lite con AMD Shutdown Manager

## Resultado y alcance

Se evalu? la IP instalada `xilinx.com:ip:dfx_axi_shutdown_manager:1.0`, revisi?n
interna `dfx_axi_shutdown_manager_v1_0_7`, con Vivado 2026.1 y
`xcvu5p-flva2104-2-i`. Dos perfiles de simulaci?n pasan seis casos cada uno,
con convertidor AMD real, inventarios retenidos y el actuador de reset local.
La s?ntesis de la IP aislada tambi?n termina y tiene revisi?n de recursos.

Este avance sustituye el aislamiento ideal del banco por una IP real.
No cierra el drenaje de XDMA, las colas del productor host ni la recuperaci?n
completa. La petici?n de cierre sigue viniendo del banco en el reloj host;
los certificados externos de participantes del actuador siguen siendo entradas
de la fixture. No hay prueba de PCIe/MIG nativo en esta tanda. La
[saturación posterior](AXIL_SHUTDOWN_SATURATION.md) mide hasta 51 operaciones
pendientes por canal con un observador U de ocho bits y queda respaldada
por separado; los bancos y recursos siguientes conservan su alcance original.

## Configuraci?n reproducible

`RESET_ACTIVE_LEVEL=0`, `RP_IS_MASTER=false`, `CTRL_INTERFACE_TYPE=0`,
`DP_PROTOCOL=AXI4LITE`, direcci?n/datos de 32 bits y `DP_AXI_RESP=0`.
Este ?ltimo cero selecciona **SLVERR** en la configuraci?n de AMD; no es
el c?digo de respuesta AXI OKAY. El wrapper generado es VHDL.

La [gu?a AMD PG377](https://docs.amd.com/v/u/en-US/pg377-dfx-axi-shutdown-manager)
describe el cierre de operaciones previas y el rechazo de nuevas. La decisi?n
sobre una escritura se toma al recibir AW; W puede haber llegado antes.
El n?cleo conserva esa decisi?n aunque cambie la petici?n de cierre.
No se ha modificado RTL de AMD ni habilitado DFX/Tandem en el dise?o de placa.

## Evidencia funcional

Banco: `fpga/tb/tb_ntrpu_axil_shutdown_pair.sv`.
Topolog?a observada: productor U ? Shutdown Manager ? interfaz H ? convertidor
AXI-Lite ? dispositivo D. El cierre real de la IP alimenta `host_source_fenced`
del monitor de ambos extremos. Un tercer inventario observa U.

Relojes: host 7 ns, gesti?n 10 ns y dispositivo 5/37 ns. Por perfil se comparan
seis escrituras y cinco lecturas arriba, frente a cuatro de cada tipo abajo.
Dos escrituras y una lectura nuevas devuelven SLVERR y nunca llegan a H/D.
Las restantes conservan datos, m?scaras, AWPROT/ARPROT y los cuatro RRESP.

Los casos cubren AW anterior al cierre con W posterior, W anterior con AW
posterior rechazado, destinos y respuestas bloqueados, accesos durante cierre,
reset con el reloj host detenido y tr?fico correcto posterior. Se comprueba
que VALID y payload permanezcan estables durante backpressure en los cinco
canales. El rechazo de lectura ignora su dato; los datos de lecturas reenviadas
se comparan incluso cuando el dispositivo devuelve SLVERR.

| Perfil dispositivo | Ciclos bloqueados AW / W / AR / B / R | Flancos host/dispositivo al ACK HELD |
| --- | --- | --- |
| 5 ns | 215 / 127 / 37 / 42 / 99 | 74 / 717 |
| 37 ns | 289 / 204 / 139 / 43 / 100 | 73 / 97 |

Los conteos de bloqueo corresponden a H para AW/W/AR y a U para B/R.
Son medidas del est?mulo, no caudal de inferencia ni rendimiento de placa.
El manager retiene su configuraci?n durante el reset de la secci?n CDC.
Los estados internos residuales del convertidor quedan vac?os antes de
reanudar tr?fico. Reabrir en el banco no autoriza reusar una sesi?n NTRD.

## Distinci?n necesaria para el coordinador

En **ambos perfiles**, `in_shutdown=1` y el certificado H/D est?n activos
mientras el inventario U contiene una W sin AW. Tambi?n pueden estar activos
con una respuesta R de rechazo pendiente. En el perfil r?pido se observa lo
mismo con las respuestas previas B/R retenidas hacia el host.

Por tanto, el certificado H/D prueba aislamiento de esa secci?n bajo cierre
persistente, pero no vaciado global. El padre necesita adem?s detener y probar
el productor, vaciar U y las colas protegidas de XDMA, y coordinar los resets
que realmente abarque. No se debe sintetizar ese permiso a partir de
`in_shutdown` ni de pines AXI moment?neamente inactivos.

El observador U del banco actual usa cuatro bits: basta para estos casos,
pero **no califica toda la capacidad de la IP**. La tanda de saturaci?n debe
usar inventario suficientemente ancho, medir la capacidad efectiva y luego
fijar dimensiones/l?mites de admisi?n en el wrapper de producci?n. No cambiar
el observador sellado ni vaciarlo con reset para eludir operaciones incompletas.

## S?ntesis aislada y l?mites

Total reportado: **518 LUT = 322 de l?gica + 196 LUTRAM**, **895 FF**,
sin BRAM, URAM, DSP, latches ni m?dulos sin resolver. El wrapper simulado
coincide con el de s?ntesis salvo siete l?neas de metadatos generados.
Esto acredita configuraci?n, no equivalencia funcional posterior a s?ntesis.

Reloj restringido a 7 ns: WNS/WHS estimados de 5,741/0,012 ns.
Faltan retardos en 154 entradas y 158 salidas y colocaci?n f?sica del reloj.
No se usan excepciones de timing. El reporte CDC no encuentra cruces porque
esta IP tiene un solo reloj; no valida los cruces del dise?o completo.

Nueve avisos revisados: cuatro registros de enable no usados en XPM memory,
cuatro feedback de empty equivalentes en XPM FIFO y un reloj OOC sin ubicaci?n.
Se preservan los hashes y fragmentos de fuente pertinentes. No se aprueba
frecuencia m?xima, timing ruteado ni pinout a partir de esta s?ntesis.

## Tandas y respaldos

Todas las copias se verificaron en `artifacts/ntrpu-board-vivado/` y NFS.

| Tanda del 2026-10-05 | Resultado | Duraci?n | Archivos |
| --- | --- | --- | --- |
| `172043Z-V4-ffn-supervised-shutdown-ip` | Generaci?n VHDL revisada | 16,199 s | 29 |
| `173146Z-V4-ffn-supervised-shutdown-pair` | Dos perfiles aprobados | 71,187 s | 119 |
| `173553Z-V4-ffn-supervised-shutdown-synth` | Recursos OOC revisados | 88,143 s | 40 |

SHA-256 de los archivos comprimidos, respectivamente:

- `4246066f637c5a0333955cd0dfd5e7a8749b86c2f146d8975a14d074a4a75d7d`
- `56d0fd1dc85f5e5903079354c1162c246b1e08c154e9810d1c7c3eaed348e0f6`
- `47fa5d487b1a293769e6fd1dcc8e7aaca678a62a07ffeadfa52808c874b6237e`

Adendas revisadas: generaci?n `50a1d1c8f28bf84dd1d3e4addc2c2244d6126944712a348c90ef93dfb26c570e`,
simulaci?n `bae626e876b94ea0c1fd65ba1dfc344dab9c7fd6cdb1fe18a31b46d1ef7edcae`,
s?ntesis `8a25d776096e3cc9b4b3d80efffcf83dc45aef1f424bc874f2bae94d35024268`.

El recolector inicial de generaci?n busc? `.v` y rechaz? la ausencia del
wrapper. La adenda recupera y verifica los `.vhd` del archivo ya sellado;
no se repiti? Vivado ni se alter? el diagn?stico. En simulaci?n se revisaron
cuatro avisos (entorno y par?metro de perfil tambi?n aplicado a glbl), sin
avisos de ejecuci?n. Se verificaron 86 archivos de bibliotecas instaladas;
el checker rechaza diez alteraciones de evidencia.

Reproducir con `tools/ntrpu_board_shutdown_ip.tcl`,
`tools/ntrpu_board_axil_shutdown_pair.tcl` o
`tools/ntrpu_board_shutdown_synth.tcl`, seg?n la etapa, siempre mediante
`vivado -mode batch -source SCRIPT -tclargs NUEVA_CARPETA`.
Consultar [reset local](LOCAL_RESET_ACTUATOR.md) y
[coordinaci?n del endpoint](ENDPOINT_RESET_COORDINATION.md).
