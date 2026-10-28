# Ingreso desde BAR hacia líneas NTRPU

Diseño del 2026-10-04; **ingreso conectado a XDMA y puente al receptor validado
en RTL/netlist y con tráfico PCIe nativo**. Cuatro repeticiones con observación
pasiva califican el transporte, los datos y cada aviso del modelo DDR genérico,
incluida la cadena nativa PCIe → CDC → MIG. Los resúmenes originales omitieron
los avisos con prefijo de modelo; se conservan con su corrección de alcance. Depende
de los bancos de [bypass](PCIE_BYPASS.md) y de los contratos existentes de
[receptor](../PEER_RX_INTEGRATION.md) y [capacidad](DDR_AXI_GUARD.md).
No modifica sus RTL ni las validaciones activas.

El [control AXI-Lite de ingreso](PCIE_INGRESS_CONTROL.md) añade descriptor
inmutable, STOP y ACK con identidad de 64 bits. Su composición de un reloj,
doble CDC y netlist tienen pruebas separadas. `20261004T223702Z` añade control
nativo por BAR0 y datos por BAR2 hasta MIG, con cuatro reservas y 53 lecturas
de control incluyendo la enumeración. Todavía utiliza pruebas externas de
aislamiento: no demuestra el drenaje de PCIe/CDC ni habilita por sí sola
reutilizar una ventana con paquetes antiguos en tránsito.

## Diferencia de interfaces observada

La variante XDMA usa dirección de 64 bits, datos de 512 e ID de cuatro
bits. El banco `164020Z` califica tres DWORDs con `AxSIZE=2`.
`165525Z` observa que máscara `0xa` y dirección PCIe `0x90000003c`
producen `AWADDR=0x10000003d`; el banco se detuvo por esperar el inicio
del DWORD. La revisión `170643Z` completa y califica las nueve máscaras,
con 24 bytes escritos, 36 leídos y centinelas intactos; respaldo verificado.

`ntrpu_axi_capacity_guard.sv` acepta únicamente `AxSIZE=6`, INCR y
dirección alineada a 64 bytes. `ntrpu_peer_rx_guard.sv` consume beats
completos y emite `WSTRB=~0`. Conservar ambos contratos: el ancho físico
de 512 bits del bypass no convierte una escritura de DWORD en 64 bytes.

## Contrato mínimo propuesto

1. **Reservar contexto.** Asociar una ventana a destino físico, longitud,
   permisos, trabajo y época mediante control autorizado. AWID solo
   identifica la respuesta AXI de ingreso; no identifica el trabajo remoto.
2. **Capturar AW y W independientemente.** Guardar dirección completa,
   tamaño, longitud, ID, strobes y LAST. Acotar la cola y mantener READY
   bajo al llenarse; no exponer a DDR una petición incompleta.
3. **Validar cada byte habilitado.** Para un beat sobre el bus de 64 bytes,
   la dirección de una lane es la base de esa línea más su índice. Derivar
   la base desde la dirección de ese beat; conservar los bits altos hasta
   validar rango, página, permiso y overflow. Rechazar strobes fuera del
   intervalo legal de la transferencia, incluso si caen dentro del BAR.
4. **Ensamblar solo dentro de la reserva.** Mantener datos y bitmap de
   bytes recibidos por línea. Nunca rellenar huecos ni leer DDR para hacer
   una modificación implícita. Definir antes del RTL si un byte duplicado
   idéntico se tolera; un duplicado con dato distinto debe invalidar el trabajo.
5. **Publicar líneas completas.** Solo con todos sus bytes autorizados y
   válidos, emitir el descriptor alineado de 64 bytes hacia el receptor.
   El éxito del trabajo espera todos los B finales y el drenaje, no solo
   la aceptación del último W por el buffer de ingreso.

El primer perfil puede rechazar formatos fuera de contrato. Debe consumir
la escritura rechazada y devolver su B una sola vez, sin accesos a DDR.
Un MemWr PCIe es publicado: ese B no informa por sí solo al emisor remoto.
El protocolo NTRPU necesita estado/finalización consultable y correlacionado.

## Casos que deben bloquear la integración

| Caso | Evidencia exigida |
| --- | --- |
| DWORD completo y máscaras `5/a/3/c/1/e` en distintas lanes | Direcciones por byte y bitmap exactos; centinelas intactos |
| Último DWORD `0xffffc` del BAR | Ningún byte habilitado fuera de la ventana; sin alias con cero |
| Hueco o LAST/longitud incorrectos | Ninguna línea parcial publicada a memoria |
| AW antes de W y W antes de AW | Asociación correcta y respuesta única con ID conservado |
| Buffer lleno y presión en el receptor | Datos estables, cuentas acotadas, sin pérdida ni sobrescritura |
| Stop/reset de aplicación con línea incompleta | Cerrar admisiones; conservar transporte y drenar; no reasignar ventana |
| Error B o pérdida de enlace después de aceptar datos | Invalidar el trabajo completo; no publicar éxito por un prefijo válido |

Una escritura antigua a la misma dirección no trae época NTRPU. Reutilizar
una reserva requiere el aislamiento/drenaje del transporte, además del
estado local vacío. El banco de cruce DDR `152800Z` ya demuestra que un
descriptor fallido puede haber escrito o publicado un prefijo válido.

## Orden de ejecución posterior

El banco de máscaras está cerrado y respaldado. Implementar
captura/validación/ensamblado en RTL aislado, con referencia por byte y
controles negativos; sintetizar y probar su netlist antes de conectarlo al
receptor. La integración inicial debe usar el dominio de transporte
conservado y una sola reserva; añadir CDC y múltiples reservas con bancos
separados. Solo después conectar MIG, medir ocupación/caudal y ampliar a
dos endpoints. Ninguno de estos pasos está aprobado por el banco de DWORDs.

## Primer bloque RTL aislado

`ntrpu_pcie_dword_decode.sv` valida un AW/W **ya asociado**: LEN=0,
SIZE=2, INCR, sin exclusividad, LAST=1 y una reserva alineada de 64 bytes.
Conserva los 64 bits de dirección, devuelve las lanes habilitadas sin desplazar
datos y rechaza strobes vacíos o fuera del DWORD. Para una dirección no
alineada, el límite superior sigue siendo el final del DWORD alineado;
no es `AWADDR+4`. Un rechazo emite bitmap cero.

La reserva debe estar previamente autorizada. El bloque no comprueba capacidad
física ni implementa AW/W, B, épocas, drenaje, ensamblado o finalización.
Por ejemplo, un ensayo con dirección alta prueba ausencia de truncamiento,
no acredita DDR existente en esa dirección.

Tanda `20261004T172307Z-V7-ffn-supervised-dword-decode`, **aprobada en su
alcance aislado**: 9.609 combinaciones con referencia escalar de direcciones
de byte, cinco mutaciones RTL detectadas y el mismo banco aprobado contra
la netlist funcional de síntesis. En cada positivo: 2.243 aceptaciones y
7.366 rechazos. Se comprueban todas las posiciones de byte, las 16 máscaras
de DWORD, lanes ilegales, los tamaños/longitudes, LAST, alineación de reserva
y cada bit alto de dirección. Ambos positivos terminan en 96,090 µs simulados.

La síntesis usa 159 LUT, 9 CARRY8 y 0 FF; no infiere latches. El máximo
reportado sin colocación es 1,308 ns bajo presupuesto combinacional de 5 ns:
es una estimación con redes sin rutear, no cierre de timing. DRC no encuentra
checks, pero advierte que la conectividad de placa no se revisa por completo
en modo OOC. Se conservan doce avisos de bits bajos constantes de la dirección
alineada y siete avisos de elaboración por `LIBRARY_PATH`.

Vivado 2026.1, VU5P-2I, 156,818 s. Respaldo local/NFS verificado a las
17:28:06 UTC: 242 archivos, 1.463.665 bytes, SHA-256
`fa46b3f86acc7e2f782f947f59a0827b711dfb2bb03e1de573d15091e75dab8d`.
Fuente `source-20261004T172257Z.zip`, SHA-256
`dffc6421a81ed589e25f1d7b17eb1c7158271ab00bdfd691faf52bb0840a5779`.

Reproducción tras cargar Vivado, desde el repositorio:

```bash
vivado -mode batch -source tools/ntrpu_board_pcie_dword_decode.tcl \
  -tclargs /ruta/nueva/decoder
```

## Ensamblador y captura AXI de una línea

`ntrpu_pcie_line_ingress.sv` añade captura independiente AW/W, un frame
pendiente y almacenamiento de 64 bytes. Usa el decodificador anterior y
una reserva autorizada por control: ventana, destino, época creciente y tag.
El control debe comprobar permisos/capacidad antes de armarla; este bloque
no sustituye al filtro de capacidad ni implementa autorización del host.

- Tolera bytes duplicados idénticos. Un duplicado distinto invalida el
  trabajo y no modifica el byte anterior. Nunca publica líneas con huecos.
- B conserva AWID y reconoce el almacenamiento de ingreso. No significa
  finalización DDR. Mientras B está pendiente, no acepta otro AW/W.
- Un formato no admitido se descarta hasta **WLAST real** y recibe una B.
  Sin LAST, conserva el frame y bloquea finalización/rearme; no inventa
  una respuesta ni deja que un contador vuelva a cero. El emisor debe
  resolver el frame o efectuar recuperación coordinada del transporte.
- Stop impide aceptar datos en el ensamblador, pero mantiene el transporte
  para drenar y responder rechazos. Una línea con VALID ya expuesto permanece
  estable hasta READY, incluso tras stop; el resultado del trabajo será error.
- La respuesta del destino debe coincidir en época/tag. Una respuesta ajena
  invalida el trabajo y no consume la finalización esperada.
- La reserva queda retenida hasta completar/drenar salida, consumir el
  resultado y recibir `source_stopped`, `ingress_isolated` y `quiesce_epoch`
  de la época actual. El siguiente armado exige una época mayor y líneas
  alineadas. Estos ACK son una **barrera externa confiable**, simulada por
  el banco; todavía falta implementarla contra el transporte real.

El reset del módulo es frío y coordinado con AXI. No usarlo como cancelación
de aplicación. La salida entrega una línea y espera una finalización con
identidad; el puente al protocolo cmd/data/cpl se valida más abajo.
Las lecturas BAR, múltiples reservas, CDC y ráfagas aceptadas no están cubiertas.

## Evidencia del ingreso

`174126Z` conserva un timeout por handshake perdido del generador de estímulos.
`174359Z` corrige el banco: pasan tres semillas RTL y seis defectos introducidos;
la síntesis termina, pero la elaboración de netlist falla por omitir `glbl`.
Ambos intentos están respaldados sin convertirlos en una aprobación completa.

`20261004T175152Z-V7-ffn-supervised-line-ingress-netlist` copia la netlist y DCP
exactos de `174359Z`, verifica hashes y repite únicamente la simulación con
`glbl`. Las tres semillas coinciden **ciclo por ciclo** con RTL. La revisión
conjunta pasa 19 escenarios por positivo, con 161 AW/161 B, 423 W y siete
líneas de 64 bytes. Recibe 478 bytes aceptados, incluidos duplicados y trabajos
descartados; publica 448 bytes por positivo. Los 19 escenarios cubren presión
en los cinco canales observados, huecos, duplicados, dirección alta, formatos
inválidos, stops, respuesta ajena y un frame de 261 beats con 260 sin LAST.

Se rechazan seis mutaciones RTL y 40 alteraciones de evidencia. Las semillas
1/7 terminan en 2.740 ciclos activos (13,900 µs incluyendo reset); la 42, en
2.579 ciclos (13,095 µs). Son estímulos con pausas deliberadas, no caudal DDR.
Referencia independiente por byte: `tools/check_ntrpu_pcie_line_ingress.py`.

Síntesis VU5P-2I: **658 LUT, 1.515 FF, sin latches ni RAM/DSP**. Para reloj
de 5 ns, WNS interno estimado +1,760 ns y WHS +0,077 ns. Hay 1.119 entradas
y 982 salidas sin retardos, falta `HD.CLK_SRC` y no se ejecutó colocación/ruteo:
**no acredita cierre a 200 MHz**. Se conserva el aviso del bit alto constante
del código de error, los de clock/OOC y ocho avisos de `LIBRARY_PATH`.

Duraciones: padre RTL/síntesis 203,200 s; replay 38,239 s. Respaldo final
verificado a las 17:55:28 UTC: 57 archivos, 1.985.399 bytes, SHA-256
`3732934ef678003472b27dc585a274f238fed7852d2e7ae55521aacebd835d5e`.
El padre, 267 archivos/2.667.894 bytes, conserva SHA-256
`25e4eac320afdaf583f1ed2c7dec8955d245b98ca13c690d6e19a47f6fcaf99d`.

El Tcl unificado ya incluye `glbl`; la evidencia anterior procede de padre
y replay separados. Para repetir solo la netlist existente:

```bash
vivado -mode batch -source tools/ntrpu_board_pcie_line_ingress_replay.tcl \
  -tclargs /ruta/al/padre/output /ruta/nueva/replay
```

## Conexión al bypass XDMA

La tanda `20261004T180851Z-V7-ffn-supervised-line-link` conecta el ingreso
RTL sin cambios al bypass del ejemplo AMD. Pasa cuatro trabajos: línea
completa, máscaras `5/a` con duplicado idéntico, duplicado conflictivo y
recuperación con máscaras `1/e` en la última línea del BAR (`0xfffc0`).
Son **83 MemWr PCIe, 83 AW/W/B y tres líneas exactas de 64 bytes**. Una B
devuelve SLVERR por el conflicto; el trabajo siguiente termina sin reset.
Acepta 198 bytes, incluidos el duplicado y cuatro bytes del trabajo fallido;
publica 192 bytes. Las ventanas PCIe se traducen sobre 4 GiB y el destino
de prueba permanece en `0x1fffffffc0`, sin truncar sus bits altos.

Se verifican 81 señales VCD: solicitudes RQ, AXI y control/salida del ingreso.
Se conserva el lector nativo de 48 señales y se comprueba por separado el
conjunto adicional, con inventario exacto. Cada byte PCIe coincide con WSTRB,
bitmap y línea de salida. RQ→B mide 192–196 ns en el modelo; no hay stalls
RQ/AW/W/B en este estímulo. La salida de líneas espera 27 ciclos y los
resultados 20, conservando VALID/datos estables. Son pausas deliberadas,
no una medida de ancho de banda sostenible.

La revisión rechaza 37 alteraciones de evidencia, verifica 76 entradas
compiladas y 134 archivos AMD intactos. Los 139 avisos de compilación coinciden
con el banco de máscaras tras normalizar la ruta y el ordinal de temporales
del compilador; se conservan línea/índice/multiplicidad y los mensajes originales.
El aviso de tiempo cero también permanece. No se afirma inocuidad fuera del
camino activo validado. Referencia: `tools/check_ntrpu_pcie_line_link.py`.

Duración 422,185 s; final simulado 73,406 µs. Respaldo local/NFS verificado
a las 18:19:56 UTC: 843 archivos, 106.322.837 bytes, SHA-256
`a91f3c8ca3f7cfc37a093bc59ab884a742d974c48cd2cbe3e0d6a9cfd93b670f`.
Fuente `source-20261004T180841Z.zip`, SHA-256
`a78b55b0a9ba7149a8f116e5f21ff8d09a2972a3b37002afac81f81bc4862eb1`.

El destino y los ACK de aislamiento siguen siendo BFMs; no hay receptor,
MIG ni protocolo de finalización consultable desde el host en esta tanda.
Un MemWr es publicado: el error B observado dentro de la FPGA no equivale
a una completion PCIe para el emisor. Sigue conectar el receptor y el filtro
de capacidad/MIG, conservando propiedad y drenaje antes de reutilizar ventanas.

## Puente al receptor: RTL y netlist validados

`ntrpu_peer_rx_guard.sv` cambia de época al rearmarse estando en cuarentena;
el ingreso exige una época mayor por reserva. `ntrpu_pcie_ingress_peer.sv`
rearma ambos con la misma época, sin admitir uno solo. Exige retiro del
resultado previo, buffer del puente vacío, aislamiento y AXI drenado.

Retener línea e identidad durante cmd/data/cpl. Una CPL de error del receptor
puede preceder al drenaje de AXI: guardarla y esperar `axi_busy=0` antes de
presentarla como `done` al ingreso. Una notificación temprana no libera memoria.

Política validada de stop de aplicación: invalidar el ingreso y dejar terminar
toda línea ya expuesta a través del receptor, reteniendo la reserva hasta B.
Preparar la cuarentena/rearme del receptor solo estando drenado; no retirar
comandos ni VALID pendientes. Probar presión en cmd/data/B, stop en cada fase,
CPL temprana y B ausente. Sin B, retener la reserva hasta recuperación coordinada.
La tanda `20261004T183931Z-V7-ffn-supervised-ingress-peer` valida este contrato:
17 escenarios por semilla (`1/7/42`), repetidos sobre netlist funcional con
igualdad ciclo a ciclo. Incluye stop en COMMAND, DATA, COMPLETION, DRAIN y
RETURN; SLVERR/DECERR; duplicados, huecos, direcciones altas, rearme inválido y
recuperación sin reset. Un BID ajeno produce CPL temprana mientras AXI sigue
ocupado. En dos variantes de ese caso y en B ausente se retiene la reserva
100 ciclos aun con aislamiento/rearme solicitados; solo la B final ID 0 permite
terminar. Ese límite de prueba no garantiza que la respuesta llegue físicamente.

Cada ejecución registra 245 AW/W/B de ingreso, 854 bytes aceptados (incluidos
duplicados y trabajos descartados), 13 escrituras completas/832 bytes al destino
y 17 resultados. Las semillas requieren 3.849, 3.859 y 3.601 ciclos a 5 ns;
son latencias de un banco con pausas intencionales, no throughput sostenido.
Se detectan seis defectos RTL intencionales y 60 alteraciones de evidencia.
Referencia independiente: `tools/check_ntrpu_pcie_ingress_peer.py`.

Reproducción en una carpeta nueva, con el entorno Vivado cargado:

```bash
vivado -mode batch -source tools/ntrpu_board_pcie_ingress_peer.tcl \
  -tclargs /ruta/nueva/ingress-peer
```

El Tcl ejecuta RTL, seis mutantes, síntesis OOC y netlist con `glbl` explícito.
Un código de salida cero no sustituye la revisión de trazas y avisos.

Síntesis OOC VU5P: 1.409 LUT, 3.041 FF, sin RAM/DSP/latches. El buffer de una
palabra se convierte en registros; se conserva el aviso de síntesis. WNS interno
sin colocar +1,757 ns y WHS +0,079 ns, con reloj de 5 ns, pero faltan retardos de
997 entradas/936 salidas y HD.CLK_SRC. DRC OOC sin incidencias no valida la placa
ni demuestra cierre a 200 MHz. Ocho avisos de elaboración LIBRARY_PATH retenidos.

Duración: 243,097 s. Respaldo local/NFS verificado a las 18:46:00 UTC:
312 archivos, 5.951.237 bytes, SHA-256
`402f059d1a7beb45b6c16bfb370ef6074eb00821f6de73a49e5f8b82fd1ca5f6`.
Fuentes `source-20261004T183921Z.zip`, SHA-256
`fb0e93502d2521908ad1495f362bed33157b510fc00c6d2097650a21eebb0cba`.

La prueba usa AXI y aislamiento externos simulados. No incluye MIG, filtro de
capacidad, CDC ni protocolo de resultados consultable por el host. El perfil
emite solamente cmd_len=0, dirección alineada y época sincronizada: descriptores
o frames arbitrarios rechazados antes de DATA quedan fuera de su alcance.
La integración nativa siguiente conserva los RTL ya calificados.

## PCIe nativo con el receptor conectado

`20261004T184808Z-V7-ffn-supervised-peer-link` pasa la revisión de **105 señales
VCD**, con 78 entradas compiladas y 134 archivos AMD intactos. Repite los cuatro
trabajos/83 MemWr del banco anterior, ahora atravesando ingreso, controlador y
receptor reales. Las tres líneas de 64 bytes llegan exactas al AXI de destino;
el duplicado conflictivo no genera escritura y el trabajo siguiente se recupera
sin reset. Las señales line/done son observaciones de la jerarquía, no BFMs.

Se observan 29 ciclos de espera en AW de destino y 32 en W, sin cambios de VALID
ni atributos/datos. La primera escritura acepta W antes que AW y las siguientes
AW antes que W. Línea→B mide 84/88/104 ns; B→done, 12 ns. RQ→B de ingreso sigue
en 192–196 ns y solo confirma staging. Son tiempos de simulación con esperas
deliberadas. Los cuatro resultados mantienen su identidad y las épocas de ambos
motores coinciden durante todo el banco. Hay 20 ciclos de presión en resultados.

La revisión rechaza 48 alteraciones de logs/VCD. Conserva 139 avisos de
compilación equivalentes al banco de máscaras y un aviso idéntico de tiempo cero;
no modifica fuentes protegidas ni generaliza esa comparación fuera del camino
ejercitado. Duración de herramienta: 424,154 s; final simulado: 73,510 µs.
Fuentes `source-20261004T184758Z.zip`, SHA-256
`f1631c6dae2b00cc8032779d207adee37d4095463e266f66f6a3fb5a51960f8f`.
Referencia: `tools/check_ntrpu_pcie_ingress_peer_link.py`.

Respaldo local/NFS verificado (`20261004T185839Z`): 846 archivos,
106.416.857 bytes, SHA-256
`534ad487c391f7f8bcb0c22eb6818a963e8310eef87c7d572eb2598b753c64e7`.

El destino AXI y el aislamiento siguen siendo BFMs; aún no hay filtro de
capacidad/MIG, protocolo host de resultados, CDC ni P2P físico en esta tanda.
Los errores de destino y stop se califican en el banco aislado, no se vuelven
a afirmar como cobertura del banco PCIe nativo de cuatro trabajos.

## Capacidad integrada: RTL y netlist

`ntrpu_pcie_ingress_guarded.sv` compone los cuatro bloques sin modificar sus
fuentes: ingreso, puente/receptor y filtro de capacidad. Conserva 64 bits hasta
la decisión del filtro y expone 37 bits solo después. La admisión permanece
abierta durante stop para drenar trabajo expuesto; el rearme exige también
`guard_idle`. El puerto de lectura del filtro se mantiene inactivo en este perfil.

La tanda `20261004T190452Z-V7-ffn-supervised-ingress-guarded` pasa **21 casos,
tres semillas y netlist con igualdad ciclo a ciclo**. Son 309 AW/W/B de fuente,
1.110 bytes aceptados (incluidos descartes/duplicados), 14 escrituras completas
de memoria/896 bytes y 21 resultados por ejecución. Los destinos `0x2000000000`,
`0x1000000000000000` y `0xffffffffffffffc0` terminan con error de destino sin
exponer AW/W a memoria. Cero y más de 4 GiB recuperan después sin reset.
Stop en cinco fases, errores de destino y B ausente durante 100 ciclos conservan
drenaje/propiedad. Se detectan cinco defectos RTL y 60 alteraciones de evidencia.

Síntesis: 1.531 LUT (1.233 lógica/298 LUTRAM), 3.091 FF y cero BRAM/URAM/DSP.
WNS interno sin colocar +1,757 ns y WHS +0,067 ns a 5 ns; 997 entradas y 913
salidas carecen de retardos. No es cierre de timing. Se conservan ocho avisos
de síntesis/reportes (buffer de una palabra a registros, cinco candidatos RAM
inaccesibles eliminados, HD.CLK_SRC y DRC OOC) y siete de LIBRARY_PATH.
La netlist valida el perfil de escritura; no extiende esa prueba al filtro con
lecturas habilitadas.

Duración 234,146 s. Respaldo local/NFS verificado 19:10:57 UTC: 292 archivos,
6.661.880 bytes, SHA-256
`125c07494f143aef237fd36c2ddeebc29d1c05bad9281e6a4ea56c5fb1265727`.
Fuentes `source-20261004T190442Z.zip`, SHA-256
`9468759110e55fa4cc4c550895bb95e1b34231d579f07a9633c1a6599b6558df`.
Reproducción: `tools/ntrpu_board_pcie_ingress_guarded.tcl` con carpeta nueva;
referencia independiente: `tools/check_ntrpu_pcie_ingress_guarded.py`.

## MIG frío y lectura de retorno

Conectar el filtro calificado a MIG y leer desde su puerto AR/R de prueba seis
ubicaciones separadas, incluida la última línea del canal. Volver a leer los
centinelas tras cada rechazo para detectar corrupción o alias; no fabricar
respuestas B/R desde un scoreboard. Conservar los 64 bits antes del filtro.

El filtro presupone respuestas AXI conformes: la inyección de BID ajeno del
banco directo del receptor no demuestra recuperación si se coloca ese filtro
entre ambos. Registrar ese contrato al componerlos. El primer banco MIG usa
solo el reloj UI y el modelo frío calificado; CDC y PCIe nativo conjunto siguen
siendo pasos separados. El modelo DIMM exacto y el reset
caliente siguen pendientes; no modificar el modelo protegido para ocultarlos.

`20261004T191519Z-V7-ffn-supervised-ingress-mig` se detuvo antes de simular:
VRFC rechazó una declaración del TB con una constante y tres nets sin inicializar.
La revisión separa esas declaraciones; no modifica RTL/IP/modelo. Diagnóstico
respaldado: 337 archivos/33.394.561 bytes, SHA-256
`0578f3af88913e3eb520b228547df47a4a4149f178cbd75f02bdf22bf57ff56a`.
La revisión R2 usa una carpeta nueva; no se convierte ese fallo en PASS.

`20261004T191958Z-V7-ffn-supervised-ingress-mig-r2` pasa la revisión:
13 trabajos, 194 AW/W/B de fuente, nueve escrituras de memoria (576 bytes)
y **63 lecturas comparadas byte a byte (4.032 bytes)**. Se escriben cero,
`0x100000040`, 32/64/96 GiB y la última línea `0x1fffffffc0`; cada trabajo
vuelve a leer todos los centinelas anteriores. Tres destinos inválidos y el
duplicado conflictivo no generan AW/W en MIG. Stop drena la escritura expuesta
y los trabajos posteriores recuperan sin reset.

Calibración a 28.051.098 ps; final a 53.178.498 ps, 4.986 ciclos de banco.
Desde el último B de fuente hasta B de memoria se observan 45–610 ns, con
el productor y la barrera de este banco; no es una medición de caudal sostenido.
Se observan activaciones en los cuatro ranks (7/3/3/6), incluidas las que pueda
originar la actividad interna del controlador. No demuestra que se haya recorrido
toda la capacidad. ECC permanece sin interrupciones; no se inyectan fallos ECC.

Son 128 entradas compiladas y 35 archivos del modelo idénticos al banco frío
anterior. Se revisan 11 advertencias de compilación/carga y se rechazan 58
alteraciones de evidencia. **Corrección posterior:** el log contiene además
144 avisos `Model:WARNING: Reading unwritten address`, omitidos por el filtro
del resumen inicial. Las comparaciones permanecen válidas; la calificación
DDR completa requiere correlacionar esos avisos. Dos aserciones AMD anidadas no son
compatibles con XSIM; se conserva esa limitación. Duración: 237,147 s.
Respaldo local/NFS verificado a las 19:29:11 UTC: 534 archivos/49.359.772 bytes,
SHA-256 `2eb183b5519c0b539fde3d0941a9c631a42ecd0f19030901e9c5122a889fd293`.
Fuentes `source-20261004T191948Z.zip`, SHA-256
`a21647725cb09a2be5de758f270f5a830771df3925ccaf1962dc83a693c86aaf`.
Referencia: `tools/check_ntrpu_pcie_ingress_mig.py`.

## Transporte asíncrono: dos perfiles

`ntrpu_pcie_ingress_async.sv` añade el convertidor AXI4 AMD al ingreso
calificado, sin modificar los seis RTL previos. Ambas tandas pasan 13
trabajos, 194 AW/W/B por cada lado del CDC, nueve escrituras de memoria
(576 bytes) y 63 lecturas comparadas (4.032 bytes). Se comprueban direcciones
de 64 bits, datos, strobes, IDs, causalidad y conservación por canal; el
verificador de memoria original permanece sin cambios. Cada tanda rechaza
85 alteraciones de evidencia y registra 130 entradas compiladas.

`20261004T193215Z-V7-ffn-supervised-ingress-cdc`: host 4 ns, fase 700 ps; UI 5 ns.
Final 66,223,498 ps, 7,595 ciclos de banco; duración 271.600 s.
Latencia observada AW 39.298–43.298 ns,
W 39.298–43.298 ns y
B 49.702–52.702 ns.
Respaldo local/NFS: 570 archivos/65,243,151 bytes, SHA-256
`c40945b2c15bf7df1d54aee2ba0dd81dfdceb61f3e754920fbfdb4a009fc847d`.

`20261004T193725Z-V7-ffn-supervised-ingress-cdc-slow`: host 7 ns, fase 1100 ps; UI 5 ns.
Final 76,598,498 ps, 9,670 ciclos de banco; duración 298.627 s.
Latencia observada AW 42.398–46.398 ns,
W 42.398–46.398 ns y
B 83.602–87.602 ns.
Respaldo local/NFS: 570 archivos/65,234,212 bytes, SHA-256
`4ef4c18147f28a037770056413194ffa26cb042ae06e606c8c5d4462ef06598c`.

**Límite del modelo:** los cinco FIFO asíncronos AMD advierten que sus
modelos funcionales no representan retardos de sincronización. Las latencias
anteriores describen este banco; no califican metastabilidad, retardos físicos
ni cierre CDC/timing. Se retienen 16 advertencias de compilación/carga y cinco
de FIFO por tanda. La revisión suplementaria identifica además **144 avisos
`Model:WARNING:` en el perfil rápido y 432 en el lento**, omitidos inicialmente.
Los PASS originales acreditan transporte/datos, no esa calificación DDR completa.
Las repeticiones observadas de la sección siguiente cierran la correlación y el
drenaje en sus propias trazas; no modifican retroactivamente los archivos originales.
Los avisos de compilación incluyen cuatro referencias adelantadas a un contador
del TB y un parámetro ofrecido a `glbl`; sus valores y relojes se verifican
independientemente en la traza. El modelo DDR y sus límites siguen intactos.

El puerto fuente usa `host_clk`; contexto, resultados, aislamiento y lectura
de retorno siguen en el dominio UI. `transport_idle` solo describe ese motor:
la barrera externa debe demostrar además el drenaje del host y del convertidor.
El banco comprueba todas las respuestas antes de declarar esa barrera; admite
una transacción host pendiente por vez. Solo se califican resets fríos
coordinados. Referencia: `tools/check_ntrpu_pcie_ingress_cdc.py`.

## Siguiente dependencia: PCIe nativo y MIG juntos

Conectar el bypass XDMA real a este mismo convertidor, ingreso y controlador
MIG. Reutilizar las comprobaciones nativas de RQ y AXI y la referencia de
lectura DDR; conservar fuentes AMD/modelo y verificar el B de memoria antes
del resultado. La barrera y el protocolo host de resultados todavía deben
integrarse; un MemWr PCIe es posted y su emisión no demuestra commit DDR.
No cerrar capacidad completa, DIMM exacto, reset caliente, placa física ni
inferencia con estas tandas acotadas.

### Primera tanda conjunta: evidencia parcial conservada

`20261004T195116Z-V7-ffn-supervised-native-mig` conecta el bypass XDMA nativo
con los siete RTL calificados, el convertidor asíncrono y MIG. Conserva los
83 MemWr del banco nativo anterior (máscaras parciales, duplicado idéntico,
conflicto y recuperación). Destinos: cero, `0x100000040` y `0x1fffffffc0`.
El banco exige ocho lecturas de centinelas después de los cuatro trabajos;
tras el conflicto debe mantenerse el contenido anterior.

Se capturan 144 señales para observar los paquetes nativos, las escrituras y
respuestas a ambos lados del CDC, las reservas/resultados en UI y los puertos
reales AW/W/B/AR/R de MIG. El control y la lectura de prueba siguen siendo
BFMs; las respuestas de memoria provienen del controlador generado. Fuentes
`source-20261004T195106Z.zip`, SHA-256
`9f3a940929dd4eb690d01d1467678c10d5e9e87296e96c04e844e022d1aabd0d`.
Terminó en 720,725 s y pasó las comparaciones, pero carece de observación causal
para sus 6.048 avisos del modelo. Estado conservado: **transporte revisado, DDR
no calificada**. Respaldo verificado: 1.364 archivos/183.256.336 bytes, SHA-256
`1f30710e2352807fe991066f517104e171227f0a05ea41198b68f0a00a8fa217`.

## Repeticiones con observación periódica y drenaje

El observador `fpga/tb/ntrpu_ddr_periodic_ingress.vh` conserva el método de
[PCIE_DDR_FFN.md](PCIE_DDR_FFN.md): solicitud → aceptación → CAS inyectado →
READ físico → ocho beats de avisos → retorno bloqueado hacia el usuario.
Reconstruye filas abiertas, escrituras previas e inversión A/B del RCD.
Se mantienen todos los diagnósticos; no hay precarga, `force`, reset adicional
ni desactivación de lecturas periódicas. Los siete RTL de producción y los
35 archivos del modelo permanecen idénticos a los bancos anteriores.

| Tanda del 2026-10-04 | Resultado funcional | Solicitudes periódicas drenadas | Avisos explicados | Alteraciones de evidencia rechazadas |
| --- | --- | ---: | ---: | ---: |
| `200944Z` MIG observado | 13 trabajos, 194 escrituras fuente, 9 escrituras DDR, 63 lecturas/4.032 bytes | 8 | 144 | 110 |
| `201023Z` CDC 4/5 ns | Mismas cuentas y datos; conservación por ambos dominios | 17 | 144 | 137 |
| `201100Z` CDC 7/5 ns | Mismas cuentas y datos; conservación por ambos dominios | 29 | 432 | 137 |
| `200726Z` PCIe nativo + CDC + MIG | 4 trabajos, 83 MemWr, 3 escrituras/192 bytes, 8 lecturas/512 bytes | 43 | 6.048 | 122 |

Las cuatro tandas pasan la revisión estricta, incluido el drenaje final.
El banco nativo conserva conflicto sin escritura DDR y recuperación sin reset,
en cero, por encima de 4 GiB y en la última línea del canal de 128 GiB.
Captura 144 señales, verifica 202 entradas compiladas y 134 archivos AMD intactos.
Final: 75.753.498 ps; ejecución: 780,610 s. La entrega de línea a B tarda 35 ns
y B a DONE 20 ns en las tres escrituras de este banco. Son medidas de simulación
con carga acotada, no caudal sostenido ni latencia física de sincronización.

Los otros finales son 53.183.498, 66.228.498 y 76.603.498 ps; duraciones
246,302, 292,323 y 314,414 s. Cada final añade un flanco de espera y comprueba
que no queda retorno periódico pendiente. Los FIFO siguen advirtiendo que no
modelan retardos de sincronización. Dos aserciones de calibración AMD no se ejecutan
en XSIM; el DIMM exacto y el reset caliente continúan pendientes.

Reproducción: `tools/ntrpu_board_pcie_ingress_{mig,cdc,cdc_slow,memory}_observed.tcl`
en una carpeta nueva. Revisión causal: `tools/check_ntrpu_ingress_periodic.py`;
revisión nativa: `tools/check_ntrpu_pcie_ingress_memory.py`, estricta por defecto.
Las auditorías y sus pruebas de alteración están en cada `review.json` del respaldo.

Respaldo MIG: SHA-256
`eeebe1a189fcb9ae680a119fd6afc1b37cce5784406c7e052ee0581955cb7dd1`;
CDC rápido: `36072b1f1327bb84febe835ba081a7eed902ae26ff4b47eefde8641a2f8e0425`;
CDC lento: `c238ef050b26fdc4c37d2dbe25e31de694e4415f230ff16173a7a9aff82e5b44`.
Los tres están verificados local/NFS. El archivo nativo también quedó verificado
local/NFS a las 20:26:20 UTC: 1.366 archivos/183.373.091 bytes, SHA-256
`c4cefa5a9ef1551377f3341dfa38565599ea6d86c9348e694db82e3359e3a874`;
conserva fuentes, trazas, revisores y manifiesto de integridad.

## Implementación exploratoria del ingreso asíncrono

`tools/ntrpu_board_pcie_ingress_async_synth.tcl` sintetiza los mismos siete RTL
y el convertidor AXI AMD, con host a 4 ns y UI a 5 ns. Exporta DCP, netlist,
recursos, timing, CDC, excepciones y constraints. Solo desactiva los relojes
provisionales del XDC OOC del IP; conserva sus restricciones internas. No añade
false paths, excepciones multiciclo ni waivers propios.

La síntesis `202525Z`, el ruteo `203200Z` y la auditoría de solo lectura
`204009Z` terminaron y tienen respaldo local/NFS verificado. Las tres tandas
conservan el mismo RTL. Contexto OOC: VU5P, host 250 MHz/UI 200 MHz y fuentes
provisionales `BUFGCE_X0Y61/X0Y60`; no son asignaciones de placa.

| Estado | LUT | Registros | WNS interno | WHS interno |
| --- | ---: | ---: | ---: | ---: |
| Sintetizado | 2.127 | 4.931 | 1,747 ns | 0,013 ns |
| Colocado/ruteado OOC | 2.055 | 4.911 | 0,856 ns | 0,011 ns |

Incluye 682 LUT de RAM distribuida, cero DSP/BRAM/URAM, cero latches y cero
black boxes. Los diez constraints de bus skew conservan un margen mínimo
de 3,174 ns tras ruteo. CDC reporta diez sincronizadores y diez resets
informativos con los waivers AMD activos; no se añaden waivers propios.

Vivado informa 6.779 nets ruteables, cero errores de ruteo y dos nets con
discontinuidades explícitas. La auditoría reabre el DCP sin modificarlo y
compara la propiedad `ROUTE` completa: los **229 alias jerárquicos PARTIAL**
pertenecen exclusivamente a `clk` (113) o `host_clk` (116). No queda una net
de datos parcialmente ruteada sin explicar. El estado global continúa siendo
`Partially Routed` por el contexto de relojes externos.

**Puertos pendientes:** 998 entradas y 913 salidas carecen de delays; los
1.000 avisos TIMING-18 son el límite del reporte. Se retienen 100 avisos de
puertos sin `HD.PARTPIN_LOCS` y un RTSTAT-10 por nets sin cargas ruteables.
Estas cifras no acreditan timing de puertos ni implementación de la tarjeta.
El siguiente montaje debe aportar relojes y presupuestos del padre y repetir
la implementación integrada, manteniendo las restricciones internas AMD.

Respaldos: síntesis `92c5f74ea812e3fa8ca89c65dbf7c2a12977a6faa72d50d322b6d3ad4e0e30b5`;
ruteo `af9e6b8eac3c3e6f28119efc71b0db106fc26ed7edfb6129555e0029c6c06aed`;
auditoría `2e27f8921aaecd23ecf3dba48d10b893af97607e0c9f5dd4e74163ee3d3ab0a6`.
Reproducción adicional: `tools/ntrpu_board_pcie_ingress_async_route.tcl` y
`tools/ntrpu_board_pcie_ingress_async_audit.tcl`.

## Replay de la netlist contra MIG

`203417Z` se detuvo antes de compilar porque Vivado interpretó las opciones de
bibliotecas como argumentos de `set_property`. El fallo queda conservado y
respaldado, SHA-256 `d20b7d9682322548c87adf52c8f667c6cf2946a4ea6cd789ec3f6643e76aa825`.
La revisión R2 usa `set_property -dict`; no altera netlist, IP ni modelo.

`203709Z` **pasa** con la netlist sintetizada real, sin compilar los siete RTL
de producción como sustitutos. Son 123 entradas compiladas, 13 trabajos,
194 transacciones por cada lado, nueve escrituras DDR y 63 lecturas/4.032 bytes.
Los **2.420 eventos observados coinciden exactamente con RTL**, incluidos
tiempos, payloads, rechazo de destinos, conflicto, stop y recuperación.
Se correlacionan los 144 avisos del modelo y se drenan 17 lecturas periódicas;
137 alteraciones de evidencia son rechazadas. Duración: 285,447 s.

El observador WREADY toma el pin real del convertidor porque síntesis renombró
su net externa. El resto del banco es idéntico. Se revisan 17 advertencias de
compilación/carga; las dos definiciones de `glbl` tienen cuerpos equivalentes.
Esta simulación funcional de primitivas no incluye SDF ni verifica metastabilidad.
Reproducción: `tools/ntrpu_board_pcie_ingress_netlist_observed_r2.tcl` con el
`ingress_async_funcsim.v` de `202525Z`. Respaldo de R2 verificado local/NFS
a las 20:47:08 UTC: 646 archivos/57.542.496 bytes, SHA-256
`83fbfa796689dfe87249f821c732727c6d17628bb37eec2b8c95314980f4e1c8`.

## Integración que falta cerrar

El [control de registros y su doble CDC](PCIE_INGRESS_CONTROL.md) pasa
contra MIG frío a host/UI de 4/5 y 7/5 ns. La netlist sintetizada conserva
7.401 eventos exactamente iguales a RTL. La tanda nativa `223702Z` conecta
ambos buses del endpoint XDMA y programa reserva, STOP, consulta de resultados
y ACK mediante PCIe. Conserva los testigos periódicos y ocho lecturas DDR
comparadas. El intento anterior queda respaldado como fallo de conteo de
la consulta de enumeración; no se cambian los estímulos al corregirlo.

El aislamiento real del transporte sigue pendiente. `transport_idle` no
acredita el drenaje del CDC ni sustituye la
barrera confiable que todavía aporta el banco. El próximo bloque debe impedir
reutilizar una ventana mientras existan escrituras publicadas antiguas; necesita
reserva/época, cierre de admisión, drenaje acreditado del origen y de ambos lados
del CDC, y resultado consultable tras los B de memoria. Probar una escritura
retrasada durante stop/rearme y demostrar que no llega a la nueva reserva antes
de habilitar esa reutilización en el host real. Una respuesta B local no
confirma por sí sola el commit al emisor PCIe. El perfil alternativo
[NTRPUF01](PCIE_INGRESS_FRAGMENTS.md) lleva identidad en cada fragmento;
su decodificador aislado ya pasa RTL/netlist, pero faltan mapeo PCIe nativo,
captura/ensamblado y el ensayo de paquetes antiguos tras rearme. No elimina
la barrera requerida por el perfil DWORD actual.
