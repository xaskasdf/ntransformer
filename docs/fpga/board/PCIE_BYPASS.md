# Ventana PCIe directa para la ruta entre NTRPU

Revisión del 2026-10-04 UTC. La configuración aislada `154440Z` y el
banco nativo de tres DWORDs `164020Z` están revisados y respaldados.
La ampliación con máscaras `170643Z` también pasa la revisión nativa y
está respaldada; su primer diagnóstico fallido se conserva por separado.
Es un paso de V4 hacia [la integración del receptor](../PEER_RX_INTEGRATION.md).
No modifica el endpoint de la FFN ni homologa P2P físico.

## Configuración comprobada

Vivado 2026.1, build 6511674, XDMA 4.2 rev4 y
`xcvu5p-flva2104-2-i`. Generación en 90,572 s, salida 0, sin advertencias
ni errores en consola; cuatro hilos y presupuesto de 600 s.

| BAR | Uso | Dirección PCIe | Apertura |
| --- | --- | --- | --- |
| 0 | AXI-Lite de control | 32 bits | 64 KiB |
| 1 | Registros DMA y MSI-X | 32 bits | 64 KiB |
| 2/3 | Master AXI-MM bypass | 64 bits, no prefetchable | 1 MiB |

La traducción del bypass usa base AXI `0x0000000100000000`; el puerto
`m_axib_*` generado tiene direcciones de 64 bits, datos de 512, strobes de
64 e IDs de cuatro bits. MSI-X conserva BAR1, tabla `0x8000` y PBA `0x8fe0`.
Los índices se verificaron en el bloque PCIE4 generado y sus wrappers,
además de la XCI de XDMA: los parámetros genéricos de BAR del nivel
superior no describen por sí solos el reparto efectivo.

Respecto de la referencia certificada `20261003T193438Z`, solo cambian
tres valores de configuración: habilitar bypass, hacerlo de 64 bits y
su base de traducción. Se revisan los proyectos principal/de ejemplo,
ocho wrappers sim/synth y 14 alteraciones negativas de evidencia.

## Reproducción y siguiente dependencia

Generador: `tools/ntrpu_board_pcie_bypass_probe.tcl`, con argumentos
`xcvu5p-flva2104-2-i DIRECTORIO_NUEVO`. El revisor es
`tools/check_ntrpu_pcie_bypass_probe.py`. La variante usa PIPE para
simulación; no implementarla como diseño físico.

La copia de fuentes es `source-20261004T154421Z.zip`, SHA-256
`d30fa5973377f167eaf70a0ddfebadd1a0602599e9f24ad1a9378155b4e84d05`.
El resultado, XCI, archivos generados y hashes se conservan por ID bajo
`artifacts/ntrpu-board-vivado/` y en el volumen NFS.

El siguiente banco debe enumerar y leer los BAR reales; emitir MemWr y
MemRd hacia la ventana; correlacionar TLP con AW/W/B y AR/R de bypass;
comprobar traducción sobre 4 GiB, límites, strobes, IDs y backpressure.
Después se conectarán autorización, propiedad/época, buffers, drenaje y
confirmación remota. Una ventana BAR no crea memoria coherente ni agrupa
automáticamente la capacidad de dos tarjetas.

AMD documenta la selección de destino según BAR en
[PG195, Target Bridge](https://docs.amd.com/r/en-US/pg195-pcie-dma/Target-Bridge).
El mapa de esta tanda procede de los archivos generados, no de asumir
una combinación predeterminada de BAR de 32 o 64 bits.

Respaldo verificado a las 15:51:51 UTC: 633 archivos, 69.285.230 bytes,
SHA-256 `44cfb8704210fcba30527775dc6046b958b6bf60acbd58e22b79d36941d2d8b6`.

## Primer banco de tráfico preparado

`ntrpu_board_pcie_bypass_sim.tcl` prepara tres escrituras y tres lecturas
de un DWORD en offsets `0`, `0x80000` y `0xfffc0`. Enumera el BAR en
`0x900000000`, distinto de la base AXI `0x100000000`, para comprobar la
traducción real. La memoria de prueba usa direcciones completas, tres
patrones independientes y centinelas para bytes no escritos; añade pausas
en AW/W/AR y respuestas demoradas. Todas las escrituras preceden a las
lecturas para detectar alias.

El ejemplo AMD conecta su esclavo de demostración a direcciones de 19
bits. El parche desconecta únicamente esos 29 puertos de bypass y conecta
la memoria del banco al master XDMA; conserva la IP y las conexiones
AXI-Lite/DMA. La comprobación local del parche pasó sobre los archivos
generados. La ejecución y su revisión nativa se describen abajo. Este banco corto
no mide caudal, transferencia de ráfagas largas ni recuperación de enlace.

### Historial de puesta en marcha del banco

| Tanda del 4 de octubre | Resultado observado | Respaldo |
| --- | --- | --- |
| `160117Z` | 20,676 s; error Tcl al pasar `-testplusarg` como valor; corregido mediante `set_property -dict` | Verificado; SHA `1025e5433dd78231a6df1cb2eef7b2409b2d8eb9118ee3652e7cc78489eb5c3b` |
| `160434Z` | 49,803 s; literales `1ps` del logger compilados como Verilog; se fijó SystemVerilog para COM/TX | Verificado; SHA `64006fa5d79793753b4b34163723c89699c4de3eef926bf3619e3f7e9450f31b` |
| `160948Z` | 7,273 s; ruta de entrada mal transcrita, antes de copiar el proyecto | Verificado; SHA `5644baa3ca44c5262eeac08fb8c6166e70476a22cafb895c4acf2a33f06fcf5c` |
| `161130Z` | 264,897 s; XSIM rechaza la referencia jerárquica a `m_axib_rlast`, declarado implícitamente por el ejemplo | Verificado; SHA `d419d30b383673942faedcef4589a61b88c3115d7f8ea41424307fdbfc749928` |
| `161947Z` | 419,521 s; tres palabras recuperadas; revisión de paquetes aplazada hasta captura nativa | Verificado; SHA `8bb895b956635df0ca10a291ed2b369966639ac214a53cd314ae0b7f1fd43d13` |
| `164020Z` | 185,566 s; replay del ejecutable anterior con VCD pasivo; PASS acotado de RQ/RC y AXI, 94 alteraciones rechazadas | Verificado; SHA `15791566ca2a7fa7db1fb74c02303134dadbef6bd91c1cb23c756fd3a032c06f` |

El lanzador de la última tanda obtiene la ruta del manifiesto de la
dependencia certificada y comprueba el `.xpr` antes de iniciar Vivado.
Los intentos anteriores conservan sus fuentes y errores; ninguno acredita
tráfico correcto. Una actualización externa de fuentes también fue
detectada por el control previo de staging y no lanzó Vivado; se tomó una
copia nueva sin modificar el trabajo paralelo.

La comprobación del parche posterior confirma que solo se añade esa
declaración; los otros cinco archivos instrumentados y los estímulos
permanecen idénticos. La tanda `161947Z` usa fuentes
`source-20261004T161937Z.zip`, SHA-256
`bc97dfe2cdb188f9b7c1f1394ef9362e2485676a9ebe2ad03b8a64847e06ae88`.
## Resultado con observación nativa

`20261004T164020Z-V4-ffn-supervised-bypass-native` conserva exactamente
375 archivos del ejecutable de `161947Z`, contrastados con su manifiesto
sellado. No recompila ni modifica RTL. El VCD independiente confirma tres
AW/W/B y tres AR/R, direcciones completas y 12 bytes en cada dirección;
los otros 60 bytes de cada palabra AXI conservan `0xa5`. Se observan 2,
7 y 2 ciclos de pausa en AW, W y AR, respectivamente, sin cambios de datos
mientras `VALID=1` y `READY=0`.
El respaldo de 398 archivos y 64.240.708 bytes quedó verificado en NFS y
localmente a las 16:50:22 UTC.

Las etiquetas solicitadas `0x95/96/97` del registro TX son anteriores a
la asignación del núcleo. Los pulsos nativos asignan `0/1/0`; cada RC
coincide con esa asignación, devuelve un DWORD, `Byte Count=4`, estado
correcto y petición completada. El payload RC nativo confirma el orden
de bytes del decodificador público. Se consultó
[PG213: etiquetas](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Tag-Management-for-Non-Posted-Transactions?contentId=I~JiLwhru37VKy8gSKn2Xw)
y el [descriptor RC](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Completion-Descriptor-Format).

La latencia RQ→RC observada es 396/404/396 ns; incluye la memoria del banco
y sus pausas. El sondeo del test host tarda 600 ns. Ninguna de estas cifras
representa latencia de DDR ni rendimiento de una placa.

Se conservan los 140 avisos originales y un `unique case` de XDMA en
tiempo cero. Los 14 avisos adicionales son la declaración tardía de RLAST,
la sensibilidad del timeout público y doce índices generados por XSIM en
código protegido inalterado. El PASS cubre las transferencias observadas;
no demuestra que esos avisos sean inocuos en otras configuraciones.
El ejemplo TX también coloca bits altos de su etiqueta en los bits
127/120 del descriptor: la traza pública no acredita ECRC en el enlace ni
interoperabilidad de etiquetas de diez bits.

## Siguiente tanda en cola

`20261004T165525Z-V4-ffn-supervised-bypass-bytes` terminó con un fallo del
banco en 413,460 s. Aunque Vivado devolvió 0, XSIM emitió
`Fatal: NTRPU_BYTES_AW_ADDRESS`; no se considera PASS. Con offset `0x3c`
y máscara `0xa`, el master expuso `AWADDR=0x10000003d`, el primer byte
habilitado. El banco esperaba incorrectamente el inicio del DWORD.
El VCD termina con una transferencia pendiente, y el revisor la rechaza.
Su respaldo quedó verificado a las 17:06:06 UTC: 839 archivos,
106.142.984 bytes, SHA-256
`ad7df8052debe40e226aefbb010027c76bc0e6d136e5b343dd47d6a623092b84`.

La revisión `20261004T170643Z-V4-ffn-supervised-bypass-bytes-r2` pasa
con la dirección esperada calculada desde el
primer bit habilitado de la máscara. Conserva los nueve estímulos,
WSTRB exacto y centinelas; no modifica la IP. Esta correspondencia parte
de la observación del modelo instalado y queda corroborada en los nueve
casos de la nueva ejecución. Depende
del respaldo verificado de `164020Z` y reutiliza la configuración
certificada `154440Z`. Sus nueve escrituras habilitan 24 bytes; las nueve
lecturas devuelven 36 bytes, incluidos los centinelas no escritos.
Los offsets son `0`, `4`, `60`, `0x80000`, `0x80020`, `0x8003c`,
`0xfffc0`, `0xfffe0` y `0xffffc`; las máscaras son `f/5/a/f/3/c/f/1/e`.
Son 447,805 s de ejecución y 82,880 µs simulados. Los stalls suman AW=7,
W=18 y AR=7 ciclos; señales estables bajo presión. RQ→RC de lectura mide
396–404 ns en este modelo, sin acreditar rendimiento físico. Se verifican
74 entradas compiladas, 134 fuentes AMD sin cambios y 61 alteraciones de
evidencia rechazadas. El inyector de fallos introduce también valores en
señales constantes del VCD; no necesita otra simulación.

RLAST se declara antes de su primer uso; se usan etiquetas de estímulo de
ocho bits y VCD desde la primera simulación. Se elimina el aviso de declaración
implícita; los otros 139 avisos y el aviso de tiempo cero coinciden con la
línea base, sin concluir inocuidad fuera del camino comprobado.
Respaldo verificado a las 17:20:55 UTC: 841 archivos, 107.508.791 bytes,
SHA-256 `d409ddafa67e85cc3afd5bea11bb5b281f1741abb3e4f269801c59139e30df1a`.

El logger actual solo admite solicitudes de
hasta 32 bytes; ampliar ráfagas requiere adaptar y validar ese observador.
Propiedad/época, integración DDR, reset, dos tarjetas y P2P siguen como
dependencias posteriores.

## Consecuencia para la integración FPGA

El [filtro de capacidad](DDR_AXI_GUARD.md) actual acepta `AxSIZE=6` y
direcciones alineadas a 64 bytes. Los DWORDs del bypass usan `AxSIZE=2`;
conectarlos directamente produciría rechazos, incluso en rangos válidos.
El receptor `ntrpu_peer_rx_guard.sv` también exige dirección alineada y
emite strobes completos. Estos contratos se conservan.

El [adaptador de ingreso propuesto](../PEER_RX_INTEGRATION.md) debe decidir
explícitamente entre rechazar escrituras pequeñas o ensamblarlas en un
buffer reservado. Para ensamblar, asociar AW/W con su ventana y época,
ubicar cada byte mediante la base de la línea y WSTRB, y mantener un mapa
de bytes válidos. No emitir una línea completa hasta haber recibido y
autorizado todos sus bytes; no rellenar huecos ni sobrescribir memoria
mediante una lectura/modificación/escritura implícita.

Probar el rechazo o ensamblado con estas mismas máscaras, AW/W separados,
presión de cola y stop a mitad de línea. Un B que acepta datos en un buffer
no certifica que llegaron a DDR ni autoriza reutilizar la ventana: la
finalización NTRPU debe esperar el drenaje y las respuestas del destino.
El [ingreso RTL de una línea](PCIE_INGRESS_PLAN.md) ya pasa captura AW/W,
ensamblado, reserva, stop y finalización en un banco independiente con netlist.
`180851Z` lo conecta además al bypass XDMA sin cambiar el RTL: 83 escrituras
PCIe producen tres líneas exactas, un conflicto rechazado y recuperación.
La revisión nativa incluye 81 señales y conserva la última línea del BAR.
La finalización del destino y el aislamiento de la reserva aún son
proporcionados por el banco de ingreso en esa tanda.

La tanda posterior `20261004T184808Z-V7-ffn-supervised-peer-link` conecta el
receptor y el controlador de épocas reales: verifica 105 señales nativas,
83 MemWr y tres escrituras completas de 64 bytes, con conflicto y recuperación.
El AXI de destino y la barrera externa siguen simulados; no incluye MIG.
El puente tiene además validación RTL/netlist de stop, respuestas erróneas,
B ajena, B ausente y rearme. Véanse [resultados y límites](PCIE_INGRESS_PLAN.md#pcie-nativo-con-el-receptor-conectado).
