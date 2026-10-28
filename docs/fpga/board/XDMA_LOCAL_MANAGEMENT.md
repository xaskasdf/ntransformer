# Gestión local del XDMA para el drenaje

Estado al 2026-10-06: **generación y banco nativo R2 revisados y
respaldados**. Continúa el [inventario nativo](XDMA_DRAIN_ACCEPTANCE.md).
Ambos perfiles de resultado retenido ya pasaron revisión y controles.

## Problema que resuelve

El banco R5 observó una consulta PCIe de control bloqueada detrás de una
B del BAR. La liberación independiente de R6 resolvió el banco, pero el
coordinador de placa necesita una vía de gestión que pueda probarse sin
depender de esa consulta. Además, el XCI actual no expone puertos de estado.

La variante añade `xdma_axilite_slave=true` y `xdma_sts_ports=true` sobre
la copia certificada con reset opcional. Conserva DMA, un H2C y un C2H,
AXI de 512 bits, PCIe x16, ambos maestros de BAR y Completion Timeout
de 50 ms. El acceso local solo alcanza registros DMA; no accede a registros
del bloque PCIe ni genera solicitudes al host.
[PG195: AXI4-Lite Slave](https://docs.amd.com/r/en-US/pg195-pcie-dma/AXI4-Lite-Slave).

## Generación y revisión exigidas

`tools/ntrpu_board_xdma_local_management.tcl` importa en una carpeta nueva
el XCI de `20261005T141836Z-V4-ffn-supervised-xdma-soft-reset-example`.
Exige SHA-256
`cb6acfd644df398ac2f4fd0e7711bc3840e148ed4f4667bcd0a77c9f513b2b77`.

La revisión posterior debe comparar parámetros de componente y modelo,
puertos de simulación/síntesis, wrapper público y conexiones del ejemplo.
Solo se admitirán diferencias explicadas por esas dos opciones; registrar
cualquier cambio derivado, aviso o generador de tráfico añadido por AMD.
No conectar el ejemplo a la composición retenida antes de esa revisión.

La tanda `20261006T165525Z-V4-ffn-supervised-xdma-local-management-frozen`
pasó la revisión de dos XCI y cuatro wrappers: solo cambian las dos opciones
indicadas y aparecen 21 puertos públicos. Los cuatro wrappers PCIe del
bloque físico siguen idénticos. Archivo SHA-256
`4f7fb26b5b030094326eab7e32e927223cfb74c343ecf4b7d554b3a1116deb44`.

De los 18 archivos públicos importados, 15 son idénticos. AMD sustituye
la RAM AXI-Lite de aplicación por un bucle `m_axil`→`s_axil`, deja abiertos
los estados y retira la excepción DRC de esa RAM. La revisión del ejemplo
es `2f2a08504193a96e36df20bf82eab1a70be507e4cf893ac6db48c7983e28415a`;
cinco alteraciones de puertos, lógica, parámetros y timeout son rechazadas.
Una primera revisión contó una excepción comentada como activa; queda
registrada como diagnóstico del revisor, corregido en R1.

## Primer intento: descubrimiento BAR0 sin respuesta

Diagnóstico respaldado: `20261006T170709Z-V4-ffn-supervised-xdma-local-management-native-frozen`.
Fuente `source-20261006T170639Z.zip`, SHA-256
`486d57292c53b1d56abbb927adcfa26724a21192ec9cda6136648bf846b760be`.
El banco privado conecta un maestro local a los puertos públicos del XDMA;
BAR0 queda sin servicio y una aserción impide solicitudes a esa apertura.
BAR1 conserva las consultas DMA y BAR2 las transferencias de memoria.
El reset opcional permanece inactivo. La simulación no quedó aprobada:
AMD busca primero el identificador XDMA leyendo BAR0. La petición nativa
salió a 88,320 µs; `m_axil_arvalid` subió a 88,488001 µs y permaneció
pendiente. El timeout del BFM terminó a 148,320 µs, aún en fase 0 y sin
lecturas/escrituras locales. Es un defecto del soporte del banco: todavía
no se había probado la independencia de gestión. El timeout de la IP
siguió en 50 ms. Duración: 721,485 s; archivo de 103.098.500 bytes,
SHA-256 `0efe911cf6e0af05f86857e8299e10d5560668b28c23018fe2795a373953289f`.

## R1: diagnóstico de cobertura; R2 aprobado en su alcance

Tanda `20261006T172223Z-V4-ffn-supervised-xdma-local-management-native-r1-frozen`.
Fuente `source-20261006T172143Z.zip`, SHA-256
`6c2b26e0b1cae3f5af208a05dfbb4a622abecbe92bcd68e1fb0406758934ec93`.
Un responder público acepta exactamente una lectura de dirección cero
en BAR0 antes de fase 1 y devuelve cero, como la RAM inicializada del
ejemplo anterior. Conserva el descubrimiento de BAR1 y prohíbe accesos
posteriores a BAR0. No es una implementación de memoria BAR0 de producción.
El revisor debe reconstruir AR/R y la completion nativa del descubrimiento,
además de todas las pruebas siguientes. R1 terminó con la marca de éxito
del banco, pero la revisión estricta encontró cero ciclos de retención de
B/R locales: READY subía antes de que llegara VALID. Se conserva como
diagnóstico de cobertura, no como aprobación completa.

La revisión diagnóstica sí reconstruyó 31 lecturas y 13 escrituras locales,
13 lecturas PCIe con sus completions, dos escrituras posted, los 64 bytes
H2C/C2H y la consulta BAR0 inicial. Las respuestas exteriores se retuvieron
durante 292 ns (BAR B), 252 ns (H2C B) y 408 ns (C2H R). Se revisaron 139
avisos, 73 entradas compiladas y 129 archivos AMD idénticos. Duración:
646,8 s; archivo SHA-256
`9b26c95d1b1e2840eb10935f7cf8bfe3056708f1c50ee1dc9693b7df097b83d5`.
Revisión diagnóstica:
`6c7423518e9853d06f3fb11942660a0ad3baf0cb475b0ee6e6038ce8ace6414b`.

R2 (`20261006T174559Z-V4-ffn-supervised-xdma-local-management-native-r2-frozen`)
espera el VALID local real y mantiene READY bajo tres ciclos antes de
aceptar. Conserva el revisor estricto, patcher, parámetros y timeout PCIe
de 50 ms. Fuente `source-20261006T174531Z.zip`, SHA-256
`883d07f9d4c5a9551e89364c9ed24082f96b0bef86233a852100f41df141f814`.
R2 pasó el revisor estricto y 12 alteraciones de evidencia: 31 lecturas y
13 escrituras locales, con 39 ciclos B y 93 ciclos R de retención real.
Las respuestas exteriores permanecieron pendientes durante 364 ns (BAR B),
312 ns (H2C B) y 504 ns (C2H R). Se reconstruyeron los 64 bytes completos
H2C/C2H, consultas PCIe, contadores y estados; se mantuvieron 129 archivos
AMD idénticos y se revisaron 73 entradas compiladas y 139 avisos.

La simulación terminó a 124,930 µs, tras 633,24 s de ejecución. El archivo
de 100.553.289 bytes y sus 769 miembros se verificaron en D: y NFS:
`0d316e1012963d596b09092b7b14c8aaea3b638dabf38ccf274d2d6a231bfa5e`.
Revisión independiente:
`2219b481f02ee8b33649649fe8173de60db8e2b3070add69249adc48a2b3774b`.
Esto valida la vía local en el banco descrito. El maestro del banco aún
era procedural: su reemplazo por RTL está en
[la tanda de integración del maestro](XDMA_LOCAL_PROVIDER_DESIGN.md).

El contrato revisado cubre lecturas locales de control/estado/contador y el borrado W1C de
Run, contrastándolos con consultas PCIe serializadas. Luego retener B
de BAR mientras la vía local consulta ambos motores: exigir handshakes
AXI-Lite completos y datos conocidos, sin temporizador que invente éxito.
Si también se bloquea, conservar la dependencia observada y revisar
la arquitectura.

Repetir con H2C B y C2H R pendientes. Correlacionar `Busy`, `Run` y pulsos
de descriptor con AXI, contadores y recepción completa del host antes de
conectar un proveedor de certificados. Los puertos incluyen Busy y Run;
sus eventos no son un certificado persistente por sí solos.
[PG195: puertos de estado](https://docs.amd.com/r/en-US/pg195-pcie-dma/Channel-0-3-Status-Ports?contentId=lvQzvWluXgPxMZm7RJbB9Q).

## Límites que debe conservar el diseño

El fin C2H documentado significa emisión de la escritura por PCIe,
no recepción en RAM del host. Mantener esa obligación separada y
vinculada a generación; no deducirla de `Busy=0`.
[PG195: C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel).
`axi_aclk` puede desaparecer: el supervisor seguirá usando su reloj de
gestión independiente y un intercambio atómico de certificados. El reset
nativo completo, su duración y la restauración posterior siguen pendientes.
[PG195: relojes y resets](https://docs.amd.com/r/en-US/pg195-pcie-dma/Clocking-and-Resets).

Cierre del driver, inventario protegido, siete certificados coherentes,
reset operativo, DIMM exacto, cuatro canales e implementación física
conservan sus criterios originales. Esta variante no los da por aprobados.
