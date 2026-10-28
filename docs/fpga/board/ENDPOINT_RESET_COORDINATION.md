# Coordinación del reset del endpoint

Estudio V4/V7 del 2026-10-05. Complementa [HOST_LOSS_GUARD](HOST_LOSS_GUARD.md).
La retención de propiedad y el reset del endpoint son contratos distintos.
El banco nativo de PERST `20261005T134131Z` pasó la revisión funcional
y 41 alteraciones negativas. Su respaldo local/NFS quedó verificado
el 2026-10-05 a las 14:58:19 UTC; no habilita reutilización de reservas.

## Interfaces observadas

La [sección AXI4 retenida](AXI4_RETAINED_SECTION.md) ya tiene una primera
composición con el Clock Converter AMD: conserva B/R locales durante reset
exclusivo del CDC y contiene una R huérfana real. Sus certificados excluyen
colas protegidas XDMA y respuestas locales; esta evidencia no modifica los
requisitos de reset del endpoint descritos aquí.

La IP del respaldo `20261004T154440Z-V4-ffn-supervised-pcie-bypass-probe`
usa XDMA 4.2, `functional_mode=DMA`, `soft_reset_en=false` y
`dma_reset_source_sel=User_Reset`. Su plantilla expone `sys_rst_n`,
`axi_aclk`, `axi_aresetn` y `user_lnk_up`, pero no `dma_bridge_resetn`.
El ejemplo conecta `axi_aresetn` a `user_resetn`.

PG195 describe `axi_aclk` como reloj derivado que puede detenerse durante
reset. Por eso gestión y retención no pueden depender de ese reloj.
El reset opcional del puente requiere tráfico detenido y una duración
relacionada con Completion Timeout; los 64 flancos de nuestros CDC no
sustituyen ese requisito. La sección de resets lo documenta para modo
Bridge: no extrapolar soporte al modo DMA por aceptar una propiedad Tcl.
Fuentes: [relojes y resets](https://docs.amd.com/r/en-US/pg195-pcie-dma/Clocking-and-Resets),
[puertos globales](https://docs.amd.com/r/en-US/pg195-pcie-dma/XDMA-Global-Ports),
PG195 v4.2, 2026-07-22, consultado el 2026-10-05.

## Separar los tres contratos

| Contrato | Evidencia necesaria | Lo que no acredita |
| --- | --- | --- |
| Propiedad invalidada | ACK registrado en fuente; época, cookie y descriptor retenidos | Drenaje de PCIe o de DDR |
| Interfaces bajo reset | Secuencia observable del endpoint y ambos extremos CDC, asociada a esta pérdida | Disponibilidad del enlace o memoria reutilizable |
| Interfaces disponibles | Reset liberado localmente, relojes presentes y nueva confirmación de peers | Reapertura de la sesión NTRD invalidada |

Mantener la vida de configuración y los resets MIG separados de PERST.
Una respuesta DDR tardía sigue perteneciendo al descriptor retenido.
Volver a ver `user_lnk_up=1` no autoriza liberar ese descriptor.

## Evitar una dependencia circular

No conectar directamente `host_reset_req` a un reset del endpoint.
En el RTL actual, `FAULT_HOLD` espera simultáneamente enlace válido,
reset del endpoint liberado y `peers_held` antes de retirar su petición.
Si esa misma petición mantiene activo el reset del endpoint, ambas partes
se esperan indefinidamente. Es una deducción del FSM actual, todavía no
una prueba de una nueva integración.

El coordinador futuro necesita fases separadas: retener invalidación,
solicitar y observar el reset permitido, completar su intervalo,
liberar el endpoint mientras mantiene los CDC/admisión cerrados,
observar su retorno y recién entonces completar la liberación local.
El guard actual tendría que incorporar ese contrato explícitamente.
Nunca introducir ACK combinacionales basados solamente en AW/W/AR idle.

## Próximos filtros de la cola

1. Auditoría de generación cerrada por R3: ver evidencia inferior.
   Generar un puerto no demuestra semántica, soporte ni reset funcional.
2. Tanda nativa de retención cerrada y respaldada: registra caída/retorno
   reales, memoria, propiedad y ausencia de ACK ficticios. Los fallos
   intermedios del verificador también quedan preservados.
3. Elegir el mecanismo documentado para la configuración exacta. Si DMA
   no permite el reset local requerido, mantener recuperación por host o
   reconfiguración; estudiar Bridge en otra variante con su contrato BAR.
4. Probar primero el secuenciador aislado: ACK antiguo, reloj detenido,
   PERST repetido, pérdida durante liberación y transacciones pendientes.
   Después integrar el endpoint real, incluyendo AW/W incompletos y B/R
   bloqueados; el banco actual drena el tráfico host antes de PERST.

La recuperación física, reenumeración, conservación de configuración PCIe,
reutilización caliente y DDR comercial permanecen pendientes.

## Auditoría de puertos en Vivado 2026.1

`20261005T140115Z-V4-ffn-supervised-xdma-reset-ports-r3` termina en
50,407 s, sin advertencias. Dos proyectos importan sus XCI por separado
y comprueban que cada directorio de generación quede dentro de su variante.
La interfaz pasa de 145 a 146 puertos: añade únicamente la entrada
`dma_bridge_resetn`. Los parámetros de componente cambian solo en
`soft_reset_en=false→true`; se conserva `functional_mode=DMA`.
La interfaz y parámetros iniciales coinciden con la referencia sellada;
sus 458 archivos de ejemplo mantienen sus hashes. El revisor rechaza
tres alteraciones de rutas, puertos o modo.

Resultado: **generación de interfaces comprobada; semántica de reset DMA
sin calificar**. No se ejecutaron simulación, síntesis ni cambios del
endpoint integrado. Fuentes: `source-20261005T140031Z.zip`, SHA-256
`2304135bebd09dccd286eebcf905e41fcbcc22133fb2e82325ffdedb8909a03d`.
Respaldo local/NFS: 58 archivos, 975.694 bytes; SHA-256
`6a48f123c2add662fa796150e3483fb6065a9cbbe48266eda22cc90c490e6a62`.

Los intentos anteriores permanecen disponibles: `135307Z` falla antes
de crear IP por usar `redirect`; `135613Z` pierde la separación de
plantillas porque `read_ip` conserva una ruta relativa de generación.
R2 se registra como procedencia incompleta y sus archivos compartidos
se preservan, con hashes, dentro de su respaldo. R3 usa `import_ip`
y comprueba esa ruta antes de generar. Ninguno de esos dos intentos
aprueba la comparación de puertos ni modifica el ejemplo original.

## Retención frente a PERST: evidencia aprobada y respaldada

`20261005T134131Z-V7-ffn-supervised-guarded-native-reset` termina con
código 0 en 2.476,610 s y 247,005 µs simulados. PERST se activa en
179,310 µs; la caída del endpoint aparece 100 ps después. La respuesta
DDR bloqueada se libera y se comprueban sus datos antes de retirar PERST
en 184,310 µs. El enlace vuelve en 246,420 µs: **62,110 µs después de
liberar PERST**, en este modelo PIPE y banco concreto.

La reserva conserva cookie `0x100000010` y época `0x100000013`; el guard
continúa en `FAULT_HOLD`, con transporte cerrado y sin ACK de peers.
El total observado es 37 frames, 17 escrituras y 29 lecturas DDR.
Esto no demuestra recuperación de la sesión ni tiempo de retorno físico.

R1 del revisor detectó 18 avisos de `board.v` desplazados cinco líneas
y un aviso nuevo de declaración posterior de `sys_rst_n`. R3 comprueba
la identidad del código en cada línea desplazada y el único registro
de reset; mantiene estricta la comparación del resto de advertencias.
R2 queda conservado por una colisión del propio algoritmo de remapeo.
La simulación y sus entradas permanecen intactas. R3 rechaza 41
alteraciones de evidencia y comprueba 221 entradas compiladas y 86
señales VCD. Respaldo: 1.474 archivos, 258.110.567 bytes; SHA-256
`46c609d750b928e5f48d9251ba7b6933749ba31ccd1b148f9c9e138bacaf95e2`.

| Suceso del caso de pérdida | Tiempo simulado |
| --- | --- |
| PERST activo | 179,310 µs |
| ACK de invalidación del propietario | 179,345 µs |
| Reset del transporte | 179,385 µs |
| Aceptación de B DDR pendiente | 179,660998 µs |
| Completion puesta en cuarentena | 179,717500 µs |
| Retorno de la lectura de comprobación | 180,230998 µs |
| PERST liberado | 184,310 µs |
| Enlace de nuevo disponible | 246,420 µs |

## Ejemplo con reset opcional y siguiente diagnóstico

`20261005T141836Z-V4-ffn-supervised-xdma-soft-reset-example` genera un
ejemplo aislado en 92,698 s, sin avisos. Sus cuatro wrappers superiores
cambian solo en el parámetro, puerto y conexión del reset; los cuatro
wrappers del bloque PCIe permanecen idénticos. El ejemplo aún conecta
`dma_bridge_resetn` a uno. Respaldo verificado: 634 archivos,
69.346.262 bytes; SHA-256
`cd2f158bbe39f8e5bf8b39c0b20035731e3d769a788568bf6c8b869e4a20cc0e`.

Está encolado un diagnóstico nativo separado, sin MIG: enumerar,
comprobar 256 flancos de referencia sin tráfico, activar el reset
opcional y observar otros 256 flancos. Finaliza manteniéndolo activo;
no intenta sustituir el intervalo de Completion Timeout ni probar su
liberación. El VCD debe corroborar reset, enlace, relojes, tráfico nativo
y cada muestra del banco. Solo podrá arrancar tras el respaldo de PERST,
sin otras simulaciones activas. Las pruebas sintéticas del verificador
son pruebas de software, no evidencia de comportamiento de XDMA.

La inspección del ejemplo sellado identifica `DMA_RESET_SOURCE_SEL=0`
y `AXI_ACLK_LOOPBACK=FALSE`. Su wrapper combina el reset del núcleo con
el reset opcional y conecta el inverso directamente a `axi_aresetn`;
`axi_aclk` toma `user_clk`. Esta observación de código permite predecir
la aserción que debe buscar el VCD, pero no califica el interior protegido,
el drenaje ni la liberación. En particular, no asumir que esa asignación
añade un sincronizador de liberación para la lógica NTRPU. La secuencia
de liberación local, los relojes presentes y las restricciones de reset
siguen siendo requisitos explícitos del coordinador futuro.

La XCI sellada selecciona `comp_timeout=50ms` y genera
`C_COMP_TIMEOUT=1`. Por tanto, el próximo contador de mantenimiento
debe derivarse de ese valor y de la frecuencia máxima del reloj de
gestión, con margen para superar el timeout. A 100 MHz exactos,
5.000.001 ciclos superan 50 ms; la tolerancia y frecuencia de placa aún
deben incorporarse. No reducir silenciosamente el timeout para acelerar
XSIM ni confundir los 256 flancos de observación con ese intervalo.

El primer intento del diagnóstico, `20261005T145900Z`, falló antes de
compilar: `set_property` interpretó `-testplusarg` como opción propia.
Se preserva con respaldo SHA-256
`48a5203b713b0b25469293df9b4c8086e733e3500bc0c814251a060c863bced8`.
La revisión R2 cambia únicamente esa llamada a la forma `-dict`; conserva
el banco, el parche y el verificador originales.

## Observación nativa de aserción cerrada

`20261005T150122Z-V4-ffn-supervised-xdma-reset-assert-r2` termina con
código 0 en 503,036 s. Enumera con 32 requests y 32 completions nativas,
comprueba 256 flancos de referencia en reposo y activa el reset en
97,530001 µs. `axi_aresetn` cae en ese mismo instante del modelo;
esto no mide propagación física. Las 256 muestras siguientes lo ven
activo y los enlaces de ambos extremos permanecen arriba durante toda
la traza posterior. La observación acaba en 100,085001 µs, manteniendo
el reset: intervalo de 2,555 µs. Cada reloj nativo aporta 639 flancos
durante ese intervalo, separados 4 ns.

El verificador original exigía una entrada VCD en el último picosegundo,
aunque allí solo se imprimía el resultado. Se conserva su diagnóstico.
El nuevo verificador mantiene cada valor hasta el siguiente cambio y
acepta únicamente ese margen terminal de 1 ps, corroborado por el
`$finish` del simulador. Rechaza 14 alteraciones de evidencia. Los 139
avisos coinciden exactamente con la referencia PCIe después de comprobar
48 localizaciones de código, ocho de ellas desplazadas.

Respaldo del diagnóstico: 842 archivos, 109.063.901 bytes, SHA-256
`c8d0bd055ce3ea61570b2373f0f84388f1d09207325fda2353f3da81d2b02d5d`.
Adenda de aprobación independiente, también verificada local/NFS:
`040f0747ca872512eb02ceecbe4bcd0f25dc5903f8c16f73556af3af00fde9bc`.
La adenda aprueba esta observación; el diagnóstico original no se reescribe.

Quedan abiertos el mantenimiento superior a 50 ms, liberación, restauración
de registros, reinicio de tráfico y coordinación con CDC/propietario/MIG.
El siguiente diseño debe separar el permiso de liberar el endpoint del
permiso de liberar los CDC, mantener la sesión invalidada y comprobar
ACK antiguos, relojes detenidos y nuevas pérdidas durante la liberación.

## Secuenciador aislado y siguiente integración

El [secuenciador](ENDPOINT_RESET_SEQUENCER.md) ya pasa nueve casos,
incluido el intervalo completo nominal de 50,00001 ms, y cuatro controles
con RTL deliberadamente incorrecto son detectados por el banco original.
Los resultados y la síntesis aislada están respaldados. Esta prueba usa
ACK controlados: no demuestra mantenimiento ni liberación del XDMA nativo.

El siguiente componente es el [monitor AXI-Lite](AXIL_DRAIN_MONITOR.md),
para impedir que un bus momentáneamente sin VALID o una respuesta
bloqueada se confundan con drenaje. Después deben implementarse la
transferencia atómica entre dominios y productores verificables de todos
los ACK. El guard y los ejemplos nativos previamente sellados permanecen
sin cambios; su integración futura requiere otra revisión y otra tanda.

El [buzón de certificados](TAGGED_RESET_MAILBOX.md) ya valida el paso
atómico de generación/tag/fase y descarta respuestas obsoletas. Su banco
conecta el monitor AXI-Lite al secuenciador mediante XPM real entre
relojes distintos. Quedan pendientes productores de aislamiento y
drenaje para todas las interfaces, sus inventarios internos y la
adaptación del guard; el retorno de este ACK parcial no autoriza reset
de todo el XDMA.

## Avance: dos interfaces y reset local

El [banco con el convertidor AMD real](AXIL_PAIR_DRAIN.md) pasa ambos perfiles.
Distingue certificado externo de estados internos de handshake; la secuencia
de reset todav?a la conduce el banco y no autoriza recuperaci?n del endpoint.

## Avance: actuador local en RTL

El [actuador de reset](LOCAL_RESET_ACTUATOR.md) pasa sus ocho casos y dos
perfiles con el convertidor AMD real. Implementa conteo y liberaci?n autom?tica;
la uni?n completa con el secuenciador y el drenaje del endpoint sigue pendiente.

## Avance: aislamiento con IP AMD

El [Shutdown Manager real](AXIL_SHUTDOWN_MANAGER.md) sustituye la entrada
de aislamiento ideal en una nueva tanda. Dos perfiles pasan, conservando
un inventario adicional arriba: cierre hacia memoria no equivale a vaciado
del host o XDMA. La [saturación](AXIL_SHUTDOWN_SATURATION.md) y la
[frontera retenida](AXIL_RETAINED_FENCE.md) posteriores están respaldadas.
La última limita admisión y produce su propio certificado de detención
hacia el manager; conserva errores locales durante reset de la sección.
Revocación por fallos posteriores, AXI4, colas internas del endpoint y
coordinación completa siguen pendientes.


## Fallos posteriores al certificado: R3

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

## Composición de origen y fallo con manager abierto

La [composición real](AXIL_FAULT_COMPOSITION.md) de admisión R2 y certificado
R4 resuelve el CDC del origen: once sincronizadores de tres etapas,
754 LUT/1.725 FF y regresión posterior al certificado aprobada. El banco
previo al certificado falla con una B huérfana que llega al manager abierto.
Antes de integrar el coordinador global se necesita contención de respuestas
con inventario causal, alarmas retenidas y conservación de transacciones
válidas. Se preserva el fallo de la IP; no se desactiva su aserción ni se
considera que el cierre con latencia CDC ya proteja ese intervalo.

## Contención incorporada en sección R3

El [protector de respuestas](AXIL_RESPONSE_GUARD.md) retiene B/R sin crédito
antes del manager, mantiene las respuestas válidas del otro canal y añade
tres fuentes registradas de alarma. Pasan 16 escenarios de vida útil y
12 de regresión con IP AMD real, cinco casos aislados y dos controles
negativos. Síntesis de la composición: 802 LUT/1.762 FF, 14 CDC-3 sin avisos
CDC. Las cinco tandas y sus adendas están respaldadas localmente/NFS.

La cola R11 exige todavía alarma exclusiva del protector con host detenido,
inyección R a través del CDC real y controles de cuarentena. Drenaje interno
XDMA/AXI4, coordinación global y reset nativo completo conservan sus
dependencias. Esta sección no autoriza reutilización NTRD ni reset global.

La continuación [AXIL_GUARD_FAULTS](AXIL_GUARD_FAULTS.md) completa los
pendientes R11: 24 escenarios de alarma exclusiva/reloj detenido y B/R
por CDC real; tres controles negativos específicos detectados. La cola R12
deja cubierta esta etapa y avanza al [inventario AXI4](AXI4_DRAIN_PLAN.md).
El inventario interno XDMA y el mecanismo nativo de reset siguen separados.

La cola R13 registra la [primera implementación AXI4](AXI4_DRAIN_MONITOR.md):
22 escenarios, cuatro mutaciones detectadas y síntesis OOC revisada.
`complete_drain` permanece abierto hasta probar capacidad de integración,
CDC real, frontera activa y contrato de colas protegidas del endpoint.

La cola R14 incorpora la [capacidad y composición CDC pasiva](AXI4_CDC_DRAIN.md):
32 escrituras/128 lecturas, tráfico 512/64/4 con D a 5/37 ns, identidades
frescas y reporte de una R huérfana con D detenido. La síntesis del par
mide 5.320 LUT/6.648 FF y 16 cruces informativos de tres etapas. No incluye
el Clock Converter de datos en ese total ni aprueba timing físico.
La R inválida atraviesa el par pasivo: el siguiente paso debe contenerla
causalmente y cerrar admisión con un grupo finito de AW/W/AR, preservando
ofertas y respuestas válidas pendientes. Solo después corresponde integrar
el contrato de colas internas XDMA y el coordinador; `complete_drain` sigue abierto.

La cola R15 incorpora el [protector causal AXI4 aislado](AXI4_RESPONSE_GUARD.md):
19 escenarios, tres controles negativos y síntesis de 2.213 LUT/2.588 FF.
Su registro por canal preserva respuestas ya capturadas; créditos posteriores
no reabren una respuesta huérfana. Todavía falta la composición real con
la [frontera activa y errores locales](AXI4_ACTIVE_BOUNDARY_PLAN.md), además
de los inventarios XDMA. La prueba pasiva R14 permanece intacta y sigue
permitiendo atravesar la respuesta huérfana en su propio banco histórico.

R16 incorpora la [admisión AXI4 retenida](AXI4_RETAINED_ADMISSION.md): trece
escenarios aislados, cuatro controles negativos y síntesis de 4.113 LUT /
3.496 FF con el protector incluido. `source_fenced`, `drain_ack` del tramo
y `local_idle` son contratos distintos. Una prueba del tramo puede ser
válida mientras B/R locales permanecen retenidas; no permite resetear XDMA.
Falta componerla con CDC real, certificados de generación/tag y reporte de
fallos con reloj de origen detenido, preservando los errores locales durante
reset exclusivo del CDC. Las dependencias nativas y de placa siguen abiertas.
