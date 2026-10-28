# Actuador RTL de reset local

## Resultado

`20261005T170319Z-V4-ffn-supervised-local-reset-r2` termina en 150,999 s:
ocho casos adversos y seis casos en cada uno de dos perfiles del convertidor
AXI4-Lite AMD real. El RTL dirige ahora aserci?n, conteo y liberaci?n; el banco
emite ?rdenes de fase. Revisi?n terminal aprobada y respaldo local/NFS.

El componente sigue siendo una pieza del coordinador, no una autorizaci?n de
reset del endpoint. El aislamiento host y los certificados externos de los
participantes son contratos controlados por el banco. Falta integrarlos con
el secuenciador, el guard, las colas de datos y XDMA.

## Interfaz y responsabilidades

`fpga/rtl/ntrpu_local_reset_actuator.sv` trabaja con un reloj de gesti?n
independiente y continuo. Todas las ?rdenes y se?ales `peers_held/peers_ready`
deben llegar ya calificadas en ese dominio.

- `LOCAL_HELD=2`: despu?s de que el padre pruebe propiedad y drenaje completo,
  una identidad nueva con `hold_request=1` afirma ambos resets, limpia los
  contadores y exige al menos 64 flancos reales por dominio. Tambi?n exige
  `peers_held` para confirmar.
- `LOCAL_READY=5`: con `hold_request=0`, espera que ambos contadores terminen y
  los participantes est?n en reset. Libera mediante tres flancos locales;
  confirma solo despu?s de observar ambos resets liberados y `peers_ready`.
- Una orden anticipada espera. Retirar una petici?n no libera un reset activo.
  Perder readiness retira el ACK, pero no fuerza un reset sobre tr?fico sin drenar.
- Cada orden incluye generaci?n y etiqueta de 64 bits. Se captura antes de
  responder; un cambio o retirada enmascara inmediatamente el ACK anterior.
  El padre debe garantizar identidades ?nicas: se retiene la ?ltima, no todo
  el historial. Una orden HELD nueva repite el conteo incluso estando en reset.
- `cold_rst_n` pertenece exclusivamente a configuraci?n. No conectar PERST
  ni resets de transporte. El componente no controla admisi?n ni sustituye
  la secuencia de arranque de placa.

El padre mantiene el aislamiento hasta terminar el mantenimiento. Este bloque
no conoce inventarios AXI ni autoriza reutilizar memoria o una sesi?n NTRD.
La regla de resets solapados sigue [AMD PG059](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).

## Pruebas y medidas

El banco aislado prueba fases desconocidas, relojes detenidos durante hold y
release, liberaci?n anticipada, participantes ausentes, limpieza de conteos
anteriores, retirada de peticiones, cambio de generaci?n y p?rdida de readiness.
Tambi?n comprueba que cada liberaci?n coincida con un flanco local.

La integraci?n conserva el [inventario y las transacciones del banco anterior](AXIL_PAIR_DRAIN.md):
cuatro escrituras y cuatro lecturas por perfil, con m?scaras, todos los RRESP,
backpressure y tr?fico posterior al reset. En el perfil lento vuelve a observarse
`0x0084a70000` en colas internas AMD aun con inventarios externos vac?os.
El actuador limpia ese estado antes de liberar.

| Dispositivo | Host/gesti?n | Flancos host/dispositivo al ACK HELD tras la parada | HELD ? READY observado |
| --- | --- | --- | --- |
| 5 ns | 7 ns / 10 ns | 73 / 715 | 90 ns |
| 37 ns | 7 ns / 10 ns | 73 / 97 | 180 ns |

Las latencias incluyen la emisi?n de la siguiente orden por el banco. No miden
recuperaci?n PCIe ni rendimiento f?sico. El perfil de 37 ns es estr?s CDC.

S?ntesis OOC para `xcvu5p-flva2104-2-i`: **91 LUT, 182 FF**, sin RAM ni DSP.
CDC informa ocho cruces CDC-9 de tres etapas; dos usan las excepciones del
XPM de reset y seis rutas custom conservan timing, sin grupos as?ncronos globales.
WNS/WHS estimados: 0,387/0,079 ns. Faltan retardos en 136 entradas y 142 salidas,
asignaci?n f?sica de clocks y revisi?n de recovery/removal tras integraci?n;
estas cifras no aprueban cierre f?sico ni relaciones entre relojes as?ncronos.

## Revisiones y trazabilidad

R1 (`20261005T170116Z-V4-ffn-supervised-local-reset`) fall? por una carrera del
monitor al cambiar la identidad en el mismo flanco que la comprobaba. R2 mueve
solo ese monitor a 1 ps despu?s del flanco activo. RTL y checker permanecen
id?nticos; se conserva el intento fallido.

El revisor gen?rico de R2 rechaz? un aviso cr?tico: la XDC del convertidor,
usado solo en simulaci?n, no encuentra instancia en el top del actuador aislado.
La adenda acepta exactamente ese caso, documentando que no se omiti? una
restricci?n del top sintetizado. Tambi?n revisa nueve avisos ordinarios:
tres de entorno, dos de par?metro aplicado a glbl, tres de clocks OOC y uno
por m?ltiples relojes. Los logs funcionales no tienen avisos.

Cambiar de top reemplaz? la primera lista de compilaci?n. La procedencia del
banco aislado se recupera de su an?lisis, elaboraci?n y finalizaci?n en consola,
contrastados con los hashes de la copia de fuentes. Las dem?s entradas SV
conservan su lista compilada. Se verificaron 42 archivos de bibliotecas AMD.
La revisi?n independiente rechaza doce alteraciones de evidencia.

- R1: 94 archivos; SHA-256 `a5c6158b02375dcdbfb2eeeb25b8a3c77e913a7a3ab826f7a36c781dd345d742`.
- R2: 135 archivos; SHA-256 `ec5912c41487c2071c9d4649fb8354e37524ca1fcb19ea5407e1bff0d8505043`.
- Adenda: `033fa705dda95d624848da86643b7140fc647a34ae37f6b1e163025bf5b1463c`.

Reproducir en directorio nuevo con
`vivado -mode batch -source tools/ntrpu_board_local_reset_actuator_r2.tcl -tclargs NUEVA_CARPETA`.
Las fuentes selladas no se editan. Consultar [secuenciador](ENDPOINT_RESET_SEQUENCER.md)
y [dependencias de recuperaci?n](ENDPOINT_RESET_COORDINATION.md).

## Controles negativos de RTL

`20261005T171232Z-V4-ffn-supervised-local-reset-mutants-r2` termina en
50,415 s. Dos copias privadas incorrectas son rechazadas por el mismo banco
R2 positivo: ignorar generaci?n falla en el caso 6; permitir release desde
COUNT falla en el caso 2. No se modifica el RTL aprobado. Los dos avisos de
entorno est?n revisados y las fuentes compiladas coinciden con sus hashes.

El primer lanzador (`20261005T171010Z-V4-ffn-supervised-local-reset-mutants`)
referenciaba por error el banco R1. El revisor lo rechaz? porque encontr? su
carrera del caso 4, no el fallo intencional esperado. Se conserva ese intento;
R2 corrige ?nicamente la ruta al banco aprobado.

- Intento fallido: 63 archivos, SHA-256 `ac1f3a2297dc6bc6f0271198845cd67cd9bb9c4adfd639d2d871414d4bbefc8f`.
- Dos controles rechazados correctamente: 113 archivos, SHA-256 `c27548caeda3fff74751d8b92f4d43ef6caf8e999b6345215f8659048c0277c8`.

Ambos respaldos est?n verificados localmente y en NFS. Reproducir con
`tools/ntrpu_board_local_reset_mutants_r2.tcl` en directorio nuevo; un PASS
de este banco significa que rechaz? las copias incorrectas, no que esas copias
sean utilizables.

## Avance: aislamiento con IP AMD

El [Shutdown Manager real](AXIL_SHUTDOWN_MANAGER.md) sustituye la entrada
de aislamiento ideal en una nueva tanda. Dos perfiles pasan, conservando
un inventario adicional arriba: cierre hacia memoria no equivale a vaciado
del host o XDMA. Saturaci?n y coordinaci?n completa siguen pendientes.
