# Supervisor V2: controles asincronos independientes

Estado al 2026-10-11: **84 casos RTL, dos casos nativos AMD y revision interna
CDC OOC aprobados**. Sigue pendiente la integracion y verificacion fisica.
`ntrpu_axi_bridge_guard_v2.sv` es una variante aislada. El guardian anterior,
cliente, mapper, escritor, top y cola principal conservan sus fuentes.
La [auditoria anterior](GUARD_CDC.md) sigue describiendo V1 y sus hallazgos.
SHA-256 del RTL V2 utilizado en todas las tandas:
`bb9a758d8cffeba6b96d238ddc295de11e126ee64dc9057a34a460bb1b054a8d`.

## Cambio de circuito

V1 combinaba disponibilidad, armado y fallo antes de controles asincronos;
Vivado senalaba CDC-10 y dos LUTAR-1. V2 usa cuatro cadenas independientes:
enlace, reset AXI, reset de control y PERSTn. Cada entrada puede borrar su
cadena de gestion sin reloj; su ultimo registro controla una cadena AXI.
La recuperacion necesita tres flancos de gestion y despues tres flancos AXI.
Asi la gestion observa un pulso de perdida antes de que pueda reabrirse la
admision. Despues del armado, el fallo queda retenido.

`armed` cruza como dato mediante tres FF. Su reset procede de la unica
liberacion `cold_a` del dominio AXI. `fault` sigue siendo la unica fuente
del PRE de `stop_a`. La conjuncion final controla `client_link_up`; no conduce
CLR/PRE. Los clientes existentes consumen disponibilidad en logica de
admision y en procesos sincronicos, sin convertirla en reset asincrono.

Se mantienen el watchdog request/echo, el contexto pendiente y la separacion
entre fallo operativo y reset de configuracion. No hay rearme automatico.
`mgmt_clk` debe seguir funcionando independientemente de AXI. El timeout
debe superar la ida y vuelta de sincronizacion al reloj AXI minimo previsto.

La topologia sigue las guias AMD sobre
[reset asincrono sincronizado](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Asynchronous-Reset-Synchronizer)
y [logica combinacional antes de un cruce](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Combinatorial-Logic).

## Evidencia de sintesis

Vivado 2026.1, `xcvu47p-fsvh2892-2-e`, modo OOC; gestion 100 MHz, AXI 250 MHz.
La tanda `cdc-20261011T055943Z` termino en 82,556 s.

| Comprobacion | V2 |
| --- | --- |
| Utilizacion estimada | 31 LUT logicas, 57 FF; 34 primitivas LUT antes del ajuste por combinacion |
| Sincronizadores | 14 cadenas de 3 FF, 42 ASYNC_REG |
| CDC internos | 3 CDC-3 y 5 CDC-9 informativos; cero criticos/advertencias |
| Logica en CLR/PRE | Solo una fuente por cadena; inversores LUT1 permitidos; cero LUTAR |
| Rutas del mismo reloj | 1804 conservadas con identicos margenes |
| Etapas internas | 28 conexiones, 56 comprobaciones setup/hold positivas e iguales |
| Minimos entre etapas | Setup +3,709 ns; hold +0,079 ns |
| WNS / WHS globales | +3,446 / +0,073 ns, estimaciones de sintesis |
| Controles negativos del revisor | 16 corrupciones rechazadas |

El ensayo agrega tres `set_max_delay -datapath_only` de C a D de la primera
etapa: request/echo y armado, limitados al periodo destino (4/10/4 ns).
Agrega quince pares false-path de C a CLR/PRE: doce para disponibilidad y
tres para fallo. No corta recovery/removal de `cold_a` a `armed_a`, ni las
rutas entre etapas. No hay grupos de reloj generales, waivers o excepciones
ignoradas. Se guardan reportes antes/despues, inventario de pines, DCP y netlist.
El DCP conserva la base sin las excepciones experimentales.

Los cinco controles externos y ocho salidas aun carecen de retardos de
integracion: permanecen trece TIMING-18. `report_cdc` no cubre por completo
los puertos sin esos retardos, como explica
[UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_cdc).
El inventario comprueba sus conexiones, no su comportamiento fisico.

## Bancos reproducibles

`run_guard_v2.py --phase rtl` congela el escritor y banco anteriores y cambia
solo la instancia a V2; Tcl asume la captura VCD para XSIM. Ejecuta los 17
escenarios anteriores en cuatro relaciones/fases de reloj: 68 casos revisados
por el revisor original. Suma 16 casos de pulsos nativos: cuatro entradas con
AXI activo, detenido bajo o alto, y cuatro pares solapados. Cada pulso dura
100 ps en simulacion digital; esto **no caracteriza el ancho fisico minimo**.
El revisor reconstruye eventos y comprueba cierre continuo sin reapertura,
fallo retenido, reset frio liberado y recuperacion del reloj sin rearme.

La tanda `rtl-20261011T060025Z` paso **84/84** en 544,110 s. Los pulsos
cerraron la admision antes del siguiente flanco; la gestion registro el fallo
6,999 ns despues, para la fase ensayada. Los casos detenidos cubrieron 48
flancos de gestion y 224 flancos AXI tras retornar. Las paradas del banco
original se detectaron en 33-38 ciclos de gestion con timeout de prueba 32.
El revisor de pulsos rechazo trece corrupciones de copias de trazas reales.
Esto prueba esas fases y escenarios digitales; no es una prueba formal.

La preparacion nativa reutiliza [CLOCK_LOSS.md](CLOCK_LOSS.md). Congela V2 bajo
su nombre propio y un alias de archivo que requiere el runner anterior;
ambos tienen bytes identicos y el modulo instanciado se llama explicitamente
`ntrpu_axi_bridge_guard_v2`. No sustituye el archivo V1 del repositorio.

`native-20261011T055946Z` paso ambos casos en 870,879 s: compilacion 2,343 s,
elaboracion 366,495 s y simulaciones 260,658/239,747 s. El log de elaboracion
confirma que compilo V2. Los 623 archivos de la semilla y el board original
permanecieron iguales. El mismo revisor V1 comprueba los VCD publicos.

| Inyeccion en salida AXI | Bajo | Alto |
| --- | --- | --- |
| Deteccion del watchdog | 670,999 ns | 672,999 ns |
| Flancos de gestion durante parada / recuperacion | 128 / 64 | 128 / 64 |
| Transiciones AXI durante parada | 0 | 0 |
| Flancos AXI ascendentes tras retorno | 160 | 160 |
| Contexto pendiente y fallo retenidos | Si | Si |

Se retiene la tercera solicitud: sesion 9, identificador 3, generacion 2,
origen `0x1000`, 64 bytes. No se acepta fuente ni se fabrica resultado o ACK;
el reset frio sigue liberado. Escritor y mapper permanecen congelados durante
la parada y registran fallo al volver el reloj. El marcador previo de 192
bytes es obligatorio, pero este revisor no reconstruye ese payload de forma
independiente: el resultado nuevo acredita el contrato de perdida de reloj.

Desde una carpeta nueva dentro del espacio privado `ntrpu-side` de la VM:

```bash
python3 -B run_guard_v2.py NUEVA_RTL --phase rtl --settings VIVADO_SETTINGS --execute-native
python3 -B run_guard_v2.py NUEVA_CDC --phase cdc --settings VIVADO_SETTINGS --execute-native
python3 -B review_guard_v2_cdc.py NUEVA_CDC/reports NUEVO_REVIEW.json
python3 -B check_guard_v2_cdc.py NUEVA_CDC/reports NUEVOS_CONTROLES.json
```

Para preparar localmente la prueba AMD y ejecutarla en la VM:

```bash
python3 -B prepare_guard_v2_native.py NUEVA_PREPARACION
# Tras copiar esa carpeta a la VM:
python3 -B NUEVA_PREPARACION/run_clock_loss_native.py NUEVA_NATIVA \
  --seed SEMILLA_COMPILADA/work --example EJEMPLO_R3/output/example \
  --settings VIVADO_SETTINGS --execute-native
```

Los runners no transfieren archivos ni contactan la VM: requieren lanzamiento
explicito. Usan dos CPU, nice 10 y plazos por proceso propio. La tanda actual
asigna pares distintos de CPU a RTL, sintesis y prueba nativa.

## Respaldo y restauracion

La evidencia queda en `D:/AI/artifacts/ntrpu-axi-bridge-guard-v2/`:

- `rtl-20261011T060025Z`: 267 archivos inventariados; archivo remoto de 179047 bytes.
- `cdc-20261011T055943Z`: 47 archivos; archivo remoto de 101310 bytes.
- `native-20261011T055946Z`: 46 archivos; archivo remoto de 821115 bytes.
- Dos intentos iniciales fallidos: biblioteca XSIM no explicitada y apertura
  duplicada de VCD. Se conserva su evidencia; no cuentan como casos aprobados.
- `20261011-guard-v2-bundle.zip`: fuentes, reportes, DCP, VCD, logs, archivos
  remotos y manifiestos SHA-256. El recibo es `20261011-backup-receipt.json`.

El bundle permite repetir la revision sin Vivado:

```bash
python3 -B BUNDLE_RESTAURADO/source/verify_guard_v2_bundle.py \
  BUNDLE_RESTAURADO NUEVO_RECIBO_FUERA_DEL_BUNDLE.json
```

Comprueba hashes, identidad V2 en las tres tandas, 84 revisiones RTL, dos
revisiones nativas, estructura/STA y 29 corrupciones rechazadas. El ZIP se
restaura en una carpeta nueva y se repite ese chequeo. Las dependencias para
regenerar el ejemplo AMD se identifican en `DEPENDENCIES.json`; sus archivos
anteriores permanecen en `ntrpu-axi-bridge-native`. No se duplica la instalacion
de Vivado ni la cache compilada de esta corrida. `PROCESS_COMPLETION.json`
registra que terminaron nuestros diez PID de runners y ultimas etapas; no se
detienen los trabajos del hilo principal ni la VM.

## Limites y siguiente integracion

La composicion con productores PCIe y consumidores reales, junto con su
sintesis global y contratos fisicos pendientes, se documenta por separado
en [GUARD_INTEGRATION.md](GUARD_INTEGRATION.md).

V2 permanece como candidato experimental. Faltan restricciones de fuentes
y consumidores reales, colocacion, ruteo, anchos minimos, recovery/removal
de entradas externas y analisis fisico de metastabilidad. La conjuncion de
salida y sus consumidores deben revisarse juntos en el top de integracion.
La parada nativa fuerza la salida publica AXI; no reproduce una averia fisica
de GT/REFCLK ni detiene necesariamente todos los relojes internos de la IP.
No valida un lector DDR/HBM, ACK real entre tarjetas ni aislamiento host
exhaustivo. La evidencia V1 de trafico host no se atribuye automaticamente a V2.
