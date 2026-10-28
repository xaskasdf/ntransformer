# Integracion del supervisor con el puente nativo

Tanda privada del 2026-10-11. El objetivo es revisar el circuito que
forma V2 con sus productores y consumidores reales. La validacion anterior
del supervisor aislado sigue en [GUARD_V2.md](GUARD_V2.md).

## Composicion sintetizable

`ntrpu_axi_bridge_guarded_client.sv` instancia V2 y el cliente existente.
Conserva escritor, mapper y maestro AXI-Lite sin cambios. El perfil nativo
usa direcciones AXI de 64 bits, datos de 512 bits, una apertura de 4 KiB en
`0x10000000` y origen local de hasta 39 bits. No incorpora un lector DDR/HBM.

`prepare_guard_integration.py` adapta una copia del ejemplo AMD ya probado.
Sustituye exactamente las 35 referencias al banco de simulacion por cables
internos al cliente. Mantiene la BRAM del ejemplo como destino del trafico
host y deja el control AXI-Lite exclusivamente conectado al mapper. Expone
los puertos de configuracion, comandos, fuente y ACK para integracion futura.
No son pines de PCB asignados ni perifericos ya implementados.

El nuevo top recibe de la IP AMD `axi_aclk`, `user_lnk_up`, `axi_aresetn` y
`axi_ctl_aresetn`. No inserta otro buffer en el reloj AXI. PERSTn llega por
el IBUF del ejemplo. La configuracion fria permanece independiente de PERSTn.
La gestion usa otro reloj, `mgmt_clk_in`, con BUFGCE habilitado permanentemente;
su fuente fisica y pin de placa siguen pendientes.

## Contratos que faltan en el padre

| Frontera | Dominio y obligacion |
| --- | --- |
| `cfg_*`, `cmd_*`, `src_*`, `ack_*`, resultados y rechazos | Productores/consumidores sincronicos al AXI nativo; falta presupuesto temporal y adaptacion desde otros relojes |
| Estado del cliente, mapper y ownership | AXI; sincronizar antes de observarlo desde otro dominio |
| `guard_clock_alive`, `guard_armed`, `guard_fault*` | Gestion; definir su consumidor y cruce si no comparte reloj |
| Reset de configuracion | Asercion externa controlada; liberacion mediante las cadenas de V2; no usarlo para descartar una operacion pendiente |
| `mgmt_clk_in` | Independiente de PCIe; objetivo nominal de 100 MHz, pendiente oscilador/pin/jitter |
| Referencia PCIe | Circuito IBUFDS_GTE4 y restricciones internas del proveedor; pendiente validacion electrica de placa |

No se usan los pines ni las excepciones generales del XDC de placa del
ejemplo. Las restricciones internas generadas para ambas IP se conservan.
No se agregan retardos ficticios de E/S para obtener un reporte limpio.

## Pruebas y flujo

La composicion guardada paso **44 escenarios locales**: once casos con
cuatro semillas del banco de mapping anterior. Incluye remap sin reset,
192 bytes por semilla, errores B/R, readback, timeout con respuesta tardia,
perdida de enlace y apertura incorrecta. Se conserva el revisor anterior.
Estos casos usan Icarus y un destino de registros modelado; no ejecutan PCIe.

El proyecto privado de Vivado importa ambas XCI desde la semilla congelada,
genera sus productos y sintetiza conjuntamente la IP, V2 y el cliente.
La sintesis global se selecciona mediante
[`GENERATE_SYNTH_CHECKPOINT=false`](https://docs.amd.com/r/2024.1-English/ug912-vivado-properties/GENERATE_SYNTH_CHECKPOINT).
Se guardan fuentes, XCI, DCP, netlist, excepciones, inventario de frontera,
CDC y caminos desde el supervisor hasta el cliente. El reporte CDC acotado
incluye cruces con cualquiera de sus extremos en la composicion, segun
[UG906](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Report-Clock-Domain-Crossings).

```bash
python3 -B run_guard_integration.py NUEVA_SALIDA \
  --example EJEMPLO_NATIVO_R3/output/example \
  --settings VIVADO_SETTINGS --execute-native
```

El runner requiere una carpeta nueva bajo `ntrpu-side`, dos CPU, nice 10 y
un limite de 1800 s para su proceso Vivado. Comprueba que las entradas y las
cinco fuentes de la semilla no cambiaron. Conserva tambien los fallos y
exporta un archivo con manifiesto SHA-256. No opera sobre la cola principal.

## Sintesis nativa y limite fisico

La decision posterior de priorizar el nodo VU47P/HBM y el candidato PCIe
de implementacion se registran en [GUARD_PHYSICAL.md](GUARD_PHYSICAL.md).
Los resultados de esta seccion siguen correspondiendo a la sintesis original.

La sintesis conjunta termino en 618 s, con dos CPU y nice 10. El netlist
incluye PCIe/GT, la BRAM del ejemplo y el cliente protegido; no quedan cajas
negras. El AXI real proviene de `bufg_gt_userclk` (BUFG_GT, 250 MHz); gestion
proviene de `management_buffer` (BUFGCE, contrato de 100 MHz).

El perfil completo ocupa 46.322 LUT, 46.543 FF y 88 RAMB36. La composicion
guard/cliente ocupa 1.695 LUT y 1.531 FF; el guard conserva 42 ASYNC_REG en
14 cadenas y 57 FF totales. Son recursos de este ejemplo PCIe, no del motor
de inferencia ni de una memoria de 512 GiB.

La revision CDC acotada al cliente protegido identifica tres CDC-3 y ocho
CDC-9 informativos. No cubre las entradas externas sin input delay ni
acredita todos los cruces internos de la IP AMD. El proyecto completo conserva
un TIMING-9 global pendiente de localizar y seis LUTAR-1 identificados dentro
de la IP. No se agregaron waivers.

La estimacion global de sintesis tiene WNS +0,531 ns, WHS -0,842 ns y WPWS
-0,092 ns: **timing no cerrado**. Existen 1.589 bits de entrada y 511 de
salida sin retardos de interfaz. Son fronteras de fabric del bloque OOC,
no miles de pines de PCB. `check_timing` no encontro registros sin reloj,
relojes multiples, lazos ni endpoints internos sin restriccion maxima.

No se ejecuta placement/routing con pines supuestos. DRC identifica
`refclk_ibuf` sin LOC (AVAL-326), PERSTn sin IOSTANDARD explicito (NSTD-1) y referencia
PCIe/PERSTn sin ubicacion (UCIO-1). Antes de implementarlo en la tarjeta:

1. Aprobar FPGA/package, lanes GT, referencia diferencial, PERSTn y voltaje
   del banco, junto con oscilador y pin independientes para gestion.
2. Integrar los productores/consumidores de fabric o definir presupuestos
   temporales verificables; sincronizar los estados que cambien de dominio.
3. Reabrir el DCP auditado, aplicar ese contrato, ejecutar opt/place/route y
   revisar setup, hold, pulsos, DRC y MTBF con colocacion real. No bajar la
   severidad de DRC ni cortar clocks completos para superar este paso.

## Auditoria de las fronteras reales

`audit_guard_integration.tcl` reabre `native_scoped.dcp`, comprueba el origen
de cada camino y agrega nueve excepciones, exclusivamente desde el pin C
de `user_lnk_up_reg` (FDCE) o `user_reset_reg` (FDPE) a los tres CLR de cada
cadena de entrada correspondiente. No corta PERSTn externo ni el reset frio.
Ambos resets AXI proceden de `user_reset_reg` a traves de inversores LUT1;
se conservan sus dos cadenas independientes.

Los nueve caminos aparecen temporizados antes y explicitamente ignorados
despues. Desaparecen TIMING-6/7 entre gestion y AXI. Permanecen las tres
max-delay del primer D y las quince excepciones internas anteriores; no se
agregan grupos de reloj asincronos generales. Las 56 comprobaciones setup/hold
de los 28 enlaces entre etapas conservan exactamente sus resultados:
setup minimo +3,562 ns, hold minimo -0,060 ns, todavia sin colocacion.
El peor hold informado guard-cliente es -0,040 ns.

`control_consumers.tsv` traza desde el driver primitivo hasta los pines
secuenciales del cliente: 3.522 para reset frio, 1.844 para link y 1.844 para
stop, todos en AXI. Son pines de control/datos alcanzables, no FF unicos.
El inventario inicial `boundary.tsv` parte del pin de jerarquia y sus tres
salidas diagnosticas solo muestran el puerto externo; el inventario del
auditado corrige esa limitacion de observacion.

La auditoria tarda 159 s y conserva el checkpoint original sin cambios.
El revisor offline verifica hashes, origen del checkpoint, CDC, excepciones,
etapas, consumidores y las 44 trazas locales; rechaza doce corrupciones de
evidencia. Distingue caminos no temporizados usando
[`-no_report_unconstrained` y `-user_ignored`](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_timing_paths).

```bash
python3 -B run_guard_integration_audit.py NUEVA_SALIDA \
  --checkpoint TANDA_NATIVA/output/native_scoped.dcp \
  --settings VIVADO_SETTINGS --execute-native
python3 -B review_guard_integration.py NATIVA_RESTAURADA AUDITORIA_RESTAURADA \
  MAPPING_RESTAURADO
```

## Evidencia local

Directorio local: `D:/AI/artifacts/ntrpu-axi-bridge-guard-integration/`.
La tanda `composed-mapping-20261011T064723Z` contiene las 44 pruebas locales.
`native-20261011T064613Z` contiene la sintesis conjunta de la VM, restaurada
y verificada: 252 archivos, archivo de 82.981.231 bytes, SHA-256
`3e2230088c7f6ff3a39c37aba487621185189ffbee8479d566fc3ffc3a9b5ecd`.
Se conserva la semilla, productos de ambas IP, netlist, DCP e informes.
`audit-20261011T070351Z` contiene la auditoria final: 25 archivos, archivo de
18.300.384 bytes, SHA-256
`9c9a6fb8ba0f3ac3677855b6ac70f54927b88110751f3050f5c5ade9266c3310`.
Se conservan dos intentos anteriores fallidos del auditor: expectativas
incorrectas sobre el pin devuelto por all_fanin y sobre caminos ignorados
devueltos por get_timing_paths. No fueron fallos RTL ni cuentan como aprobados.

`20261011-guard-integration-bundle.zip` agrupa fuentes, las tres tandas,
ambos intentos fallidos, DCP, productos de IP y recibos. El manifiesto raiz
incluye todos los archivos; el recibo externo registra SHA-256 del ZIP y
la verificacion despues de restaurarlo en una carpeta nueva. Repetir sin VM:

```bash
python3 -B BUNDLE_RESTAURADO/source/review_guard_integration.py \
  --bundle BUNDLE_RESTAURADO
```

`PROCESS_COMPLETION.json` confirma que finalizaron nuestros ocho PID. No se
detuvieron procesos del hilo principal ni la instancia.

Para repetir las 44 pruebas, usar las fuentes congeladas en
`composed-mapping-20261011T064723Z/source` y ejecutar desde su directorio padre:

```bash
iverilog -g2012 -Wall -s tb_ntrpu_axi_bridge_mapping -o mapping.vvp \
  source/bound_mapping_tb.sv source/ntrpu_axi_bridge_guarded_client.sv \
  source/ntrpu_axi_bridge_guard_v2.sv source/ntrpu_axi_bridge_endpoint_client.sv \
  source/ntrpu_axi_bridge_writer.sv source/ntrpu_axi_bridge_mapper.sv \
  source/ntrpu_xdma_local_master.sv
# Ejecutar cada semilla desde una carpeta nueva para conservar su VCD.
vvp ../mapping.vvp +seed=0
```

Repetir con semillas 1, 2 y 3 y revisar los logs con el revisor congelado.
El banco local usa AXI_ADDR_BITS=48; la sintesis nativa usa 64. Esta tanda
no repite la simulacion PCIe nativa anterior ni acredita aislamiento host
exhaustivo, inferencia integrada o una tarjeta de 512 GiB.
