# R12 con reloj interno: implementación y replay

Revisado el 2026-10-08. El netlist ruteado con Quick pasó los cinco bancos
binarios, con **78.712 filas y 30 controles de evidencia**. La lectura del
DCP conservó circuito, restricciones, rutas y timing. **Setup y hold fallan**;
este resultado no califica 250 MHz ni una FPGA de placa.

## Intentos y trazabilidad

El primer intento falló antes de colocar por una consulta de objetos Tcl.
R1 corrigió solo esa consulta: colocó el diseño y agotó 3.600 s durante
corrección de hold. Se conservan el código 124 y el checkpoint colocado;
el resumen intermedio del router no es STA final.

Quick abrió ese checkpoint sin repetir síntesis, optimización ni colocación.
Conservó 4 ns, E/S cero y ausencia de excepciones o `HD.CLK_SRC` supuesto.
Es un diagnóstico sin optimización dirigida por timing. El DCP exportado
tiene SHA-256 `79c90a2e9ff64a712d482823d2766681829d1b9144d845bac8d2b87ebc72c8b4`.
La lectura posterior es independiente y no modifica el diseño.

## Timing y recursos

| Estado | WNS ns | TNS ns | WHS ns | THS ns |
| --- | ---: | ---: | ---: | ---: |
| Colocado, estimado | −6,192 | −80.700,734 | −3,066 | −61.096,625 |
| Quick final, guardado y reabierto | −17,680 | −166.119,219 | −3,043 | −52.900,992 |

Fallan 16.675/29.696 endpoints de setup y 27.032/29.696 de hold.
El peor setup interno termina en `remaining_reg[228][12]/CE` desde
`allocated_tags_reg[219]/C`: 21,640 ns, 25 niveles, 87,546 % de rutas.
El hold registro→registro también falla: −0,035 ns. No atribuir los fallos
solo a los puertos OOC. Recursos: 51.828 LUT, 10.733 FF y 1.895 CARRY8;
sin BRAM, URAM, DSP ni latches.

## Circuito, reloj y frontera

El EDIF reexportado coincide salvo timestamp: 73.363 primitivas, sin ciclos
combinacionales detectados; pasan ocho controles heredados y trece de reloj.
Respecto al DCP optimizado, se añaden diez FDRE llamados
`producer/observing_reg_replica*`; no hay celdas eliminadas ni cambios de
tipo. Ese inventario no equivale por sí solo a una prueba funcional general.

BUFGCE en `BUFGCE_X1Y23`, CE desde VCC, salida ruteada a 10.733 pines C.
Vivado informa `ROUTED_FULLY=1`, cero gaps y cero errores de rutas;
hay 40.676 objetos ROUTED, 3.124 HIERPORT y 34.017 INTRASITE.
La entrada del buffer sigue como HIERPORT sin ruta física; los 1.830 puertos
carecen de ubicaciones de partición. DRC informa cero violaciones con
comprobaciones OOC incompletas. Se conservan avisos de frontera, origen
externo de reloj, jerarquía y los conteos de metodología
TIMING-15=1.000, TIMING-16=1.000 y XDCH-2=1.829.

El reloj de XDMA procede de un BUFG_GT, confirmado en dos archivos
respaldados: [origen nativo](PCIE_INVENTORY_NATIVE_CLOCK_ORIGIN.md).
La ubicación del BUFGCE diagnóstico no define restricciones del reloj padre.

## Replay exacto y respaldo

Netlist: `83afdbe26128f8eda588900257edcdfe771cda245b3d9eaa39ac8cb47d306429`.
Principal 22.330 filas, ownership 46.356, RQ 8.850, prefix 544 y lanes 632.
Coinciden todas las filas públicas, cancelaciones y estados posteriores a
fallos con R12 RTL. DUT exclusivamente netlist, referencia independiente R7,
arranque 200 ns, muestreo 1 ns, sin SDF. Los 30 controles rechazan corrupción.
Duración observada **608.553 s**;
WDB **51134361 bytes**. No es rendimiento de hardware.

Se verificó la fuente instalada BUFGCE antes/después y el uso de
`unisims_ver`. Permanecen la discrepancia documentada de `SIM_DEVICE` y
la identidad no demostrada entre fuente y biblioteca precompilada.

| Tanda | ID | SHA-256 del archivo |
| --- | --- | --- |
| initial_port_failure | `20261008T213458Z-V4-ffn-supervised-pcie-inventory-clock-placement` | `c5d794d1fcac71a220f3a2c02a9a1a68110780f04c0c5bb4a33bee081dd35fe6` |
| default_timeout | `20261008T213940Z-V4-ffn-supervised-pcie-inventory-clock-placement-r1` | `980131e4dc07c74b998949ba3c3627bf5618fcf883b5bc1f7741b5684b56d169` |
| quick_route | `20261008T224646Z-V4-ffn-supervised-pcie-inventory-clock-quick-route` | `d08113a2a9a42a0696a3f709e6a5b1bb0a6340c91e0b9027d9582b014b93316b` |
| readback | `20261008T225600Z-V4-ffn-supervised-pcie-inventory-clock-route-audit` | `5a6e63f96502f5d4e3016d77783eaba9a2453a2283973187225506688808b9a5` |
| binary_netlist | `20261008T230251Z-V4-ffn-supervised-pcie-inventory-clock-quick-route-binary-netlist` | `01aea62db5b5956cdb1264543d151a767c9bf6b3b7ca5803ff44d489376f2781` |

Cada archivo permanece en NFS y en
`D:/AI/ntransformer/artifacts/ntrpu-board-vivado/backups/`, con verificación
por miembros y hashes. R101 enlaza revisiones, análisis y checkpoint;
los fallos y el R100 anterior permanecen intactos.

## Siguiente trabajo

Preparar el [ensayo de dependencia de tags RC](PCIE_INVENTORY_CLOCK_QUICK_ROUTE_FOLLOWUP.md)
antes de otra implementación larga. La segmentación completa sigue pendiente.
Continúan abiertos el endpoint nativo, reset/CDC, DIMM exacto, cuatro canales
DDR/512 GiB, inferencia integrada, restricciones eléctricas y ECAD.
