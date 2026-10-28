# Frontera sintetizada de observación XDMA

## Resultado revisado del 2026-10-07

La tanda `20261007T151433Z-V4-ffn-supervised-xdma-observation-boundary-synth`
terminó en **708,783 s**. Sintetizó únicamente XDMA en una copia del ejemplo
nativo, con Vivado 2026.1 y el perfil congelado. No conectó el monitor.

Los **756 archivos originales** permanecen idénticos. La revisión compara
153 fuentes/configuraciones de la copia contra el original, liga los 122
archivos del orden de compilación al archivo respaldado y verifica que
el script real lee el XCI de la copia. Exportó 136.537 celdas, sin cajas
negras, y 117.926 filas de pines de las jerarquías seleccionadas.

## Señales observadas

Frontera: `inst/pcie4_ip_i`. Los anchos se comprueban por cada bit y dirección.

| Puerto | Ancho | Conexión en el nivel superior |
| --- | ---: | --- |
| `s_axis_rq_tdata` / `s_axis_rq_tuser` | 512 / 137 | Todos los bits |
| `s_axis_rq_tvalid` | 1 | Presente |
| `s_axis_rq_tready` | 4 | Solo bit 0 |
| `m_axis_rc_tdata` / `m_axis_rc_tuser` | 512 / 161 | Todos los bits |
| `m_axis_rc_tvalid` / `m_axis_rc_tready` | 1 / 1 | Presentes |
| `user_clk` / `user_reset` / `user_lnk_up` | 1 cada uno | Presentes |
| `cfg_bus_number` | 8 | Sin nets en esta consulta |

La ausencia de net en el nivel superior **no prueba** que falte un driver
interno. En otra copia del DCP, inspeccionar segmentos y drivers antes de
conectar cargas pasivas. Preservar las conexiones de transporte, comparar
sus extremos y verificar CDC/timing. No reconstruir READY ni fijar el
Requester ID a `0x01a0` sin una justificación verificable para producción.

La inspección `20261007T161209Z` terminó en **111,124 s** sobre una copia
idéntica. Revisó 1.357 bits y 152.749 filas de extremos, con las 136.537
celdas y conexiones superiores anteriores intactas. Rechazó cinco
corrupciones de evidencia. Hay un driver único por bit, pero **128 bits
están ligados a constantes**: tener driver no demuestra utilidad funcional.
`cfg_bus_number[7:0]`, READY[1:3], RC TKEEP y RC TLAST provienen de GND
en esta frontera sintetizada. READY[0] proviene de un FDRE activo.
No conectar el monitor a esos ceros como sustituto de las señales requeridas.

UG835 documenta que la consulta predeterminada usa el lado superior;
`-boundary_type lower/both` y `-segments` permiten inspeccionar la jerarquía.
[Referencia de get_nets](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_nets).

## Candidato en la primitiva PCIe

La tanda `20261007T162248Z` reabrió el mismo DCP en **74,701 s** y enumeró
los **7.457 pines** de su única `PCIE40E4`. Sus ocho `CFGBUSNUMBER` tienen
nets propias y driver en la primitiva; sus únicos extremos observados son
los propios pines de salida. Esto ofrece un candidato para un ensayo de
carga pasiva en otra copia, sin usar el `cfg_bus_number` superior ligado a cero.
En esa tanda todavía no se añadieron cargas ni se probó una captura funcional.

PG213 describe el número de bus capturado de una escritura de configuración
Type 0; eso no basta para establecer el Requester ID completo.
[Control de configuración](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Configuration-Control-Interface).
`CFGEXTFUNCTIONNUMBER` corresponde a una transacción de configuración y
solo es válido con su indicación asociada: no es un reemplazo del BDF.
[Interfaz extendida](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Configuration-Extend-Interface).
PG213 permite usar cualquier bit de READY porque representan el mismo valor;
una futura normalización desde READY[0] debe ligarse al perfil y contrastarse
con la captura nativa antes de admitirse.
[Interfaz RQ](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Request-Interface).

## Conexión y reapertura revisadas

El ensayo `20261007T164306Z` terminó en **321,811 s** en una copia del DCP.
Añadió ocho puertos lógicos de salida a los nets reales mediante
`connect_net -hierarchical`. La revisión verifica las **136.537 celdas** y
los **152.749 extremos de 1.357 bits** antes/después y contra la evidencia
anterior; no se añadieron celdas primitivas ni drivers de transporte.
Los ocho puertos aparecen como salidas en el netlist exportado. Se
rechazaron seis corrupciones de evidencia. Es viabilidad estructural,
sin muestreo, monitor completo ni Requester ID completo.
Configuración congelada:
`1d8010b72c381c2826982c6d1347c319c50da42576f572e2ca7b295516a8f90f`.

La tanda `20261007T165658Z` reabrió el DCP exportado idéntico en
**116,896 s** y confirmó las ocho conexiones, sus direcciones de salida,
las celdas y todos los extremos de transporte. Enumeró **5.816 bits de
puertos superiores** y rechazó otras seis corrupciones. No modificó
conexiones ni ejecutó una simulación funcional. DCP persistido:
`3e31c9bc18c5b201b2392cadcfc860100cbb2a0b438617bf06d539f610d2eb28`.

El siguiente paso requiere cargas de muestreo y contraste funcional con
el endpoint nativo, además de procedencia verificable de Device/Function.
No fijar un BDF de laboratorio ni usar datos de la propia RC como identidad
esperada. La integración del monitor depende también de su cierre temporal.

## Límites y respaldo

El resumen temporal del ensayo coincide con el DCP inicial: WNS
**+0,644 ns**, WHS **−0,758 ns** y WPWS **−0,008 ns**. Hold y ancho de pulso
fallan; 1.213 entradas y 1.679 salidas carecen de retardos I/O. No confundir
esas cifras de XDMA OOC con el timing del monitor R7.

CDC reporta 161 `CDC-3`, 23 `CDC-9`, **288 `CDC-15` y un `CDC-26`**;
omite puertos sin input delay y no existe informe CDC inicial para comparar.
El DRC OOC conserva `NSTD-1` y `UCIO-1`, reportando 53 puertos sin estándar
ni ubicación; ese subconjunto no es el inventario completo de puertos.
Vivado advierte que no ejecuta todos los controles de conectividad OOC.
Resolverlos en la integración del top; no rebajar severidades. Los logs
de simulación copiados son evidencia histórica, no tráfico nativo nuevo.
No hay aprobación funcional del tap, reset ni placa.

Archivo D:/NFS: **253.554.483 bytes**, SHA-256
`21a03212875cb16b4d1364c5bc86c9c49acec3ae933d931545962666c6361683`.
Revisión independiente:
`16103222856079d609973fe9e253b5d706c9547fe726d81461e6bc79efd2ce45`.
DCP: `bce78d71c97604c1883aa0fc316df29020931dff9f27d186c376d7762c7df567`.

| Inspección | Archivo D:/NFS SHA-256 | Revisión SHA-256 |
| --- | --- | --- |
| Segmentos, 25.499.560 bytes | `215b7b9a37c1c7b50239b3e2a5bfa54583bd83220567359f15708316b5f30a93` | `624aef19ed9d7494c5366bc8cc29e0b2a4e1debeed3ae3a49791e7a407098192` |
| Primitiva, 22.162.038 bytes | `a2db5b55e5d8f6b95e67411b446654b9a2f2bcf049c5f593666236ca6f7cbeec` | `838c95703f2a159989dcf949ed34b9cb6682d47bfb05edb008f58fee7be2f967` |
| Ocho salidas, 81.306.775 bytes | `6d47866bdd31911b9d26c09b72897d123f8831ab259250eb26f8e2ad7fe64cbd` | `1a0e9c8d50e1c5bb723fc111e587de8f2a5c2f4a85c138b25137d6aa49fb3658` |
| Reapertura, 24.523.019 bytes | `b5ebc000c4da9752fda4d2d2cb05eab0b71ecb97573e541d4f1ee8d20b397b42` | `f9cf07312d4390911e32af3e16df4e627d9fbfd41253f36d5aacec8d3a27066a` |

Siguen pendientes fuente/host, siete contratos coherentes, reset nativo
de 50 ms y los criterios de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
