# Síntesis del host retenido y supervisor

Revisión del 2026-10-06. Son resultados de **síntesis fuera de contexto**
para `xcvu5p-flva2104-2-i`, Vivado 2026.1, con dos hilos. No califican
colocación, ruteo, interfaces externas, XDMA ni la placa de 512 GiB.

Actualización: [dos perfiles de netlist y ensayo de restricciones específicas](RETAINED_RESET_TIMING_RESULTS.md)
ya revisados; `TIMING-6/7` desaparecen en ese ensayo. Las observaciones
originales de este documento se conservan. [Diagnóstico y revisión R2 del banco DATA](RETAINED_DATA_PENDING.md).

## Recursos y timing observado

| Bloque | LUT | LUTRAM incluidas | FF | WNS | WHS | WPWS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Host AXI retenido | 7.899 | 640 | 14.848 | 1,465 ns | 0,012 ns | 1,968 ns |
| Supervisor de dos memorias | 102 | 0 | 149 | 4,092 ns | 0,077 ns | 2,225 ns |

Ninguno usa BRAM, URAM o DSP; no hay cajas negras. Se compararon los
reportes archivados con el inventario de celdas y las fuentes congeladas:
28 archivos RTL para el host y dos para el supervisor. Los conteos FF
coinciden con las primitivas del netlist.

Los clocks OOC son host 7 ns, fuente 5 ns y gestión 10 ns; el supervisor
añade dos clocks UI de 5 ns. No hay endpoints internos sin restricción,
registros sin reloj ni lazos combinacionales. **El WNS positivo del
supervisor no resuelve su diagnóstico entre relojes independientes.**
Faltan retardos externos en 1.373 entradas/3.041 salidas del host y
11/11 del supervisor. Tampoco se ha fijado `HD.CLK_SRC`.

## Restricciones efectivas del host

La auditoría abrió el DCP original, sin cambiar restricciones ni guardarlo
de nuevo. El hash del DCP se verificó antes y después. DRC de síntesis:
cero comprobaciones fallidas; no hay waivers efectivos. CDC informa
67 cruces `CDC-3` y diez `CDC-9`, todos informativos. Parte de los cruces
pertenece a lógica protegida del convertidor y tiene profundidades
distintas: no se afirma que todos sean de tres etapas.

Las 13 restricciones de bus skew presentan margen positivo, mínimo
4,851 ns. La metodología informa `TIMING-18` por los límites OOC; sus
1.000 entradas de reporte no sustituyen el conteo completo de `check_timing`.

Se consultaron las RAM de los monitores propios, fuera de las dos IP
proveedoras. De 152 primitivas, 93 tienen rutas restringidas desde CLK,
atravesando O, con margen mínimo 2,556 ns. Para otras 59, el recorrido
nativo termina en el mismo pin O, sin fanout posterior; no se presentan
como rutas temporizadas. Diez contenedores de macro no exponen CLK/O y
se registran aparte. Esto comprueba rutas efectivas de los monitores;
no prueba todas las rutas físicas ni las colas protegidas.

El primer auditor no distinguía esos contenedores ni los destinos de
las 59 consultas. Se conserva como diagnóstico. La revisión R2 registra
los objetos exactos y evita consultas con selectores vacíos.

## Pendientes del supervisor

El DCP confirma `opened_reg` como FDRE, INIT=0, reloj de gestión y R
conectado a GND. `hold_reset`, `clear_counters` y `block_admission`
son FDSE con INIT=1. La entrada de reset pasa por tres FDPE con
`ASYNC_REG`; no resetea asíncronamente el registro de primera apertura.

La metodología detecta seis `TIMING-6` y seis `TIMING-7`: los cruces
gestión/fuente y gestión/UI aún se temporizan como relojes relacionados.
Debe inventariarse cada ruta y revisar las restricciones específicas de
las cadenas `ntrpu_reset_release`, manteniendo su INIT=0. No se añadieron
grupos asíncronos generales ni se sustituyó el circuito por una XPM con
distinta semántica de arranque.

También permanece sin waiver `CDC-26` desde `external_reset_async` hacia
el primer PRE de `management_reset`. El texto menciona LUTRAM aunque el
netlist no tiene RAM. La discrepancia está documentada, no descartada.

`tools/ntrpu_board_retained_start_followup.tcl` prepara dos trabajos:
inventario de rutas entre clocks (`timing`) y simulación del netlist con
el banco original de 13 casos (`5000`/`37000`). **Todavía no ejecutados.**
La simulación no incorpora el RTL original del supervisor. Sus resultados
funcionales tampoco sustituirán la revisión de metastabilidad y timing.

## Evidencias inmutables

Todos los IDs tienen archivo local y NFS con manifiesto verificado.
Los revisores independientes se guardan en el checkpoint de continuación.

| Tanda | ID abreviado (prefijo `20261006T`) | SHA-256 del archivo |
| --- | --- | --- |
| Host sintetizado | `131357Z-V4-ffn-supervised-retained-host-synth-r2-frozen` | `01876682c6621eeaee0dd5ad02697b4dda29af76ffd3e1be46789ca27e0ac394` |
| Supervisor sintetizado | `132220Z-V4-ffn-supervised-retained-start-synth-frozen` | `29e93710239a757292f5ce392b65a1be04c823f9497080af45c107f320e77667` |
| Auditoría host inicial | `133002Z-V4-ffn-supervised-retained-host-constraints-frozen` | `695fb813d832b5efe8a1556e819537342e3d3a31e6b7d37d2d7d853a768f57ad` |
| Auditoría supervisor | `133215Z-V4-ffn-supervised-retained-start-constraints-frozen` | `f55e440a00d1488c2020d1e9b52ca43b8bbaf5bf703d3b7112c920059eb0b93b` |
| Auditoría host R2 | `133436Z-V4-ffn-supervised-retained-host-constraints-r2-frozen` | `8f95fe80f9350c7baef3369363decc4bc5b23eea4f12e4d27ad0595d55222eae` |

El primer banco DATA pendiente quedó conservado como diagnóstico por
una espera del testbench sobre un reloj detenido. La revisión R2 está
en ejecución y conserva el RTL de los bancos anteriores. Véase su
[estado y evidencia](RETAINED_DATA_PENDING.md). La calificación física
del supervisor permanece abierta.
