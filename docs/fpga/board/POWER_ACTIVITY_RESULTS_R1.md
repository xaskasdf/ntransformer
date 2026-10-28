# Resultado nativo de la captura DDR R1

Fecha: 2026-10-16. **Dos SAIF exportados y revisados; sin nueva estimación
de consumo.** Tanda `20261016T122423Z-V8-ddr-activity-vivado-r3`:
Vivado/XSim 2026.1, dos CPU, 412,01 segundos, código cero y fuentes intactas.
Se ejecutó después de revisar y respaldar BRAM; la cola continuó con Q/K/V.

## Ventanas y tráfico observado

El banco declaró actividad preparada a 35,466498 µs. Las ventanas fueron
36–56,48 y 56,48–76,96 µs: **20,48 µs y 4.096 ciclos UI cada una**.
Los contadores incluyen operaciones pendientes al cruzar cada borde.

| Medición | PRBS 0 | PRBS 1 |
| --- | ---: | ---: |
| AW / B aceptados | 24 / 23 | 6 / 7 |
| AR / RLAST aceptados | 11 / 12 | 7 / 6 |
| Beats de escritura / lectura | 764 / 285 | 1.036 / 1.106 |
| Bytes escritos según WSTRB | 32.075 | 63.658 |
| Bytes leídos | 18.240 | 70.784 |
| Ciclos con espera de escritura | 2.322 | 957 |
| Ciclos con espera de lectura | 0 | 0 |
| Escrituras pendientes, inicio → fin | 0 → 1 | 1 → 0 |
| Lecturas pendientes, inicio → fin | 1 → 0 | 0 → 1 |
| Transiciones SAIF agregadas | 85.711.970 | 132.513.018 |

Ambos SAIF contienen el mismo inventario de **577.280 señales**. Resets,
calibración, modo PRBS y señales de error permanecieron en sus estados
esperados; el reloj UI tiene 8.192 transiciones por ventana. Los cuatro
pines diferenciales CK exportados tienen período de 1.250 ps. El log
identifica 72 modelos x4 de 16 Gb: dos slots, dos ranks y 18 dispositivos
por rank. Esto corresponde al modelo genérico, no a la calificación de un
DIMM comercial ni de la tarjeta de 512 GiB.

La segunda ventana tiene un 54,6 % más de transiciones agregadas y cambia
la actividad de 186.440 señales. Hay 101.498 y 101.008 señales con X/Z/TB,
respectivamente, incluyendo buses en alta impedancia. No se establece
estabilidad estadística. Hay actividad CS/ODT de los cuatro ranks, pero
su ocupación SAIF no prueba cobertura temporal de comandos o refresh.

## Revisión y procedencia

El revisor original quedó conservado como diagnóstico. Confundía dos
líneas Tcl copiadas en el log con errores ejecutados y dividía nombres
SAIF con espacios escapados. `review_ntrpu_ddr_activity_r3.py` reutiliza
el analizador estricto de actividad y reconoce exclusivamente esas dos
formas de eco. Pasó 13 controles negativos en Windows/Linux y revisó la
captura nativa sin errores. No se repitió Vivado.

Los observadores originales se detuvieron por comparar el hash del informe
Linux LF con su copia Windows CRLF. Una revisión separada verificó ambos
hashes, la igualdad exacta tras normalizar sólo CRLF y los 585 archivos;
siete controles rechazan alteraciones de contenido. Los registros originales
se conservan.

La revisión de identidad R3 verificó 132 fuentes: 129 copias archivadas,
startup, banco observador y `glbl.v` de la instalación Vivado 2026.1. Este
último difiere del importado antiguo, incluido GRESTORE y fuerzas de reset;
su equivalencia de comportamiento sigue abierta. La geometría activa viene
de los parámetros explícitos de `ddr4_rdimm_wrapper` en el banco. El wrapper
SDRAM preparado está intacto pero no se compiló: sus tres macros no determinan
esta simulación. Pasaron 31 controles de identidad en Windows/Linux.

## Respaldo y siguientes criterios

Captura completa: **585 archivos restaurados en D: y copia NFS verificada**,
179.178.407 bytes, SHA-256
`203dd7b2de98042c6775b5e3f2a8e9b342d9caeb1f6c90070ae4f06bb93a28ff`.
El directorio es `artifacts/ntrpu-board-vivado/20261016T122423Z-V8-ddr-activity-vivado-r3/`.

| Archivo | SHA-256 |
| --- | --- |
| `prbs_0.saif` | `8cd7532f0e8ac25ca05057d128efdc24691f5cce37cbc83d6c43694342710fee` |
| `prbs_1.saif` | `418687f10e7261d3bf7430326a735656c2b993085c2bcd37cea5f6623c9ebda7` |
| Revisión nativa de actividad | `eb0f6a42fc1516f8f50008a7eca8724001ecc3948ab141397f01fbbf87dba941` |
| Revisión nativa de identidad R3 | `ca72b52bee891a2dd9e0687f140d44e0ad0f604eb10c3d9dc832d35ea1bd3dd3` |

Las revisiones suplementarias, incluida la identidad R2 diagnóstica, tienen
respaldo separado y restauración comprobada. El traspaso consolidado está
en `artifacts/ntrpu-board-vivado/ddr-activity-results-handoff-20261016-r1/`.

Antes de potencia: completar bibliotecas/includes y la variante global,
clasificar X/Z con el netlist, medir comandos por rank y refresh mediante
una nueva captura temporal, y comprobar la asociación SAIF/DCP por ventana
con redes no asociadas y condiciones térmicas explícitas. El
[helper de comandos](POWER_RANK_TRACE_R1.md) está preparado; no se integró
a R1 ni se añadió otra tanda Vivado durante esta revisión.
