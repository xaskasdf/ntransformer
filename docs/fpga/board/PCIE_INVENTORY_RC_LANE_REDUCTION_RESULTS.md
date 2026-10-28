# Monitor PCIe: reducción de bytes RC por carril

R12 pasó RTL y el replay de su propia netlist el 2026-10-08. Frente a R8:
**+297 LUT**, **+12 CARRY8** y
**+0.374 ns de WNS**. **R12** queda como referencia
de mejor WNS medido. La preferencia no acredita timing físico.

## Cambio comprobado

El productor R12 parte de R8 y reemplaza únicamente las contribuciones
RC escritas por slot EOP por valores de 13 bits en sus carriles DWORD.
Un árbol de sumas de 15 bits produce el mismo total. Permanecen las
condiciones procedurales de propiedad, terminal y normal; también RQ,
prefijos, tablas, contadores, latencia, reset y estados posteriores a fallos.
La comparación exacta de fuentes delimita el cambio; selector R3 por hash.
[Contrato](PCIE_INVENTORY_RC_LANE_REDUCTION_CONTRACT.md).

## Validaciones

Se conservaron 1.153.296 comprobaciones heredadas, incluidas 994.288 X/Z,
y 32.768 adicionales, incluidas 32.256 X/Z, comparando 840 bits con las
referencias independientes. También pasaron selector 89.632; RQ
615.536/5.712 X/Z y 8.850/240; RC 45.056/6.624 X/Z y 544/60;
propiedad integrada 46.356/3.872; principal 22.339/68 y siete mutantes.

Se retienen las 2.048 pruebas adicionales de máscaras de R11, incluidas
1.920 con X/Z en la máscara. El helper real de suma pasa **70.464**
comparaciones contra una acumulación secuencial independiente, con
**832 X/Z**. Incluye todos los subconjuntos de 16 términos máximos;
los subconjuntos de más de cuatro términos prueban sólo la aritmética
modular del helper, no representan beats PCIe legales.

El suplemento nuevo pasa **984 filas/92 escenarios**, incluidos 32 con
entradas X/Z: EOP duplicados/desordenados, continuidad retenida, terminal,
tag y conteos desconocidos, y tráfico posterior a fallos. Compara
**251.904 palabras de tabla** en RTL. Sus **632 filas/60 escenarios**
binarios son el prefijo exacto de entradas conocidas y pasan en RTL y
netlist propia. No se concretizan X/Z. Cuatro controles de evidencia de
la suma y seis por modo de integración rechazan resultados alterados.

La propia netlist reproduce 22.330/67 y los cuatro suplementos binarios
completos, con todas las filas públicas idénticas. El DUT mapeado no
compila RTL del productor, selector, wrapper o validador. Las tablas se
comparan en RTL; ambos modos conservan estados posteriores a fallos.
Arranque binario de 200 ns, sin SDF. Los casos X/Z no son prueba universal.

## Área y timing OOC

Vivado 2026.1, VU5P FLVA2104 -2-i, `RuntimeOptimized`, período de 4 ns,
I/O cero y sin excepciones temporales. Perfil 256 tags/RQ2/RC4.

| Métrica | R8 | R9 | R10 | R11 | R12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUT | 51937 | 45541 | 53671 | 56863 | 52234 |
| FF | 10729 | 10729 | 10729 | 10729 | 10729 |
| CARRY8 | 1890 | 354 | 1890 | 2168 | 1902 |
| WNS, ns | -7.442 | -8.652 | -7.789 | -8.414 | -7.068 |
| TNS, ns | -72175.352 | -83415.094 | -85617.805 | -72725.289 | -70828.984 |
| Setup: endpoints incumplidos | 15866 | 15863 | 15872 | 15849 | 15867 |
| WHS, ns | -0.075 | -0.075 | -0.075 | -0.075 | -0.075 |
| THS, ns | -140.037 | -139.511 | -139.541 | -139.621 | -140.893 |
| Hold: endpoints incumplidos | 1992 | 1891 | 1894 | 1892 | 2035 |

Ruta peor: `rc_user[76]` → `producer/remaining_bytes_reg[20]/D`, **11.079 ns/43 niveles**.
La contribución lógica pasa de 2.900 a 3.058 ns y la de
ruteo estimado de 8.553 a 8.021 ns. Son estimaciones previas
a colocación/ruteo; no mediciones del retardo físico de la tarjeta.
DCP acíclico: 74523 primitivas, 92994 nets y ocho
controles del auditor. Las 12 categorías de `check_timing` son cero;
los incumplimientos setup/hold se conservan en la tabla. No hay
colocación/ruteo ni frecuencia física medida.

Se conservan R7, R8, R9, R10 y R11 con sus diferencias de área y timing. La
preferencia sólo compara WNS entre candidatos funcionalmente revisados;
no afirma cierre físico ni que una sola métrica resuma todos los costes.

## Coste de ejecución y respaldo

Se mantuvieron los bancos heredados, las 2.048 pruebas de máscaras y
el inicio XSIM de R8. Se añaden la suma y la integración por carril:
17 inicios XSIM en RTL y cinco en cada modo binario. La unidad de ascendencia
completa empleó 00:00:41 de pared y
42520 ms de CPU de XSIM. El banco RTL
tardó 337.209 s, frente a 305,695 s de R8; la netlist
685.270 s, frente a 614,293 s. Son observaciones
individuales, sujetas a carga, E/S y al banco ampliado; no una medida
aislada de aceleración.

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T185736Z-V4-ffn-supervised-pcie-inventory-join-rc-lane-reduction-rtl` | 337.209 | `3e3d6f4c00bb341020a0f8981440b94a52e2e085ac0b42c3206d59b5d630fb54` |
| `20261008T190613Z-V4-ffn-supervised-pcie-inventory-join-rc-lane-reduction-synth` | 298.344 | `08aa059cb2dbd7753fed8d65fc58fba203446d015a219e8b696742bef8e49428` |
| `20261008T191320Z-V4-ffn-supervised-pcie-inventory-join-rc-lane-reduction-binary-rtl` | 139.501 | `6b24214adb94323333a6ba4ac75d2c7a94029f5c7b2da3f5849fcfba0bf53610` |
| `20261008T191748Z-V4-ffn-supervised-pcie-inventory-join-rc-lane-reduction-binary-netlist` | 685.270 | `6b598b5d416436fba190c6efe4d71e6bb8a1423954b936707c0ea0209f368c48` |

Fuentes: `source-20261008T185651Z.zip`, 2296 archivos,
SHA-256 `6b838f778081a3d14e6f9f53d0a5f4415a942cab9745a7e35f85e9d8adec67c0`. Paquetes y miembros verificados localmente
y en NFS bajo `artifacts/ntrpu-board-vivado/<tanda>/`. Revisiones funcionales,
de síntesis, DCP y ejecución: `build/cloud-setup/20261002-board/<tanda>-*.json`.
Netlist propia: `05f3855f6fd1b261ba02f3adbe12e01d765b04f429b870bd40e13bfb745b2733`.

## Pendientes

Revisar el siguiente cambio desde las métricas, conservando todas las
variantes. Endpoint nativo/BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD siguen abiertos.
La captura nativa heredada sólo contiene tres lecturas de 32/64/32 bytes;
los bancos ampliados son sintéticos. [Cola completa](VIVADO_QUEUE.md).
