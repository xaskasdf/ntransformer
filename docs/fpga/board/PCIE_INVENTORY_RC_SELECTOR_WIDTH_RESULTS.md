# Monitor PCIe: selector con prioridad de cinco bits

R9 pasó RTL y el replay de su propia netlist el 2026-10-08. Frente a R8:
**-6396 LUT**, **-1536 CARRY8** y
**-1.210 ns de WNS**. **R8** queda como referencia
de mejor WNS medido. La preferencia no acredita timing físico.

## Cambio comprobado

El selector R4 sólo sustituye el `integer newest` de R3 por
`reg signed [4:0] newest`. Representa −1 o posiciones 0–15. La revisión
compara el texto completo para excluir otros cambios. El productor R9
conserva R8, salvo un comentario. Capacidad, latencia, máscaras,
propiedad y aritmética permanecen iguales.
[Contrato](PCIE_INVENTORY_RC_SELECTOR_WIDTH_CONTRACT.md).

## Validaciones

Se conservaron 1.153.296 comprobaciones heredadas, incluidas 994.288 X/Z,
y 32.768 adicionales, incluidas 32.256 X/Z, comparando 840 bits con las
referencias independientes. También pasaron selector 89.632; RQ
615.536/5.712 X/Z y 8.850/240; RC 45.056/6.624 X/Z y 544/60;
propiedad integrada 46.356/3.872; principal 22.339/68 y siete mutantes.

La propia netlist reproduce 22.330/67 y los tres suplementos binarios
completos, con todas las filas públicas idénticas. El DUT mapeado no
compila RTL del productor, selector, wrapper o validador. Las tablas se
comparan en RTL; ambos modos conservan estados posteriores a fallos.
Arranque binario de 200 ns, sin SDF. Los casos X/Z no son prueba universal.

## Área y timing OOC

Vivado 2026.1, VU5P FLVA2104 -2-i, `RuntimeOptimized`, período de 4 ns,
I/O cero y sin excepciones temporales. Perfil 256 tags/RQ2/RC4.

| Métrica | R7 | R8 | R9 |
| --- | ---: | ---: | ---: |
| LUT | 42362 | 51937 | 45541 |
| FF | 10729 | 10729 | 10729 |
| CARRY8 | 306 | 1890 | 354 |
| WNS, ns | -8.301 | -7.442 | -8.652 |
| TNS, ns | -73934.203 | -72175.352 | -83415.094 |
| Setup: endpoints incumplidos | 15866 | 15866 | 15863 |
| WHS, ns | -0.075 | -0.075 | -0.075 |
| THS, ns | -139.633 | -140.037 | -139.511 |
| Hold: endpoints incumplidos | 1893 | 1992 | 1891 |

Ruta peor: `rc_user[76]` → `producer/remaining_bytes_reg[20]/D`, **12.663 ns/50 niveles**.
La contribución lógica pasa de 2.900 a 2.680 ns y la de
ruteo estimado de 8.553 a 9.983 ns. Son estimaciones previas
a colocación/ruteo; no mediciones del retardo físico de la tarjeta.
DCP acíclico: 66281 primitivas, 73846 nets y ocho
controles del auditor. Las 12 categorías de `check_timing` son cero;
los incumplimientos setup/hold se conservan en la tabla. No hay
colocación/ruteo ni frecuencia física medida.

R9 reduce área respecto de R8, pero R7 usa menos LUT y CARRY8 y tiene
mejor WNS que R9 en estas síntesis. R9 se conserva como experimento
revisado; la reducción de anchura por sí sola no mejora el compromiso
área/timing de las variantes anteriores.

## Coste de ejecución y respaldo

Se mantuvieron los mismos bancos y el inicio XSIM de R8. La unidad
completa empleó 00:00:42 de pared y
43410 ms de CPU de XSIM. El banco RTL
tardó 305.374 s, frente a 305,695 s de R8; la netlist
460.222 s, frente a 614,293 s. Son observaciones
individuales, sujetas a carga y E/S; no una medida aislada de aceleración.

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T164400Z-V4-ffn-supervised-pcie-inventory-join-rc-selector-width-rtl` | 305.374 | `2f9749fba7ddc00b03dadffae3b6d0a0148114bcce48390149f82d90a1ea6294` |
| `20261008T165220Z-V4-ffn-supervised-pcie-inventory-join-rc-selector-width-synth` | 286.324 | `7b8f78e6ee01f5f61e58f471533b56dd80fc1eba91206de51e2b42ea6df8bc42` |
| `20261008T165832Z-V4-ffn-supervised-pcie-inventory-join-rc-selector-width-binary-rtl` | 123.720 | `81f4530da5e61f7b689bb995213be511817020fc7401123e6192e618670a30b5` |
| `20261008T170316Z-V4-ffn-supervised-pcie-inventory-join-rc-selector-width-binary-netlist` | 460.222 | `d46f970852287a2b8142d0af52eb1a8ba90a08effd6acebf0129b9f582a3a5b9` |

Fuentes: `source-20261008T164323Z.zip`, 2262 archivos,
SHA-256 `0aca80e1f0fe5da761ece8f1c4e5b84a983456bfa900da3fd7f78c84f44d7df0`. Paquetes y miembros verificados localmente
y en NFS bajo `artifacts/ntrpu-board-vivado/<tanda>/`. Revisiones funcionales,
de síntesis, DCP y ejecución: `build/cloud-setup/20261002-board/<tanda>-*.json`.
Netlist propia: `710285d79c56b79daccd8c2ba25527979111b507e651076ad928f81d8fb96cb9`.

## Pendientes

Revisar el siguiente cambio desde las métricas, conservando todas las
variantes. Endpoint nativo/BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD siguen abiertos.
La captura nativa heredada sólo contiene tres lecturas de 32/64/32 bytes;
los bancos ampliados son sintéticos. [Cola completa](VIVADO_QUEUE.md).
