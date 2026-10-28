# Monitor PCIe: destinos constantes de la máscara RC

R10 pasó RTL y el replay de su propia netlist el 2026-10-08. Frente a R8:
**+1734 LUT**, **+0 CARRY8** y
**-0.347 ns de WNS**. **R8** queda como referencia
de mejor WNS medido. La preferencia no acredita timing físico.

## Cambio comprobado

El selector R5 reemplaza una escritura de bit con índice dinámico por
destinos literales de headers anteriores: bit 4, bits 8–9 y bits 12–14.
La revisión compara el texto completo contra el selector R3 para excluir
otros cambios. El productor R10 conserva R8, salvo un comentario.
`newest` sigue siendo un `integer` de 32 bits. Capacidad, latencia,
propiedad, RQ y aritmética permanecen iguales.
[Contrato](PCIE_INVENTORY_RC_STATIC_MASK_CONTRACT.md).

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

| Métrica | R7 | R8 | R9 | R10 |
| --- | ---: | ---: | ---: | ---: |
| LUT | 42362 | 51937 | 45541 | 53671 |
| FF | 10729 | 10729 | 10729 | 10729 |
| CARRY8 | 306 | 1890 | 354 | 1890 |
| WNS, ns | -8.301 | -7.442 | -8.652 | -7.789 |
| TNS, ns | -73934.203 | -72175.352 | -83415.094 | -85617.805 |
| Setup: endpoints incumplidos | 15866 | 15866 | 15863 | 15872 |
| WHS, ns | -0.075 | -0.075 | -0.075 | -0.075 |
| THS, ns | -139.633 | -140.037 | -139.511 | -139.541 |
| Hold: endpoints incumplidos | 1893 | 1992 | 1891 | 1894 |

Ruta peor: `rc_user[79]` → `producer/remaining_reg[150][12]/D`, **11.799 ns/47 niveles**.
La contribución lógica pasa de 2.900 a 2.634 ns y la de
ruteo estimado de 8.553 a 9.165 ns. Son estimaciones previas
a colocación/ruteo; no mediciones del retardo físico de la tarjeta.
DCP acíclico: 76366 primitivas, 94680 nets y ocho
controles del auditor. Las 12 categorías de `check_timing` son cero;
los incumplimientos setup/hold se conservan en la tabla. No hay
colocación/ruteo ni frecuencia física medida.

Se conservan R7, R8 y R9 con sus diferencias de área y timing. La
preferencia sólo compara WNS entre candidatos funcionalmente revisados;
no afirma cierre físico ni que una sola métrica resuma todos los costes.

## Coste de ejecución y respaldo

Se mantuvieron los mismos bancos y el inicio XSIM de R8. La unidad
completa empleó 00:00:42 de pared y
43370 ms de CPU de XSIM. El banco RTL
tardó 305.482 s, frente a 305,695 s de R8; la netlist
624.392 s, frente a 614,293 s. Son observaciones
individuales, sujetas a carga y E/S; no una medida aislada de aceleración.

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T172339Z-V4-ffn-supervised-pcie-inventory-join-rc-static-mask-rtl` | 305.482 | `a359b6f62aebab0f8291d4dd271fbd40c023ae30ff1435a913df0213ad6795a4` |
| `20261008T173404Z-V4-ffn-supervised-pcie-inventory-join-rc-static-mask-synth` | 318.722 | `cd4ff93cab3abe9937534bd93964226bafb133dbdaead970896236098eecf60f` |
| `20261008T174107Z-V4-ffn-supervised-pcie-inventory-join-rc-static-mask-binary-rtl` | 122.828 | `487632c2bc5bedd8d6d7970dca972c94fd548581297e69224c6f70c74cc09962` |
| `20261008T174544Z-V4-ffn-supervised-pcie-inventory-join-rc-static-mask-binary-netlist` | 624.392 | `d6f7ca8ff9658972a19c1634d34468a51afb9b7ee439fa5ea4254e805c8f82d6` |

Fuentes: `source-20261008T172303Z.zip`, 2271 archivos,
SHA-256 `bf94eea721e07a303f439bef901a6db605f20fddc5b9154e94976204e27d8476`. Paquetes y miembros verificados localmente
y en NFS bajo `artifacts/ntrpu-board-vivado/<tanda>/`. Revisiones funcionales,
de síntesis, DCP y ejecución: `build/cloud-setup/20261002-board/<tanda>-*.json`.
Netlist propia: `a4f5eec8dd57b729ac44dea1ea74073a7f3c91a9c914a603411226ea3633a753`.

## Pendientes

Revisar el siguiente cambio desde las métricas, conservando todas las
variantes. Endpoint nativo/BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD siguen abiertos.
La captura nativa heredada sólo contiene tres lecturas de 32/64/32 bytes;
los bancos ampliados son sintéticos. [Cola completa](VIVADO_QUEUE.md).
