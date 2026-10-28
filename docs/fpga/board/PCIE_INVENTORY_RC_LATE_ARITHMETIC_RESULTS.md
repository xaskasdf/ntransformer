# Monitor PCIe: selección tardía de aritmética RC

R11 pasó RTL y el replay de su propia netlist el 2026-10-08. Frente a R8:
**+4926 LUT**, **+278 CARRY8** y
**-0.972 ns de WNS**. **R8** queda como referencia
de mejor WNS medido. La preferencia no acredita timing físico.

## Cambio comprobado

El productor R11 conserva el cuerpo aritmético de R8 en la función
`rc_prefix_constant`, con el único cambio de nombre. La nueva
`rc_prefix_state` calcula 16 resultados para máscaras constantes y
selecciona después con una máscara efectiva construida por `if`.
Los bits X/Z no habilitan una operación. La comparación de fuentes
excluye otros cambios. El selector R3 y `newest` de 32 bits, capacidad,
latencia, propiedad, RQ y estados posteriores a fallos permanecen iguales.
[Contrato](PCIE_INVENTORY_RC_LATE_ARITHMETIC_CONTRACT.md).

## Validaciones

Se conservaron 1.153.296 comprobaciones heredadas, incluidas 994.288 X/Z,
y 32.768 adicionales, incluidas 32.256 X/Z, comparando 840 bits con las
referencias independientes. También pasaron selector 89.632; RQ
615.536/5.712 X/Z y 8.850/240; RC 45.056/6.624 X/Z y 544/60;
propiedad integrada 46.356/3.872; principal 22.339/68 y siete mutantes.

El banco aritmético conserva las 45.056 comprobaciones originales y añade
2.048 casos en ocho contextos, incluidos 1.920 con X/Z en la máscara
y 2.000 con alguna entrada desconocida. Se verifica el top realmente
compilado, su llamada al DUT y cuatro controles de evidencia adicionales.

La propia netlist reproduce 22.330/67 y los tres suplementos binarios
completos, con todas las filas públicas idénticas. El DUT mapeado no
compila RTL del productor, selector, wrapper o validador. Las tablas se
comparan en RTL; ambos modos conservan estados posteriores a fallos.
Arranque binario de 200 ns, sin SDF. Los casos X/Z no son prueba universal.

## Área y timing OOC

Vivado 2026.1, VU5P FLVA2104 -2-i, `RuntimeOptimized`, período de 4 ns,
I/O cero y sin excepciones temporales. Perfil 256 tags/RQ2/RC4.

| Métrica | R8 | R9 | R10 | R11 |
| --- | ---: | ---: | ---: | ---: |
| LUT | 51937 | 45541 | 53671 | 56863 |
| FF | 10729 | 10729 | 10729 | 10729 |
| CARRY8 | 1890 | 354 | 1890 | 2168 |
| WNS, ns | -7.442 | -8.652 | -7.789 | -8.414 |
| TNS, ns | -72175.352 | -83415.094 | -85617.805 | -72725.289 |
| Setup: endpoints incumplidos | 15866 | 15863 | 15872 | 15849 |
| WHS, ns | -0.075 | -0.075 | -0.075 | -0.075 |
| THS, ns | -140.037 | -139.511 | -139.541 | -139.621 |
| Hold: endpoints incumplidos | 1992 | 1891 | 1894 | 1892 |

Ruta peor: `rc_user[76]` → `producer/remaining_bytes_reg[20]/D`, **12.425 ns/51 niveles**.
La contribución lógica pasa de 2.900 a 2.957 ns y la de
ruteo estimado de 8.553 a 9.468 ns. Son estimaciones previas
a colocación/ruteo; no mediciones del retardo físico de la tarjeta.
DCP acíclico: 81118 primitivas, 102404 nets y ocho
controles del auditor. Las 12 categorías de `check_timing` son cero;
los incumplimientos setup/hold se conservan en la tabla. No hay
colocación/ruteo ni frecuencia física medida.

Se conservan R7, R8, R9 y R10 con sus diferencias de área y timing. La
preferencia sólo compara WNS entre candidatos funcionalmente revisados;
no afirma cierre físico ni que una sola métrica resuma todos los costes.

## Coste de ejecución y respaldo

Se mantuvieron los bancos heredados y el inicio XSIM de R8, añadiendo
2.048 casos aritméticos. La unidad de ascendencia
completa empleó 00:00:41 de pared y
42640 ms de CPU de XSIM. El banco RTL
tardó 303.491 s, frente a 305,695 s de R8; la netlist
683.237 s, frente a 614,293 s. Son observaciones
individuales, sujetas a carga, E/S y al banco ampliado; no una medida
aislada de aceleración.

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T180852Z-V4-ffn-supervised-pcie-inventory-join-rc-late-arithmetic-rtl` | 303.491 | `f1e93601be0d29e2b7a7fdad3f4833e47b64a9eb10511aa05c8ccf5c9f1f6986` |
| `20261008T181604Z-V4-ffn-supervised-pcie-inventory-join-rc-late-arithmetic-synth` | 319.735 | `5040fb49e03bd95e93bc6ef94b0209a6547c86f5340c82afbe1d1bb3e5973b1d` |
| `20261008T182354Z-V4-ffn-supervised-pcie-inventory-join-rc-late-arithmetic-binary-rtl` | 124.056 | `beb64c86e4ee17a40b9ece37e6e6f00af535212a838c35342c6e5a1eb796d55c` |
| `20261008T182829Z-V4-ffn-supervised-pcie-inventory-join-rc-late-arithmetic-binary-netlist` | 683.237 | `8c6ec750d97395f1d63564e35ffe75747369973f6ae29b3f9d9ab9ed6ee3b96a` |

Fuentes: `source-20261008T180813Z.zip`, 2279 archivos,
SHA-256 `88531ed7ed6b3414d41e4fc753c86c2cf28a1b665338764e0cc2a434548632a8`. Paquetes y miembros verificados localmente
y en NFS bajo `artifacts/ntrpu-board-vivado/<tanda>/`. Revisiones funcionales,
de síntesis, DCP y ejecución: `build/cloud-setup/20261002-board/<tanda>-*.json`.
Netlist propia: `1363f395ed604439d6762428f360b3e8343acd8bd56b5ac35f52e57b64cf5e2a`.

## Pendientes

Revisar el siguiente cambio desde las métricas, conservando todas las
variantes. Endpoint nativo/BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD siguen abiertos.
La captura nativa heredada sólo contiene tres lecturas de 32/64/32 bytes;
los bancos ampliados son sintéticos. [Cola completa](VIVADO_QUEUE.md).
