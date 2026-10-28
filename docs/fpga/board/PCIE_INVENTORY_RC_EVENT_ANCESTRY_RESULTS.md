# Monitor PCIe: ascendencia por eventos EOP

El 2026-10-08 R8 pasó los bancos completos RTL y de su propia netlist.
WNS OOC **-7.442 ns**; **R8** conserva el mejor
WNS de las variantes comparadas. Esta preferencia no certifica timing físico.

## Cambio y pruebas

El productor `ntrpu_pcie_inventory_scheduler_r8.sv` conserva R7, salvo un
comentario. El selector R3 obtiene cada contexto del último SOP anterior
o simultáneo y admite únicamente su primer EOP efectivo. Las máscaras
provienen de los orígenes de escrituras previas; no se propagan entre ellas.
Propiedad, aritmética, capacidad y latencia permanecen iguales.
[Contrato](PCIE_INVENTORY_RC_EVENT_ANCESTRY_CONTRACT.md).

- 1.153.296 comparaciones heredadas, incluidas 994.288 X/Z y 155.168 filas
  binarias de referencia independiente; 840 bits comparados contra la
  recurrencia de propiedad y el selector secuencial R1.
- 32.768 comparaciones adicionales de seis controles simultáneos,
  incluidas 32.256 X/Z, con contexto retenido, EOP repetidos y discontinue.
  Ocho configuraciones del banco incluyen seis disposiciones distintas
  de SOP/EOP; las dos restantes repiten disposiciones. Tres controles
  negativos verifican la presencia de sus resultados.
- Banco principal 22.339/68, siete mutantes y 17 controles de evidencia;
  propiedad integrada 46.356/3.872; RQ 615.536/5.712 X/Z y 8.850/240;
  selección 89.632; aritmética RC 45.056/6.624 X/Z y 544/60.
- RTL binario y netlist propia: 22.330/67 más los tres suplementos íntegros.
  Todas las filas públicas coinciden. El DUT mapeado no compila RTL del
  productor, selector, wrapper ni validador. Arranque de 200 ns y sin SDF.

Las pruebas X/Z no constituyen demostración formal universal. Se conserva
la comparación de estados posteriores a fallos y de tablas en RTL.

## Área y timing estimado

Vivado 2026.1, VU5P FLVA2104 -2-i, `RuntimeOptimized`, 4 ns, I/O cero,
sin excepciones temporales; 256 tags, dos encabezados RQ y cuatro RC.

| Métrica | Original | R5 | R6 | R7 | R8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUT | 48189 | 44857 | 42403 | 42362 | 51937 |
| FF | 10729 | 10729 | 10729 | 10729 | 10729 |
| CARRY8 | 260 | 308 | 306 | 306 | 1890 |
| WNS (ns) | -9.877 | -8.806 | -8.903 | -8.301 | -7.442 |
| TNS (ns) | -109536.984 | -127995.078 | -90778.977 | -73934.203 | -72175.352 |
| Setup: endpoints incumplidos | 15851 | 15868 | 15865 | 15866 | 15866 |
| WHS (ns) | -0.075 | -0.075 | -0.075 | -0.075 | -0.075 |
| THS (ns) | -139.02 | -139.603 | -139.723 | -139.633 | -140.037 |
| Hold: endpoints incumplidos | 1936 | 1896 | 1961 | 1893 | 1992 |

Peor ruta: `rc_user[76]` → `producer/remaining_bytes_reg[20]/D`, **11.453 ns/45 niveles**.
DCP acíclico: 75004 primitivas, 93317 nets y ocho
controles del auditor. Las 12 categorías de `check_timing` son cero;
los incumplimientos setup/hold indicados arriba permanecen abiertos.
No hay colocación/ruteo ni frecuencia física medida.

## Coste de simulación

Se verificó la ejecución del Tcl sin captura automática de ondas en las
15 sesiones del banco completo y cuatro de cada replay binario.
La unidad ampliada empleó **00:00:41** de pared
y **42760 ms** de CPU de XSIM.
El conjunto WDB del banco RTL ocupa **489778 bytes**.
El banco completo tardó **305.695 s**, frente a
645,496 s de R7. Cambiaron diseño, pruebas y captura de ondas: esta comparación
no aísla una causa ni demuestra una aceleración atribuible a una sola opción.
El replay de netlist tardó **614.293 s**, frente a
419,170 s de R7, y sus WDB suman **47387980 bytes**.
Por tanto, la reducción de tiempo del banco RTL no se extiende a cada etapa.

## Evidencia y respaldo

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T155949Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-rtl` | 305.695 | `a0066da6f4ce0ea2014951e04b4abe1a7c5c9a041c797840b2d3969cafa6bf59` |
| `20261008T160701Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-synth` | 307.559 | `3e073eb32daa4ce6c3b2cc595973ee89e2bb816ec792eaf89b89387fdfe731f1` |
| `20261008T161453Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-binary-rtl` | 123.399 | `c418a24bcbe06e40ae5801cb83c08338555a735a60d2e39a5328fd19ce670caa` |
| `20261008T161905Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-binary-netlist` | 614.293 | `b7fec3653cef5624e47383a2932d7fb75e14d726699d1e1594358c90269e1ec3` |

Fuentes: `source-20261008T155914Z.zip`, 2249 archivos,
SHA-256 `74605ae680b7f752148e56a3ec89aa52cc5185c528540025a9c59ab34f654d70`. Cada tanda conserva paquete local/NFS
verificado en `artifacts/ntrpu-board-vivado/<tanda>/`.
Netlist propia: `6b1ed7e6b9360ee2096a50648c7328ba8f34ecef821f10a072b3ca8ee71181de`.

Revisiones en `build/cloud-setup/20261002-board/`:

- `20261008T155949Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-rtl-independent-inventory-join-review.json`: `ece94fa50cb1f99aeccda52048ee4f57883ab1bbec3488737fb90793c09c3b69`.
- `20261008T160701Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-synth-independent-synthesis-review.json`: `c05f2404a72781f61d86c8e36465781ca659bc1a7c6b0b224e102c10e9bad065`.
- `20261008T161453Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `a330abe52fef9b75e411f246c1e26ccf7f04900c2acb88dd99c7e374e07f37dc`.
- `20261008T161905Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `8a64f5ebef4a399db834378a6871420b0dcdbe0ea0a187a745ffb11a8e26dfd2`.
- `20261008T155949Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-rtl-xsim-runtime-review.json`: `7e387c3a2afa0ab3d390d7e5aa6c40d8297801e15d5e553f415f42de012d7ede`.
- `20261008T161453Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-binary-rtl-xsim-runtime-review.json`: `5566b7f78e0f30a91b91299df8171c67fc9acb8ee291421ca840841a5d1acf1c`.
- `20261008T161905Z-V4-ffn-supervised-pcie-inventory-join-rc-event-ancestry-binary-netlist-xsim-runtime-review.json`: `a19002f1f21685fcaa757ea36ceb82c12e1d460651d0191e101ff78732a98f6e`.

## Pendientes

Continuar desde la ruta medida y la variante preferida, sin suprimir casos.
Siguen abiertos endpoint nativo, BDF independiente, CDC/reset de 50 ms,
DIMM exacto, cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD.
La captura nativa heredada sólo incluye tres lecturas de 32/64/32 bytes;
los bancos ampliados son sintéticos. [Cola completa](VIVADO_QUEUE.md).
