# Monitor PCIe: propiedad RC en paralelo

El 2026-10-08 el productor R7 y selector R2 pasaron los bancos completos
y el replay de su propia netlist. WNS OOC **-8.301 ns**:
**+0.602 ns** frente a R6 y **+0.505 ns** frente a R5.
**R7** queda como referencia de mejor WNS entre estas variantes.
La comparación no acredita timing físico ni mejora global por sí sola.

## Cambio y semántica

`ntrpu_pcie_inventory_scheduler_r7.sv` comparte la predecodificación
`rc_ownership` del selector R2. Cada encabezado consulta la propiedad
registrada y antecedentes terminales del mismo tag, sin realimentar las
liberaciones calculadas por otros encabezados. Conserva la aritmética RC
de R5, admisión RQ de R6, tablas, capacidad y latencia. La liberación en
SOP sigue siendo independiente del error del descriptor. El EOP retenido
no vuelve a liberar el tag; RC sigue precediendo a la reutilización RQ.
[Contrato previo](PCIE_INVENTORY_RC_OWNERSHIP_CONTRACT.md).

## Comparaciones verificadas

- Propiedad y ascendencia: **1.153.296 comprobaciones**, **994.288 con X/Z**,
  contra recurrencia secuencial y selector R1 conservado. Incluye 155.168
  filas de un modelo de eventos independiente, las 15 particiones de tags
  y todas las combinaciones de ocho señales 0/1/X/Z. Se conservan las
  89.632 entradas anteriores y sus 800 bits de salida. Tres controles de
  evidencia y dos de procedencia de compilación.
- Integración nueva: **46.356 filas/3.872 escenarios**, **11.867.136 palabras
  de tabla**, 34741 retiros, 3871
  cancelaciones y 3872 revocaciones. Compara
  todo estado posterior a errores con el inventario de referencia R7;
  éste es un módulo distinto del productor nuevo. Cuatro controles.
- Regresiones retenidas: banco principal **22.339/68**, siete mutantes y
  17 controles; RQ **615.536/5.712 X/Z** más **8.850/240**; selección
  **89.632**; aritmética RC **45.056/6.624 X/Z** más **544/60**.
- RTL binario y netlist propia: **22.330/67** más los tres suplementos
  **46.356/3.872**, **8.850/240** y **544/60**. Todas las filas públicas
  coinciden. Cada ejecución tiene 12 controles principales y 12 adicionales.
  El DUT mapeado no compila productor, selector, wrapper ni validador RTL.

Arranque binario de 200 ns, muestreo de 1 ns y sin SDF. X/Z aprobado en
estos casos no equivale a prueba formal universal. Vivado reemplaza el PRJ
de cada top al cambiar de banco: la revisión R1 comprueba los registros
archivados de análisis con rutas exactas, módulos, elaboración y finalización,
además de hashes de fuentes. El verificador R0 que exigía ese PRJ se conserva.

## Área y timing estimado

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, 4 ns, I/O cero y
sin excepciones temporales; 256 tags/RQ2/RC4.

| Métrica | Original | R5 | R6 | R7 |
| --- | ---: | ---: | ---: | ---: |
| LUT | 48189 | 44857 | 42403 | 42362 |
| FF | 10729 | 10729 | 10729 | 10729 |
| CARRY8 | 260 | 308 | 306 | 306 |
| WNS, ns | -9.877 | -8.806 | -8.903 | -8.301 |
| TNS, ns | -109536.984 | -127995.078 | -90778.977 | -73934.203 |
| Endpoints setup incumplidos | 15851 | 15868 | 15865 | 15866 |
| WHS, ns | -0.075 | -0.075 | -0.075 | -0.075 |
| THS, ns | -139.02 | -139.603 | -139.723 | -139.633 |
| Endpoints hold incumplidos | 1936 | 1896 | 1961 | 1893 |

Peor ruta: `producer/allocated_tags_reg[243]/C` → `producer/remaining_bytes_reg[20]/D`, **12.284 ns/49 niveles**.
DCP: 62594 primitivas y 70050 nets, sin ciclos; ocho
controles del auditor. Las 12 categorías de `check_timing` están limpias.
Esto no elimina los incumplimientos setup/hold que aparecen en la tabla.
Sin colocación/ruteo ni frecuencia física medida.

## Evidencia y respaldo

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T145423Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-rtl` | 645.496 | `c2fcad6d8776a9011cbb37a1b5e75411b9022a661f79a9fb2a6e3c1746a688f4` |
| `20261008T151053Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-synth` | 291.866 | `af731f1df5e26371871c37ab9e21a885158e6b014e6141939809ae6bdb14c672` |
| `20261008T151716Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-binary-rtl` | 128.402 | `c4a2f63c82d80bc91981d5b4c5afd152e49fcfaa4025e38f829d4d8538626a4c` |
| `20261008T152147Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-binary-netlist` | 419.170 | `ab0d10cd083e3ac33f0582ce0314505278ed3296f0ea1a9b8d512731688b8493` |

Fuentes: `source-20261008T145341Z.zip`, 2224 archivos,
SHA-256 `0a0282275fd3bd986c0038d91099ae855b37e8008713463691ba676da0e92f1a`. Respaldos en
`artifacts/ntrpu-board-vivado/<tanda>/`, con almacenamiento físico en D: y
copia NFS verificados. Netlist propia:
`d0c78c6faa30bd53329c9c5c61870de53ec3b4ca4f01c6fac160418ff0f3e272`.

Revisiones en `build/cloud-setup/20261002-board/`:

- `20261008T145423Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-rtl-independent-inventory-join-review.json`: `67b48a5cb950ee740f298d64e6d9df38d7b4a7cde03a8d925b67887016a58b51`.
- `20261008T151053Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-synth-independent-synthesis-review.json`: `42cedd3dae0f81b3189903dc13fb6539fb85066920b9a87b419c8271734edd0f`.
- `20261008T151716Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `7f78b201b4d60b111730a9f3fff64d29059fd7978321d3b32f91e4819ffc2b8e`.
- `20261008T152147Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `0d0c3443a5d128cc70ca6bbb52bb159a7aaf151d3c08e2d0c5fb86757d5edbfc`.

## Pendientes

Continuar desde la ruta medida y mantener los bancos completos. Siguen
abiertos identidad/BDF independiente, conexión nativa, CDC/reset nativo de
50 ms, DIMM exacto, cuatro DDR/512 GiB, inferencia real, ECAD y timing físico.
La captura nativa heredada sólo contiene tres lecturas de 32/64/32 bytes;
los bancos ampliados son sintéticos. [Cola completa](VIVADO_QUEUE.md).
