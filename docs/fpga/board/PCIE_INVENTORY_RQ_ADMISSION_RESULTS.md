# Monitor PCIe: dos admisiones RQ efectivas

El 2026-10-08 R6 pasó los bancos RTL y el replay de su propia netlist.
WNS OOC: **-8.903 ns**, diferencia **+0.974 ns**
respecto del original y **-0.097 ns** respecto de R5. R5 conserva el mejor WNS estimado entre las variantes medidas. Esto no acredita timing físico.

## Cambio validado

`ntrpu_pcie_inventory_scheduler_r6.sv` conserva los prefijos RC de R5 y
reemplaza las cuatro etapas de admisión por posición por dos eventos SOP
efectivos, ordenados por lane. `rq_admission_flags` combina duplicados y
preserva punteros desconocidos, condiciones de cuatro estados y count
firmado. El parser de payload/EOP y la prioridad del slot original siguen
vigentes. No cambia el contador global ni la latencia; no incorpora R3.
[Contrato previo](PCIE_INVENTORY_RQ_ADMISSION_CONTRACT.md).

## Simulación y controles

- Admisión real: **615.536 comparaciones** contra la exploración original
  por lane, incluidas **5.712 inyecciones X/Z**. Los casos de capacidad
  cubren counts almacenados 0–511 y cero a cuatro liberaciones RC.
  Dos controles rechazan evidencia incompleta.
- Integración RQ nueva: **8.850 filas/240 escenarios**, **2.265.600 palabras
  de tabla** en RTL. Orden, duplicados, máscaras, read/write, capacidad
  254/255/256 y reutilización después de RC. Todas las filas coinciden con
  R7, incluidas las posteriores a fallos. Cuatro controles de evidencia.
- Selector real: **89.632 casos** contra un modelo de lista de eventos
  regenerado; dos controles de evidencia.
- Función real: **45.056 comparaciones** contra la recurrencia original,
  incluidas **6.624 inyecciones X/Z**, bajo las 16 máscaras binarias.
  Dos controles verifican el rechazo de evidencia incompleta.
- Integración adicional: **544 filas/60 escenarios**, fragmentos del mismo
  tag, contexto retenido, cero, counts inválidos, reutilización RQ y punteros
  malformados. RTL compara **139.264 palabras de tabla** contra R7 congelado
  en la misma ejecución; las expectativas Python sólo cubren estado fiable
  anterior a errores. Se comparan también todos los estados posteriores.
- Banco completo: **22.339 filas/68 escenarios**, **5.718.784 palabras**,
  siete mutantes reales y 17 controles de corrupción de evidencia.
- RTL binario y netlist propia: **22.330 filas/67 escenarios**, más las
  **544 filas RC y 8.850 filas RQ adicionales**. Todas las filas públicas
  coinciden. Cada banco tiene 12 controles principales y ocho adicionales. La netlist no compila
  el productor, selector, wrapper ni validador RTL como DUT.

El banco RC adicional retira 365
registros, cancela 59
por cold reset y verifica 60
revocaciones. Startup binario de 200 ns y muestreo de 1 ns, sin SDF.
Los casos X/Z aprobados no constituyen una demostración formal universal.

## Caracterización OOC

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, reloj de 4 ns,
I/O cero, sin excepciones temporales; perfil completo 256 tags/RQ2/RC4.

| Métrica | Original | Prefijos R5 | Admisión RQ R6 |
| --- | ---: | ---: | ---: |
| LUT | 48189 | 44857 | 42403 |
| FF | 10729 | 10729 | 10729 |
| CARRY8 | 260 | 308 | 306 |
| WNS, ns | -9.877 | -8.806 | -8.903 |
| TNS, ns | -109536.984 | -127995.078 | -90778.977 |
| Endpoints setup incumplidos | 15851 | 15868 | 15865 |
| WHS, ns | -0.075 | -0.075 | -0.075 |
| THS, ns | -139.02 | -139.603 | -139.723 |
| Endpoints hold incumplidos | 1936 | 1896 | 1961 |

Peor ruta: `producer/allocated_tags_reg[179]/C` → `producer/remaining_bytes_reg[20]/D`: **12.886 ns**, **53 niveles**.
El DCP contiene 63419 primitivas y 70876 nets auditadas
sin ciclos combinacionales; ocho controles validan el auditor. Las 12
categorías de `check_timing` tienen cero incidencias. La revisión conserva
el inventario exacto de advertencias y los fallos setup/hold observados.
No hay colocación/ruteo ni frecuencia física medida.

## Evidencia y recuperación

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T140516Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-rtl` | 234.504 | `c675c8b19ff1dc2ef23a89ee4905e4787365995f0ceebe63fbf4af945e3ada36` |
| `20261008T141223Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-synth` | 295.487 | `c56432934c5847fff6b6ce3207ce9e6c3ed5e850686c39b10f8723579996b0a1` |
| `20261008T141841Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-binary-rtl` | 89.322 | `331e2c306e351b9cabc787acb83ab75cf57b8c78e25e096da66e0084463534e6` |
| `20261008T142356Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-binary-netlist` | 251.224 | `44ad21a4a28f849a1a94f6fca21a98c6015121d17230e94a28745826f5a16793` |

Fuentes: `source-20261008T140419Z.zip`, 2196 archivos,
SHA-256 `ddf508d5f2bbdfff2e3dbd373e777d68df2e0cc1e13b943582ffd1aa1a8b923c`. Archivos y miembros verificados en
`artifacts/ntrpu-board-vivado/<tanda>/`, respaldo físico en D: y copia NFS.
Netlist reproducida: `36833e7856a4cc3dedfdbff534ac5fc31a6542d0f4f0c0618a844e4003f5a5ce`.

Revisiones en `build/cloud-setup/20261002-board/`:

- `20261008T140516Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-rtl-independent-inventory-join-review.json`: `c4cab0ebab109d99c53592fa5d17319e7e4585b8a729f86ea74250627818f1d8`.
- `20261008T141223Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-synth-independent-synthesis-review.json`: `d8100fe2b3e8f1bd633cfa1bc2fdc18573cbc49e45491ed5d4638395ead63e11`.
- `20261008T141841Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `172c4b2c8d916e5f734af75bbed7ab847e3b225366293e1cbe4bfc2dafcd1ea5`.
- `20261008T142356Z-V4-ffn-supervised-pcie-inventory-join-rq-admission-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `3722f56b813d25807de6ddeba46a2c0fae00d499e0f2ee3701f2367c9563e59e`.

## Trabajo pendiente

El [siguiente experimento](PCIE_INVENTORY_RQ_ADMISSION_FOLLOWUP.md) parte de
la ruta medida. Mantener capacidad, ausencia de pausas, latencia y semántica
posterior a fallos. Toda variante requiere el banco completo, ambos
suplementos y replay de su propia netlist. Siguen abiertos conexión nativa con identidad/BDF
independiente, CDC/reset y reset nativo de 50 ms, DIMM exacto, cuatro DDR/
512 GiB, inferencia con pesos reales, ECAD e implementación física.
La captura nativa reutilizada sólo contiene tres lecturas de 32/64/32 bytes;
el banco amplio es sintético. Requisitos completos: [cola](VIVADO_QUEUE.md).
