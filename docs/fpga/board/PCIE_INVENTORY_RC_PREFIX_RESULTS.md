# Monitor PCIe: prefijos de escrituras RC efectivas

El 2026-10-08 R5 pasó los bancos RTL y el replay de su propia netlist.
WNS OOC: **-8.806 ns**, diferencia **+1.071 ns**
respecto del original. R5 queda como preferido por WNS estimado; el TNS también debe considerarse. Esto no acredita timing físico.

## Cambio validado

`ntrpu_pcie_inventory_scheduler_r5.sv` y `ntrpu_pcie_rc_update_select_r1.sv`
calculan el estado previo a cada encabezado desde la tabla propia o el contexto
retenido y una máscara de escrituras anteriores efectivas. Se elimina el
encadenamiento aritmético de esas escrituras, conservando el cálculo del
fragmento actual, prioridad por lane, liberación terminal, RC antes de RQ,
contadores y latencia. El candidato no incorpora el contador registrado R3.
[Contrato previo](PCIE_INVENTORY_RC_PREFIX_CONTRACT.md).

## Simulación y controles

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
  **544 filas adicionales**. Todas las filas públicas coinciden. Cada banco
  tiene 12 controles principales y cuatro adicionales. La netlist no compila
  el productor, selector, wrapper ni validador RTL como DUT.

El banco adicional retira 365
registros, cancela 59
por cold reset y verifica 60
revocaciones. Startup binario de 200 ns y muestreo de 1 ns, sin SDF.
Los casos X/Z aprobados no constituyen una demostración formal universal.

## Caracterización OOC

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, reloj de 4 ns,
I/O cero, sin excepciones temporales; perfil completo 256 tags/RQ2/RC4.

| Métrica | Original | Selector R4 | Prefijos R5 |
| --- | ---: | ---: | ---: |
| LUT | 48189 | 42176 | 44857 |
| FF | 10729 | 10729 | 10729 |
| CARRY8 | 260 | 260 | 308 |
| WNS, ns | -9.877 | -10.047 | -8.806 |
| TNS, ns | -109536.984 | -127754.758 | -127995.078 |
| Endpoints setup incumplidos | 15851 | 15865 | 15868 |
| WHS, ns | -0.075 | -0.075 | -0.075 |
| THS, ns | -139.02 | -139.748 | -139.603 |
| Endpoints hold incumplidos | 1936 | 1952 | 1896 |

Peor ruta: `producer/allocated_tags_reg[51]/C` → `producer/remaining_bytes_reg[20]/D`: **12.789 ns**, **51 niveles**.
El DCP contiene 66369 primitivas y 73829 nets auditadas
sin ciclos combinacionales; ocho controles validan el auditor. Las 12
categorías de `check_timing` tienen cero incidencias. La revisión conserva
el inventario exacto de advertencias y los fallos setup/hold observados.
No hay colocación/ruteo ni frecuencia física medida.

## Evidencia y recuperación

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T132016Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-rtl` | 196.340 | `014f20dda1e970828efb9e05c3dbdf20e0098492d512e290c568411baea20e0c` |
| `20261008T132618Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-synth` | 301.466 | `f72d2b97c3b6ce22a337013af0db63c42ee860c6887a4ad7cc1fc102ffb10625` |
| `20261008T133232Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-binary-rtl` | 67.170 | `e5e7f91cf0547139762c57c3dacbae4b1228291d75c400daf582c640eb531fba` |
| `20261008T133626Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-binary-netlist` | 177.983 | `5a8689ba38e35d1bd0cf240a9e6c4473a1f8313062fe791985653417b921728a` |

Fuentes: `source-20261008T131939Z.zip`, 2179 archivos,
SHA-256 `aba7fd5cb7272632826a690d1324a69fa9955e73355a8f0e976ff89b44647d53`. Archivos y miembros verificados en
`artifacts/ntrpu-board-vivado/<tanda>/`, respaldo físico en D: y copia NFS.
Netlist reproducida: `eac3dd36b858e1084af4b00597fe6c6e79b8e1028b05cf0087d6856780d32779`.

Revisiones en `build/cloud-setup/20261002-board/`:

- `20261008T132016Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-rtl-independent-inventory-join-review.json`: `165fab1500021eb3fc0a5b31e6b2bbe346c0a7c0a5fd1f5c928dff238603e04e`.
- `20261008T132618Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-synth-independent-synthesis-review.json`: `226607fd383366f6159a68f7be37a009901e0beaa740562b7fe5863c7117449a`.
- `20261008T133232Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `c226453ea63637c3fc4f34160f31274de2523c08c2dbf561dc6d74dd0b1f780b`.
- `20261008T133626Z-V4-ffn-supervised-pcie-inventory-join-rc-prefix-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `227306c1020a521df251e2e51a0f6933ea5ff5ad261180d87cd903001edba2ca`.

## Trabajo pendiente

El [siguiente experimento](PCIE_INVENTORY_RC_PREFIX_FOLLOWUP.md) reduce la
cadena de admisión RQ a sus dos eventos efectivos, preservando punteros
malformados y la semántica posterior a fallos. Optimizar la ruta medida sin reducir capacidad, introducir pausas ni perder
semántica posterior a fallos. Toda variante requiere banco completo y replay
de su propia netlist. Siguen abiertos conexión nativa con identidad/BDF
independiente, CDC/reset y reset nativo de 50 ms, DIMM exacto, cuatro DDR/
512 GiB, inferencia con pesos reales, ECAD e implementación física.
La captura nativa reutilizada sólo contiene tres lecturas de 32/64/32 bytes;
el banco amplio es sintético. Requisitos completos: [cola](VIVADO_QUEUE.md).
