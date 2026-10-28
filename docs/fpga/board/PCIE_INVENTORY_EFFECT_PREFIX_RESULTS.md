# R15: efectos en las consultas PCIe crudas

Revisado el 2026-10-09. R15 pasa el banco RTL completo y la comparación
binaria con **su propia netlist**, pero empeora la estimación de timing.
**R12 conserva la preferencia** como referencia del monitor.

## Cambio y alcance

`rc_prefix_state` aplica el álgebra revisada de efectos de 57 bits a las
consultas seleccionadas por R3, desde la tabla y el contexto propios.
El resto del productor R12 permanece idéntico: puertos, registros,
capacidad, latencia y reducción de bytes por lane. No introduce etapas
nuevas ni acredita el pipeline completo.

## Validación funcional

El banco principal pasa 22.339 filas y 68 escenarios, con 5.718.784
comparaciones de palabras de tabla. Conserva todos los bancos auxiliares
R12, las inyecciones X/Z y cada estado posterior a fallo. Se mantienen
los siete mutantes heredados. Un octavo invierte el orden dentro de la
función real del productor y falla en el caso dirigido: `0007004` frente
a `000000b`; tres controles rechazan evidencia alterada de ese caso.

Los cinco bancos binarios comparan **78.712 filas** entre RTL, netlist y
referencia independiente: principal 22.330, propiedad 46.356, RQ 8.850,
prefijos 544 y lanes 632. Los 30 controles de evidencia son rechazados.
Se verifican fuentes compiladas y artefacto mapeado; no se compila un
productor, wrapper o validador RTL oculto en esa prueba. Arranque de
200 ns, muestreo a 1 ns, sin SDF: no demuestra timing físico ni X/Z en puertas.

## Síntesis comparable

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, OOC a 4 ns,
I/O cero y sin excepciones. Mismas restricciones que R12.

| Medida | R12 | R15 |
| --- | ---: | ---: |
| LUT | 52.234 | 52.880 |
| FF | 10.729 | 10.729 |
| CARRY8 | 1.902 | 1.922 |
| WNS (ns) | −7,068 | −9,544 |
| TNS (ns) | −70.828,984 | −90.078,938 |
| WHS (ns) | −0,075 | −0,075 |

R15 tiene cero latches, BRAM, URAM y DSP. La peor ruta llega desde
`rc_user[76]` a `remaining_bytes_reg[20]/D`: 53 niveles, 13,555 ns
(3,524 de lógica y 10,031 de ruteo estimado). El grafo auditado es
acíclico: 76.688 primitivas, 95.424 redes y ocho controles negativos.
Setup y hold siguen fallando; el DCP no está colocado ni ruteado.

## Evidencia y siguiente paso

| Etapa | Tanda | SHA-256 del archivo |
| --- | --- | --- |
| pcie_inventory_effect_prefix_rtl | `20261009T102343Z-V4-ffn-supervised-pcie-inventory-join-effect-prefix-rtl` | `f45630f6799e4b5bf089e113b727c215d5333570c9bab3f8eb937c15e84835a2` |
| pcie_inventory_effect_prefix_synthesis | `20261009T103314Z-V4-ffn-supervised-pcie-inventory-join-effect-prefix-synth` | `d2e343c28877ce34de9b36a5726dc6cf6e557b9a169192aff0e462d9a02b385f` |
| pcie_inventory_effect_prefix_binary_rtl | `20261009T103940Z-V4-ffn-supervised-pcie-inventory-join-effect-prefix-binary-rtl` | `e20bfe720164588aca6b73a229a5e0d0920a3ad371873dc625822e68bca1b9e6` |
| pcie_inventory_effect_prefix_binary_netlist | `20261009T104547Z-V4-ffn-supervised-pcie-inventory-join-effect-prefix-binary-netlist` | `0b777b73823f6bb6c426e563726cbe042c4876ae56d928e4b3689ce8454089ad` |

Archivos y fuentes verificados en D:/NFS, certificados bajo
`artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`. R106 registra
las cuatro tandas; los lectores y configuraciones están en
`build/cloud-setup/20261002-board/`. La comparación archivada es
`pcie-effect-prefix-characterization-comparison.json` y conserva esta regresión.

Sigue el [modelo de contexto simbólico](PCIE_SYMBOLIC_CONTEXT_DESIGN.md)
para separar consultas y aplicación mediante registros reales. No repetir
ruteo de esta variante sin segmentar. Se mantienen todos los criterios de
[VIVADO_QUEUE.md](VIVADO_QUEUE.md), incluido el diseño DDR de 512 GiB.
La [limpieza local R26](D_BACKUP_CLEANUP_R26.md) preservó archivos y hashes;
no modificó la VM ni canceló validaciones.
