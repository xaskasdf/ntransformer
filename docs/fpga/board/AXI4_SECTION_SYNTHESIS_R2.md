# Síntesis y restricciones de la sección AXI4 candidata R2

## Medición integrada

`20261006T015121Z-V4-ffn-supervised-axi4-section-synth-r2`, Vivado 2026.1 sobre `xcvu5p-flva2104-2-i`, sintetiza admisión,
guard, CDC AMD real y certificado con los módulos candidatos R2.
La inspección `20261006T015614Z-V4-ffn-supervised-axi4-section-r2-constraints` abre el mismo DCP y no lo modifica.

| Recurso | Original | Candidato R2 |
| --- | ---: | ---: |
| LUT totales | 10.631 | 10.603 |
| LUT de lógica | 9.813 | 9.785 |
| LUTRAM incluidas | 818 | 818 |
| Flip-flops | 13.771 | 13.771 |
| BRAM / URAM / DSP | 0 / 0 / 0 | 0 / 0 / 0 |

No hay latches ni cajas negras. La diferencia es de 28 LUT: 25 en el guard
y tres en el par de monitores. El CDC conserva 998 LUT y 3.122 FF.
Son recursos de esta sección; no representan el motor completo NTRPU.

## Timing y CDC: alcance conservado

A H/D/gestión=7/5/10 ns, las estimaciones fuera de contexto son
WNS=1,523 ns, WHS=0,013 ns y WPWS=1,968 ns. Faltan retardos externos de
1.438 entradas y 1.587 salidas; READY tiene slack no acotado.
Sus máximos estimados de camino son 1,405 ns de salida y 2,366 ns de entrada.
No se deduce Fmax ni cierre de timing físico.

El informe conserva 38 CDC-3 informativos, diez CDC-9 informativos y
**un CDC-15 nativo sin resolver ni eximir**. Los 25 caminos CDC-3 propios
mantienen tres etapas y los mismos extremos; seis fuentes de alarma de
admisión y doce de monitores salen directamente de registros. El bit 6
de admisión se optimiza a cero: no es un detector dinámico de fallos.

Las excepciones RAM del proveedor conservan su ámbito. Se revisan 93
caminos internos propios con slack finito, mínimo 2,052 ns; diez restricciones
de bus skew tienen mínimo estimado 4,851 ns. Un false path privado elimina
un camino antes acotado; este control se descarta sin escribir el DCP.
No se deben reaplicar globalmente los selectores impresos del proveedor.

Se conservan los límites de informes: supresión tras cien avisos de puerto
sin uso, mil TIMING-18 y comprobaciones de conectividad DRC omitidas en OOC.
La inicialización y los fanouts de reset coinciden con el original
(cold=11.751; host=20; dispositivo=20). No prueba equivalencia de netlists.

## Reproducción y respaldo

Top: `fpga/tb/ntrpu_axi4_section_synth_top_r2.sv`.
Flujo: `tools/ntrpu_board_axi4_section_synth_r2.tcl`; inspección conservada:
`tools/ntrpu_board_axi4_section_constraints_r4.tcl`.

- Archivo de síntesis: `7570d03558540f5a71b561c0d469afc220f9bd87e2a0ba3e21d3be2bece7a811`.
- Archivo de inspección: `da9ed480fd889d7ceed28b87095548259494e7a6a9e9f286c9e7b28b429bef03`.
- DCP revisado: `128038a0308a9ca39408fdb98266b4e0eb5fd0def2d6f6748236302263e4d098`.
- Adenda: `115ef74aac092a8ca0600c770128d148941093c0e187d938cffd725b186888c2`.

Fuentes, bibliotecas instaladas, reportes, netlist y checkpoint se contrastan
por hash y están respaldados en NFS y localmente. La
[corrección del índice inicial](AXI4_SECTION_FAULT_LIFETIMES.md) ya tiene
medición propia. Faltan restricciones de placa, colocación/ruteo, cierre del
CDC-15, inventario XDMA, reset global y DIMM exacto. No autoriza reutilización
de propiedad ni fabricación. Ver [pruebas de solicitudes](AXI4_SECTION_REQUEST_FAULTS.md).
