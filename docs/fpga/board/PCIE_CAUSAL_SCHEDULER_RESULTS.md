# Productor del estado PCIe: resultados iniciales

## Resultado y alcance

El productor `fpga/rtl/ntrpu_pcie_inventory_scheduler.sv` posee su tabla,
bitmap, contadores y contextos. Recibe únicamente RQ/RC y configuración;
la referencia R7 está en el testbench. Se conserva el
[contrato de orden y compromiso](PCIE_CAUSAL_SCHEDULER_CONTRACT.md):
RC antes de RQ, liberación en cabecera terminal y contexto de payload
independiente hasta EOP. No se añaden pausas del observador.

La regresión RTL coincide con la referencia incluso después de fallos.
**La síntesis no cumple 4 ns.** La equivalencia funcional de este banco
no demuestra equivalencia formal ni califica el monitor PCIe completo.

## Bancos y controles

| Tanda del 2026-10-08, UTC | Alcance | Resultado |
| --- | --- | --- |
| `041911Z`, RTL R1 | 22.339 filas, 68 escenarios, 5.718.784 comparaciones de palabras, 89.356 posiciones de consulta | Revisado |
| `042837Z`, síntesis | VU5P `xcvu5p-flva2104-2-i`, reloj 4 ns, I/O 0 ns, sin excepciones | Revisado; setup y hold fallan |
| `043941Z`, RTL binario | 22.330 filas, 67 escenarios, 5.716.480 comparaciones de palabras | Revisado |
| `044305Z`, netlist binario | Mismo banco y netlist propio de `042837Z` | Revisado; salidas idénticas al RTL |

Cada fila compara estado público, contexto retenido y respuestas de
metadatos; el RTL compara además las 256 palabras internas. La secuencia
diagnóstica ejercita wrap de 32 bits. Se mantienen picos de 256 tags,
257 lecturas y 1.052.672 bytes, con RQ2/RC4.

Omitir RQ/RC y corromper tabla/contexto provocó fallos reales en las filas
4, 132, 0 y 204. El revisor rechazó 12 alteraciones de evidencia del banco
completo y 9 de cada banco binario. XSIM emitió `VRFC 10-3149` para el
`force` sobre la tabla: se observó desigualdad en tag 0/fila 0, pero el
log no imprime el valor forzado exacto. Se conserva esa limitación.

El replay mapeado duró 93,305 segundos. Todas las filas de estado público,
contexto, consultas y fallos tempranos coinciden con el RTL binario,
omitiendo solamente el contador de comprobaciones internas de tabla.
No se compiló el productor RTL como DUT mapeado; no se aplicó SDF.
Las dos definiciones `glbl` se cotejaron y tienen cuerpos idénticos.

## Recursos y tiempos medidos

- 43.932 LUT, 9.760 FF y 119 CARRY8; sin latches, BRAM, URAM ni DSP.
- Setup: WNS **−10,451 ns**, TNS −126.328,086 ns; 15.863/28.035 extremos fallan.
- Hold: WHS **−0,075 ns**, THS −104,104 ns; 1.563/28.035 extremos fallan.
- Ruta peor: `allocated_tags_reg[51]` → `remaining_bytes_reg[20]`;
  **60 niveles**, 14,434 ns: lógica 3,152 ns y ruteo estimado 11,282 ns.
- Síntesis: 588,194 segundos. Grafo final: 64.849 primitivas y 68.474 nets,
  sin ciclos combinacionales; ocho controles del auditor rechazados.

Son estimaciones fuera de contexto, sin colocación/ruteo ni retardo del
reloj padre. Permanecen los avisos `TIMING-16` y `XDCH-2`. Las 41 conversiones
`Synth 8-589` impiden interpretar el replay binario como equivalencia X/Z.

## Evidencia y siguiente cambio

Los IDs completos comienzan por `20261008T`, seguidos de la hora de la
tabla y `-V4-ffn-supervised-pcie-causal-scheduler-` más `rtl-r1`, `synth`,
`binary-rtl` o `binary-netlist`. Certificados, fuentes y archivos de cada
tanda están en `artifacts/ntrpu-board-vivado/<ID>/`, respaldados en D: y
NFS. Los revisores y sus diagnósticos se guardan en
`build/cloud-setup/20261002-board/` y se incorporan al checkpoint siguiente.

La siguiente revisión debe reducir la cadena de decisiones y aritmética
del compromiso de estado, preservando los estados posteriores a fallos.
Después se integrarán validación completa y retiro público alineado.
El productor no certifica vacío ni ACK. Siguen abiertos identidad host/BDF,
CDC/reset nativo de 50 ms, DIMM exacto, cuatro DDR/512 GiB, inferencia con
pesos reales e implementación física.
