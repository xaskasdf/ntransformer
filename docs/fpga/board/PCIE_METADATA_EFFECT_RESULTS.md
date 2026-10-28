# Unidad de efectos de metadatos: resultados

Revisado el 2026-10-09. La unidad de 57 bits pasa el banco RTL y el replay
de su propia netlist. **No es el pipeline completo**; R12 sigue como
referencia del monitor. Contrato: [formato y semántica](PCIE_METADATA_EFFECT_UNIT_CONTRACT.md).

## Comportamiento comprobado

Composición ordenada anterior→posterior, reemplazo de los 30 bits de
metadatos, suma de counts limitada a 2049 y separación entre cero, uno
y varios consumos. Un flanco registra el efecto compuesto y su aplicación;
las burbujas retienen datos y reset frío domina.

El banco pasa **17.114 casos**, incluidas 4.308 inyecciones X/Z activas
y enmascaradas. Compara contra asignaciones secuenciales independientes;
12.352 filas binarias se recalculan además en Python. Comprueba 277 pausas,
176 resets y estabilidad antes del flanco. El DUT con orden invertido falla
en el primer caso dirigido; seis alteraciones de evidencia son rechazadas.

RTL y netlist propia coinciden con la referencia en **12.805 filas binarias**
(25.610 observaciones). Se verifican vectores regenerados, fuentes realmente
compiladas y ausencia de DUT RTL oculto en el replay mapeado. Cuatro controles
rechazan evidencia corrupta. Arranque 200 ns, muestreo 1 ns y sin SDF:
equivalencia finita de este banco, sin afirmación física ni de X/Z en puertas.

## Medidas de síntesis

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, OOC a 4 ns,
retardos I/O cero y ninguna excepción. **220 LUT, 88 FF y 12 CARRY8**;
cero latches, BRAM, URAM y DSP. El DCP contiene 355 primitivas y 639 redes,
con grafo combinacional acíclico y ocho controles de auditoría.

Setup: **WNS +0,377 ns**, TNS 0 y 0/350 endpoints fallidos.
Hold: **WHS −0,042 ns**, THS −3,305 ns y 149/350 endpoints fallidos.
El reloj externo carece de contexto OOC para estimar skew y los DRC son
incompletos por ese modo. No hay colocación/ruteo ni cierre físico; este
margen aislado no se transfiere al monitor completo.

## Diagnósticos y respaldo

El primer replay se detuvo en 14,404 s antes de simular: `set_property`
interpretó `-testplusarg` como opción propia. La revisión R2 usa `-dict`;
conserva RTL, vectores y netlist idénticos. El intento fallido permanece
archivado. También se conservan las revisiones iniciales del lector:
normalización de X/Z y resolución de rutas relativas de compilación.

| Etapa | Tanda | SHA-256 del archivo |
| --- | --- | --- |
| pcie_metadata_effect_rtl | `20261009T095927Z-V4-ffn-supervised-pcie-metadata-effect-rtl` | `3669264e05749b8041da2e0bd434f25c472691c62b7233fc3fe6b495fff8313e` |
| pcie_metadata_effect_synthesis | `20261009T100354Z-V4-ffn-supervised-pcie-metadata-effect-synth` | `80508339b90704baab806b0c15177d60227f23c81c315fc0a9fecde31575e639` |
| pcie_metadata_effect_binary_setup | `20261009T100842Z-V4-ffn-supervised-pcie-metadata-effect-binary` | `f4c07ff101b3a10c889b80d12d9cd2ff5bb6a379e46efca31467c29437f9ce80` |
| pcie_metadata_effect_binary_replay | `20261009T101251Z-V4-ffn-supervised-pcie-metadata-effect-binary-r2` | `4adfc56c0363804b7ea145125f28255ae79cfbfe7ca770fd45e9c51b9dec1b4b` |

Fuentes y todos los miembros están verificados en D:/NFS bajo
`artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`. Revisiones/configuraciones:
`build/cloud-setup/20261002-board/`. R105 enlaza los cuatro resultados, incluido
el diagnóstico. No se modificó `AGENTS.md` ni se repitió la limpieza R25.

Sigue [la integración con tráfico crudo](PCIE_METADATA_EFFECT_INTEGRATION_PLAN.md).
Selección, versiones/wrap, contexto retenido, monitor completo, CDC/reset,
DDR/512 GiB y cómputo siguen sujetos a [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
