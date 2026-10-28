# Prefijos de framing y máscaras de payload RC

## Hipótesis R6

R6 combina las máscaras acotadas de [R5](PCIE_RC_EVENT_PAYLOAD.md) con los
prefijos de framing registrados en [R4](PCIE_RC_EVENT_PREFIX.md). El peor
camino de R5 comienza en `end_map_stage_reg[5]`, atraviesa el contador de
DWORD y la acumulación de fallos, y termina en `packet_checked_reg[0]`.
Adelantar los predicados que solo dependen de SOP/EOP podría acortarlo.

Cada DWORD captura su cabecera seleccionada, presencia de inicio, ausencia
de EOP previo, actividad del fragmento nuevo y distancia desde SOP.
`context_seen` y `context_active` siguen leyéndose en el ciclo de
validación: pueden cambiar con el beat inmediatamente anterior.

Se conservan las dos etapas, RC4 por ciclo, wrap de once bits, respuestas
posteriores a fallos, reset frío y manejo de `discontinue`. No se añaden
burbujas ni se reducen capacidad o caudal para mejorar el timing.

## Validación requerida

Los bloques de aritmética y captura son idénticos a sus respectivas
fuentes R5 y R4; sus hashes están en
`pcie-rc-event-combined-composition-r6.json`. Esto no demuestra equivalencia
de la composición. Se exige repetir el banco de cuatro estados de
22.339 filas/68 escenarios, sus dos mutaciones y diez controles.

Después deben pasar las mismas 304.604 filas binarias: 22.330 filas de
referencia, 370 de límites y 281.904 de fragmentación. Los modelos
independientes y la comparación simultánea con R1 renombrado permanecen
intactos. La síntesis usa la misma parte, reloj de 4 ns, E/S explícitas
y ninguna excepción temporal. Se audita el EDIF final y se reproduce
su propio netlist con los tres bancos y quince controles del verificador.

## Estado y criterio de comparación

Validación funcional completada. La referencia de comparación fue R5: 6.295 LUT, 1.219 FF,
99 CARRY8, WNS −0,189 ns y peor camino de 4,171 ns. R4 aislada empeoró
el timing; combinar ambas ideas no garantiza una mejora.

Los bancos RTL de cuatro estados y binarios ya pasaron la revisión
independiente: 22.339 y 304.604 filas respectivamente, con dos mutaciones
reales y diez/quince controles. La síntesis y el netlist propio también
están revisados: las 304.604 filas binarias coinciden con RTL y modelos
independientes; sus quince controles rechazaron las corrupciones introducidas.
R6 es la mejor revisión temporal medida, sin cierre de setup ni hold.

Siguen abiertos la integración completa del inventario, identidad host,
CDC/reset nativo, DIMM exacto, cuatro DDR/512 GiB, inferencia con pesos
reales y cierre físico. Una comprobación local de paquete no certifica
vacío global ni autoriza ACK.

## Síntesis medida

| Métrica | R5 | R6 |
| --- | ---: | ---: |
| LUT | 6.295 | 6.161 |
| FF | 1.219 | 1.302 |
| CARRY8 | 99 | 99 |
| WNS | −0,189 ns | −0,093 ns |
| Peor camino | 4,171 ns | 4,075 ns |
| Niveles lógicos | 15 | 15 |

R6 ahorra 134 LUT y mejora el WNS en 0,096 ns, a coste de 83 FF.
Falla setup en 4 de 3.214 endpoints (TNS −0,343 ns) y hold en 541
(WHS −0,075 ns; THS −31,413 ns). No usa latches, BRAM, URAM ni DSP.
Son estimaciones OOC con la misma parte y restricciones, sin colocación
ni ruteo; no califican la implementación física.

El peor camino ahora va de `context_delivered_reg[2]` a
`packet_checked_reg[2]`: 0,995 ns de lógica y 3,080 ns de ruta estimada.
El EDIF final contiene 8.427 primitivas y 11.383 redes, sin lazos
combinacionales. Pasaron los ocho controles del auditor. Hay seis avisos
de puertos internos constantes en cero, sin avisos críticos; se conservan
los avisos de conversión X/Z y la eliminación del registro de payload
usado solo para detectar X. La reproducción sintetizada es binaria, sin SDF.

## Siguiente experimento encolado

Evaluar un registro de 14 bits para el extremo exclusivo del payload
retenido, cargado en la misma transición de cabecera que sus campos de
contexto. Debe conservar `end = delivered + low[1:0]` tras inicialización,
reset frío, retención y cada cabecera, incluso para estados inválidos.
No adelantar el muestreo de contexto dinámico ni borrar campos retenidos
al encontrar EOP. Exigir los mismos bancos y comparar contra R6.

## Tandas y conservación

| ID | Estado |
| --- | --- |
| `20261008T023036Z-V4-ffn-supervised-pcie-rc-event-combined-rtl` | Cuatro estados y mutaciones revisados y respaldados; 68,056 s de herramienta |
| `20261008T023440Z-V4-ffn-supervised-pcie-rc-event-combined-binary-rtl` | Tres bancos binarios revisados y respaldados; 180,979 s |
| `20261008T024137Z-V4-ffn-supervised-pcie-rc-event-combined-synth` | Síntesis y EDIF revisados y respaldados; 286,177 s |
| `20261008T024744Z-V4-ffn-supervised-pcie-rc-event-combined-binary-netlist` | Tres bancos binarios revisados y respaldados; 467,295 s |

Las fuentes y resultados se conservan por tanda en
`artifacts/ntrpu-board-vivado/<ID>/`, con respaldo físico en D: y NFS.
Los tiempos anteriores corresponden a la herramienta, no a inferencia
en hardware. No se modifican las revisiones o ejecuciones anteriores.

El netlist reproducido tiene SHA-256
`f0cfba38d9f0c53d9d99f9e1a3b72b62b5ad40469cdf83ffb40cadb5c8a78c38`.
Los tres bancos compilaron ese circuito, sin el RTL del DUT; R7 y R1
renombrado suministraron las referencias. Esto no establece equivalencia
general X/Z, timing con SDF ni integración del inventario completo.
