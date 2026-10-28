# Siguiente experimento: destinos constantes de la máscara RC

R9 cambió únicamente `newest` a cinco bits con signo y pasó RTL y su
propia netlist. WNS **-8.652 ns**, frente a
−7,442 ns de R8, con **45541 LUT** y
**354 CARRY8**. R8 conserva el mejor
WNS medido; las métricas OOC no califican la implementación física.
[Resultados](PCIE_INVENTORY_RC_SELECTOR_WIDTH_RESULTS.md).

## Cambio aislado

Partir de R8, sin modificar la anchura seleccionada de `newest`,
latencia, capacidad, propiedad, RQ, aritmética ni prioridades.
Sustituir sólo la escritura dinámica
`header_prefix_mask[header_index*4+event_origin[scan]]=1`
por destinos constantes de headers estrictamente anteriores.
Mantener la rama procedural de origen retenido y el resto de condiciones.
No introducir excepciones temporales.

El candidato R10 ya está preparado en
`fpga/rtl/ntrpu_pcie_inventory_scheduler_r10.sv` y
`fpga/rtl/ntrpu_pcie_rc_update_select_r5.sv`, sin ejecutar ni calificar.
Su [contrato](PCIE_INVENTORY_RC_STATIC_MASK_CONTRACT.md) exige cerrar R9
y sellar R95 antes de crear y lanzar los nuevos bancos congelados.

## Fundamento y límites

El origen es −1 o el último SOP presente en una posición no posterior al
EOP. Si un evento termina antes del header destino, su origen pertenece
a un header anterior. El comprobador
`tools/check_ntrpu_pcie_rc_prefix_origin_bounds.py` agotó las
16 máscaras SOP × 16 posiciones EOP × 4 destinos: **1.024 casos**.
La máscara permitida `0x7310` coincide con la unión de las 155.168 salidas
binarias de referencia; los dos controles negativos fueron rechazados.

El comprobador se ejecutó por separado contra R3 y R4 y liga cada
resultado al selector por hash y fragmentos de fuente.
Es un modelo finito de posiciones decodificadas: no demuestra equivalencia
universal ni califica una reescritura RTL. El resultado elegido para
R8 es `pcie-rc-selector-width-r8-origin-bound.json`.
Los nombres de nets y la presencia de LUT no justifican rutas falsas.

## Criterios de aceptación

Conservar las 1.153.296 comparaciones heredadas y 32.768 conjuntas,
incluidos todos los X/Z y los 840 bits comparados. Ejecutar los bancos
RC/RQ, aritmética, propiedad integrada, principal, mutantes y controles
de evidencia completos. Exigir síntesis VU5P/4 ns/RuntimeOptimized,
auditoría acíclica y replay binario de la propia netlist, sin RTL del DUT.
Comparar recursos, setup/hold y ruta crítica con R7, R8 y R9; respaldar
también fallos antes de decidir una preferencia.

Si persiste el camino aritmético, estudiar selección tardía de casos de
prefijo constantes en una revisión separada. No mezclar experimentos.
La [cola completa](VIVADO_QUEUE.md) mantiene abiertos endpoint nativo,
BDF, CDC, reset de 50 ms, DIMM exacto, cuatro DDR/512 GiB, inferencia
real y diseño físico/ECAD.
