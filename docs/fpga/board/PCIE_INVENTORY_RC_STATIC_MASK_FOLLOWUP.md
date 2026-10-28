# Siguiente experimento: selección tardía de aritmética RC

La síntesis revisada de R10 produce **53671 LUT** y
**1890 CARRY8**, con WNS **-7.789 ns**.
R8 conserva el mejor WNS, −7,442 ns, y usa 51.937 LUT. Los destinos
constantes eliminan los drivers con nombres de bits 5–7 de la máscara,
pero no mejoran área o timing en esta medición. La propia netlist debe
aprobarse y respaldarse antes de cerrar R10 y publicar la cola R96.

## Evidencia y alcance

R10: `rc_user[79]` → `producer/remaining_reg[150][12]/D`, 11,799 ns y 47 niveles.
R8: `rc_user[76]` → `remaining_bytes_reg[20]/D`, 11,453 ns y 45 niveles.
Ambos caminos relacionan la selección RC con el estado de bytes. Los
nombres orientan la inspección; no autorizan excepciones temporales.
Los retardos son estimaciones OOC, sin colocación/ruteo.

Partir de **R8 y selector R3**, conservando `newest` de 32 bits y la
máscara dinámica. No combinar los cambios R9 y R10 con esta primera
medición. Preparar los resultados de la función `rc_prefix_state` para
cada máscara constante y seleccionar el resultado después del cálculo.
Mantener el cuerpo aritmético anterior como helper sin simplificarlo.
Vivado podría compartir lógica de otra forma; comprobar la estructura y
las rutas obtenidas, sin asumir que la escritura del RTL obliga al mapeo.

## Cuatro estados y aceptación

La máscara efectiva debe construirse con `if (mask[bit])`: los bits X/Z
no habilitan una operación. Conservar los 16 casos de la función, signos,
conteos cero desalineados, desbordamientos y propagación de desconocidos
de entradas utilizadas; las cantidades no utilizadas no deben contaminar
el resultado seleccionado.

El banco nuevo `fpga/tb/tb_ntrpu_pcie_rc_prefix_mask_four_state.sv` conserva
íntegramente las 45.056 comprobaciones anteriores, incluidas 6.624 X/Z,
y agrega **2.048 casos**: las 256 máscaras de cuatro estados en ocho
contextos. Hay 1.920 casos con máscara desconocida y 2.000 con alguna
entrada desconocida. **Está preparado, aún no compilado ni simulado.**
Compara con la recurrencia secuencial independiente; no es prueba universal.

Exigir todos los bancos heredados, mutantes y controles de evidencia,
síntesis VU5P/4 ns/RuntimeOptimized sin excepciones, DCP acíclico y replay
de la propia netlist con los tres suplementos completos. No cambiar
capacidad, latencia, RQ, propiedad, estados posteriores a fallos, reset o
empty. Medir recursos, setup/hold y ruta crítica; respaldar también fallos.

Decisión ligada por SHA-256 a la revisión de `20261008T173404Z-V4-ffn-supervised-pcie-inventory-join-rc-static-mask-synth` en
`build/cloud-setup/20261002-board/pcie-rc-static-mask-next-decision.json`.
La nueva aritmética todavía no está implementada ni calificada.
La [cola completa](VIVADO_QUEUE.md) mantiene los pendientes nativos,
DDR/512 GiB, inferencia real y diseño físico/ECAD.
