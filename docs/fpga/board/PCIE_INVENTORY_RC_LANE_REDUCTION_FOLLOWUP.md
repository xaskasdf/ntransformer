# Siguiente paso: colocación y ruteo diagnóstico

R12 obtuvo **52234 LUT**, **1902 CARRY8** y
WNS **-7.068 ns**; R8 mantiene la comparación de
-7.442 ns. **R12** tiene el mejor WNS OOC entre ambos.
La síntesis está revisada y respaldada; completar el replay propio de
R12 y verificar el punto de recuperación R98 antes del siguiente trabajo.

## Pregunta que resolverá la tanda

Los reportes actuales usan retardos de interconexión estimados. Implementar
el bloque aislado permitirá medir sus rutas internas con colocación real,
sin atribuir esos números a la tarjeta completa. Reutilizar el DCP
`20261008T190613Z-V4-ffn-supervised-pcie-inventory-join-rc-lane-reduction-synth`; no cambiar RTL ni resintetizar dentro de esta comparación.

## Contexto y comprobaciones

El [contrato de implementación](PCIE_INVENTORY_PLACEMENT_DIAGNOSTIC_CONTRACT.md)
conserva el reloj de 4 ns, los retardos de E/S cero y la ausencia de
excepciones temporales. Registrar el origen padre supuesto
`HD.CLK_SRC BUFGCE_X0Y60`, su existencia/tipo y el timing antes y después
de añadirlo. No representa una ubicación aprobada para la placa.

Ejecutar optimización, colocación y ruteo predeterminados; conservar
checkpoints intermedios y fallos. Auditar las rutas internas y los gaps
del reloj padre por separado, manteniendo visibles DRC y setup/hold.
Validar DAG y la netlist exportada con los cinco bancos binarios:
22.330/67, 46.356/3.872, 8.850/240, 544/60 y 632/60.
R12 ya tendrá su banco binario correspondiente al completar la tanda actual.
El replay funcional conserva arranque de 200 ns y muestreo de 1 ns, sin SDF.

Respaldar cada tanda y revisar sus hashes. La [cola general](VIVADO_QUEUE.md)
mantiene abiertos el endpoint nativo, CDC/reset, DIMM exacto, cuatro DDR,
inferencia real, restricciones de placa y ECAD. Esta decisión prepara una
medición; no acredita cierre físico ni una frecuencia de operación.
