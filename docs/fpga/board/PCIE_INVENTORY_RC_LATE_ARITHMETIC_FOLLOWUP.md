# Siguiente experimento: suma de bytes RC por carril

R11 produjo **56863 LUT**, **2168 CARRY8** y
WNS **-8.414 ns**. R8 sigue siendo la referencia por WNS:
-7.442 ns y 51937 LUT. La revisión de síntesis
está respaldada. Completar el replay propio y el punto de recuperación R97
antes de ejecutar el siguiente candidato.

## Ruta observada e hipótesis

La ruta peor sigue siendo `rc_user[76]` → `producer/remaining_bytes_reg[20]/D`:
**12.425 ns y 51 niveles**. El reporte atraviesa selección RC,
aritmética, bytes entregados y lógica del contador. Los nombres de nets
orientan la inspección; no prueban por sí solos la causa del retardo.
Los valores son estimaciones OOC, sin colocación/ruteo.

En R8, `rc_removed_span[rc_end_slot[lane]]` se actualiza dentro del
bucle de 16 carriles y después se suman cuatro slots. El siguiente
experimento guardará cada contribución en su carril y la sumará con un
árbol balanceado. Su propósito es estudiar si Vivado evita parte de la
selección encadenada. Puede aumentar área o empeorar timing: medir ambos.
Conservar el selector R3 y la aritmética R8, sin combinar R9–R11.

## Correspondencia y validación

El modelo finito de posiciones pasó 88.741 combinaciones, incluidas
25.696 con duplicados y 18.836 con una clase de puntero desconocido;
rechazó prioridad duplicada incorrecta y alias de puntero desconocido a cero.
La correspondencia inyectiva entre carriles y slots efectivos sustenta
la conservación de los sumandos. El modelo no sustituye simulación RTL.

El [contrato R12](PCIE_INVENTORY_RC_LANE_REDUCTION_CONTRACT.md) exige
comparación exacta de fuentes, prueba del helper real, casos integrados
dirigidos, todos los bancos anteriores y replay de la propia netlist.
No cambiar capacidad, latencia, estados posteriores a fallos ni reset.
Respaldar también los fallos. R12 todavía no está implementado ni calificado.

Decisión vinculada por SHA-256 a `20261008T181604Z-V4-ffn-supervised-pcie-inventory-join-rc-late-arithmetic-synth` en
`build/cloud-setup/20261002-board/pcie-rc-late-arithmetic-next-decision.json`.
La [cola general](VIVADO_QUEUE.md) mantiene pendientes endpoint nativo,
CDC/reset, DIMM exacto, cuatro DDR/512 GiB, inferencia real y diseño físico.
