# Origen del reloj para integrar el inventario PCIe

Revisión del 2026-10-08, sin nueva ejecución de Vivado ni cambios de IP.
Complementa el [diagnóstico de reloj aislado](PCIE_INVENTORY_CLOCK_BOUNDARY_NOTES.md).

## Conectividad comprobada

En el DCP XDMA sintetizado y en su reapertura posterior con ocho salidas
pasivas de número de bus, `inst/pcie4_ip_i/user_clk` tiene un único driver
primitivo: **BUFG_GT**, salida de
`inst/pcie4_ip_i/inst/ntrpu_xdma_pcie4_ip_gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O`.
La misma red llega al puerto superior `axi_aclk`.

Las 44.431 filas de extremos coinciden en ambas evidencias: un driver,
un puerto de salida y 44.429 entradas de primitivas. Estas entradas se
distribuyen en 37.905 pines `C`, 6.416 `CLK`, 54 `CLKARDCLK` y 54
`CLKBWRCLK`. No son 44.429 registros distintos. Todavía no incluyen
las cargas del monitor integrado.

Se verificaron ambos archivos de respaldo, sus reportes, los bytes de
los DCP y el tipo del driver en el inventario de celdas. La conexión
permanece tras añadir las salidas pasivas y reabrir el checkpoint.
Los XDC guardados contienen el reloj OOC de 4 ns en `axi_aclk` y
restricciones del proveedor sobre el divisor de `bufg_gt_userclk`.
Esto no califica las relaciones de relojes generados ni fija una ubicación
física del buffer para la integración.

## Consecuencia para la siguiente implementación

El BUFGCE real del ensayo aislado de R12 permite estudiar su distribución
interna. No representa automáticamente el árbol de reloj del endpoint.
Al integrar, conectar el monitor a la red de usuario de XDMA y revisar
el árbol resultante; no trasladar el BUFGCE del diagnóstico ni encadenar
otro buffer sin una decisión explícita y validación del diseño padre.

La siguiente tanda de integración debe capturar driver, ubicación, ruta,
relojes heredados/generados, divisores y relaciones temporales con RQ/RC.
Conservar las restricciones del proveedor. Cualquier contexto OOC derivado,
incluido `HD.CLK_SRC`, requiere evidencia de ese padre. No usar el sitio
del BUFGCE aislado como sustituto. Estas comprobaciones pertenecen a la
integración ya pendiente en [VIVADO_QUEUE.md](VIVADO_QUEUE.md); no requieren
repetir la síntesis histórica sólo para volver a identificar el driver.

## Evidencia y límites

Revisión local:
`build/cloud-setup/20261002-board/xdma-inventory-clock-origin-review-20261008T221822Z.json`,
SHA-256 `d3058dfea93e408e522d89788832b6b65f566e745bb2c2943a964ea2fd7fd10a`.
Vincula las tandas `20261007T161209Z` y `20261007T165658Z`, con archivos
SHA-256 `215b7b9a37c1c7b50239b3e2a5bfa54583bd83220567359f15708316b5f30a93`
y `b5ebc000c4da9752fda4d2d2cb05eab0b71ecb97573e541d4f1ee8d20b397b42`.

No prueba muestreo del monitor, cierre temporal, comportamiento nativo,
reset, CDC, DDR de 512 GiB ni una frecuencia física de placa.
