# Implementación diagnóstica con BUFGCE interno

## Dependencias y entrada

Ejecutar después de R100 y su checkpoint certificado: auditoría de los DCP,
conectividad/parametrización del reloj y replay de los cinco bancos binarios
R12, con 78.712 filas y 30 controles. Conservar el preflight detenido.

Entrada única: `optimized_clock.dcp` de `20261008T205100Z`, SHA-256
`cbb55388614920f547bf3b4e9cfaac3943e40272c69d0e66eca81fbe8d71d440`.
Parte `xcvu5p-flva2104-2-i`, top `ntrpu_pcie_inventory_join`.

## Ejecución y evidencia

Usar `tools/ntrpu_board_pcie_inventory_clock_placement.tcl` con argumentos
`OPTIMIZED_CLOCK_DCP NEW_OUTPUT`. Carpeta nueva, máximo cuatro hilos y
3.600 segundos. Verificar herramientas activas antes de iniciar; preservar
trabajos ajenos. Ejecutar `place_design` y `route_design` predeterminados.
No repetir síntesis ni `opt_design` ni modificar RTL, latencia o capacidad.

Conservar período de 4 ns, E/S cero e incertidumbre heredada. No añadir
`HD.CLK_SRC`, excepciones, asignaciones de pines ni cambios de severidad.
El BUFGCE real ya está en el DCP; su ubicación la decide el placer.

Guardar ayuda instalada, checkpoint de entrada, estado colocado y estado
final/parcial, EDIF, netlist funcional y restricciones. Registrar ubicación,
parámetros y conexiones del buffer; cargas de reloj; timing global y
FDRE→FDRE para setup/hold; recursos, DRC, metodología, fanout, estados
de ruteo y puertos sin ubicaciones de partición. Preservar y respaldar
errores, límites de reporte y ejecuciones que agoten tiempo.

## Revisión y límites

Separar el tramo externo `clk`→BUFGCE de la distribución interna a FDRE.
Comparar con R1 considerando el contexto de reloj distinto. Un código de
salida cero sólo acredita ejecución; los slacks negativos siguen fallando.

Antes de aprobación funcional, reabrir el DCP exacto, auditar su EDIF/DAG,
restricciones y cambios de primitivas, y reproducir su propia netlist en
los cinco bancos. El replay anterior no valida cambios del placer/router.
La frontera OOC, el endpoint nativo, DDR/512 GiB, reset físico y placa
permanecen pendientes incluso si mejora el timing interno.
