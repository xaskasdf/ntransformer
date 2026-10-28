# Auditoría del buffer de reloj desde DCP

El preflight `20261008T205100Z` insertó un BUFGCE y un VCC, pero terminó
con error al comparar listas de objetos de registros entre dos diseños.
Los inventarios exportados contienen los mismos 10.723 nombres de FDRE;
no hay registros añadidos o eliminados respecto de `optimized.dcp` de R1.
La causa exacta de la desigualdad Tcl queda por aclarar. Ese resultado
autoriza sólo una lectura de los checkpoints conservados.

## Lectura y evidencia

Usar `tools/ntrpu_board_pcie_inventory_clock_audit.tcl` con
`REFERENCE_DCP CANDIDATE_DCP NEW_OUTPUT`. No ejecutar optimización,
colocación, ruteo ni cambios de propiedades. Registrar los hashes de
ambos DCP y exportar EDIF, inventarios, puertos, restricciones, reloj y
conectividad. Materializar `get_property NAME` antes de cerrar cada
diseño; conservar también el resultado de comparar los objetos crudos.

Exigir nombres de registros y puertos idénticos. Verificar un BUFGCE:
I desde `clk`, CE desde VCC, O exclusivamente a todos los C de FDRE.
Comparar las restricciones antes y después de auditar. Conservar
período de 4 ns, E/S cero y ausencia de excepciones; la candidata no
debe heredar el origen de reloj externo supuesto de R1.

## Auditoría lógica y simulación

Auditar el DAG completo tratando BUFGCE conservadoramente como lógica
combinacional. Revisar inversión, habilitación, modo CE y conexiones
con controles negativos. Los atributos omitidos exigen cotejo con la
biblioteca instalada, que se preserva por hash, y con la netlist
funcional exportada. La guía
[UG974 2026.1](https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/BUFGCE)
describe CE, inversión y sus valores predeterminados; registrar cualquier
diferencia entre guía, biblioteca y parámetros realmente usados.

Revisar diferencias de conectividad y parámetros respecto de R1
optimizado, no sólo conteos. Sólo después habilitar replay de los cinco
bancos binarios completos R12 y sus controles. Un replay aprobado
habilita preparar una implementación separada; no demuestra timing,
frontera de placa, endpoint nativo ni DDR/512 GiB. Conservar y respaldar
todas las tandas, incluido el preflight detenido.
