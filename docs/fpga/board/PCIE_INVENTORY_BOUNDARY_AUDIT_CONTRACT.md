# Auditoría del DCP implementado del inventario PCIe

Esta tanda de lectura complementa la implementación diagnóstica R1;
no vuelve a optimizar, colocar ni rutear. Exige su archivo terminal
respaldado, el DCP por hash y una auditoría preliminar del EDIF exportado
en la misma implementación. Ese resultado sólo habilita esta lectura.

El auditor inicial del EDIF embebido falló al asumir compresión Zstd
directa; el DCP implementado tiene un encabezado codificado de Xilinx.
Conservar ese fallo. Abrir el DCP exacto con Vivado, exportar un EDIF de
texto y su inventario de primitivas, y repetir la auditoría DAG con sus
ocho controles. Vincular hashes de DCP, archivo, fuentes y exportación;
comparar la conectividad con el EDIF original. No habilitar el replay
con la auditoría preliminar ni inferir conectividad sólo por conteos.

## Evidencia adicional

Comparar los puertos y restricciones del DCP de síntesis R12 con el
DCP implementado. El único contexto temporal añadido permitido es
`HD.CLK_SRC BUFGCE_X0Y60`; mantener reloj de 4 ns, E/S cero y ausencia
de excepciones. Exportar las restricciones otra vez al terminar y
comprobar que la lectura no las cambió.

Registrar todos los estados de ruteo y los objetos implicados en los
DRC. Para rutas incompletas, guardar puertos, drivers, cargas y la
correspondencia con la ruta del reloj padre. Distinguir objetos
jerárquicos de conexiones físicas: no contar alias como cables nuevos.
No degradar DRC ni tratar los gaps del padre como rutas completas.
Exportar `HD.PARTPIN_LOCS` de cada puerto: el aviso `Route 35-198` indica
que, sin ubicaciones de partición, la frontera del módulo carece de
ruteo exportable y el timing hacia/desde esos puertos es incompleto.
No interpretar el límite de mensajes del log como cantidad de puertos
afectados ni presentar esas rutas como interconexión física completa.

Añadir reportes de las peores rutas registro → registro, setup y hold,
con los objetos consultados identificados. Permiten distinguir la
realimentación interna de los caminos que parten de puertos externos.
Son subconjuntos de diagnóstico; conservar el resumen temporal completo
y sus fallos. No cambiar restricciones para producir estos reportes.
Registrar las 30 redes de mayor fanout, sus cargas y timing, y su
distribución entre SLR en un reporte separado. La
[ayuda AMD](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_high_fanout_nets)
documenta esas opciones; no combinar `-load_types` y `-slr`.
Relacionar esos datos con las rutas observadas antes de atribuir una
causa al retardo; un nombre de net o fanout alto no basta para probarla.

## Corrección de la captura de ayuda

El primer intento falló por `redirect`; R1 eliminó ese comando, pero
descartó el valor devuelto por `help`, dejando vacías sus marcas en el
log. Conservar ambos intentos y registrar esa carencia expresamente.
La [referencia AMD 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/help)
indica que `help` devuelve el texto solicitado. Esta auditoría lo guarda
con `puts $archivo [help $comando]`, exige contenido no vacío y registra
la versión instalada. Es evidencia obtenida después de implementar;
no afirmar que los logs anteriores ya contenían esa ayuda.

Capturar las opciones reales de
[`report_route_status`](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_route_status)
y sus indicadores de colocación completa, ruteo parcial/completo y
errores. Guardar también los fallos de la auditoría.

## Puerta siguiente

Revisar archivo, fuentes, restricciones, avisos, DRC y rutas junto con
el DAG del DCP implementado. Sólo entonces congelar el replay de su
netlist exacta contra los cinco bancos binarios de R12. El replay es
funcional, sin SDF; los fallos setup/hold, el reloj padre supuesto,
endpoint nativo, DDR/512 GiB y la implementación de placa siguen siendo
criterios separados. [Cola completa](VIVADO_QUEUE.md).
