# Exportación acotada del inventario de rutas

Fecha: 2026-10-16. **Helper Tcl probado con dobles de comandos; todavía
no integrado ni ejecutado en Vivado.** No ocupa el slot ni modifica la cola.

Actualización 12:16 UTC: la exportación original terminó correctamente
a las 12:14:59, dentro del límite. La revisión aprobó 576.845 filas sin
`gap` explícito y el netlist completo. El inventario ocupa 103.161.109 bytes;
Vivado informó un pico de memoria de 123.283,590 MB. No hizo falta recuperar
informes ni repetir implementación. El helper queda como candidato para
medir y reducir el coste de exportación en una revisión futura.

## Motivo

La tanda `20261016T101133Z-V7-projected-root-bram-vivado-r3` terminó
colocación, ruteo y optimización, pero a las 12:08 UTC continuaba dentro
de la exportación del inventario, con unos 38 GB de RSS. El código consulta
`ROUTE_STATUS` y `ROUTE` de todas las redes antes de escribir sus filas.
La relación causal exacta entre esa consulta y la memoria sigue pendiente
de medición; no se ha interrumpido ni cambiado su límite de una hora.

El checkpoint físico y los informes parciales ya existen. La observación
`projected-root-bram-physical-preliminary-20261016T120632Z.json`, SHA-256
`29029cc47a6bb0410b7cbd550615ff192ed1368839fc68177e1f655b5f2f44d8`,
registra 87.158/87.158 redes ruteables conectadas, cero errores, setup
+0,056 ns y hold +0,010 ns a 5 ns. Recursos: 48.256 LUT, 45.208 registros,
84,5 BRAM y 134 DSP. **Son informes parciales; falta revisión terminal,
inventario completo, prueba del netlist y respaldo de toda la tanda.**

## Variante preparada

`tools/ntrpu_route_inventory_bounded_r1.tcl` exporta las mismas columnas,
en el mismo orden, consultando ambas propiedades en lotes de 64 redes.
Mantiene la búsqueda de `gap` sin distinguir mayúsculas y minúsculas en
cada cadena ROUTE; no omite redes ni sustituye la comprobación por un resumen.
Libera las cadenas del lote y vacía el archivo tras cada lote. Comprueba
cardinalidad, nombres únicos y estados presentes. Usa creación exclusiva
y conserva un archivo parcial cerrado si una consulta falla.

El banco `tests/test_ntrpu_route_inventory_bounded_r1.tcl` compara la salida
completa contra un oráculo escalar independiente: 2.051 redes y lotes de
1, 7, 64 y 4.096. Verifica el tamaño y número de consultas y rechaza once
casos de archivos existentes, límites inválidos, propiedades incompletas,
estados vacíos, fallos de consulta y duplicados. Pasó en Tcl nativo Linux,
CPU 2–3. No demuestra reducción de RSS ni equivalencia con propiedades
extraídas por Vivado.

## Respaldo y siguiente paso

Tanda `20261016T121017Z-route-inventory-bounded-r1`, 17 archivos y
restauración verificada en D:/NFS. ZIP de resultados SHA-256:
`7077aead989fe0d5e4a936944bd90885f525eea13a5a61e2014379943f13c67c`.
Revisión: `build/cloud-setup/20261002-board/route-inventory-bounded-r1-review.json`.

Una integración posterior requiere una revisión nueva y un turno propio.
Si la exportación actual expira, preservar primero su diagnóstico y
checkpoint; recuperar sólo los informes faltantes desde ese checkpoint,
sin repetir aritmética o implementación ni omitir criterios del revisor.
DDR conserva su dependencia de la revisión y respaldo de BRAM.
