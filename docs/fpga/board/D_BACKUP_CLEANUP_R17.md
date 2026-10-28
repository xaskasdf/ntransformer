# Limpieza adicional de respaldos en D:

El 2026-10-06 se recuperaron **26.013.696 bytes de asignación NTFS
(26,01 MB)**, conservando las rutas y los contenidos de los respaldos.

| Operación | Alcance | Espacio recuperado |
| --- | --- | ---: |
| Consolidación por enlaces duros | 7 índices duplicados | 14.766.080 bytes |
| Compresión NTFS | 8 índices físicos, sin contar dos veces los enlaces | 11.247.616 bytes |

## Alcance y conservación

Se trabajó únicamente con `INDEX.json` y `BASE_INDEX.json` de tandas
cerradas anteriores al **6 de octubre, 20:00 UTC**, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
Los paquetes de resultados y fuentes, instaladores, licencias y datos
recuperados se conservaron. Las extracciones grandes redundantes ya se
habían retirado en las limpiezas anteriores.

El espacio libre observado terminó en **105,57 GB**, antes del pequeño
paquete de auditoría. La variación del espacio libre también refleja
actividad concurrente; el ahorro indicado se calculó por archivo.

## Verificación

Se verificaron rutas absolutas sin puntos de reanálisis, SHA-256,
certificados históricos y miembros de los ZIP correspondientes. Cada
reemplazo conservó una copia temporal hasta verificar los archivos.
La auditoría comprobó identidades NTFS compartidas, ausencia de temporales
y conservación de los demás archivos inventariados.

La compresión preservó hashes, tamaños, fechas e identidades. `AGENTS.md`,
`README.md`, `CMakeLists.txt`, `VIVADO_QUEUE.md`, la cola R58 y su estado
de seguimiento conservaron sus hashes. La consulta SSH fue de lectura:
no había procesos Vivado, XSIM o XELAB en ese momento; no se modificó la VM.

Los enlaces duros comparten contenido: copiar fuera del respaldo antes
de editar. No son copias independientes ante un fallo de D:. La compresión
se puede revertir mediante `compact /U "ruta exacta"` si hay espacio.

## Evidencia

Planes, registros, scripts y auditorías:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r17-20261006T204011Z\`.
El ZIP incluye `FILES.json`; `VERIFIED.json` registra su hash y la
verificación de todos sus miembros. Continúa la [tanda R16](D_BACKUP_CLEANUP_R16.md).
