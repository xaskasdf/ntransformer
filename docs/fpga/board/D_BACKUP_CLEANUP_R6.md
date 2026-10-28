# Limpieza adicional de respaldos antiguos en D:

El 2026-10-06 se compactaron **10 respaldos de entrega del 3 y 4 de
octubre**, posteriores al alcance de la [limpieza anterior](D_BACKUP_CLEANUP_R5.md).
Se retiraron **5.346 archivos sueltos** que ocupaban 2.143.321.256 bytes.
Para mantener cada respaldo recuperable desde D:, se copiaron allí sus
ZIP originales verificados, que suman 364.091.215 bytes.

El ahorro lógico neto es **1,78 GB**. El espacio libre observado pasó de
**104,63 a 106,41 GB**; la diferencia también puede reflejar la asignación
de archivos y actividad concurrente del disco.

## Alcance y comprobaciones

Solo se eliminaron archivos enumerados dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\<fecha>-handoff\`.
Se conservaron todos los archivos comprimidos existentes, fuentes,
manifiestos, índices, certificados y README. Los checkpoints recientes,
respaldos iniciales de migración, instaladores, licencias y
`recovered_projects.7z` quedaron fuera de esta limpieza.

Antes de borrar se comprobaron los SHA-256 de los ZIP originales en C:,
sus copias en D: y sus copias remotas en NFS contra los certificados
históricos. Se leyeron todos los miembros de cada ZIP y se compararon
con su manifiesto y los archivos sueltos. PowerShell volvió a verificar
los hashes y las rutas absolutas; se rechazaron directorios y puntos de
reanálisis. Después se comprobaron nuevamente los archivos conservados.

## Evidencia y restauración

Cada carpeta conserva `HANDOFF_COPIES_REMOVED.json` con la ruta del ZIP y
su SHA-256. Para restaurar el contenido suelto, comprobar ese hash y extraer
el ZIP en una carpeta nueva. Los enlaces históricos a archivos sueltos
pueden requerir esa extracción.

El plan y los registros usan el prefijo
`build/cloud-setup/20261002-board/d-cleanup-r8-20261006T112932Z`.
La evidencia consolidada y su certificado local/NFS se guardan en
`artifacts/ntrpu-board-vivado/handoffs/20261006T112932Z-d-handoff-cleanup-review/`.
