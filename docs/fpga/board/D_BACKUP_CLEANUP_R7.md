# Limpieza de instantáneas históricas en D:

El 2026-10-06 se retiraron **2.469 archivos sueltos redundantes** de
14 instantáneas del 3 y 4 de octubre. Suman **209.681.445 bytes lógicos
(209,68 MB)**, adicionales a las [limpiezas anteriores](D_BACKUP_CLEANUP_R6.md).
El espacio libre observado pasó de **105,38 a 105,57 GB**; esta medida
también refleja la asignación de archivos y la actividad concurrente.

## Alcance

Solo se procesaron carpetas históricas `2026100[34]T*-pcie-in-progress`
y `2026100[34]T*-v7-in-progress` bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
El nombre «in-progress» describe el momento de aquella captura; sus
certificados históricos mantienen ese estado y no se consideran pruebas
terminadas por esta limpieza.

Se conservaron todos los ZIP, fuentes, manifiestos, índices, certificados
y README. Los respaldos recientes, proyectos y validaciones actuales,
instaladores, licencias, respaldos iniciales de migración y
`recovered_projects.7z` quedaron fuera del alcance.

## Verificación y recuperación

Antes de eliminar se comprobaron los SHA-256 de cada ZIP local y remoto
contra su certificado. Se leyeron todos sus miembros y se compararon con
`FILES.json` y con las copias sueltas. PowerShell verificó nuevamente las
rutas absolutas, tamaños y hashes; rechazó directorios y puntos de
reanálisis y eliminó únicamente archivos enumerados. Después se
revalidaron los archivos conservados.

Cada carpeta conserva `CHECKPOINT_COPIES_REMOVED.json`, que identifica
su ZIP y SHA-256. Para recuperar los archivos sueltos, comprobar el hash
y extraer ese ZIP en una carpeta nueva. Los enlaces históricos a copias
sueltas pueden requerir esa restauración.

Los planes, comprobantes y registros usan el prefijo
`build/cloud-setup/20261002-board/d-cleanup-r9-20261006T141438Z`.
La evidencia consolidada, con manifiesto y certificado local/NFS, queda
en `artifacts/ntrpu-board-vivado/handoffs/20261006T141438Z-d-historical-snapshot-cleanup-review/`.
