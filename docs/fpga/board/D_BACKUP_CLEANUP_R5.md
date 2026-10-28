# Limpieza de copias sueltas de checkpoints antiguos en D:

Limpieza completada el 2026-10-06, entre las 10:46 y las 10:59 UTC.

## Alcance

Esta pasada complementa la [limpieza de extracciones](D_BACKUP_CLEANUP_R4.md).
Se retiraron copias sueltas de 59 checkpoints del 4 y 5 de octubre bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`:
62.370 archivos sueltos, equivalentes a **37.930.188.147 bytes lógicos
(37,93 GB)**. El espacio libre observado en D: pasó de **66,63 a
104,67 GB** (97,48 GiB finales). La diferencia incluye la disposición de
archivos en el disco y puede reflejar actividad concurrente.

Se conservan todos los ZIP de checkpoints, comprimidos de resultados,
ZIP de fuentes, manifiestos `FILES.json`, índices, certificados y README.
También se conservan los checkpoints del 6 de octubre, los respaldos
originales de migración, los instaladores, licencias y `recovered_projects.7z`.

## Verificación

Antes de cada eliminación se verifica el SHA-256 del ZIP local contra su
certificado y su copia remota en NFS. Se leen todos los miembros del ZIP,
se comprueban contra su manifiesto y se comparan los archivos candidatos
con sus copias archivadas. PowerShell vuelve a verificar los hashes y las
rutas absolutas antes de eliminar exclusivamente los archivos enumerados.
Se rechazan directorios y puntos de reanálisis; no se eliminan
carpetas completas. Los archivos conservados se verifican nuevamente.

## Evidencia y restauración

Cada checkpoint procesado conserva `CHECKPOINT_COPIES_REMOVED.json` con
los hashes y la referencia de su ZIP. Para recuperar los archivos sueltos,
comprobar el hash del ZIP indicado y extraerlo en una carpeta nueva. Los
enlaces históricos a archivos sueltos pueden requerir esa restauración.

Los planes, pruebas por archivo y registros usan el prefijo
`build/cloud-setup/20261002-board/d-cleanup-r7-20261006T104653Z`.
La revisión final se guarda, junto con su manifiesto y certificado
local/NFS, en
`artifacts/ntrpu-board-vivado/handoffs/20261006T104653Z-d-checkpoint-cleanup-review/`.
