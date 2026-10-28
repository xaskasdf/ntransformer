# Limpieza de cachés redundantes en D:

El 2026-10-08 se retiraron **305 ZIP redundantes de fuentes**, liberando
**2.148.409.344 bytes de asignación NTFS (2,15 GB; 2,00 GiB)** antes de
guardar la evidencia. Las 305 copias certificadas permanecen intactas.

## Alcance

Se eliminaron únicamente los archivos del traslado documentado en
[Conservación de cachés locales](LOCAL_SOURCE_CACHE_RELOCATION.md), bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\source-cache-relocated\20261008T014123Z\`.
Su contenido lógico sumaba 2.147.776.515 bytes. La carpeta vacía se conserva.

Cada archivo eliminado tiene una copia idéntica en `backups/<run>/`,
referenciada por el índice certificado del checkpoint `20261008T030140Z`.
Se preservaron ese checkpoint, los paquetes de simulación, las colas,
el candidato R7, los instaladores, las licencias y los datos recuperados.
Esta operación local no accedió a la VM.

## Verificación

Antes de eliminar se verificaron los SHA-256 de ambos archivos de cada
par, sus tamaños, los certificados de fuentes y la lectura completa de
los ZIP conservados mediante CRC. El índice se cotejó con su miembro
del paquete certificado del checkpoint.

PowerShell validó las rutas absolutas, sus directorios exactos, el límite
temporal y la ausencia de puntos de reanálisis. Volvió a comprobar cada
par antes de eliminar su duplicado, sin borrados recursivos.

La auditoría posterior confirmó la ausencia de los 305 duplicados y la
conservación de hashes, tamaños, fechas, identidades y enlaces de las
copias retenidas. También verificó el checkpoint y los archivos protegidos
del repositorio y de la cola.

## Evidencia y recuperación

Plan, diario, auditoría y herramientas están en
`build/cloud-setup/20261002-board/`, con prefijo
`d-cleanup-r23-20261008T031805Z`.
Su respaldo se guarda en
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r23-20261008T031805Z\`.

El plan relaciona cada ruta eliminada (`duplicate`) con su copia conservada
(`retained`) y SHA-256. Para recuperar una caché, copiar ese ZIP desde
`retained` a la ruta requerida y verificar su hash. No repetir el script
de eliminación ni el traslado anterior.

El espacio libre observado tras la auditoría fue **199,92 GB**; puede
variar por actividad concurrente. Continúa la [tanda R22](D_BACKUP_CLEANUP_R22.md).
