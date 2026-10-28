# Limpieza adicional de respaldos del 5 de octubre

El 2026-10-06 se retiraron **57 copias extraídas redundantes** de tandas del
5 de octubre, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\<run>\<run>`.

Se retiraron **16.997 archivos**, equivalentes a **7.931.280.516 bytes
lógicos (7,93 GB)**. El espacio libre observado en D: pasó de 54,96 a
62,88 GB; la diferencia también refleja actividad concurrente del equipo.
Esta pasada es adicional a la [limpieza anterior](D_BACKUP_CLEANUP_R2.md).

## Verificación y contenido conservado

Antes de cada eliminación se comprobó el SHA-256 del comprimido local y
remoto, la integridad del ZIP de fuentes y la correspondencia de todos los
archivos extraídos con el tar y su manifiesto. Los enlaces duros del tar
se resolvieron contra miembros previamente verificados del mismo archivo.

PowerShell volvió a comprobar hashes, inventario y límites de la ruta
absoluta. Se rechazaron puntos de reanálisis en la extracción y sus
ancestros. No se eliminó ningún comprimido, ZIP de fuentes ni certificado
histórico `VERIFIED.json`. Se conservaron las tandas del 6 de octubre,
los proyectos, instaladores, licencias y datos de recuperación.

## Evidencia y restauración

Cada carpeta exterior conserva `EXTRACTION_REMOVED.json`, que identifica
el comprimido y las fuentes necesarios para restaurar la extracción en
una carpeta nueva. Los registros y pruebas por archivo tienen el prefijo
`build/cloud-setup/20261002-board/d-cleanup-r5-20261006T084504Z`.

La eliminación terminó en las 57 carpetas. El generador original falló
después al construir su resumen por una referencia obsoleta a `previous`.
La revisión independiente reconcilia los 57 comprobantes con el inventario,
vuelve a verificar los hashes conservados y genera
`d-cleanup-r5-20261006T084504Z-independent-summary.json`.

El paquete de evidencia y su certificado local/NFS se guardan en
`artifacts/ntrpu-board-vivado/handoffs/20261006T084504Z-d-cleanup-oct5-review/`.
El ZIP incluye el manifiesto `FILES.json`, los scripts usados, las pruebas
por archivo y los certificados de restauración.
