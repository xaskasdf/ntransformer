# Nueva limpieza de respaldos en D:

El 2026-10-06 se retiraron **447 copias extraídas redundantes** de tandas del
2 al 4 de octubre, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\<run>\<run>`.

Se liberaron **46.953.302.761 bytes lógicos, aproximadamente 46,95 GB**.
El espacio libre observado pasó de **8,68 GB a 55,79 GB**; esta medida
también refleja actividad concurrente del equipo.

## Contenido conservado

Permanecen los **447 comprimidos originales**, sus ZIP de fuentes, los
certificados históricos `VERIFIED.json` y los respaldos remotos. También se
conservan las tandas recientes, los proyectos activos, instaladores,
licencias y `recovered_projects.7z`.

Antes de cada eliminación se comprobó el SHA-256 del comprimido local y
remoto, la integridad del ZIP de fuentes y la correspondencia de cada
archivo extraído con el tar y su manifiesto. PowerShell volvió a comprobar
inventario, hashes, límites de la ruta absoluta y ausencia de puntos de
reanálisis antes de retirar la extracción.

La pasada general retiró 440 carpetas. Una revisión adicional resolvió
los enlaces duros internos de otras seis, comprobándolos contra miembros
del mismo archivo ya verificados. La última contenía un caché Python de
13.339 bytes ajeno al manifiesto: se conservó por separado, con hash
verificado, antes de eliminar su carpeta. En total se retiraron 111.056
archivos respaldados y se trasladó ese caché; ningún contenido único se perdió.

## Evidencia y restauración

Cada carpeta exterior conserva `EXTRACTION_REMOVED.json`, con el
comprimido, las fuentes y los hashes necesarios para restaurar el contenido
en una carpeta nueva. Los certificados históricos no se modificaron.

El registro completo, los comprobantes por archivo y el caché preservado
se reúnen en
`artifacts/ntrpu-board-vivado/handoffs/20261006T072059Z-d-cleanup-followup/`.
El paquete contiene `FILES.json` y un certificado de verificación local/NFS.
Los registros de ejecución están en `build/cloud-setup/20261002-board/`,
con prefijos `d-cleanup-r2-20261006T063703Z`,
`d-cleanup-r3-20261006T071803Z` y `d-cleanup-r4-20261006T072051Z`.

Las nuevas tandas mantienen el respaldo comprimido sin crear otra copia
extraída automáticamente.
