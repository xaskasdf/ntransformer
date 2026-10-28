# Limpieza de extracciones restantes en D:

El 2026-10-06 se retiraron **21 copias extraídas redundantes**, adicionales
a las [limpiezas anteriores](D_BACKUP_CLEANUP_R3.md): 18 tandas cerradas del
6 de octubre, anteriores a las 04:00 UTC, y tres de las primeras pruebas
del 2 de octubre.

Se eliminaron **5.832 archivos**, equivalentes a **4.874.087.161 bytes
lógicos (4,87 GB)**. El espacio libre observado en D: pasó de **62,27 a
67,03 GB**. Esta diferencia también puede reflejar actividad concurrente
y la disposición de los archivos en el disco.

## Alcance y verificación

Las únicas rutas retiradas fueron extracciones bajo:

- `D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\<run>\<run>`
- `D:\AI\ntransformer\artifacts\ntrpu-board-vivado\extracted\<run>`

Antes de eliminar cada árbol se verificaron el SHA-256 del comprimido
local y remoto, el ZIP de fuentes y la igualdad de cada archivo extraído
con el tar y su manifiesto. Se exigió inventario idéntico, sin archivos
locales adicionales. PowerShell volvió a comprobar los hashes, la ruta
absoluta y la ausencia de puntos de reanálisis antes de eliminarla.

Permanecen todos los comprimidos, fuentes, certificados históricos y
respaldos remotos. Los comprimidos de las tres primeras pruebas siguen
en sus carpetas originales de `artifacts/ntrpu-board-vivado/` en C:.
No se modificaron proyectos activos, instaladores, licencias, respaldos
iniciales de la migración ni `recovered_projects.7z`.

## Registro y restauración

Cada carpeta que conserva el comprimido contiene `EXTRACTION_REMOVED.json`
con sus hashes y la referencia de restauración. Para consultar nuevamente
el contenido, verificar el hash y extraer el tar en una carpeta nueva.
Los enlaces antiguos a las tres extracciones retiradas en D: pueden
seguir presentes en C:; su ausencia de destino no implica pérdida del tar.

Los registros y pruebas por archivo usan el prefijo
`build/cloud-setup/20261002-board/d-cleanup-r6-20261006T095156Z`.
La revisión independiente reconcilia las 21 eliminaciones y vuelve a
verificar los archivos conservados. Su paquete, con manifiesto y
certificado de verificación local/NFS, se guarda en
`artifacts/ntrpu-board-vivado/handoffs/20261006T095156Z-d-cleanup-residual-review/`.
