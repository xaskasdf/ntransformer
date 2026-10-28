# Consolidación de fuentes históricas en D:

El 2026-10-06 se consolidaron **172 copias idénticas de ZIP de fuentes**,
agrupadas en 106 contenidos, de respaldos del 2–5 de octubre. Se liberaron
**657.810.651 bytes lógicos (657,81 MB)**. El espacio libre observado pasó
de **105,14 a 105,80 GB**; la diferencia también refleja asignación del
sistema de archivos y actividad concurrente.

## Alcance

La operación se limitó a `source-*.zip` bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\2026100[2345]T*\`.
Se reemplazaron copias físicas repetidas por **enlaces duros NTFS** al
mismo archivo. Todas las rutas originales siguen disponibles con el mismo
tamaño y SHA-256; no hace falta restaurarlas para leerlas.

Se conservaron sin consolidar cinco copias cuyo certificado local no
identificaba ese contenido. Los resultados, respaldos recientes, datos
de migración, instaladores, licencias y `recovered_projects.7z` quedaron
fuera del alcance. La cola y las validaciones en curso no se modificaron.

## Verificación

Antes de reemplazar cada copia se contrastó su hash con un certificado
histórico y se verificó el ZIP completo, incluidos los hashes de
`SNAPSHOT_FILES.json`. PowerShell comprobó las rutas absolutas y rechazó
puntos de reanálisis. El reemplazo fue atómico y conservó temporalmente
el original hasta verificar ambos archivos supervivientes.

La revisión posterior comprobó todos los hashes, la identidad compartida
de los archivos, los certificados y la ausencia de temporales. También
comprobó tamaños y fechas de modificación de otros 3.945 archivos de
respaldo. `AGENTS.md`, `CMakeLists.txt` y `README.md` conservaron sus hashes.

## Conservación y evidencia

Los ZIP de fuentes son inmutables: editar un archivo enlazado modificaría
el contenido visible por todas sus rutas. Para trabajar sobre una versión,
copiarla a una ubicación nueva. Los enlaces comparten almacenamiento en
D: y no constituyen copias independientes ante un fallo de ese disco.

El plan, registro y revisión usan el prefijo
`build/cloud-setup/20261002-board/d-cleanup-r10-20261006T150139Z`.
La evidencia consolidada queda en
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r10-20261006T150139Z\`.
Esta tanda continúa la [limpieza anterior](D_BACKUP_CLEANUP_R7.md).
