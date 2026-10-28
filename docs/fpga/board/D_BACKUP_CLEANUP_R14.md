# Limpieza de respaldos cerrados en D:

El 2026-10-06 se consolidaron **25 copias redundantes**, agrupadas en
21 contenidos idénticos. Se liberaron **110.260.224 bytes de asignación
NTFS (110,26 MB)**; todas las rutas y contenidos originales se conservaron.

## Alcance

Se revisaron `source-*.zip`, `INDEX.json` y `BASE_INDEX.json` bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
La selección excluyó las tandas desde el **6 de octubre a las 12:00 UTC**
y cualquier ejecución activa. Las copias consolidadas corresponden a
tandas cerradas de esa mañana, pendientes de las limpiezas anteriores.

Se sustituyeron copias físicas idénticas por enlaces duros NTFS.
Los paquetes de resultados, fuentes únicas, respaldos recientes,
instaladores, licencias y datos recuperados permanecen disponibles.

## Verificación

Antes de cada sustitución se verificaron rutas absolutas, ausencia de
puntos de reanálisis, tamaños y SHA-256. Los contenidos se contrastaron
con certificados y manifiestos históricos: manifiesto completo de fuentes
o miembro del ZIP certificado con comprobación CRC de todo el paquete.
El original temporal se retiró únicamente después de verificar el reemplazo.

La auditoría posterior confirmó identidad NTFS compartida y contenido
idéntico, sin temporales pendientes. Los otros **4.244 archivos** de respaldo
conservaron sus tamaños y fechas de modificación; `AGENTS.md`,
`CMakeLists.txt`, `README.md` y el estado de la cola conservaron sus hashes.
No se accedió a la VM.

El espacio libre observado pasó de **105,61 a 105,72 GB**; la variación
también refleja actividad concurrente y metadatos del sistema de archivos.

## Conservación y evidencia

Los archivos enlazados deben tratarse como inmutables. Para editarlos,
copiarlos primero fuera del respaldo: todas sus rutas comparten contenido
y no constituyen copias independientes frente a un fallo de D:.

Plan, registro, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r14-20261006T183110Z\`.
El ZIP contiene `FILES.json`; `VERIFIED.json` permite verificar el paquete.
Continúa la [limpieza anterior](D_BACKUP_CLEANUP_R11.md).
