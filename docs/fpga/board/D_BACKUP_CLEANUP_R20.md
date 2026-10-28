# Limpieza adicional de respaldos en D:

El 2026-10-06 se liberaron **15.331.328 bytes de asignación NTFS
(15,33 MB, antes del pequeño paquete de auditoría)**. Se revisaron 8.659
archivos de los directorios de respaldo del proyecto.

## Cambios realizados

| Operación | Alcance | Espacio recuperado |
| --- | --- | ---: |
| Enlace duro NTFS | Un índice idéntico entre checkpoints cerrados | 2.166.784 bytes |
| Compresión NTFS | Once archivos; compresión reaplicada a un índice compartido | 13.164.544 bytes |

El alcance quedó limitado a `INDEX.json`, `BASE_INDEX.json` y
`NEXT_STATE_AFTER_CHECKPOINT.json` de checkpoints certificados dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
El checkpoint más reciente incluido fue `20261006T222400Z`.

Se conservaron todas las rutas y contenidos, los paquetes históricos,
fuentes, instaladores, licencias y datos recuperados. Las extracciones
grandes redundantes ya habían sido retiradas en tandas anteriores.

## Verificación

Se comprobaron los certificados, los hashes de sus paquetes y la igualdad
de cada índice con su miembro del ZIP o certificado de estado. PowerShell
validó las rutas absolutas, el límite de fecha, la lista exacta de archivos
y la ausencia de puntos de reanálisis antes de modificar cada archivo.

La auditoría posterior confirmó SHA-256, identidad NTFS del enlace duro,
ausencia de temporales y metadatos intactos de los otros 8.646 archivos.
Los archivos protegidos del repositorio, la cola y los fuentes congelados
de C2H mantuvieron sus hashes. No se accedió a la VM ni se modificaron sus
procesos o la tanda en ejecución.

El espacio libre observado terminó en **104,52 GB** antes de guardar la
evidencia. La actividad concurrente del equipo afecta esa cifra; el ahorro
indicado arriba se calculó por archivo.

## Evidencia y conservación

Inventario, plan, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r20-20261006T223749Z\`.
El paquete incluye `FILES.json`; `VERIFIED.json` registra su SHA-256.

Copiar los archivos enlazados fuera del respaldo antes de editarlos.
Las fechas originales quedan en el plan. La compresión puede revertirse
con `compact /U "ruta exacta"`. Continúa la [tanda R19](D_BACKUP_CLEANUP_R19.md).
