# Compresión de respaldos históricos en D:

El 2026-10-08 se comprimieron **51 archivos de metadatos** de 17 checkpoints
certificados. Se liberaron **82.554.880 bytes de asignación NTFS (82,55 MB)**
antes de guardar la evidencia. No se eliminó ningún archivo.

## Alcance y conservación

La tanda abarca `INDEX.json`, `BASE_INDEX.json` y
`NEXT_STATE_AFTER_CHECKPOINT.json` entre `20261007T174547Z` y
`20261008T143724Z`, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.

Se conservaron sus contenidos, rutas, fechas de modificación, identidades
y enlaces. El checkpoint más reciente, `20261008T153402Z`, su anexo de
fuentes y la cola operativa quedaron fuera de la compresión. Se excluyó
el directorio `20261007T155805Z` por carecer de certificado de checkpoint.

## Validación

Antes de comprimir se cotejaron los paquetes con sus SHA-256 certificados,
los índices con sus miembros del ZIP y los estados con sus certificados.
PowerShell comprobó las rutas absolutas, el límite temporal, los hashes y
la ausencia de puntos de reanálisis.

La auditoría posterior pasó: los 51 hashes permanecen idénticos y los
metadatos de los **9.685 archivos** inventariados se conservaron. También
se verificaron los paquetes certificados, `AGENTS.md`, `README.md`,
`CMakeLists.txt`, la cola y el último certificado de recuperación.

## Evidencia y reversión

Plan, diario y auditoría tienen el prefijo
`d-cleanup-r25-20261008T154930Z` en `build/cloud-setup/20261002-board/`.
Su paquete de evidencia y `VERIFIED.json` se guardan en
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r25-20261008T154930Z\`.

La compresión es transparente; puede revertirse mediante
`compact /U "ruta exacta"`. No repetir la aplicación de esta tanda.
El espacio libre observado al terminar fue **189,98 GB** y puede variar
por actividad concurrente. Esta operación no accedió a la VM.

Antecedentes: [compresión R22](D_BACKUP_CLEANUP_R22.md) y
[eliminación de cachés R23](D_BACKUP_CLEANUP_R23.md). La eliminación R24
ya estaba registrada en `d-cleanup-r24-20261008T105050Z-summary.json`;
no se repitió.
