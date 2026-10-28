# Limpieza de metadatos históricos en D:

El 2026-10-07 se comprimieron **21 archivos** de siete checkpoints
certificados. Se recuperaron **29.962.240 bytes de asignación NTFS
(29,96 MB)** antes de guardar la evidencia de esta tanda. No se eliminaron
archivos ni se cambiaron sus contenidos.

## Alcance

La operación abarcó `INDEX.json`, `BASE_INDEX.json` y
`NEXT_STATE_AFTER_CHECKPOINT.json` de checkpoints comprendidos entre
`20261007T132527Z` y `20261007T170635Z`, dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.

Se conservaron sin cambios el checkpoint más reciente (`174547Z`), los
paquetes de simulación y las colas operativas. El directorio `155805Z`
se excluyó porque no contiene `CHECKPOINT_VERIFIED.json`. Las instantáneas
históricas que registran procesos activos no se interpretaron como estado
actual de la VM.

## Verificación

Antes de comprimir se comprobaron los hashes de paquetes y certificados,
los índices frente a sus miembros del ZIP y los certificados de estado.
PowerShell validó las rutas absolutas, la ausencia de puntos de reanálisis,
el límite temporal y los hashes de cada archivo.

La auditoría posterior confirmó los hashes, tamaños, fechas, identidades
y enlaces de los 21 archivos; también comprobó los metadatos de los
**4.694 archivos** inventariados en el directorio de respaldos. Los
paquetes certificados y los archivos protegidos del repositorio permanecen
intactos. No se accedió a la VM ni se modificaron instaladores, licencias
o datos recuperados.

## Evidencia

Plan, registro, auditoría y scripts:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r22-20261007T182105Z\`.
El paquete incluye `FILES.json`; `VERIFIED.json` registra su SHA-256 y
el ahorro neto después de guardar la evidencia.

La compresión es transparente y puede revertirse con
`compact /U "ruta exacta"`. El espacio libre observado después de comprimir
fue **207,06 GB**; la actividad concurrente puede modificar esa cifra.
Continúa la [tanda R21](D_BACKUP_CLEANUP_R21.md).
