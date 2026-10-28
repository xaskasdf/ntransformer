# Limpieza adicional de respaldos en D:

El 2026-10-07 se revisaron **13.853 archivos** de los directorios de
respaldo del proyecto. La limpieza liberó **25.227.264 bytes de asignación
NTFS (25,23 MB)**, antes de guardar la evidencia de esta tanda.

## Cambios realizados

| Operación | Alcance | Espacio recuperado |
| --- | --- | ---: |
| Enlaces duros NTFS | Dos índices idénticos entre checkpoints | 4.354.048 bytes |
| Compresión NTFS | 23 archivos físicos de metadatos | 20.873.216 bytes |

El alcance se limitó a `INDEX.json`, `BASE_INDEX.json` y
`NEXT_STATE_AFTER_CHECKPOINT.json` certificados, dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
El último checkpoint incluido fue `20261006T232611Z`.

Se conservaron todas las rutas y contenidos. Los estados históricos que
registraban procesos activos se trataron como instantáneas certificadas;
no se modificaron las colas operativas ni se interpretaron esos registros
como prueba del estado actual de la VM.

## Verificación

Se validaron los hashes de los paquetes y certificados, la igualdad de
los índices con sus miembros del ZIP y los certificados de estado.
PowerShell comprobó rutas absolutas, ausencia de puntos de reanálisis,
fecha límite y hashes antes de modificar cada archivo.

La auditoría posterior confirmó hashes, identidades NTFS, ausencia de
temporales y metadatos intactos de otros **4.418 archivos** de respaldos
antiguos. Los archivos protegidos del repositorio conservaron sus hashes.
No se accedió a la VM ni se tocaron las tandas recientes de HBM,
instaladores, licencias o datos recuperados.

El espacio libre observado después de la limpieza fue **215,93 GB**.
La actividad concurrente afecta esa cifra; el ahorro atribuido a esta
tanda se calculó por archivo.

## Evidencia y conservación

Inventario, plan, registro, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r21-20261007T130551Z\`.
El paquete contiene `FILES.json`; `VERIFIED.json` registra su SHA-256.

Copiar los índices enlazados fuera del respaldo antes de editarlos.
La compresión puede revertirse con `compact /U "ruta exacta"`.
Continúa la [tanda R20](D_BACKUP_CLEANUP_R20.md).
