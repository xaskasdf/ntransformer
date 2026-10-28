# Limpieza adicional de respaldos en D:

El 2026-10-06 se recuperaron **174.993.408 bytes de asignación NTFS
(174,99 MB)** en dos pasos, conservando todas las rutas y contenidos:

| Operación | Alcance | Espacio recuperado |
| --- | --- | ---: |
| Consolidación por enlaces duros | 28 copias redundantes, 24 contenidos distintos | 136.630.272 bytes |
| Compresión NTFS | 29 índices físicos, sin contar dos veces los enlaces | 38.363.136 bytes |

## Alcance y conservación

Solo se modificó el almacenamiento de `source-*.zip`, `INDEX.json` y
`BASE_INDEX.json` de tandas cerradas anteriores al **6 de octubre,
17:00 UTC**, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.

Los paquetes de resultados, respaldos recientes, fuentes únicas,
instaladores, licencias y datos recuperados se conservaron. La VM y la
cola de Vivado no se modificaron. El espacio libre observado terminó en
**105,76 GB** antes de guardar el pequeño paquete de auditoría.

## Verificación

Las copias se contrastaron mediante SHA-256 y certificados históricos;
se verificaron los manifiestos de fuentes y los ZIP correspondientes.
Cada reemplazo mantuvo un original temporal hasta comprobar las rutas
conservadas. La auditoría confirmó identidad NTFS compartida, hashes
idénticos y ausencia de temporales pendientes.

La compresión conservó hashes, tamaños lógicos, fechas e identidades.
Se comprobaron también los archivos restantes del inventario, los hashes
de los archivos protegidos del repositorio y el estado de la cola.

Los enlaces duros comparten contenido: copiar fuera del respaldo antes
de editar. No representan copias independientes frente a un fallo de D:.
La compresión se puede revertir con `compact /U "ruta exacta"` si hay
espacio disponible.

## Evidencia

Planes, registros, scripts y auditorías:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r15-20261006T192138Z\`.
El paquete incluye `FILES.json` y su certificado `VERIFIED.json`.
Continúa la [limpieza R14](D_BACKUP_CLEANUP_R14.md).
