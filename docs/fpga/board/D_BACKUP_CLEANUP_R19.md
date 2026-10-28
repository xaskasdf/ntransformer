# Limpieza adicional de respaldos en D:

El 2026-10-06 se liberaron **10.752.000 bytes de asignación NTFS
(10,75 MB)**. La revisión de 8.363 archivos encontró tres índices
duplicados adicionales entre los checkpoints cerrados de las 19:50,
20:25, 21:07 y 21:40 UTC.

| Operación | Alcance | Espacio recuperado |
| --- | --- | ---: |
| Enlaces duros NTFS | Tres copias de índices idénticos | 6.447.104 bytes |
| Compresión NTFS | Tres índices físicos | 4.304.896 bytes |

Se conservaron todas las rutas y contenidos. Los paquetes de resultados
y fuentes, respaldos de migración, instaladores, licencias y datos
recuperados permanecen disponibles. Las grandes extracciones redundantes
ya habían sido retiradas en tandas anteriores; no se eliminaron respaldos
únicos para aumentar el ahorro.

## Verificación

Se validaron los cuatro paquetes de checkpoint contra sus certificados,
todos sus miembros contra `FILES.json` y cada índice mediante SHA-256.
Antes de modificar archivos se comprobaron rutas absolutas dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`, sin puntos de
reanálisis. Cada reemplazo conservó el original hasta verificar su copia.

La auditoría posterior confirmó identidades NTFS compartidas, ausencia de
temporales, hashes íntegros y conservación de los paquetes certificados.
`AGENTS.md`, `README.md`, `CMakeLists.txt` y `VIVADO_QUEUE.md` conservaron
sus hashes. No se accedió a la VM ni se modificó su trabajo.

El espacio libre observado terminó en **105,23 GB**. Su variación incluye
actividad concurrente; el ahorro anterior se calculó por archivo.

## Conservación y evidencia

Los índices enlazados comparten contenido: copiarlos fuera del respaldo
antes de editarlos. Las fechas originales quedan registradas en el plan.
La compresión se puede revertir con `compact /U "ruta exacta"`.

Inventario, plan, registro, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r19-20261006T215623Z\`.
El paquete contiene `FILES.json`; `VERIFIED.json` registra su hash.
Continúa la [tanda R18](D_BACKUP_CLEANUP_R18.md).
