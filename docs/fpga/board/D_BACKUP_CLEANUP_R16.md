# Limpieza adicional de respaldos en D:

El 2026-10-06 se liberaron **47.800.320 bytes (47,80 MB)** mediante
compresión NTFS de 54 archivos históricos de simulación bajo
`D:\AI\artifacts\ntrpu-*`. Se conservaron todos los archivos y sus rutas.

## Alcance y resultado

Se comprimieron binarios de XSIM, objetos, bases de depuración y algunos
textos pendientes de tandas antiguas. Los 81.415.740 bytes de contenido
siguen siendo accesibles de forma transparente. Los paquetes ZIP, TAR/GZ
y DCP ya comprimidos se excluyeron de esta operación.

El espacio libre pasó de 105.596.325.888 a **105.644.126.208 bytes
(105,64 GB)**, antes de guardar el pequeño paquete de auditoría.
El ahorro se calculó por asignación NTFS, con clústeres de 4.096 bytes.
Los duplicados detectados durante el inventario no se modificaron.

## Verificación y conservación

Antes y después se comprobaron SHA-256, tamaños, fechas, identidades y
número de enlaces de los 54 archivos. Se validaron rutas absolutas y
ausencia de puntos de reanálisis. Otros 3.553 archivos del mismo árbol
conservaron sus tamaños, fechas, identidades y enlaces.

No cambiaron los hashes de `AGENTS.md`, `CMakeLists.txt`, `README.md`,
`VIVADO_QUEUE.md`, el estado de seguimiento ni la cola R57. No se accedió
a la VM. Los paquetes de resultados y fuentes, respaldos de migración,
licencias, instaladores y datos recuperados quedaron fuera del alcance.

Windows descomprime estos archivos al leerlos. La operación se puede
revertir con `compact /U "ruta exacta"` si hay espacio disponible.

## Evidencia

Inventarios, plan, registro, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r16-20261006T200218Z\`.
El ZIP incluye `FILES.json`; `VERIFIED.json` registra su hash y la
verificación de todos sus miembros. Continúa la [tanda R15](D_BACKUP_CLEANUP_R15.md).
