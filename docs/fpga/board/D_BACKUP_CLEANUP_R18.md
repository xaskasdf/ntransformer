# Limpieza adicional de respaldos en D:

El 2026-10-06 se recuperaron **29.032.448 bytes de asignación NTFS
(29,03 MB)** al consolidar 12 copias físicas redundantes en 9 grupos.
Se conservaron todas las rutas y contenidos.

## Alcance

Se revisaron 8.343 archivos en los directorios de respaldo del proyecto.
Las sustituciones se limitaron a checkpoints `.dcp`, netlists `.v` y logs
de tandas del 4–5 de octubre en `D:\AI\artifacts\`, dentro de:

- `ntrpu-candidate-route`
- `ntrpu-registered-dispatch`
- `ntrpu-multibuffer-elastic`
- `ntrpu-candidate-dispatch`

Los duplicados comparten ahora un archivo mediante enlaces duros NTFS.
Las extracciones grandes redundantes ya se habían retirado en tandas
anteriores. Los paquetes históricos, respaldos recientes, instaladores,
licencias y datos recuperados permanecen disponibles.

## Verificación

Se comprobaron rutas absolutas sin puntos de reanálisis, tamaños y SHA-256
antes y después de cada sustitución. Los nueve contenidos coinciden con
miembros de cuatro paquetes históricos certificados, revisados incluyendo
sus ZIP anidados. Una entrada repetida en un ZIP antiguo tenía exactamente
el mismo contenido en ambas apariciones; quedó registrada en el plan.

Cada original temporal se retiró después de verificar el reemplazo.
La auditoría confirmó identidades NTFS compartidas, ausencia de temporales
y conservación de metadatos de los otros 3.868 archivos del árbol.
Los archivos protegidos del repositorio y la cola R59 conservaron sus
hashes. No se accedió a la VM ni se modificaron sus validaciones.

El espacio libre pasó de 105.389.477.888 a **105.418.510.336 bytes
(105,42 GB)** antes del pequeño paquete de auditoría.

## Conservación y evidencia

Los archivos enlazados son inmutables: copiarlos fuera del respaldo antes
de editarlos. Sus fechas pasan a ser las del archivo compartido; las fechas
originales se conservan en el plan. No representan copias independientes
ante un fallo del disco.

Plan, registro, inventario, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r18-20261006T211702Z\`.
El paquete incluye `FILES.json`; `VERIFIED.json` registra su SHA-256.
Continúa la [tanda R17](D_BACKUP_CLEANUP_R17.md).
