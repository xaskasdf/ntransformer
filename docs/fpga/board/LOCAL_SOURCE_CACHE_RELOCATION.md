# Conservación de cachés locales de fuentes

**Limpieza posterior en C: (2026-10-09):** se retiraron otras 30 cachés
redundantes del 8 de octubre, conservando sus fuentes certificadas en D:
y NFS. Se recuperaron aproximadamente 1,00 GB. Véase
[C_SOURCE_CACHE_CLEANUP_R1.md](C_SOURCE_CACHE_CLEANUP_R1.md).

**Actualización:** la [tanda R23](D_BACKUP_CLEANUP_R23.md) retiró las 305
copias trasladadas después de volver a verificar sus copias certificadas.
Estas últimas permanecen en `backups/` y permiten restaurar cualquier
caché. El registro siguiente describe el traslado anterior a esa limpieza.

El 2026-10-08 se trasladaron 305 ZIP antiguos de fuentes desde
`build/cloud-setup/20261002-board/` a
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\source-cache-relocated\20261008T014123Z\`.
Su tamaño total es **2.147.776.515 bytes**. No se eliminaron archivos.
Este traslado recupera espacio en C: y consume espacio en D:; no es ahorro
de capacidad total ni una nueva eliminación de respaldos en D:.

Cada ZIP tenía una copia certificada en D:, conservada en su ubicación
original. Se excluyeron las fuentes desde `source-20261007T185000Z.zip`
y todos los archivos distintos del patrón seleccionado. Las rutas usadas
por los certificados de respaldo no cambiaron.

La auditoría posterior comprobó las 305 entradas del plan y del diario,
la ausencia de las rutas antiguas y los hashes SHA-256 y tamaños tanto de
los archivos trasladados como de sus copias certificadas. Las validaciones
y sus resultados permanecen intactos. El espacio libre observado tras
esa auditoría fue 4,72 GB en C: y 199,08 GB en D:; la actividad concurrente
impide atribuir esas variaciones únicamente al traslado.

Plan, diario y revisión están bajo `build/cloud-setup/20261002-board/`:

- `pcie-rc-event-source-cache-relocation-plan-20261008T014139Z.json`
- `pcie-rc-event-source-cache-relocation-plan-20261008T014139Z.moves.jsonl`
- `pcie-rc-event-source-cache-relocation-20261008T015100Z-review.json`

También se conservaron seis archivos de evidencia y herramientas, con
manifiesto SHA-256, en
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\pcie-rc-event-source-cache-relocation-20261008T015100Z\`.

El diario permite restaurar individualmente un ZIP si una herramienta
antigua requiere su ruta de caché. No repetir el script de traslado:
exige un destino nuevo y las rutas de origen originales.

La limpieza posterior de extracciones en D: está documentada en
[la tanda R27](D_BACKUP_CLEANUP_R27.md).
