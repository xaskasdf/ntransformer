# Limpieza local de respaldos: R26

El 2026-10-09 se comprimieron con NTFS **36 archivos de índices y estados**
de 12 checkpoints históricos. Se recuperaron **67.768.320 bytes (67,77 MB)**
de asignación antes de guardar la evidencia. No se eliminaron archivos.

## Alcance

Se procesaron `INDEX.json`, `BASE_INDEX.json` y
`NEXT_STATE_AFTER_CHECKPOINT.json` entre `20261008T153402Z` y
`20261009T093903Z`, dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
La compresión conserva las rutas y es transparente para los lectores.

El checkpoint `20261009T101647Z`, los archivos ZIP/tar.gz certificados,
las fuentes originales, las licencias y las validaciones en curso quedaron
fuera de la operación. Las tandas de limpieza anteriores no se reaplicaron.

## Verificación y evidencia

El plan cotejó los paquetes con sus hashes certificados, los índices con
los miembros de sus ZIP y los estados con sus certificados. PowerShell
verificó rutas absolutas, ámbito, fechas límite y ausencia de puntos de
reanálisis antes de comprimir cada archivo.

La auditoría posterior confirmó los 36 hashes, tamaños, fechas, identidades
y enlaces; comprobó también los metadatos de los **10.017 archivos** del
inventario previo y la identidad de los paquetes certificados. `AGENTS.md`,
`README.md`, `CMakeLists.txt`, `VIVADO_QUEUE.md` y el certificado del último
checkpoint conservaron sus hashes. La limpieza no accedió a la VM.

Plan, diario y auditoría: prefijo `d-cleanup-r26-20261009T104630Z` en
`build/cloud-setup/20261002-board/`. La copia de evidencia y su certificado
se guardan bajo `D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\`
con ese mismo identificador. Para descomprimir un archivo concreto, usar
`compact /U "ruta exacta"`; no repetir el aplicador de esta tanda.

Antecedente: [compresión R25](D_BACKUP_CLEANUP_R25.md).
