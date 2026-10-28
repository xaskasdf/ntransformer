# Limpieza local de respaldos: R27

El 2026-10-09 se retiraron **cuatro extracciones redundantes**, con
**4.504 archivos** y **2.637.475.840 bytes de asignación NTFS (2,64 GB)**.
El contenido lógico eliminado fue de 2.627.776.388 bytes. Los paquetes
comprimidos locales y remotos, las fuentes y los certificados se conservaron.

## Alcance

Solo se eliminaron los subdirectorios `<tanda>/<tanda>` bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`:

| Tanda | Archivos | Asignación liberada |
| --- | ---: | ---: |
| `20261005T121701Z-V7-ffn-supervised-guarded-mig-r2` | 710 | 410.497.024 bytes |
| `20261005T124640Z-V7-ffn-supervised-guarded-native` | 1.477 | 872.501.248 bytes |
| `20261005T134131Z-V7-ffn-supervised-guarded-native-reset` | 1.475 | 998.166.528 bytes |
| `20261006T034802Z-V4-ffn-supervised-xdma-stop-inventory-r5` | 842 | 356.311.040 bytes |

## Verificación

Se cotejaron los SHA-256 de los cuatro TAR/GZ locales y remotos, todos
los miembros de sus manifiestos y cada archivo extraído. Se verificaron
también los ZIP de fuentes y su contenido, sin extraerlos nuevamente.
PowerShell volvió a comprobar las rutas absolutas, la ausencia de puntos
de reanálisis, el inventario exacto y los hashes antes de cada eliminación.

La auditoría posterior dio `PASS`: los paquetes, fuentes, certificados,
`AGENTS.md`, `README.md`, `CMakeLists.txt`, `VIVADO_QUEUE.md`, la cola R108,
el estado de seguimiento y el certificado del checkpoint 915 conservaron
sus hashes. La VM solo recibió consultas de lectura; no se alteraron
procesos ni resultados remotos. El espacio libre observado quedó en
**177,41 GB**, antes del pequeño paquete de evidencia.

## Evidencia y restauración

Plan, diario y auditoría usan el prefijo `d-cleanup-r27-20261009T124217Z`
en `build/cloud-setup/20261002-board/`. Su paquete verificado se guarda
con el mismo identificador bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\`.

Cada carpeta conservada contiene `EXTRACTION_REMOVED.json`. Para restaurar,
verificar el SHA-256 allí indicado y extraer el TAR/GZ en una carpeta nueva.
No repetir el aplicador. Los diagnósticos de preparación se conservan;
ambos ocurrieron antes de cualquier eliminación.

Antecedente: [compresión R26](D_BACKUP_CLEANUP_R26.md).
