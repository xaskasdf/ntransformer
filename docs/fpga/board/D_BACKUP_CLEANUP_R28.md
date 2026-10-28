# Limpieza local de respaldos: R28

El 2026-10-09 se comprimieron con NTFS **15 archivos de metadatos** de
cinco checkpoints cerrados. Se recuperaron **30.375.936 bytes (30,38 MB)**
de asignación antes de guardar la evidencia. No se eliminaron archivos.

## Alcance y verificación

Se procesaron `INDEX.json`, `BASE_INDEX.json` y
`NEXT_STATE_AFTER_CHECKPOINT.json` entre `20261009T101647Z` y
`20261009T132534Z`, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
El checkpoint vigente `20261009T135757Z` quedó fuera de la compresión.

El plan cotejó cada índice con su ZIP certificado y cada estado posterior
con su certificado. El aplicador verificó rutas absolutas, fechas, ámbito,
ausencia de puntos de reanálisis y hashes antes y después de comprimir.
La auditoría posterior dio `PASS`: contenido, tamaño, fecha, identidad y
enlaces de los 15 archivos se conservaron. También se cotejaron los
metadatos de los 5.659 archivos del inventario previo y los hashes de los
paquetes certificados y archivos protegidos del repositorio.

La inspección no encontró más extracciones completas de tandas bajo
`backups/`; la carpeta `extracted/` está vacía. La carpeta histórica
`20261004T084804Z-ffn-supervised-ddr-cold-observation/tools/` se conservó.
No se intervinieron los artefactos de otras líneas de trabajo.

## Evidencia y restauración

Plan, diario y auditoría: prefijo `d-cleanup-r28-20261009T143126Z` en
`build/cloud-setup/20261002-board/`. El paquete y su certificado se guardan
bajo `artifacts/ntrpu-board-vivado/cleanup-reviews/` en D:.

La compresión es transparente; para descomprimir un archivo concreto,
usar `compact /U "ruta exacta"`. No repetir el aplicador.
El espacio libre observado tras la auditoría fue **174,47 GB**; puede
cambiar con las transferencias posteriores. Esta operación no accedió a
la VM ni alteró simulaciones, instaladores, licencias o datos recuperados.

Antecedentes: [extracciones R27](D_BACKUP_CLEANUP_R27.md) y
[cachés de fuentes en C:](C_SOURCE_CACHE_CLEANUP_R1.md).
