# Limpieza de cachés locales de fuentes: R2

El 2026-10-09 se retiraron **27 ZIP redundantes** del directorio
`build/cloud-setup/20261002-board/`. Se liberaron **999.829.504 bytes de
asignación NTFS** en C:. Los respaldos de D: y NFS permanecen intactos.

## Alcance y comprobaciones

Solo se consideraron fuentes `source-20261009T*.zip` anteriores a las
14:33 UTC, pertenecientes a tandas cerradas del checkpoint 931.
Cada archivo tenía una copia certificada bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\` y otra en NFS.
Se verificaron los tres hashes, todos los miembros del manifiesto de
fuentes y la pertenencia del certificado al checkpoint.

El aplicador PowerShell volvió a comprobar las rutas absolutas, el límite
de fecha, la ausencia de puntos de reanálisis y los hashes. Eliminó archivos
individuales mediante `Remove-Item -LiteralPath`; no realizó borrados
recursivos. La auditoría posterior dio `PASS`.

La fuente activa `source-20261009T170025Z.zip`, las fuentes recientes,
la cola R119, `AGENTS.md`, `README.md`, `CMakeLists.txt` y `VIVADO_QUEUE.md`
conservaron sus hashes. El espacio libre observado quedó en
1.296.723.968 bytes; puede variar por actividad concurrente.

## Evidencia y restauración

Plan, diario y auditoría: `closed-source-cache-r2-20261009T171401Z`.
El paquete de evidencia se conserva en `cleanup-reviews/` tanto en D:
como en NFS. Para restaurar una caché, buscar su entrada en el plan,
verificar el SHA-256 de `retained` y copiarla a `path`. No repetir el
aplicador: rechaza un diario existente.

Esta tanda complementa la [limpieza R1](C_SOURCE_CACHE_CLEANUP_R1.md).
