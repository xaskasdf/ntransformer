# Limpieza de cachés locales de fuentes: R1

El 2026-10-09 se retiraron **30 ZIP redundantes** de
`build/cloud-setup/20261002-board/`, todos con el patrón
`source-20261008T*.zip`. Se liberaron **999.751.680 bytes de asignación
NTFS**, aproximadamente 1,00 GB, en C:. Esto no elimina respaldos de D:.

## Copias conservadas y comprobaciones

Cada ZIP conserva una copia certificada bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\` y otra en NFS.
Se verificaron los SHA-256 de las tres copias, el contenido completo del
manifiesto de fuentes y la pertenencia del certificado al checkpoint 920.

El aplicador PowerShell comprobó de nuevo las rutas absolutas, el patrón
de fecha, la ausencia de puntos de reanálisis y los hashes antes de retirar
cada archivo. No realizó eliminaciones recursivas ni modificó la VM.

La auditoría posterior dio `PASS`: las fuentes y certificados conservados,
`AGENTS.md`, `README.md`, `CMakeLists.txt`, `VIVADO_QUEUE.md`, la cola R110,
el certificado del checkpoint y la fuente activa conservaron sus hashes.
La fuente `source-20261009T135718Z.zip` usada por el replay C/E/D quedó
excluida. El espacio libre observado tras la auditoría fue 1.221.091.328
bytes; la actividad concurrente puede cambiarlo.

## Evidencia y restauración

Plan, diario y auditoría usan el identificador
`closed-source-cache-r1-20261009T142354Z` en
`build/cloud-setup/20261002-board/`. Su paquete de evidencia se conserva
bajo `artifacts/ntrpu-board-vivado/cleanup-reviews/` en D:.

Para restaurar una caché que necesite una herramienta antigua, localizar
su entrada en el plan, verificar el SHA-256 de `retained` y copiar ese ZIP
a la ruta `path`. No repetir el aplicador: rechaza un diario existente.

Esta limpieza es posterior al [traslado de cachés anterior](LOCAL_SOURCE_CACHE_RELOCATION.md)
y complementa la [limpieza R27 de extracciones en D:](D_BACKUP_CLEANUP_R27.md).
