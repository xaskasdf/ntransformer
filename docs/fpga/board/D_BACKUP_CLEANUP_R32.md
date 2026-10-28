# Limpieza de fuentes históricas en D: — R32

El 2026-10-11 se consolidaron **cuatro copias físicas redundantes** de
tres ZIP de fuentes. Se liberaron **192.208.896 bytes (192,2 MB)** de
asignación NTFS antes de guardar la evidencia. La auditoría dio `PASS`;
las siete rutas implicadas conservan sus tamaños y contenidos originales.

## Alcance

Solo se seleccionaron fuentes de tandas cerradas del 11 de octubre,
desde las **12:00 hasta antes de las 18:00 UTC**, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
Las copias idénticas comparten almacenamiento mediante enlaces duros NTFS.
Se preservaron fuentes únicas, resultados, instaladores, licencias y datos
recuperados. No se repitió la [limpieza R31](D_BACKUP_CLEANUP_R31.md).

El checkpoint vigente `20261011T061920Z-ffn-supervised-checkpoint`
quedó excluido y conservó sus hashes. La tanda activa
`20261011T201441Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-winner-focus-r1`
quedó fuera del inventario y continuó ejecutándose.

## Verificación

Antes de consolidar se comprobaron rutas absolutas, ausencia de puntos
de reanálisis, certificados de cierre, paquetes de resultados y SHA-256.
Las fuentes pasaron CRC y validación del manifiesto ZIP; también se
recalcularon los hashes de sus tres copias originales en NFS.

Después se verificaron identidad compartida, hashes, tamaños y ausencia
de temporales. Otros **81 archivos** conservaron sus metadatos. Los
archivos protegidos, incluidos `AGENTS.md` y `VIVADO_QUEUE.md`, permanecieron
intactos. El espacio libre observado pasó de **153,20 a 153,39 GB**;
otras escrituras pueden cambiarlo.

## Evidencia y recuperación

Plan, diario y auditoría: prefijo `d-cleanup-r32-20261011T201531Z`
en `build/cloud-setup/20261002-board/`. El paquete de evidencia y
`VERIFIED.json` se conservan en la carpeta homónima de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\`.
El recibo `-nfs-verified.json` registra la verificación remota.

No repetir el aplicador. Los enlaces duros comparten identidad y
metadatos: para editar una fuente, copiarla primero fuera del respaldo.
