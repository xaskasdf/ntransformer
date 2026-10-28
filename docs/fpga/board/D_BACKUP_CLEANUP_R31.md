# Limpieza de fuentes históricas en D: — R31

El 2026-10-11 se consolidaron **9 copias físicas redundantes** de siete
ZIP de fuentes. Se liberaron **428.814.336 bytes (428,8 MB)** antes de
guardar la evidencia. La auditoría posterior dio `PASS`; las 16 rutas
implicadas conservan sus tamaños y contenidos originales.

## Alcance

Solo se seleccionaron fuentes de tandas cerradas del 11 de octubre,
anteriores a las **12:00 UTC**, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
Las copias idénticas comparten almacenamiento mediante enlaces duros NTFS.
Se conservaron resultados, fuentes únicas, instaladores, licencias y datos
recuperados.

El checkpoint vigente `20261011T061920Z-ffn-supervised-checkpoint` quedó
excluido; su archivo y certificado conservaron sus hashes. La tanda activa
`20261011T163442Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-full-r1`
continuó ejecutándose y no entró en el inventario de limpieza.

## Verificación

Se comprobaron rutas absolutas, ausencia de puntos de reanálisis,
certificados de cierre y paquetes de resultados. Cada fuente pasó SHA-256,
CRC y validación del manifiesto ZIP; sus siete copias originales en NFS
también se verificaron por hash.

Tras la consolidación se verificaron identidad compartida, hashes,
tamaños y ausencia de temporales. Otros **390 archivos** conservaron sus
metadatos. `AGENTS.md`, `README.md`, `CMakeLists.txt`, `VIVADO_QUEUE.md` y el
seguimiento protegido permanecieron intactos. El espacio libre observado
pasó de **156,94 a 157,37 GB**; otras escrituras pueden cambiarlo.

## Evidencia y recuperación

Plan, diario y auditoría usan el prefijo `d-cleanup-r31-20261011T163845Z`
en `build/cloud-setup/20261002-board/`. El paquete de evidencia y
`VERIFIED.json` se guardan en la carpeta homónima de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\`.
El recibo `-nfs-verified.json` registra la copia y verificación remota.

No repetir el aplicador. Los enlaces duros comparten identidad y
metadatos: para editar una fuente, copiarla primero fuera del respaldo.
Antecedente: [limpieza R30](D_BACKUP_CLEANUP_R30.md).
