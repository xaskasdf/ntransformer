# Compresión del respaldo original en D:

El 2026-10-06 se recuperaron **1.048.780.800 bytes de almacenamiento
(1,05 GB)** mediante compresión NTFS de tres archivos del respaldo del
1 de octubre. No se eliminó ningún archivo ni se cambió su formato.

## Alcance y resultado

Carpeta: `D:\AI\ntrpu-backups\20261001T164202Z\`.

| Archivo | Tamaño lógico conservado | Espacio tras compresión |
| --- | ---: | ---: |
| `local-repository.tar` | 5.294.919.680 bytes | 4.251.332.608 bytes |
| `local-files.json` | 6.542.168 bytes | 2.822.144 bytes |
| `remote-files.json` | 2.359.152 bytes | 888.832 bytes |

El ahorro incluye el redondeo de los archivos inicialmente sin comprimir
a clústeres NTFS de 4.096 bytes. El espacio libre observado al finalizar
era **106,28 GB**; puede variar por actividad concurrente.

## Verificación y uso

Antes y después se verificó el SHA-256 del TAR contra el certificado
histórico, y sus **25.718 entradas**, incluidos tamaños y hashes de los
23.771 archivos, contra `local-files.json`. Se conservaron las rutas,
contenidos, fechas de modificación e identidades originales. Los otros
14 archivos del respaldo, los archivos protegidos del repositorio y la
cola de Vivado conservaron sus hashes.

La compresión se aplicó con `compact /C /Q` a rutas exactas, comprobando
que no hubiera puntos de reanálisis. No se activó compresión recursiva.
Windows descomprime al leer; las instrucciones de `RESTORE.md` siguen
siendo válidas. Para revertirla, ejecutar `compact /U` con la ruta exacta
del archivo, disponiendo del espacio necesario.

La revisión adicional encontró unos 37 MB de archivos idénticos en
entregas antiguas de `D:\AI\artifacts`; se conservaron independientes.
No se modificaron la VM, las validaciones, los instaladores, las licencias
ni `recovered_projects.7z`.

## Evidencia

Plan, scripts, registros de `compact`, auditoría y manifiesto de hashes:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r12-20261006T162643Z\`.
El ZIP de evidencia incluye `FILES.json`; `VERIFIED.json` registra su
SHA-256 y la verificación de todos sus miembros. Esta tanda continúa la
[consolidación anterior](D_BACKUP_CLEANUP_R9.md).
