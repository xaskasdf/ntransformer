# Limpieza adicional de respaldos en D:

El 2026-10-06 se liberaron **167.051.264 bytes (167,05 MB)** mediante
compresión NTFS de 269 archivos físicos antiguos: 97 índices de respaldo
y 172 logs, netlists y otros archivos de texto. No se eliminaron archivos.

## Alcance

- `D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`: únicamente
  `INDEX.json` y `BASE_INDEX.json` de tandas anteriores al 6 de octubre.
- `D:\AI\artifacts\`: textos dentro de carpetas de tandas del 4–5 de octubre.

La selección excluyó archivos ya comprimidos y contó una sola vez cada
identidad NTFS, incluidos los índices que ya compartían enlaces duros.
Los 293.051.805 bytes lógicos seleccionados conservan su contenido y rutas.
Las grandes extracciones redundantes ya se habían retirado en limpiezas
anteriores; los paquetes históricos restantes conservan evidencia.

## Verificación

Se comprobaron rutas absolutas, ausencia de puntos de reanálisis, SHA-256,
tamaños, fechas e identidades antes y después de `compact /C /Q`, aplicado
a cada ruta exacta. D: usa NTFS con clústeres de 4.096 bytes; el ahorro
indicado mide asignación física, descontando duplicados por enlaces duros.

Se verificaron además tamaños, fechas e identidades de otros 7.552 archivos.
Los hashes de `AGENTS.md`, `CMakeLists.txt`, `README.md`, `VIVADO_QUEUE.md`
y del estado de la cola no cambiaron. No se accedió a la VM. Los paquetes
de resultados y fuentes, licencias, instaladores y datos recuperados
quedaron intactos.

Windows descomprime los archivos al leerlos. Se puede revertir con
`compact /U "ruta exacta"`, disponiendo del espacio necesario.

## Evidencia

Plan, registro por archivo, auditoría y script:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r13-20261006T174333Z\`.
El ZIP contiene `FILES.json`; `VERIFIED.json` registra su hash y la
verificación de sus miembros. Continúa la [tanda anterior](D_BACKUP_CLEANUP_R10.md).
