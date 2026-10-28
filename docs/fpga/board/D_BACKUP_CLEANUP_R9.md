# Consolidación adicional de respaldos antiguos en D:

El 2026-10-06 se consolidaron **63 copias redundantes** de respaldos del
2 al 5 de octubre: 58 índices y cinco ZIP de fuentes. El ahorro lógico
adicional es **94.070.743 bytes (94,07 MB)**. El espacio libre observado
pasó de **105,61 a 105,70 GB**; puede variar por actividad concurrente.

## Alcance y verificación

La operación se limitó a `INDEX.json`, `BASE_INDEX.json` y cinco
`source-*.zip` dentro de
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
Se sustituyeron copias idénticas por enlaces duros NTFS, conservando
todas las rutas y contenidos originales.

Los índices se contrastaron con sus ZIP certificados: SHA-256 del paquete,
CRC completo y hash del miembro en `FILES.json`. Para las fuentes se
comprobó el certificado de una copia canónica y todo su manifiesto.
Esto permitió resolver las cinco copias omitidas en la
[tanda anterior](D_BACKUP_CLEANUP_R8.md), mediante igualdad exacta de
contenido con esa copia certificada, sin modificar certificados históricos.

PowerShell verificó rutas absolutas, rechazó puntos de reanálisis y
conservó cada original temporalmente hasta comprobar el reemplazo.
La auditoría posterior pasó: hashes e identidad compartida correctos,
sin temporales, y tamaños y fechas de otros **4.078 archivos** intactos.
La cola y los archivos protegidos del repositorio conservaron sus hashes.

## Conservación y evidencia

Los archivos enlazados son inmutables; copiarlos fuera del respaldo antes
de editarlos. Comparten almacenamiento y no son copias independientes
ante un fallo de D:. Permanecen los paquetes comprimidos de los índices.

Los resultados de simulación, respaldos del día, datos de migración,
instaladores y licencias quedaron fuera del alcance. No se modificó la VM.

Plan, registro, scripts y auditoría:
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r11-20261006T154048Z\`.
El paquete incluye `FILES.json` para comprobar cada miembro.
