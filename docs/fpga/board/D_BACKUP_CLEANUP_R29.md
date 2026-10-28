# Limpieza de fuentes históricas en D: — R29

El 2026-10-09 se consolidaron **42 copias físicas redundantes** de
**17 ZIP de fuentes**, recuperando **1.315.201.024 bytes (1,315 GB)**
de asignación NTFS antes de guardar la evidencia. La auditoría dio `PASS`.
Todas las rutas originales y su contenido siguen disponibles.

## Alcance

Se seleccionaron únicamente `source-*.zip` de tandas anteriores al
**9 de octubre de 2026 a las 00:00 UTC**, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
Las copias idénticas comparten ahora almacenamiento mediante enlaces
duros NTFS. Se conservaron dos archivos asociados a checkpoints sin
certificación de cierre completo.

Los paquetes de resultados, certificados, fuentes únicas, tandas de hoy,
instaladores, licencias y datos recuperados permanecen intactos.
La cola y el checkpoint vigente de **936 tandas respaldadas** conservaron
sus hashes. No se modificó la VM ni se lanzaron o interrumpieron procesos.

## Verificación

La preparación cotejó SHA-256, certificados locales, paquetes de resultados,
manifiestos completos y CRC de los ZIP. También recalculó los hashes de
las 17 fuentes en NFS. El aplicador PowerShell validó rutas absolutas,
ausencia de puntos de reanálisis y hashes antes de cada sustitución;
retiró el original temporal después de comprobar el reemplazo.

La auditoría confirmó los enlaces, tamaños y hashes de las 59 rutas,
la ausencia de temporales y los metadatos de los otros **5.846 archivos**.
El espacio libre observado pasó de **167,48 a 168,79 GB**.

## Evidencia y conservación

Plan, diario, auditoría y scripts: prefijo
`d-cleanup-r29-20261009T190922Z` en `build/cloud-setup/20261002-board/`.
Su ZIP con `FILES.json` y certificado `VERIFIED.json` se guarda bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\`
en la carpeta del mismo identificador.

Los archivos enlazados comparten identidad y metadatos NTFS. Para editar
o recuperar una copia independiente, copiarla primero fuera del respaldo.
No repetir el aplicador. El diagnóstico de preparación inicial se conservó;
ocurrió antes de cualquier modificación.

Antecedente: [limpieza R28](D_BACKUP_CLEANUP_R28.md).
