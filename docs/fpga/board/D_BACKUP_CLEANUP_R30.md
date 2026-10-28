# Limpieza de fuentes históricas en D: — R30

El 2026-10-11 se consolidaron **10 copias físicas redundantes** de ZIP de
fuentes del 9 de octubre. Se recuperaron **437.112.832 bytes (437,1 MB)**
de asignación NTFS antes de guardar la evidencia. La auditoría dio `PASS`;
las 20 rutas implicadas conservan íntegramente su contenido.

## Alcance

La selección se limitó a `source-*.zip` dentro de respaldos anteriores al
**11 de octubre de 2026 a las 00:00 UTC**, bajo
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\`.
Las copias idénticas comparten ahora almacenamiento mediante enlaces duros.
Dos fuentes de checkpoints sin certificado de cierre completo se conservaron.

El checkpoint vigente de **945 tandas respaldadas** y la tanda activa
`20261011T062034Z-V4-ffn-supervised-pcie-symbolic-monitor-placement-r2`
quedaron fuera de la limpieza. El proceso de Vivado continuó enrutando.
Se preservaron paquetes de resultados, fuentes únicas, instaladores,
licencias, datos recuperados y archivos ajenos al proyecto.

## Verificación

Antes de cada sustitución se comprobaron rutas absolutas, ausencia de puntos
de reanálisis, certificados, paquetes de resultados y SHA-256. También se
validaron manifiestos y CRC de los ZIP, y se recalcularon los hashes de las
diez fuentes en NFS, sin modificar la VM.

La auditoría posterior confirmó hashes, tamaños, enlaces y ausencia de
temporales; verificó los metadatos de otros **5.937 archivos históricos**.
`AGENTS.md`, `README.md`, `CMakeLists.txt`, la cola documentada y el
certificado del checkpoint conservaron sus hashes. El espacio libre
observado pasó de **166,87 a 167,31 GB**; puede variar por otras escrituras.

## Evidencia y recuperación

Plan, diario y auditoría: prefijo `d-cleanup-r30-20261011T063710Z` en
`build/cloud-setup/20261002-board/`. El paquete de evidencia, manifiesto
`FILES.json` y certificado `VERIFIED.json` se conservan en
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\cleanup-reviews\d-cleanup-r30-20261011T063710Z\`.

No repetir el aplicador. Los enlaces comparten identidad y metadatos NTFS;
para editar una fuente, copiarla primero fuera del respaldo.
Antecedente: [limpieza R29](D_BACKUP_CLEANUP_R29.md).

## Comprobación posterior

El 2026-10-11 a las 12:08 UTC se volvieron a verificar los diez grupos:
SHA-256, identidad compartida de archivo, certificados y paquetes de
resultados coinciden con el plan original. Las dos fuentes excluidas siguen
presentes. Esta comprobación no eliminó archivos ni liberó espacio adicional.
Se observaron **161,43 GB libres** en D:, con la tanda 961 en curso.

Registro `d-cleanup-r30-followup-20261011T120817Z.json`, conservado junto al
plan local y en la carpeta de evidencia de D:. SHA-256
`0aa714879b16b6be39866982418ec34d3edf98ee7390d0e6492f000ce474a89f`.
