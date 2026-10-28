# Limpieza de copias extraídas en D:

El usuario solicitó otra limpieza de respaldos antiguos el 2026-10-06.
Se retiraron **12 árboles extraídos redundantes** de ejecuciones del
2 al 4 de octubre, bajo:

`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\backups\<run>\<run>`

Se eliminaron **18.734 archivos**, con **8.540.859.829 bytes lógicos**.
Durante la operación, el espacio libre pasó de 271.577.088 a
8.852.975.616 bytes, unos **8,85 GB libres**. La diferencia de espacio libre
puede incluir actividad concurrente del equipo.

## Verificaciones y contenido conservado

Antes de cada eliminación se comprobó el SHA-256 del archivo comprimido
local y su copia en NFS, la integridad y el manifiesto del ZIP de fuentes,
y la igualdad de cada archivo extraído con el tar y su `FILES.json`.
Se exigió inventario idéntico, sin contenido local adicional ni puntos
de reanálisis. PowerShell volvió a comprobar los hashes y la ruta absoluta
antes de ejecutar `Remove-Item -LiteralPath` sobre el árbol autorizado.

Permanecen el `.tar.gz`, el ZIP de fuentes, los certificados históricos y
las copias remotas. Cada carpeta conserva `EXTRACTION_REMOVED.json`, que
explica el cambio de disposición y referencia los archivos necesarios
para restaurar la extracción. Los certificados históricos no se alteraron.
No se tocaron las tandas del 5 o 6 de octubre, los proyectos activos,
instaladores, licencias ni el archivo original de recuperación.

## Evidencia y restauración

El registro y las doce pruebas por archivo están en
`build/cloud-setup/20261002-board/d-cleanup-20261006T053502Z*`.
El paquete verificable se conserva en
`artifacts/ntrpu-board-vivado/handoffs/20261006T053502Z-d-cleanup-review/`.

Para volver a consultar un árbol retirado, verificar el SHA-256 indicado
en `EXTRACTION_REMOVED.json` y extraer el tar en una carpeta nueva.
No interpretar la ausencia del árbol como pérdida del respaldo ni volver
a descargarlo mientras el archivo local verificado esté disponible.

Las nuevas tandas usan respaldo comprimido sin extracción adicional;
la disposición evita volver a duplicar los archivos de simulación.
