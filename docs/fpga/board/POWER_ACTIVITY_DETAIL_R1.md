# Revisión complementaria de actividad DDR R1

Fecha: 2026-10-16. **Preparada y probada con datos sintéticos; captura nativa
pendiente.** Complementa [POWER_ACTIVITY_R1.md](POWER_ACTIVITY_R1.md).
No cambia su banco, parámetros, paquete sellado ni lugar después de BRAM.

## Qué comprueba

El [revisor](../../../tools/review_ntrpu_ddr_activity_detail_r1.py) consume
una tanda terminada y restaurada, con `invocation.json`, el `REVIEW.json`
original y los archivos de captura. Escribe un informe nuevo fuera de la
tanda; rechaza sobrescribirlo o escribir dentro del respaldo.

- Vincula el informe original con la invocación y el SHA-256 del paquete
  DDR R1. Conserva errores de ejecución y del revisor original.
- Contrasta cada intervalo del simulador con los 20,480 µs del SAIF,
  4.096 ciclos y la última muestra UI. Comprueba continuidad, solicitudes
  pendientes, máximo de un evento por canal/ciclo, RLAST y máscaras.
- Calcula bytes aceptados por segundo y fracciones de espera. Son métricas
  del PRBS del ejemplo, no rendimiento de inferencia ni del DIMM físico.
- Revisa diez señales críticas: resets, calibración, PRBS y errores del
  generador. Comprueba además el reloj UI y sus 8.192 transiciones.
- Desglosa actividad y X/Z por jerarquía, conserva campos TB/IG y señala
  las mayores residencias desconocidas. Informa señales críticas ausentes;
  no sustituye observaciones faltantes por valores esperados.
- Compara inventarios y transiciones por señal. Dos ventanas pueden tener
  el mismo total de transiciones y cambios opuestos en señales distintas.

El parser sigue la estructura SAIF del ejemplo de
[AMD UG900](https://docs.amd.com/api/khub/documents/ocAvD5R2tJvgu72ywMfsKA/content).
Se comprobó la existencia de los identificadores en el `example_top` y el
generador archivados; esto no demuestra su visibilidad tras elaboración.

## Límites que permanecen

Se preparó una [observación temporal por rank](POWER_RANK_TRACE_R1.md),
probada con formas de onda sintéticas y un VCD histórico. Distingue tres
codificaciones REF del chip trazado y señala los otros ranks sin cobertura.
La captura R1 no exporta ese VCD: integrar el helper requerirá una revisión
nueva después de inspeccionar el resultado pendiente. No modifica esta tanda.

Se exporta la ocupación de CS, CKE y ODT de los cuatro índices de rank,
con `commands_observed=null` y `refresh_observed=null`. Un SAIF agregado no
conserva la correlación temporal entre reloj, selección y comando necesaria
para reconstruir transacciones o refresh. La ocupación de un pin no prueba
cobertura de comandos de ese rank. Hace falta una traza temporal adicional.

No hay umbral automático de representatividad. Dos ventanas similares no
demuestran estabilidad estadística. Persisten la revisión de diagnósticos
vendor, identidad compilada, causas de X/Z y correspondencia con el DCP.
`ready_for_power_estimate`, `board_power_qualified` y `exact_dimm_qualified`
permanecen siempre en `false`, incluso si la revisión estructural pasa.

## Ejecución después del respaldo de R1

El coordinador existente ejecutará primero su revisor R2 sellado y terminará
el respaldo. Desde 2026-10-16 08:13 UTC, el
[observador complementario](../../../tools/ntrpu_ddr_activity_followup_r1.py),
PID 16980, espera ese estado terminal, tanto para una exportación como para
un diagnóstico respaldado. No adquiere el slot ni escribe la cola compartida.
Su límite de espera es de doce horas; no cambia el límite de la captura.

Revalida el TAR completo, inventario y archivos restaurados, identidad del
paquete y recibos, además del hash del respaldo NFS. Aplica la revisión a
la copia restaurada y comprueba que sus archivos sigan intactos. Guarda el
informe y sus fuentes en un respaldo adicional D:/NFS, con restauración
verificada y referencia al archivo original. Un fallo de transferencia se
conserva para inspección; no relanza Vivado ni repite escrituras remotas.

Estado: `build/cloud-setup/20261002-board/ddr-activity-followup-r1-monitor.json`.
**Todavía no hay revisión de resultados nativos de esta captura.** El comando
manual siguiente sigue disponible para inspección; no duplicarlo mientras
el observador esté esperando o procesando la misma tanda.

```powershell
python tools/review_ntrpu_ddr_activity_detail_r1.py `
  <tanda-restaurada> --output <carpeta-nueva>/activity-detail.json
```

Conservar informe, fuentes y referencias al archivo original en un respaldo
adicional, con SHA-256 y restauración verificados. Un diagnóstico no autoriza
repetir Vivado ni adelantar etapas de la cola.

## Evidencia de preparación

Pasaron **39 rechazos de evidencia corrupta o escritura indebida y 22
comprobaciones diagnósticas**, con resultados idénticos en Python 3.14 local
y Python 3.10.12 en Linux. Incluyen resets transitorios, X, reloj incorrecto,
desacuerdo temporal, actividad que supera el ancho del bus, recibos ajenos,
ejecución activa y conservación byte a byte de entradas.

La comprobación Linux usó CPU 2–3 y un límite de 120 segundos; no adquirió
el slot Vivado. Un primer lanzador tuvo un error de escape antes de ejecutar
el código remoto; su diagnóstico se conserva. La corrección R2 no cambió
el revisor ni la captura encolada.

Evidencia en `build/cloud-setup/20261002-board/`:
`ddr-activity-detail-r1-controls-final.json`,
`ddr-activity-detail-r2-python-validation.json` y
`ddr-activity-detail-r1-launch-diagnostic.json`.
Paquete de revisión: SHA-256
`bed05f376f411060a53be91ebd5fd8b4b3f44a243269cb75d7b75176576a2453`.
Fuentes, diagnósticos y documentación se respaldan en
`artifacts/ntrpu-board-vivado/ddr-activity-detail-handoff-20261016-r1/`.

El observador pasó **15 pruebas de respaldo y postprocesamiento**, con los
mismos casos en Python 3.14 Windows y 3.10.12 Linux. Rechazan archivos
alterados, inventarios incompletos, rutas ajenas, recibos contradictorios e
invocaciones activas; verifican la conservación de diagnósticos y el respaldo
complementario mediante transporte local simulado. No son pruebas de Vivado.
Evidencia: `ddr-activity-followup-r1-python-validation.json` en la carpeta de
estado, SHA-256 `aaee25b5d6c7d9d3a87d1ebfd6a3a01fdc7e0fe577fc112880b6e95ab7687491`.
Fuentes y pruebas: ZIP `06e60d13ada2999191cac3691869d3cef4230865db644efa92ef5b640f9a85df`.
Integración respaldada en
`artifacts/ntrpu-board-vivado/ddr-activity-followup-handoff-20261016-r1/`.
