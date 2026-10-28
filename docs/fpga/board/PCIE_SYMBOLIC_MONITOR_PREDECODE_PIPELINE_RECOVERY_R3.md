# Cola explícita R3 tras el cierre del consumidor

## Causa y evidencia

La tanda 989 terminó, se archivó y posteriormente aprobó su revisor
original. El coordinador R2 se había detenido al buscar
`pcie-symbolic-monitor-predecode-bound-late-focus-r2-owner.json`.
Una sustitución del prefijo de logs también alteró ese nombre; el runner
conservó correctamente el owner **R1**. Los sucesores se detuvieron antes
de ejecutar sus tandas. Todos esos registros y scripts se conservan.

## Ejecución corregida

`continue_symbolic_monitor_predecode_pipeline_recovery_r3.py` administra
esta secuencia con rutas explícitas, comprobadas contra los runners:

1. Verificar y respaldar el PASS archivado del consumidor 989.
2. E completo: 183.058 filas nuevas; solo cambia E respecto a RQ.
3. Síntesis propia E, auditoría EDIF y comparación con 984.
4. Replay de la netlist E: 174.436 filas binarias nuevas.
5. Consumidor de límites sobre winner: 309.884 muestras nuevas.

Los owners y configuraciones de los runners siguen siendo R1; los
registros de coordinación nuevos son R3. Cada fase crea su registro
exclusivamente y publica el cierre después de revisión y respaldo.
Los congeladores R3 solo actualizan enlaces de dependencia y agregan
la procedencia de esta recuperación; los runners y revisores no cambian.

El controlador verifica las recetas y hashes de cada congelador, los
nombres de owner/configuración y las salidas esperadas de los revisores.
No inicia si existe algún owner o configuración de las fases pendientes.
Un error conserva el estado y sella evidencia para inspección; no vuelve
a ejecutar una tanda completada ni reinicia por timeout de observación.

## Respaldo y alcance

La cápsula R12 añade los futuros snapshots del consumidor winner a la
cobertura R11. Mantiene revisión y archivos terminales en D: y NFS.
El manifiesto `pcie-symbolic-monitor-predecode-pipeline-recovery-r3.json`
tiene SHA-256
`84de7dbe09c58e484e56aeb90029bffa7719ef9234663bd067aee3d1487d6823`.

Esta recuperación habilita ejecución ordenada; no concede equivalencia,
mejora temporal ni calificación de placa a las fases todavía pendientes.
