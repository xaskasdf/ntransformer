# RMSNorm: corrección del prelanzamiento y recuperación de la cola

El 2026-10-12 se reprodujo localmente un error del coordinador R1 antes
de que ejecutara Vitis. La sustitución global de `SOURCE` introducía el
diccionario de configuración también dentro del marcador de confirmación,
produciendo `SyntaxError` en el programa Python remoto. No fue un fallo RTL.

## Corrección y conservación

El coordinador HLS R2 sustituye únicamente la primera aparición del
placeholder. El fragmento corregido se analizó con la configuración sellada
real. La misma comprobación se aplicó al nuevo lanzador RoPE antes de sellarlo.
Se mantuvieron sin cambios kernel, testbench, Tcl, fixture y paquete de
fuentes RMSNorm: SHA-256
`95be19e7a604837d2dfc296517fbcca510f6de79efc782cb5519662e2dd0bb5a`.

Se retiraron únicamente los dos coordinadores R1 que todavía esperaban,
tras comprobar sus PID, líneas de comando, hashes, estados y ausencia de
un ID de ejecución remota. Los registros originales no se reescribieron;
por ello siguen mostrando su último estado de espera. El retiro efectivo
figura en `llama32-rmsnorm-prelaunch-r2-retirement.json`.
La FFN y las ocho etapas físicas/máscaras continuaron ejecutándose o esperando.

## Controladores vigentes

Ambos quedaron activos a las 02:08 UTC:

- `queue_llama32_rmsnorm_hls_r2.py`, PID inicial 24888: espera el replay
  de máscaras, ejecuta la misma tanda HLS y conserva revisión/respaldo.
- `queue_llama32_rmsnorm_transactions_r2.py`, PID inicial 175188: espera
  el PASS HLS R2 respaldado y comprueba las transacciones reales.

El revisor de transacciones R2 conserva exactamente las funciones de
parseo/cálculo que rechazaron 22 corrupciones en R1. Cambian los registros
de dependencia y el nombre del resultado; no se relajan criterios.
Los PID son históricos: verificar el proceso y su identidad al reanudar.

La preparación, diagnóstico y retiro quedaron verificados en D: y NFS
antes de activar R2, en `llama32-q8-working-20261012T020815Z.zip`, SHA-256
`0464a047d07ffae74fb2d95a6f17cff8ab8ffa74fae20e5f90b76b30eb685072`.
El estado posterior se registra en [VIVADO_STATUS](board/VIVADO_STATUS.md).
RMSNorm C/RTL, recursos HLS y timing físico permanecen pendientes.
