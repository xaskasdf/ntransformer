# Atención HLS: extensión acotada de la ejecución original

La tanda `20261013T165530Z-V7-llama32-attention-hls-r1` empezó a las
16:55:33 UTC del 2026-10-13. Su contrato conserva 291 transacciones y
1.063 tokens. Al completar 234 transacciones solo había procesado 451
tokens: quedaban 612 en tiles mayores. El porcentaje de transacciones
completadas no representaba el trabajo de cálculo restante.

Para conservar el banco completo, a las 17:16:41 UTC se amplió el plazo
efectivo de 2.400 a **5.400 segundos desde el inicio original**. El límite
queda en **18:25:33 UTC**. No se modificaron fuentes, datos, criterios de
aprobación ni dependencias; tampoco se reiniciaron las herramientas.

## Procesos y mecanismo

- Wrapper Python original: PID **3563**, inicio `99562358`.
- Hijo/grupo Vitis original: PID/PGID **3564**, inicio `99562372`.
- XSIM original: PID **4659**, dentro del grupo 3564.
- Guardián: PID **7694**, acotado además por `timeout`, PID 7693.

Solo el wrapper que esperaba a Vitis está detenido con `SIGSTOP`; el
grupo de herramientas sigue ejecutándose. El guardián lo reanudará cuando
el hijo termine o al vencer el límite efectivo. En este último caso, el
wrapper aplicará su limpieza original de timeout. Un error del guardián
también reanuda al wrapper.

El mecanismo conserva el comprobado para
[normalización Q8](../LLAMA32_NORM_Q8_C_RTL_RESULTS_R1.md). Se verificó
que el código de espera de CPython instalado mantiene el mismo hash;
además se fijaron los PID, instantes de creación, grupos, directorio,
comandos y fuente de esta tanda. El guardián nuevo comprueba directamente
el hash del plan y sus identidades antes de detener al wrapper.

## Evidencia

Plan: `llama32-attention-hls-wait-extension-r1-plan.json`, SHA-256
`4783fa5c9738f181972e1415d754c48ecf81e81901c6311868a144280af3b820`.

Guardián: `tools/extend_ntrpu_attention_hls_wait_r1.py`, SHA-256
`cd1b3a5d85e2ba58ce9fac3f13ff136e2f8e73f2518a58ec5445e6e43a2801be`.

Cápsula D:/NFS: `20261013T171632Z-attention-hls-wait-extension-r1.zip`,
SHA-256 `03e0ee7d62a88e6557899aa113815e48999eead88db5b4b38e7dbd9ec27a11f3`.

El directorio de la tanda contiene `supervision-r1/` y
`budget-extension-r1.json`; ambos entrarán en su archivo terminal.
`invocation.json` conserva el límite inicial para registrar la historia
sin reescribirla. Mientras esta extensión esté activa, el estado `T` de
3563 es esperado; comprobar también guardián, XSIM y progreso del log.
La aprobación aritmética sigue pendiente del resultado real y su revisión.
