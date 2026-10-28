# Norma→Q8: ampliación acotada del plazo

La tanda `20261013T155333Z-V7-llama32-norm-q8-hls-r1` empezó el
2026-10-13 a las 15:53:37 UTC, con límite inicial de 2.400 segundos.
El banco contiene 61 transacciones: 49 correctas y 12 rechazos. A las
16:22:38 seguía calculando la transacción 25; el ritmo observado requería
más de 40 minutos para conservar el banco completo.

Se amplió el límite efectivo a **5.400 segundos desde el inicio original**:
17:23:37 UTC. No se reiniciaron Vitis/XSIM, no se recompiló el núcleo y no
se modificaron fuentes, fixtures, orden de cola ni criterios numéricos.
El `invocation.json` original conserva su plazo inicial; el registro
`budget-extension-r1.json` documenta expresamente el plazo efectivo.

## Supervisión durante la extensión

El wrapper Python original no ofrece una API para cambiar su espera.
El guardián `extend_ntrpu_owned_hls_wait_r1.py` transfiere temporalmente
esa supervisión: detiene únicamente al proceso Python **4182379**,
que estaba esperando, y deja activo al grupo Vitis **4182380**. El
simulador **4183222** conserva su identidad y trabajo. Antes de actuar,
se verificaron PID, instante de creación, relación padre/hijo, grupos
separados, línea de comandos, directorio y hashes de la invocación.

El guardián **4187932** observa al hijo original. Cuando termina, reanuda
el wrapper para que recoja su verdadero código de salida y complete el
registro habitual. Al llegar al nuevo límite, lo reanuda para ejecutar
la limpieza de timeout original. Una excepción o SIGTERM del guardián
también reanuda al wrapper. Un `timeout` externo acota al propio guardián.

Se inspeccionó el orden real de `waitpid` de CPython instalado y se
probaron dos procesos desechables: finalización después del plazo
original con resultado 0, y agotamiento del nuevo plazo con resultado
124. El grupo de herramientas no recibió señales de pausa. El archivo
terminal de la tanda incluirá el guardián, plan y registros de extensión.

## Evidencia y seguimiento

- Plan: `llama32-norm-q8-wait-extension-r1-plan.json`, SHA-256
  `8b1c5fce2f21239d79bbe59f3d018d0f971ef404f93a020614cfdbd5b645aa12`.
- Cápsula D:/NFS verificada: `20261013T162231Z-norm-q8-wait-extension-r1.zip`,
  SHA-256 `7681afa46dc9b4e7d9554b346d6a74bef95b165d3386425c379ae123006e84c8`.
- Inspección viva a las 16:26:04: mismo grupo y simulador, guardián activo,
  progreso 28/61. Registro SHA-256
  `5356cd33b57034bf43c30ae103c6704f018234338a35d0156911baa9336fa331`.

Mientras la extensión esté activa, el estado `T` de 4182379 es esperado;
no implica que la simulación esté parada. Consultar también guardián,
grupo Vitis y avance del log. Una desconexión de observación no autoriza
reiniciar la tanda. La aprobación depende del resultado real y de la
revisión/respaldos existentes, todavía pendientes al escribir esta nota.
