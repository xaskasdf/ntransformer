# FFN real: revisión pendiente de ciclos por fase

La FFN completa con pesos Q8 de Llama continúa en XSIM. Se preparó un
revisor suplementario que solo actuará después de su PASS numérico y del
respaldo verificado. No modifica entradas ni requiere otro turno de Vivado.

## Qué comprobará

- Reapertura del archivo cerrado y repetición de la revisión numérica.
- Identidad por ruta y hash de 49 entradas de compilación de la aplicación
  y once unidades generadas para simular las operaciones flotantes.
- Copias HLS, fixture, registros de ejecución y log contra sus archivos sellados.
- Partición de ciclos en gate, up, SwiGLU/Q8 y down, contrastada con muestras
  de fase y bytes comprometidos cada 100.000 ciclos.
- Tráfico aceptado por fase: 16.480 AR, 886.784 beats R y 2.432 beats W,
  con 56.754.176 bytes leídos y 155.648 escritos para D2048/H8192.

La tabla de resultados separará pesos leídos una vez, relecturas de
activaciones e intermediarios. Las fracciones de beats y pausas pueden
solaparse; no son porcentajes de ocupación del MAC ni ancho de banda DDR.
Las muestras de progreso acotan las transiciones, pero no proporcionan
una medición independiente del ciclo exacto de cada cambio de fase.

## Preflight realizado

El revisor reprodujo el banco sintético cerrado
`20261003T113023Z-V7-ffn-full-llama-shape`: 20.332.664 ciclos, distribuidos
en 6.454.374 / 6.455.250 / 980.311 / 6.442.729. Rechazó catorce alteraciones
de contadores, progreso, tráfico y cierre. Son datos del banco histórico;
no se presentan como resultado de los pesos reales actualmente en ejecución.

También se comprobaron las 60 rutas de compilación del banco actual;
cuatro alteraciones de identidad fueron rechazadas. Es una observación
de entradas compiladas, no una aprobación numérica de la simulación viva.

Los archivos de coordinación están en `build/cloud-setup/20261002-board/`,
con prefijo `llama32-full-ffn-metrics-r1`. Una revisión completa generará
`LLAMA32_FULL_FFN_METRICS_RESULTS_R1.md` y se respaldará en D: y NFS.
Hasta entonces no hay medición final de esta FFN real, inferencia completa
ni timing físico validado.
