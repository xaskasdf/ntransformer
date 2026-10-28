# Llama Q8: revisión independiente de métricas GEMV

Preparación 2026-10-12 UTC. La simulación existente sigue intacta.
Este análisis local espera su PASS numérico y respaldo antes de
aceptar métricas; puede ejecutarse mientras avanza la FFN siguiente.

## Comprobaciones

Se recuperan las tres semillas del archivo verificado, junto con
fixture, testbench compilado y los doce archivos HLS copiados.
Se repite la comparación numérica y se vinculan los 28 trabajos de
cada semilla a sus slices Q/K/V/O/gate/up/down de las capas 0/15.

El comprobador de contadores existente verifica que los seis estados
particionen cada latencia, los límites y la intersección de ocupación
de ambos MAC, y los bloques Q8, beats AXI y activaciones exactos.
La envoltura nueva rechaza campos repetidos, faltantes, extras o
inválidos, identidades desordenadas y geometrías distintas al fixture.
Catorce corrupciones se ensayan sobre el log real de la primera semilla.

## Entregables y límites

El JSON suplementario conserva contadores por trabajo/semilla y
min/mediana/max de ciclos por slice. El informe separa fases de
ocupación y esperas: estas últimas pueden solaparse y no se suman
como porcentajes independientes. Bytes útiles/ciclo usa ciclos dentro
de trabajos, excluyendo huecos entre ellos.

No se convierte el reloj del testbench en frecuencia física validada,
ni estas medidas en ancho de banda DDR o tokens/s. Las activaciones
son sintéticas y los pesos son slices del Q8 original; no es una
inferencia completa ni un ensayo de placa.

`queue_llama32_gemv_metrics_r1.py` espera revisión y respaldo del GEMV,
ejecuta únicamente análisis local y respalda el suplemento en D: y NFS.
No altera las dependencias ni resultados de la cola de Vivado.
