# Cancelación registrada entre post-O y FFN

Fecha: 2026-10-16 10:12 UTC. **Protocolo nativo aprobado; efecto sobre timing aún sin medir.**

## Evidencia que motiva el cambio

La síntesis archivada de atención/FFN tiene una ruta de 66 niveles, desde
`held_command_epoch_reg[9]` hasta `handoff/completion_status_reg[0]`.
Suma 15,449 ns: 3,566 ns de lógica y 11,883 ns de redes **sin colocar**.
WNS: −10,467 ns con reloj solicitado de 5 ns. La revisión de cada incremento
reconcilia exactamente esos totales; no son retardos de routing físico.

La ruta cruza validación de identidad, cancelación, validez de resultados,
FFN y residual. `handoff/source_bad` aparece a los 7,681 ns acumulados y
alimenta cancelación del consumidor, cuyo resultado vuelve al handoff.
Registrar esa propagación puede acortar la cadena; no prueba una frecuencia.

## Variante y contrato

[`ntrpu_post_ffn_handoff_r3.sv`](../../fpga/rtl/ntrpu_post_ffn_handoff_r3.sv)
deriva `child_cancel_valid` sólo de `fault`, `revoked` y `cancel_seen`, estados
retenidos. Un fallo presente en el flanco se refleja después de ese flanco,
y permanece hasta el retiro. El consumidor puede continuar hasta ese punto;
las operaciones ya admitidas deben drenarse. No se prometen rollbacks AXI.

La invalidación pública por cancelación, época o productor alterado sigue
siendo combinacional e inmediata. El padre conserva las reservas y no libera
respuestas ajenas. No cambian certificados, arbitraje ni aritmética. El
wrapper experimental reemplaza únicamente ese hijo; las fuentes en ejecución
y las demás tandas de la cola permanecen intactas.

## Prueba y límites

El [banco R3](../../fpga/tb/tb_ntrpu_post_ffn_cancel_r3.sv) conecta R2 y R3 a
los mismos pares controlados. Compara las otras 43 salidas antes/después del
flanco y verifica la demora máxima de cancelación. Pasaron 522 trabajos por
semilla (1, 7, 42): **1.566 trabajos**, con 22 demoras observadas por semilla.
Incluye fallos transitorios y tardíos, contrapresión, identidades completas,
retención, protección de regiones y reutilización.

Se rechazaron **43 variantes RTL** por aserciones explícitas, incluidas una
cancelación sin registrar y una demora de dos flancos. El revisor rechazó
324 alteraciones de las trazas. Son controles de protocolo, no pruebas de
aritmética ni demostración formal para toda entrada posible. Dos errores
de expansión de macros del banco quedaron archivados; el candidato RTL
permaneció igual. Las cuatro tandas están restauradas y verificadas en D:/NFS.

## Tanda Vivado pendiente

Paquete `20261016T100531Z-post-ffn-cancel-vivado-r1`, PID 29932, después de la prueba aritmética
de capa completa. Compila 56 RTL y 122 Verilog HLS originales, seis familias
y 27 scripts IP; el banco de comparación usa otro proyecto. Dos CPU, máximo
3.600 segundos, sin reintento automático. ZIP fuente:
`4aa0135d036a9583fd4cc1e28020c76b15d34d471512e466b4667b1c823918a3`.

Se revisarán identidad del top, fuentes reales, recursos, ausencia de latches,
cajas negras y bucles, 50 rutas críticas y las tres semillas XSim. Se publicará
la diferencia de WNS/recursos frente a la síntesis archivada, incluso si empeora.
No se adopta el candidato en la capa completa sin revisar esos resultados y
validar su integración. Routing y actividad DDR siguen pendientes.

Revisión nativa SHA-256:
`c5cc265c72856f5616a8c70971276bc836be93895b34c36e6c0ad75f14a9aac9`.
Análisis de ruta SHA-256:
`32720bf3a412142759f8f8f76bebff62077b5ed9a2c281fd4b31f6fdacb44d42`.
Respaldo: `artifacts/ntrpu-board-vivado/post-ffn-cancel-handoff-20261016-r1/`.
