# Secuenciador aislado de reset del endpoint

Prototipo V4/V7 del 2026-10-05. Implementación:
`fpga/rtl/ntrpu_endpoint_reset_sequencer.sv`. Complementa el
[contrato del endpoint](ENDPOINT_RESET_COORDINATION.md). La tanda
`20261005T153531Z-V4-ffn-supervised-endpoint-sequencer` está aprobada
para el contrato aislado y respaldada localmente y en NFS.
Todavía no está conectado al guard, XDMA, CDC AXI ni MIG existentes.

## Interfaz y propiedad

El módulo funciona con `mgmt_clk` independiente. Sus entradas deben llegar
ya calificadas a ese dominio; el bus de ACK no implementa un cruce CDC.
`cold_rst_n` corresponde exclusivamente a la vida de configuración.
El guard de arranque sigue siendo responsable de mantener cerrada la
admisión antes del primer `start` de este módulo.

Al recibir `start`, fija `invalidate_req` y `block_admission` hasta la
siguiente configuración. Solo permite pedir resets después del ACK de
propiedad retenida y de una confirmación independiente de drenaje.
`complete` indica finalización de mantenimiento de interfaces; **no
autoriza reutilizar reservas ni reabrir la sesión NTRD**.

Cada solicitud lleva `req_kind` y un identificador de 64 bits `req_tag`.
El ACK debe coincidir con ambos. Los identificadores no se reciclan dentro
de una configuración: el agotamiento bloquea el secuenciador y conserva
sus salidas. Un ACK coincidente tampoco sustituye las observaciones
`endpoint_held` y `endpoint_available` exigidas por su fase.

## Orden de las fases

| Solicitud | Evidencia exigida al productor del ACK |
| --- | --- |
| INVALIDATE | Propietario invalidado y reserva retenida |
| QUIESCE | Tráfico terminado según el contrato de todos los participantes |
| LOCAL_HELD | Ambos extremos CDC y sus interfaces enfrentadas bajo reset |
| ENDPOINT_HELD | Reset permitido del endpoint observado para esta solicitud |
| ENDPOINT_READY | Endpoint, enlace y reloj disponibles tras liberar su reset |
| LOCAL_READY | Liberación local completada y reconocida por todos los peers |

La liberación del endpoint ocurre con el reset local todavía solicitado.
Así se evita esperar que el guard retire una petición que impide volver
al enlace. Una nueva pérdida durante la liberación vuelve a QUIESCE con
otro identificador; no vuelve a afirmar un reset ya liberado basándose
solo en el ACK antiguo. Las salidas que seguían activas permanecen así.

## Intervalo de mantenimiento

El contador empieza en cero cuando se acepta ENDPOINT_HELD. Cuenta
periodos completos posteriores; una pérdida de esa observación obliga
a obtener otro ACK y empezar de nuevo. Un `external_hold` persistente
impide liberar el endpoint incluso después de alcanzar el límite.

La configuración XDMA examinada fija Completion Timeout en 50 ms.
`HOLD_CYCLES=5000001` corresponde a 50,00001 ms con un reloj exactamente
de 100 MHz. Para placa debe usarse la frecuencia máxima garantizada y
sus tolerancias: `floor(timeout × frecuencia_máxima) + 1`, más el margen
que corresponda. Esta cifra nominal no aprueba restricciones de placa.

## Banco y límites de la evidencia

`tools/ntrpu_board_endpoint_reset_sequencer.tcl` ejecuta nueve casos:
ACK incorrectos/antiguos, ACK de propietario ausente, drenaje ausente,
pérdida de observación de reset, PERST repetido, parada/retorno de reloj,
pérdida durante liberación, agotamiento de identificadores y el contador
completo de 5.000.001 ciclos. El caso de reloj utiliza el detector XPM
existente. Los ACK de acciones son estímulos controlados del banco.
Después se sintetiza únicamente el secuenciador fuera de contexto.

La integración deberá implementar productores de ACK verificables;
copiar el identificador actual sobre un nivel antiguo no demuestra la
acción. Los cruces de solicitudes/ACK requieren transferencia atómica,
y los mensajes externos requieren identificación de la configuración
para descartar respuestas de una vida anterior. También quedan pendientes
CDC físicos, reset completo del XDMA nativo, restauración de registros,
tráfico tras liberación y el enlace con el propietario/MIG reales.

## Resultados y controles negativos

Los nueve casos pasan en XSIM: 61 observaciones de estado, 45 eventos y
nueve ACK antiguos/incorrectos rechazados. El contador completo mantiene
el reset durante 5.000.001 periodos de 10 ns. La síntesis OOC reporta
114 LUT y 198 FF; no usa BRAM, URAM, DSP ni latches. Hay 137 primitivas
LUT antes de considerar el empaquetado reflejado por utilización.

La estimación de síntesis da setup +8,165 ns y hold +0,066 ns. No califica
timing físico: 74 entradas y 134 salidas carecen de retardos externos y
falta ubicar la fuente del reloj. Se revisaron 84 avisos: 82 corresponden
a la extensión con ceros del contador de 23 bits, uno al entorno de XSIM
y uno al reloj OOC. No hay avisos funcionales durante la simulación.

Respaldo: 71 archivos, 33.962.017 bytes, SHA-256
`3ba413d7735df1ca915cb377486705baa6f78afc16d6a7e8c0177b8df169188e`.
Adenda de revisión terminal:
`0826c5178d7434bb9a2cbfd3d625e119daf174f7ac76bee8682f503bd4cf0afe`.

La tanda `20261005T154758Z-V4-ffn-supervised-endpoint-sequence-mutants`
compila cuatro copias RTL deliberadamente incorrectas contra el banco
original sin cambios. Las cuatro fallan en la aserción esperada: omitir
el identificador del ACK, liberar un ciclo antes, saltarse QUIESCE y
omitir disponibilidad antes de liberar la lógica local. Las copias
originales permanecen intactas; estos fallos son controles negativos,
no variantes funcionales aprobadas. Respaldo: 217 archivos, 836.991 bytes,
SHA-256 `dd4b40948848a17ad109af7dba7f6358fb448ab4528900b8d2205b1fd511dfd7`.

Siguiente componente: [seguimiento de drenaje AXI-Lite](AXIL_DRAIN_MONITOR.md).
Su certificado cubre una interfaz; nunca sustituye el drenaje de todos
los participantes exigido por QUIESCE.

## Avance: actuador local en RTL

El [actuador de reset](LOCAL_RESET_ACTUATOR.md) pasa sus ocho casos y dos
perfiles con el convertidor AMD real. Implementa conteo y liberaci?n autom?tica;
la uni?n completa con el secuenciador y el drenaje del endpoint sigue pendiente.
