# Secuenciador de parada por registros XDMA

Estado al 2026-10-06: RTL R1 probado con el maestro AXI-Lite real y un
modelo de respuestas de registros; síntesis fuera de contexto revisada.
Pendientes la comparación funcional RTL/netlist y la integración con la IP
XDMA nativa. No autoriza drenaje global ni reset del endpoint.

## Contrato y revisión vigente

Implementación: `fpga/rtl/ntrpu_xdma_stop_registers_r1.sv`, módulo
`ntrpu_xdma_stop_registers`. Opera en `axi_aclk`, con propietario exclusivo,
cierre del host e inventario del
[ledger de descriptores](XDMA_DESCRIPTOR_LEDGER.md) coherentes en ese dominio.
Captura una identidad fresca `{generation, tag}` y cantidades por canal.

Solicita W1C Run en `0x000c` y `0x100c`, dato `1`, strobes `0xf`. Tras
ambas respuestas B reales lee control, estado y contador de H2C/C2H:
`0x0004/0x0040/0x0048` y `0x1004/0x1040/0x1048`. No usa registros RC
que borrarían diagnósticos. Las máscaras y su procedencia están en
[el contrato del proveedor](XDMA_INVENTORY_PROVIDER_PLAN.md).

Run/Busy o un contador menor provocan otra ronda completa de seis lecturas.
Un contador decreciente o superior al inventario produce fallo retenido.
También se retienen errores AXI/CSR, bits reservados, identidad incorrecta,
timeout y revocación del contexto. VALID/payload se conservan mientras
esperan READY, incluso después de fallar. Un timeout no inventa B/R.

`snapshot_valid` exige ambas paradas, una ronda completa sin errores,
Run=Busy=0, contadores iguales a lo publicado y ledger contabilizado.
Después de capturar deja de sondear; cambios de cierre, inventario,
readiness o propiedad revocan la salida. La pérdida de contabilidad o un
cambio del total completado después de capturar deja fallo retenido.
Errores CSR que aparezcan después requieren vigilancia del proveedor
superior: este resultado es una observación local retenida.

`release_valid` corresponde a retirada global por el supervisor, nunca a
la mera aparición del snapshot. `cold_rst_n` es configuración inicial;
los resets entre casos del banco no prueban recuperación operacional.

## Diagnósticos conservados

- `20261006T200813Z`: el banco original falló al compilar una expresión
  ternaria adyacente a un literal hexadecimal. No ejecutó simulación.
- `20261006T200958Z`: el banco R1 imprimió 89 éxitos, pero la revisión
  independiente encontró la primera dirección W1C indefinida a **146 ns**.
  El decodificador `always @*` no se evaluó al inicio con `step` inicializado
  por declaración; las comparaciones lógicas del banco dejaron pasar X.
- La revisión RTL R1 usa una asignación continua. El banco R2 añade
  `$isunknown` en solicitudes ofrecidas y comparaciones estrictas.
  No se alteraron fuentes ni resultados de los intentos anteriores.

## Simulación revisada

Tanda `20261006T201404Z-V4-ffn-supervised-xdma-stop-registers-r2-frozen`.
Vivado/XSim 2026.1 build 6511674, reloj de 4 ns y maestro con `WAIT_LIMIT=32`.

| Evidencia observada | Resultado |
| --- | ---: |
| Casos, incluidos todos los bits CSR de ambos canales | 89 |
| Solicitudes aceptadas, respuestas AXI y reportes confrontados | 589 de cada tipo |
| Capturas válidas, incluidas las luego revocadas | 17 |
| Ciclos con solicitud esperando READY / con fallo ya retenido | 99 / 93 |
| Ciclos de espera AW / W / AR | 174 / 2 / 189 |
| Latencia máxima solicitud → B/R, sin los dos casos de timeout | 24 ns |
| B/R tardías reales después de timeout | una de cada tipo, máximo 144 ns |
| Duración simulada | 30,2561 µs |
| Alteraciones de evidencia rechazadas | 19 |

La primera sesión conserva tres obligaciones durante varias rondas; la
segunda empieza con identidad nueva sin reset frío. Las comprobaciones
reconstruyen cada transacción desde los puertos públicos, sin leer el FSM.
También contrastan los snapshots con las respuestas crudas. XSIM solo
emite transiciones: el revisor R3 consulta el valor retenido en los tiempos
de los mensajes, incluido el último mensaje 100 ps después de la última
transición. No desplaza eventos ni modifica la traza.

El único aviso fue `XSIM 43-3431`, relativo al entorno del compilador.
Estas latencias pertenecen al modelo de registros; no miden PCIe físico
ni el Completion Timeout nativo de 50 ms.

## Síntesis revisada

Tanda `20261006T201715Z-V4-ffn-supervised-xdma-stop-registers-synth-frozen`,
parte `xcvu5p-flva2104-2-i`, controlador aislado:
**806 LUT, 414 FF**, sin RAM/DSP/latches/blackboxes. WNS/WHS estimados:
**+1,880/+0,077 ns** a 250 MHz. Se conservaron los registros completos de
identidad, inventario, snapshots y los 14 bits de fallo.

Se revisaron 103 avisos: 100 mensajes de salidas constantes, limitados por
el máximo de mensajes, y tres avisos de jerarquía plana, origen del reloj
sin ubicación y alcance DRC fuera de contexto. El netlist confirma los
59 bits constantes previstos. Se rechazaron seis alteraciones de reportes.
Los 699 bits de entrada y 551 salidas no constantes carecen de retardos
externos. La ausencia de bucles o fallos internos no cierra timing de placa.

## Comparación RTL/netlist revisada

Tanda `20261006T204759Z-V4-ffn-supervised-xdma-stop-registers-replay-r2-frozen`.
Los mismos 89 casos conservan resultados y tiempos: **15.536 observaciones**
coinciden desde 120 ns, con 589 solicitudes/respuestas/reportes en cada modo.
La comparación contempla exclusivamente los **100 ps** de propagación
comprobados en FDRE/FDSE de AMD. No acredita timing físico ni equivalencia
formal para todas las entradas posibles. Se rechazaron **40 alteraciones**
de evidencia y se comprobaron los siete archivos realmente compilados.

El modo netlist usa las dos síntesis conservadas: secuenciador y maestro
con límite 32. No se modificaron sus netlists ni el RTL funcional. Se
verificaron las tres definiciones idénticas de `glbl`, incluido su GSR
de 100 ns; los tres avisos corresponden al entorno y a esa definición
automática duplicada.

Los intentos `20261006T203247Z` y `20261006T204343Z` quedan rechazados:
al mezclar maestro RTL con secuenciador sintetizado, cinco LUT con entradas
permanentemente en cero mantuvieron salidas X. El segundo intento registra
sus entradas y salidas internas. Los bits de fallo 7/8 pasan a X a
250,1 ns y el banco termina a 254 ns. Con ambos bloques sintetizados,
la inicialización de sus celdas produce una simulación íntegramente definida.
No se enmascararon las X ni se alteraron las pruebas para aceptarlas.

## Integración nativa revisada

Tanda `20261006T205127Z-V4-ffn-supervised-xdma-stop-registers-native-frozen`.
El secuenciador RTL, maestro con `WAIT_LIMIT=65536` y ledger de 32 bits
controlan los puertos públicos del XDMA. El banco publica dos descriptores
reales, arranca ambos canales y entrega la propiedad de registros al RTL
cuando el maestro está libre. Las respuestas H2C B y C2H R permanecen
retenidas durante las dos escrituras W1C y la primera ronda completa.

| Medida de simulación | Resultado |
| --- | ---: |
| Accesos locales completos, incluida preparación | 29 lecturas y 13 escrituras |
| Accesos autónomos del secuenciador | 2 escrituras y 18 lecturas |
| Rondas CSR completas | 3 |
| Inicio aceptado → snapshot válido | 1,108 µs |
| Retención efectiva BAR B / H2C B / C2H R | 412 / 1.200 / 852 ns |
| Descriptores publicados / completados | uno por canal |
| Bytes reconstruidos desde PCIe y comparados en el host | 64 |
| Alteraciones de evidencia rechazadas | 44 |

La primera ronda observa ambos Busy=1 y contadores cero; la segunda,
H2C terminado con C2H todavía pendiente; la tercera, ambos completados.
Se verifican solicitudes, AW/W/AR, B/R reales, reportes con identidad y
cada palabra de los snapshots. La retirada de cierre del host bloquea
inmediatamente la observación y conserva fallos `0x200c` en el secuenciador
y `0x100` en el ledger después de restaurar el nivel.

La contabilidad de descriptores aparece a 107,144001 µs, la recepción de
64 bytes termina a 107,248 µs y el snapshot a 107,260 µs. Los **103,999 ns**
entre contabilidad y recepción confirman nuevamente su independencia.
Esta muestra no garantiza que todo snapshot futuro ocurra después de la
recepción; el contrato del host sigue siendo obligatorio.

Se conservaron 129 archivos AMD sin cambios y se revisaron sus 139 avisos.
El revisor R1 toma el valor anterior a la retirada de cierre cuando esta
comparte timestamp con el mensaje de recepción. El revisor de controles R2
altera las imágenes ya publicadas, no valores transitorios de inicialización.
Ambas correcciones conservan la simulación original. El cierre del host
sigue siendo una entrada del banco; no acredita un driver real.

## Cadena H2C precargada revisada

Tanda `20261006T212408Z-V4-ffn-supervised-xdma-stop-registers-chain-r1-frozen`.
Se publican dos descriptores adyacentes en `0x100/0x120` antes de Run,
con fuentes `0x400/0x500`, destinos `0x1000/0x1100` y cargas distintas de
64 bytes. Se verifican sus 64 bytes de descriptor y 128 bytes de datos
mediante tres lecturas PCIe, seis beats de completion y dos escrituras AXI.
Ambas AW/W ya fueron aceptadas cuando termina la primera escritura W1C.
Borrar Run conserva las dos obligaciones; la segunda B permanece retenida
mientras el secuenciador observa el primer descriptor completado.

| Ronda | Busy H2C | Completados H2C | Snapshot válido |
| --- | ---: | ---: | --- |
| 1 | 1 | 0 | No |
| 2 | 1 | 1 | No |
| 3 | 1 | 1 | No |
| 4 | 0 | 2 | Sí |

Se completan **35 lecturas y 9 escrituras** locales: 18 operaciones de
preparación y 26 autónomas, con respuesta real y reporte identificado.
El máximo solicitud→B/R es 48 ns. El inicio del secuenciador ocurre a
105,396 µs; el snapshot, a 106,844 µs: **1,448 µs** simulados.
Las retenciones efectivas de B son 436 ns para la primera respuesta H2C,
672 ns para la segunda y 412 ns para BAR durante la preparación.

Cada Done nativo sigue a su B real: B a 105,824/106,504 µs y Done a
105,856/106,536 µs. El ledger conserva 2→1→0 obligaciones y comienza
su aprobación a 106,536001 µs. Retirar el cierre invalida la aprobación
y conserva fallos `0x200c/0x100` después de restaurar el nivel.
C2H permanece sin tráfico; este ensayo no acredita recepción en el host.

La revisión reconstruye las interfaces públicas sin consultar el estado
interno del secuenciador. Rechaza **47 alteraciones de evidencia**, incluidas
segunda publicación con índice repetido, segundo payload incorrecto,
Done perdido y aprobación insertada con un solo descriptor completado.
Se verificaron 76 entradas compiladas, 129 archivos AMD sin cambios y
139 avisos asociados a sus declaraciones originales. La ejecución duró
601,355 s; mantiene `WAIT_LIMIT=65536` y Completion Timeout nativo de 50 ms.

El intento `20261006T212100Z` queda como diagnóstico de compilación:
tres literales hexadecimales tocaban el operador ternario `?`. R1 separa
esos tokens sin cambiar el RTL funcional. Ambos intentos están respaldados
en D:/NFS. Esta cadena no homologa cadenas arbitrarias, driver, drenaje
global, reset operativo o diseño físico.

## Respaldo y siguiente tanda

Cada intento conserva fuentes, logs, manifiestos y archivo verificado en
D: y NFS, accesible por `artifacts/ntrpu-board-vivado/<run>/`. Las revisiones
independientes y la cola se incluyen en el checkpoint incremental.

La comparación del netlist, integración nativa y cadena H2C de dos elementos
están revisadas en su alcance acotado. La [composición DMA/BAR](XDMA_DMA_BAR_INVENTORY.md)
también pasó: una obligación BAR persiste 36 ns después del snapshot local,
hasta B real. Dos observadores de 32 escrituras mantienen ACK en cero sin
cierre del productor. Sigue extender esa composición a C2H y recepción
host; después se reúnen inventarios PCIe y se cruza el certificado revocable
al join coherente.

Mantener Completion Timeout en 50 ms y no forzar estado interno de AMD.
La recepción del host sigue siendo un contrato separado: el ledger nativo
ya mostró contabilidad **103,999 ns antes** de recibir todos los bytes.
Permanecen abiertos driver real, cadenas generales, inventario PCIe/BAR,
join coherente, reset nativo, DIMM exacto/cuatro DDR/512 GiB, inferencia con
pesos reales e implementación física.
