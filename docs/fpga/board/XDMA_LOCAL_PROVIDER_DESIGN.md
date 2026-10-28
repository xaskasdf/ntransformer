# Proveedor de gestión local XDMA

Estado al 2026-10-06: contrato unitario, síntesis e integración del maestro
con XDMA nativo revisados y respaldados. Comparación funcional RTL/netlist
aprobada para límites de 32 y 65.536 ciclos, dentro del alcance descrito abajo.
Esta preparación continúa [la gestión local](XDMA_LOCAL_MANAGEMENT.md),
sin dar por cerrado el inventario protegido del endpoint.

## Acceso a registros

`ntrpu_xdma_local_master.sv` acepta una transacción por vez en `axi_aclk`.
Captura dirección, dato, byte enables e identidad de 128 bits mediante
`req_valid/req_ready`; devuelve dato, respuesta AXI e identidad mediante
`rsp_valid/rsp_ready`. La identidad es opaca: el proveedor superior debe
validar generación/tag y rechazar reutilizaciones.

AW y W se aceptan independientemente. Cerrar `admit` impide solicitudes
nuevas y conserva las ya ofrecidas. El límite positivo `WAIT_LIMIT` mide
espera de transacción, no tiempo de PCIe Completion Timeout. Al vencer,
`timeout_latched` queda retenido y bloquea futuras admisiones; la transacción
sigue esperando su B/R real. No se retira VALID, se fabrica una respuesta
ni se permite reset. La respuesta recibida posteriormente conserva el
indicador de timeout y permanece estable bajo backpressure.

`cold_rst_n` pertenece únicamente a configuración. El bloque no detecta
un reloj detenido: el supervisor independiente y el mailbox deben mantener
esa obligación. No conectar su resultado directamente a `complete_ack`.

## Secuencia requerida del proveedor

| Operación | H2C canal 0 | C2H canal 0 | Uso |
| --- | --- | --- | --- |
| Control W1C | `0x000c` | `0x100c` | Escribir solo bit 0 para borrar Run |
| Control | `0x0004` | `0x1004` | Confirmar Run borrado |
| Estado | `0x0040` | `0x1040` | Busy y errores; conservar bits de diagnóstico |
| Completados | `0x0048` | `0x1048` | Comparar con obligaciones de esa ejecución |

Las variantes W1C permiten borrar Run sin reescribir el resto de control.
[PG195: control C2H W1C](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Control-0x0C).
Los estados incluyen fallos de descriptor, acceso, alineamiento y longitud;
Busy y los eventos de parada/completado no sustituyen su revisión.
[Estado H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Status-0x40),
[estado C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Status-0x40).

Primero cerrar productores con identidad y barrera verificadas; después
parar ambos motores sin esperar a que uno quede vacío para solicitar la
parada del otro. Observar control/estado/contadores y correlacionar los
puertos públicos con las transacciones pendientes. Los contadores se
reinician al subir Run: no son números de generación ni acreditan por sí
solos cancelación de descriptores.
[Contador H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Completed-Descriptor-Count-0x48),
[contador C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Completed-Descriptor-Count-0x48).

El C2H termina su solicitud al recibir confirmación de emisión por PCIe;
la recepción en RAM del host requiere su propia evidencia.
[PG195: C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel).
Un proveedor válido debe combinar estas obligaciones con BAR, secciones
locales, errores retenidos, cierre real del driver y cruces coherentes.

## Prueba unitaria revisada

Tanda `20261006T173555Z-V4-ffn-supervised-xdma-local-master-unit-frozen`.
XSim terminó diez casos en 1.564 ns simulados (31,514 s de ejecución). La
revisión independiente reconstruyó todos los handshakes desde VCD: AW/W
en ambos órdenes y simultáneos, errores SLVERR/DECERR, retención de identidad
y cierre de admisión sin cancelar solicitudes ofrecidas. Los cinco timeouts
ocurren exactamente a 32 ciclos de 4 ns (128 ns); cada operación termina
solo al aceptar su B/R real. El reporte del caso 5 soporta 41 ciclos de
backpressure sin timeout. Se rechazaron 14 corrupciones de evidencia.

El único aviso, `XSIM 43-3431`, corresponde al entorno del compilador; se
verificaron las rutas y hashes compilados, la elaboración y la traza.
Archivo: `5477ac8766966acbe635d6675cd0cd3e368d48036435ca254c9f7c58df929982`.
Revisión: `040d0be8bc0e642fd727b476138654c8c86203e6e30ff1e99295217bfca5d64b`.
No se instancia XDMA ni se valida reset operativo en esta prueba.

## Síntesis OOC revisada

Tanda `20261006T175837Z-V4-ffn-supervised-xdma-local-master-synth-frozen`.
Vivado 2026.1 sintetizó ambas variantes para `xcvu5p-flva2104-2-i`:

| WAIT_LIMIT | LUT empaquetadas | FF | WNS estimado | WHS estimado |
| --- | ---: | ---: | ---: | ---: |
| 32 | 42 | 279 | 2,795 ns | 0,083 ns |
| 65.536 | 44 | 290 | 2,795 ns | 0,066 ns |

Los temporizadores conservan 5/16 bits, la identidad 128 bits y los 521
bits de puertos. Los FF coinciden entre reporte y celdas; no hay cajas negras,
latches, RAM ni DSP. Se revisaron los DCP y su reloj de 4 ns, sin excepciones
temporales. Se rechazaron seis alteraciones de reportes/celdas.

Los 28 avisos corresponden a protección AXI constante (24), falta de sitio
de reloj para estimar skew (2) y cobertura DRC limitada por OOC (2). No hay
endpoints internos sin reloj/restricción, pero faltan retardos de 242 entradas
y 272 salidas no constantes. Los márgenes no acreditan timing físico ni
equivalencia funcional del netlist; falta integración, colocación y ruteo.

Archivo: `c727c41172fdedd890041e85783b3d99dc40a3a126f63214ae8602e07bf99e3e`.
Revisión: `fc43b40e939f66abfa81d818c084574f767e12ab40321bb077d3d26426c745e6`.

## Integración nativa revisada

Tanda `20261006T180930Z-V4-ffn-supervised-xdma-local-master-native-frozen`.
Fuente `source-20261006T180830Z.zip`, SHA-256
`ad91be615b34013c5f6fda1d23f81e3f5fa70ce81eb0c78f566b65ebeca327e8`.
El RTL conecta directamente los puertos locales del XDMA. El banco completó
44 accesos, altera el payload del solicitante tras aceptarlo, cierra admisión
y retiene cada reporte hasta consumirlo con la identidad correcta.

El maestro acepta B/R nativos cuando llegan; el backpressure se aplica en
su reporte retenido. Un verificador separado exige esa conexión directa y
correlaciona solicitud, bus y reporte. El revisor estricto de retención B/R
de R2 permanece intacto y su aprobación es una dependencia de esta tanda.
La revisión correlacionó las 44 solicitudes con respuestas nativas y reportes:
31 lecturas y 13 escrituras, latencia máxima solicitud→B/R de **48 ns**,
y retención de al menos tres ciclos en cada reporte. Las respuestas externas
BAR/H2C/C2H quedaron pendientes durante 412/352/568 ns, respectivamente.
Se recuperaron los 64 bytes H2C/C2H. Son mediciones de este banco simulado.

Se rechazaron 18 corrupciones de evidencia; se revisaron 139 avisos, 74
entradas compiladas y 129 archivos AMD sin cambios. B/R locales tienen cero
ciclos de backpressure por la conexión directa; R2 cubre su retención aparte.
Archivo: `4083c3886595795835f14a59fbd03f03d57f9ca91bfb5c63ec5c41ffa3dfeea9`.
Revisión: `0d7042f86cf133787444c908f754db0aa28f3e184ebbb855dab7a863c3b5ecad`.
Esto valida el transporte local del maestro, no el inventario global ni reset.

## Comparación funcional RTL/netlist

Tanda `20261006T183936Z-V4-ffn-supervised-xdma-local-master-replay-r2-frozen`.
Las cuatro variantes completaron diez escenarios cada una, usando sin cambios
los netlists de la síntesis OOC. El revisor reconstruyó solicitudes,
handshakes y respuestas, y rechazó 22 alteraciones de evidencia.

| Límite | Timeout de transacción a 250 MHz | Retención del reporte sin timeout | Muestras iguales por ciclo |
| --- | ---: | ---: | ---: |
| 32 | 128 ns | 41 ciclos | 783 |
| 65.536 | 262,144 µs | 65.545 ciclos | 786.831 |

En ambos límites se comprobaron los cinco sitios de timeout y el retiro
exclusivamente mediante respuestas reales. La comparación comienza tras la
configuración del simulador, a 120 ns. Los modelos AMD FDRE/FDSE introducen
`assign #100 Q = Q_out` con resolución de 1 ps: se permite solo ese
asentamiento de salidas de celdas y se comparan valores y handshakes por ciclo.
Cambios tardíos, desplazamientos de entradas y valores desconocidos después
de inicialización se rechazan. Las trazas crudas se conservan; no se afirma
igualdad de retardos de propagación ni equivalencia formal.

Se revisaron seis avisos: cuatro del entorno del compilador y dos de
redefinición de `glbl`. Las definiciones embebidas y compiladas de `glbl`
son idénticas byte por byte, incluido GSR de 100 ns. Los intentos
`20261006T182325Z` y `20261006T183413Z` permanecen como diagnósticos de
configuración Tcl y búsqueda de puertos, respectivamente. R2 usa una constante
de preprocesador del banco para evitar el override global de parámetros.

Archivo: `a7d39eafe4e8c838c49e5e796150ed269853f6b0d5fa2e2d557d74b44a95847d`.
Revisión: `6035f55d2a636441631c96798569d5b739a80e37f31980c323f24d125cad5a69`.
El tiempo de espera local es independiente del Completion Timeout PCIe
nativo de 50 ms; este último permanece sin reducir y sin calificar aquí.

## Tandas siguientes

El banco revisado `tb_ntrpu_xdma_local_master.sv` cubre diez escenarios: lectura
retenida; AW antes de W, W antes de AW y simultáneos; respuestas SLVERR y
DECERR; cierre de admisión; reporte retenido más allá del límite; timeout
en AR, AW parcial, W parcial, B y R. Cada timeout debe conservar la operación
hasta su respuesta real. Los reinicios entre grupos son vidas nuevas de
configuración del banco, no recuperación operativa.

Avanzar el
[proveedor de inventario](XDMA_INVENTORY_PROVIDER_PLAN.md), vinculándolo al
cierre real de productores y al retiro de transacciones. Integrar restricciones
reales de los puertos. Mantener el timeout nativo de 50 ms, los CDC
pendientes, DIMM exacto, cuatro canales, inferencia e implementación física.
