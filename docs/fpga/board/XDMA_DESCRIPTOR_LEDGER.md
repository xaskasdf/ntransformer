# Registro de obligaciones de descriptores XDMA

`fpga/rtl/ntrpu_xdma_descriptor_ledger.sv` registra publicaciones y
completaciones H2C/C2H en `axi_aclk`. Su contrato unitario pasó el 2026-10-06;
su síntesis OOC, comparación RTL/netlist e integración con el banco XDMA
nativo también están revisadas. La conexión al driver real sigue pendiente.

## Contrato de publicación y propiedad

`publish_valid` es una notificación de **publicación efectivamente realizada**,
una por ciclo, con canal, índice consecutivo desde cero y `{generation, tag}`.
Debe conectarse al punto donde el descriptor adquiere visibilidad y propiedad
para DMA, después de las barreras necesarias. No tiene `ready`: no es una
solicitud que este bloque pueda posponer ni una intención de software.

Los puertos de estado XDMA aportan Run, Busy y un evento Descriptor Done por
descriptor, incluidos eventos en ciclos consecutivos. El bit de evento se
cuenta por ciclo, no solamente en su flanco ascendente.
[PG195: puertos de estado](https://docs.amd.com/r/en-US/pg195-pcie-dma/Channel-0-3-Status-Ports).

Cada sesión se arma con los canales detenidos y una identidad estrictamente
creciente por orden `{generation, tag}`, igual que el coordinador existente.
No se permite un segundo flanco ascendente de Run por canal dentro de esa
sesión: podría reiniciar el contador nativo. Bajar Run conserva todas las
obligaciones. Una sesión nueva requiere liberación previa y otra identidad.

## Cierre y revocación

La entrada `host_closed` debe ser un certificado coherente y revocable del
productor. Se exige haber observado su nivel bajo antes de aceptar el cierre
de la identidad actual; un nivel alto heredado no sirve para otra sesión.
Tras cerrar, publicaciones tardías conservan su contabilización y activan un
fallo. Pérdida de readiness, retiro del cierre o cambio de identidad revocan
la aprobación; restaurar el nivel no borra el fallo retenido.

`accounted` significa únicamente que las publicaciones registradas tienen
completaciones correspondientes, ambos canales están detenidos y ociosos,
y permanece vigente el cierre. No autoriza reset ni recepción de escrituras
posted en el host. `release_valid` debe provenir del retiro global completo;
el banco unitario representa esa orden como estímulo, no demuestra su origen.
El supervisor independiente sigue siendo necesario si `axi_aclk` se detiene.

| Bit de `faults` | Condición retenida |
| --- | --- |
| 0 | Pérdida de readiness durante la sesión |
| 1 | Identidad o índice inválido; armado conflictivo/repetido |
| 2 | Publicación o arranque sin propietario |
| 3 | Completion sin obligación válida o sin Run asociado |
| 4 | Run repetido, después del cierre o sin descriptor |
| 5 | Publicación posterior al cierre |
| 6 | Capacidad del contador agotada, sin wrap |
| 7 | Liberación prematura |
| 8 | Revocación del cierre aceptado |

Los contadores saturan; un fallo nunca elimina pendientes ni permite liberar
la sesión. `cold_rst_n` se reserva a configuración, no a recuperación operativa.

## Evidencia unitaria

Tanda `20261006T190004Z-V4-ffn-supervised-xdma-descriptor-ledger-frozen`:

| Variante | Casos | Fin simulado | Uso |
| --- | ---: | ---: | --- |
| 32 bits | 15 | 664 ns | Configuración de producción, sin alcanzar wrap |
| 4 bits | 16 | 768 ns | Mismos casos y saturación al llegar a 15 |

La revisión independiente reconstruyó publicaciones, eventos Done, identidades,
contadores y revocación desde VCD. Incluye publicación/completion simultáneas,
dos canales, sesión nueva sin reset y conservación de un pendiente con
Run=Busy=0. Rechazó **22 corrupciones de evidencia**. Solo hubo dos avisos
`XSIM 43-3431`, uno por variante; las cuatro entradas compiladas se verificaron.

Archivo: `451af5a2e7e050c5fb7284da9b3c1bfa636593353f8983ba8d1b644fbfe7de57`.
Revisión: `d1b38c105a7648d59664bfab80bcbd12eecdeec37770145da63a25340e8aa5e9`.

La variante de 4 bits prueba saturación de la lógica parametrizada; no es una
observación de wrap del contador nativo de 32 bits. La asociación de CSR con
la sesión se comprobó después en el banco nativo. Faltan el secuenciador W1C
y los productores, PCIe/BAR, recepción host y mailboxes del
[proveedor completo](XDMA_INVENTORY_PROVIDER_PLAN.md).

## Síntesis OOC revisada

Tanda `20261006T190536Z-V4-ffn-supervised-xdma-descriptor-ledger-synth-frozen`,
Vivado 2026.1, `xcvu5p-flva2104-2-i`, reloj declarado de 4 ns:

| Contadores | LUT empaquetadas | FF | WNS estimado | WHS estimado |
| --- | ---: | ---: | ---: | ---: |
| 32 bits | 563 | 273 | 1,749 ns | 0,077 ns |
| 4 bits | 329 | 161 | 1,841 ns | 0,077 ns |

Los cuatro contadores retienen todos sus bits, la identidad conserva 128 bits
y permanecen los nueve fallos retenidos. Celdas y reportes coinciden; todos
los FF tienen INIT=0, sin latches, cajas negras, RAM ni DSP. Se verificaron
los DCP y su XDC embebido sin excepciones, además de seis controles negativos.

Se observaron 145 avisos: 40 por los diez bits de estado no usados en este
bloque, 100 por salidas constantes de la variante reducida, dos por ubicación
de reloj ausente, dos por cobertura DRC limitada y uno por diseño plano con
muchas primitivas. Vivado limitó los avisos de constantes a 100; se comprobaron
directamente los **112 bits constantes** del netlist para cubrir esa supresión.
Los bits de estado sin carga corresponden a IRQ, eventos latched/packet y
reservados; el ledger consume únicamente Run, Descriptor Done y Busy.

Faltan retardos de 430 entradas usadas y de 335/223 salidas no constantes;
no se especificó sitio de reloj para estimar skew. Los márgenes solo cubren
rutas internas estimadas: no hay colocación/ruteo ni aprobación física.
Los netlists funcionales se conservaron y se compararon en la tanda siguiente.

Archivo: `13b16caea5680dde5b892aeb3210d566355e71c55eec43d2eb9b68efe83fd46b`.
Revisión: `6780e61510d8536fb147cdbac9cb8fdfc4b13447483f459b5fb984e1793a9e05`.

## Comparación RTL/netlist

Tanda `20261006T192532Z-V4-ffn-supervised-xdma-descriptor-ledger-replay-frozen`:
los mismos 15/16 casos pasaron en RTL y en los netlists originales de
32/4 bits. Los eventos y casos de la prueba unitaria se conservaron, desplazados
120 ns para respetar la configuración global del simulador.

Coincidieron **333/385 muestras públicas por variante**; el fin simulado fue
784/888 ns. El revisor acepta únicamente los 100 ps de propagación de salidas
de las celdas FDRE verificadas, sin desplazar estímulos ni relojes. Es igualdad
por ciclo con asentamiento explícito, no igualdad temporal de las trazas crudas
ni equivalencia formal para cualquier estímulo.

Se rechazaron **62 corrupciones de evidencia**. Las ocho entradas compiladas
y los netlists se vincularon a sus archivos certificados. Los seis avisos
corresponden a cuatro advertencias de entorno XSIM y dos redefiniciones `glbl`:
se verificó que las definiciones compiladas y embebidas fueran idénticas,
incluido GSR de 100 ns. No se modificaron los netlists ni las fuentes unitarias.

Archivo: `1a768fb23b1990985b17893cf23dbc880715eb92f9057814e617e832074da36a`.
Revisión: `9113f7c8e7b530632e85269b8f1ada42efa44354238bbc5e61b392184b4ab2f2`.

## Integración con XDMA nativo

Tanda `20261006T193312Z-V4-ffn-supervised-xdma-descriptor-ledger-native-frozen`.
El productor del banco escribe cada descriptor completo en `DATA_STORE`,
control al final, con motores detenidos; después notifica al ledger. Se
comparan los 32 bytes de cada descriptor durante toda su propiedad y en las
completions PCIe que los entregan al XDMA. Los puertos nativos aportan Done,
Run y Busy; ocho lecturas CSR del contador coinciden con los eventos observados.

| Medida simulada | H2C | C2H |
| --- | ---: | ---: |
| Publicación del descriptor | 104,156 µs | 104,164 µs |
| Descriptor Done nativo | 105,764 µs | 111,760 µs |
| Publicación → Done, incluidas esperas del banco | 1,608 µs | 7,596 µs |
| Respuesta DMA retenida | B: 352 ns | R: 572 ns |

El banco también retuvo B de BAR durante 412 ns. Completó 44 accesos locales
(31 lecturas y 13 escrituras) mediante el maestro RTL y comparó los 64 bytes
devueltos al host. Al retirar el cierre del productor se revoca `accounted`;
restaurarlo conserva el fallo `0x100`, identidad y contadores. Fin: 125,314 µs.

La revisión verificó 75 entradas compiladas, 129 archivos AMD sin cambios,
139 advertencias heredadas y **29 corrupciones rechazadas**. El primer
revisor tenía un control que alteraba memoria inicial todavía no publicada;
R1 selecciona los bytes del descriptor publicado. Se conserva el diagnóstico,
sin cambiar las trazas, el verificador positivo ni repetir la simulación.

**Contabilidad de descriptores y recepción host son momentos distintos:**
`accounted` aparece a 111,760001 µs cuando el receptor aún tiene cero bytes;
la recepción completa ocurre a 111,864 µs, **103,999 ns después**. El VCD y
los paquetes aceptados verifican ambos puntos. Es una observación de este
banco, no una cota general de latencia ni una medida de la placa.

Solo se probó un descriptor por canal, con cierre del productor simulado.
No hay driver coherente, barreras de memoria del sistema, cadena integrada
al ledger ni autorización de drenaje global/reset. `release_valid` queda en
cero. El Completion Timeout nativo sigue en 50 ms y no expira en esta tanda.

Archivo: `63ebb08e38dbde3737e53a21998b283f99fa22046330d84bea41baf4bf93ccdd`.
Revisión: `c9880404b3a416d6e5df37ab0f5381aa1e185336e2b14a7ac9b2062f8a78ebc4`.
El registro adicional `descriptor-versus-host-receipt.json` conserva la
separación temporal entre contabilización y recepción.
