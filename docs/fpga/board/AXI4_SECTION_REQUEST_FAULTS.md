# Solicitudes AXI4 fallidas con respuestas sanas retenidas

## Resultado revisado

Cinco causas externas pasan en los perfiles host/dispositivo **7/5 y
7/37 ns**, gestión a 10 ns, usando el Clock Converter AMD real con bus
512/64/4 y sincronizadores de tres etapas. Son diez casos vigentes:
ocho de `20261006T020548Z-V4-ffn-supervised-axi4-section-requests` y dos corregidos de `20261006T022749Z-V4-ffn-supervised-axi4-section-requests-r2`. Los seis módulos de producción
son los candidatos RTL R2 y el mailbox original, sin cambios en esta tanda.

| Causa inyectada por el bus fuente | Obligación comprobada |
| --- | --- |
| AW cambia mientras READY=0 | Retener el descriptor capturado |
| W cambia mientras READY=0 | Retener datos, strobes y WLAST capturados |
| AR cambia mientras READY=0 | Retener el descriptor capturado |
| WLAST en el segundo beat de una escritura de cuatro | Señalar longitud incorrecta |
| Falta WLAST en el beat 256 | Señalar el límite sin desbordar el contador |

Las respuestas locales sanas permanecen válidas durante el fallo y terminan
al liberar READY. Las solicitudes locales no se reenvían al CDC. Se detiene
el reloj host inmediatamente después del registro de la alarma: gestión
la observa en **22.5–27.5 ns simulados**, revoca el certificado y
rechaza una nueva generación/tag. Se conservan payloads completos, IDs,
RLAST, strobes, inventarios pendientes y los tres mailboxes.

## Corrección del estímulo de 256 beats

El banco original usaba AWLEN=255/AWSIZE=6: 16 KiB que cruzaban una página
de 4 KiB. Por ello sus dos casos de ausencia de WLAST quedan como evidencia
de entrada malformada con dos defectos; no aíslan WLAST.
[ARM IHI 0022H, A3.4.1](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf)
establece el límite de página de las ráfagas AXI.

El banco R2 cambia solamente su nombre, AWSIZE a 4 y los strobes rotativos
de 16 bytes: 256 × 16 = 4.096 bytes desde `0x100008000`, dentro de una
página. Conserva las aserciones y la ausencia deliberada de WLAST.
El comprobador R2 exige geometría válida en todos los descriptores,
rechaza ambas trazas antiguas y vuelve a aprobar las otras ocho sin
repetir Vivado. Los archivos originales permanecen intactos.

## Controles independientes

La revisión original rechaza **190 alteraciones de evidencia**. La revisión
R2 rechaza **40**, incluido un AWSIZE que
vuelve a cruzar 4 KiB. La selección inicial de una alteración del payload
retenido estaba fuera del intervalo de retención; se conserva ese borrador
y se corrige el selector, sin cambiar el comprobador ni el RTL sellados.

`20261006T022112Z-V4-ffn-supervised-axi4-section-request-mutants` ejecuta dos copias privadas de admisión con una expresión cambiada:

- Sin registro de alarma: el AW bloqueado cambia y ocho flancos después
  la alarma sigue a cero y el certificado activo. El banco termina por
  su límite de 100 µs; la traza causal, además del timeout, demuestra el fallo.
- RVALID condicionado por cualquier fallo: una R local sana desaparece
  con RREADY=0 después del fallo W. La aserción original y el inventario
  independiente detectan la retirada, con 16 beats locales aún pendientes.

Estos controles cubren los dos casos concretos indicados. No cambian el
Completion Timeout nativo de 50 ms ni demuestran cobertura de todos los fallos.

## Evidencia y siguientes pasos

| Tanda | SHA-256 del archivo |
| --- | --- |
| Matriz original | `34c1fea6788105d99d6d29ac104f83b296824acffca1b6991584422c04b0d833` |
| Controles RTL negativos | `a17e17323b070a6b29dea584fcefda5c44fb855d0187c0486933504ab43e0fe4` |
| Descriptor válido de 256 beats | `0462826bf3019e9e695d384ffc95f506cefddff02ea04e2080b060b5844f2d37` |

Fuentes, logs, revisiones y manifiestos están verificados en NFS y
`artifacts/ntrpu-board-vivado/`. Adenda R2: `498c012f3f200b2be8dc2787558a1e27145d24249c58172496b8118fd84c926d`.
Ejecutores: `tools/ntrpu_board_axi4_section_requests{,_r2}.tcl`;
revisores: `tools/check_ntrpu_axi4_section_requests{,_r2}.py`.

Las escrituras malformadas o sin pareja permanecen pendientes: no se afirma
recuperación. Las [causas W durante reenvío](AXI4_FORWARDED_FAULTS.md)
ya pasan ocho casos y un control negativo; sigue pendiente demostrar la
alcanzabilidad de capacidad defensiva del guard. El inventario XDMA,
reset global, CDC-15, DIMM exacto y diseño físico siguen abiertos. El FIFO
funcional AMD omite retardos de sincronización de datos; estas medidas no
son rendimiento físico. Ver [cola principal](VIVADO_QUEUE.md) y
[síntesis del candidato](AXI4_SECTION_SYNTHESIS_R2.md).
