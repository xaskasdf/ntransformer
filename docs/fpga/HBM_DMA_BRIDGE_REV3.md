# Puente HBM–DMA: límites de admisión precalculados

Estado: **`REV3_BEHAVIOR_PASS`**, pendiente de síntesis y timing físico.
Tanda local: `D:/AI/artifacts/ntrpu-hbm-dma-rev3/20261009T052346043474Z`;
resultado seleccionado: `rtl2`.

## Cambio y contrato

La [frontera registrada anterior](HBM_DMA_BOUNDARY.md) dejó setup del lector
en −0,339 ns a 200 MHz. Su peor camino atraviesa la validación de rangos
hasta el enable de los contextos. Esta variante sustituye únicamente el
wrapper por `fpga/experimental/hbm_dma_rev3/ntrpu_hbm_dma_bridge.sv`.
Conserva la lane de revisión 1 y el FIFO de revisión 2. Los archivos
anteriores permanecen intactos; no compilar dos versiones del mismo módulo.

Al aceptar una configuración válida se guardan cuatro límites de 65 bits:
fin exclusivo de ventana, fin publicado y capacidades de dirección
`2^64 - source_base` / `2^64 - target_base`. Se actualizan junto al lease;
una configuración inválida o bloqueada no los altera. El reset frío los borra.

Cada solicitud calcula `offset + bytes` con acarreo, en paralelo para fuente
y destino. Comparar esos fines con los límites reemplaza las cadenas de
restas de admisión. No se añade latencia, cola ni propiedad provisional:
`submit_valid && submit_ready` sigue siendo el único punto de aceptación.
Se conservan reservas, watermark, rechazo retenido, parada y la obligación
de CPL local, DMA atribuido y ACK antes de liberar el contexto.

## Aritmética y prioridades

Para longitud positiva, `offset + bytes <= size` equivale a comprobar
`offset <= size` y `bytes <= size - offset`, usando un fin de 65 bits.
La ventana conserva además su límite inferior. El límite físico exclusivo
puede ser exactamente `2^64`: la última dirección válida sigue siendo
`2^64 - 1`. Los fines de ventana/publicación también pueden superar `2^64`
sin truncarse; las comprobaciones de tamaño y dirección siguen aplicándose.

Tras validar rango y longitud máxima de 4096, la página se comprueba con
`source_start[11:0] + bytes[12:0] <= 4096`. La suma tiene 13 bits; su máximo
relevante es 8191. Una transferencia que termina exactamente en el borde
se permite. Se conserva la comprobación de apertura HBM de 34 bits y la
alineación de 64 bytes. La prioridad permanece:
`HANDLE → PERMISSION → REPLAY → RANGE → UNPUBLISHED → SPAN → OVERLAP`.

## Validación funcional

El banco congelado instancia el wrapper anterior como referencia, con las
mismas entradas y las mismas implementaciones de lane/FIFO. Compara los
26 grupos de salidas un picosegundo después de los flancos de ambos relojes,
una vez liberados sus resets. Es comparación muestreada en simulación;
no demuestra equivalencia formal ni ausencia de glitches físicos.

| Evidencia de `rtl2` | Resultado |
| --- | --- |
| Escenarios anteriores del puente | 144 pasan |
| Corridas anteriores de FIFO, tres anchos | 72 pasan |
| Casos nuevos, 24 fases de reloj | 72 pasan |
| Vectores aritméticos contra ambos RTL y oráculo Python | 10.740 pasan |
| Transferencias / bytes reproducidos contra el modelo | 110 / 108.416 |
| Mutaciones de RTL detectadas | 4 de 4 |

Los vectores usan enteros Python sin desbordamiento, contención de intervalos
y comparación de páginas absolutas. Cubren cero, alineación, 4 KiB, 34/64
bits, acarreo, ventanas/publicación y prioridades combinadas. Los casos con
estado añaden aceptación consecutiva, replay, solapamiento entre canales,
rechazo bajo contrapresión, agotamiento de crédito, stop, configuración
inválida/ocupada y reemplazo/reset de límites. La memoria destino continúa
contrastándose con el modelo de transacciones anterior.

Las mutaciones deben compilar y fallar por una comprobación semántica:
perder el acarreo de solicitud, truncar el fin de ventana, rechazar el borde
exacto de página y cambiar límites mediante configuración inválida. No se
aceptan timeouts o errores de compilación como detección. `rtl1` conserva
la primera tanda aprobada, de 10.416 vectores; `rtl2` añade el caso de ventana
que cruza `2^64` y los cuatro controles de mutación. El RTL candidato es igual
en ambas tandas. La ejecución seleccionada dura 32,413 s localmente.

## Reproducción y respaldo

```text
python -B tools/prepare_ntrpu_hbm_dma_rev3.py NUEVO_REPO --donor TANDA_REV2
python -B NUEVO_REPO/tools/run_ntrpu_hbm_dma_rev3.py NUEVA_SALIDA --toolchain TOOLCHAIN
```

El donante es `D:/AI/artifacts/ntrpu-hbm-dma-rev2/20261009T022945511244Z`.
Se utiliza el Icarus restaurado de
`D:/AI/artifacts/ntrpu-hbm-dma-rtl/20261009T005545469986Z/restore-toolchain`.
Los runners verifican hashes de fuentes y toolchain antes/después, exit
status, marcadores completos y trazas. Las advertencias permitidas son
únicamente las conocidas sobre sensibilidad de arrays de Icarus.

Los recibos `BACKUP_*.json` conservan las tandas en el almacén local
`D:/AI/artifacts/ntrpu-content-store`. La tanda restaurada se recompila y
repite la regresión completa; `OUTCOME.json` registra esa comparación y los
156 archivos protegidos. `handoff/` incluye fuentes nuevas y procedencia.
La recompilación restaurada termina en 32,787 s, con resultados iguales y
5.564 líneas de comportamiento idénticas en 77 logs. Los cuatro mutantes
vuelven a fallar en sus comprobaciones semánticas previstas.

## Siguiente medición física

Sintetizar solo el candidato y sus tres dependencias RTL, excluyendo la
instancia de referencia y los bancos. Volver a revisar los 45 controles
CDC, 1.318 bits capturados y 12 buses Gray sobre el nuevo netlist; no reutilizar
un DCP rev2 como evidencia de rev3. Construir la misma frontera registrada
a 200/250 MHz, manteniendo presupuestos y alcance, y medir setup, hold,
ancho de pulso, fanout, recursos y ruteo. Revisar también los caminos de
configuración hacia los registros nuevos.

La expresión RTL declara 260 bits nuevos de límites; el coste sintetizado
y la mejora de timing todavía no están medidos. El resultado físico anterior
continúa abierto. Esta tanda no integra IP HBM/XDMA ni califica la tarjeta.
