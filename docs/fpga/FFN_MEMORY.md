# FFN completa mediante memoria AXI

Actualizado el 2026-10-03 UTC. `ntrpu_ffn_memory.sv` conecta un único
[motor de matrices](MATRIX_MEMORY.md) y el [adaptador SwiGLU/Q8](SWIGLU_MEMORY.md):
**x → gate → up → SwiGLU/cuantización → down**. Las cuatro fases comparten
AXI de 512 bits y direcciones de 64 bits, con un solo reloj/reset.
Los RTL HLS de MAC y activación son los previamente validados, sin modificaciones.

Este documento conserva el contrato y las métricas de la revisión **binary64**.
La raíz vigente usa `OUTPUT_FP32=1`; su contrato, validaciones y recursos
se registran en [FFN con salida FP32](FFN_FP32.md). Los modos `sim`/`synth`
seleccionan explícitamente `OUTPUT_FP32=0` para reproducir esta referencia.

## Descriptor y regiones

El host registra ocho buffers, cada uno con dirección, base, tamaño y época.
`job_sequence`, `job_epoch` y `expected_epoch` se capturan al aceptar el job.

| Índice | Contenido | Espacio mínimo desde la dirección inicial |
| --- | --- | --- |
| 0 | Entrada x, activación ABI1 | `ceil(D/32)*64` bytes |
| 1, 2 | Pesos gate y up, layout 2 | `H*ceil(D/32)*34` cada uno |
| 3 | Pesos down, layout 2 | `D*ceil(H/32)*34` |
| 4, 5 | Resultados gate y up, binary64 | `ceil(H/8)*64` cada uno |
| 6 | Activación SwiGLU cuantizada, ABI1 | `ceil(H/32)*64` |
| 7 | Resultado down, binary64 | `ceil(D/8)*64` |

`D` y `H` están entre 1 y 8192. Cada matriz tiene `tile_k` entre 1 y 32;
se mantiene el requisito de tiles compactos múltiplos de 64 bytes.
Una forma dentro del límite dimensional puede no satisfacer ese requisito.
Por ejemplo, Brandon `D=256,H=720` admite `tile_k={4,4,8}`.
El host debe rellenar con ceros los K de pesos y activaciones fuera de la
dimensión lógica. No se añade padding de filas silenciosamente.

Antes del primer comando hijo, el secuenciador comprueba **los ocho buffers**:
alineación de 64 bytes, suma de direcciones con 65 bits, límites, épocas,
formato y solapamientos. Cada destino debe quedar fuera de todas las demás
regiones registradas, incluso de sus partes no utilizadas. Esta primera
versión no admite reutilización de intermediarios en el mismo job.

## Secuencia, errores y publicación

Las fases 0/1/2/3 son gate/up/activación/down. Los hijos reciben
`job_sequence+fase`, con wrap de 32 bits. El puerto AXI cambia de propietario
solo después de la completion del hijo anterior, que exige drenar su tráfico.
No hay CDC ni accesos concurrentes de los dos motores en esta versión.

`done_status`: 0 correcto; 1 descriptor/formato; 2 época; 5 completion hija
inconsistente; 6 error del hijo; 7 cancelado; 8 timeout del hijo.
`done_child_status`, `done_phase` y `done_hls_status` conservan el diagnóstico.
El controlador rechaza identidades, conteos y longitudes inválidos, incluido
un falso éxito de activación con retorno HLS no nulo.

`done_stages` cuenta fases completadas con éxito. `done_bytes` cuenta bytes
W de down; `done_stage_bytes` suma los reportados por completions hijas válidas.
Estos conteos no prueban persistencia. Una cancelación puede llegar cuando un
grupo de matriz ya está comprometido y terminar de escribirlo antes de responder.
**Solo una completion raíz con estado 0 permite publicar la salida.** Los
intermediarios y prefijos de jobs fallidos o reiniciados deben descartarse.

Los hijos conservan watchdog de 65.536 ciclos. El controlador no fuerza el
abandono de un comando ofrecido ni de transacciones pendientes. No tiene un
watchdog raíz independiente: un hijo que no acepte o no complete puede exigir
reset coordinado de motores y peer AXI. Un reset local aislado no recupera
de forma segura una transacción de memoria pendiente.

La salida down de esta revisión es **binary64**. La integración posterior
del [conversor FP32](FP32_OUTPUT.md) al escritor está documentada y
validada separadamente en [FFN_FP32](FFN_FP32.md).

## Validación realizada

- Controlador: **166 casos × 3 semillas = 498**, con stubs de protocolo.
  Incluye formas Brandon/Llama y máximo 8192×8192, 28 solapamientos,
  corrupción de completions, descriptores externos sobrescritos tras captura,
  backpressure, wrap, 16 cancelaciones y ocho resets por semilla.
  No demuestra cálculo numérico a esas dimensiones.
- Conjunto real RTL/HLS: **21 casos × 3 semillas = 63**. Formas 32×32,
  64×96 y 64×80; error R y B, cancelación esperando B y reset tras W en
  cada fase; recuperación y destino final inválido sin tráfico previo.
  Todos los intermediarios se escriben en el modelo RAM antes de leerse.
- Pesos sintéticos dispersos: gate es `257+2*bloque`, up tiene valores
  INT8 no uniformes y máximo 127. SiLU redondea a gate; la referencia ABI1
  y down se reconstruye independientemente en Python, byte por byte.
  El banco compara máscaras, guardas, direcciones altas y cruces de página.
- Los revisores rechazan 11 alteraciones de evidencia del controlador y
  13 del conjunto. Los 17 archivos RTL son idénticos entre síntesis y
  simulación, con watchdog 65.536 en ambos. Las dos copias HLS se cotejan
  con sus entradas originales.

Los avisos `Wavedata 42-489` impiden mostrar las RAM grandes en el visor
automático, pero no impiden simularlas ni comparar sus bytes. `XSIM 43-3431`
se conserva con las variables de entorno del compilador. No se silenciaron.

Se verificaron **74.176 bytes físicos** entre las tres semillas y 672 valores
down de jobs correctos. Los 74.304 bytes W incluyen otros 128 aceptados antes
de AW y descartados por reset; la diferencia se comprueba expresamente.

| D / H sintéticos | Ciclos, semillas 1 / 7 / 42 | R beats | W bytes / down bytes |
| --- | --- | ---: | ---: |
| 32 / 32 | 5368 / 5363 / 5359 | 62 | 832 / 256 |
| 64 / 96 | 19386 / 19425 / 19409 | 348 | 2240 / 512 |
| 64 / 80 | 18515 / 18509 / 18512 | 310 | 1984 / 512 |

Son ciclos de simulación del comando completo, con preflight, pausas AXI
y seis ciclos de espera del consumidor de completion. Para 64/96 equivalen
a 96,930–97,125 µs al reloj supuesto de 5 ns. No son tokens/s ni una medida
de DDR física. Los contadores internos `p0`–`p3` incluyen control y pueden
atribuir el primer ciclo al propietario anterior; no son tiempos exclusivos
de aritmética. La prueba de recuperación 64/96 añade otros tres jobs correctos.

## Cadena con pesos reales de Brandon

Tres muestras de la FFN de capa 0, `D=256,H=720`, pasan de extremo a
extremo con los mismos 17 archivos RTL de síntesis/ruteo. Pesos de
`xaskasdf/brandon-tiny-10m-instruct`, revisión
`ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`; SHA-256 del checkpoint
`2ff727cc8921d21e5e7b153babd288efc5524b166e9417e1b439ac8c11fe44cf`.
Gate/up reciben **la misma x sintética**; los intermediarios posteriores
provienen de las escrituras anteriores del hardware. Las entradas no son
activaciones capturadas de una inferencia completa.

| Muestra | Ciclos del comando | Error relativo L2 frente a FFN FP32 |
| --- | ---: | ---: |
| 1 | 307309 | 1,149197% |
| 2 | 307377 | 1,369425% |
| 3 | 307289 | 1,503418% |

Cada muestra realiza 418 AR / 9980 R y 82 AW / 235 W / 82 B:
638.720 bytes leídos y 15.040 escritos. Los **45.120 bytes físicos**, que
incluyen 5088 valores FP64 y 69 registros ABI1, coinciden exactamente con
una referencia Python escalar independiente. Los 768 resultados down,
convertidos a FP32 fuera del hardware, coinciden bit a bit con la FFN Q8
nativa. La diferencia L2 es cuantización respecto a FP32, no error RTL.
Las latencias equivalen a 1,536445–1,536885 ms al reloj supuesto de 5 ns;
incluyen pausas/control y no miden DDR física ni tokens/s.

`prepare_ntrpu_ffn_memory_fixture.py` empaqueta los pesos y entradas;
`check_ntrpu_ffn_memory_fixture.py` decodifica los bytes y reconstruye MAC,
SiLU y cuantización sin importar el productor ni HLS. Se rechazan seis
fixtures alterados incluso con hashes actualizados y 13 alteraciones del
log. El manifiesto de 37 archivos tiene SHA-256
`bf01bf1a0aeb709d0da6b14bb0856b0395b03f490bf976224c37a01f0e0be33a`.

## Síntesis e implementación

VU5P `xcvu5p-flva2104-2-i`, Vivado 2026.1 SW6511674, reloj de 5 ns:

| Alcance | LUT | FF | DSP | RAMB36 / RAMB18 | WNS / WHS estimados |
| --- | ---: | ---: | ---: | ---: | --- |
| Secuenciador raíz | 2701 | 3165 | 2 | 0 / 0 | +1,438 / +0,039 ns |
| Conjunto FFN | 38917 | 32200 | 93 | 39 / 3 | +0,849 / +0,039 ns |

El conjunto incluye 488 LUTRAM y 244 SRL dentro del total LUT. Los caches
gate/up con varias escrituras por ciclo siguen en registros. Se conservan
42 avisos de registros de salida BRAM no absorbidos y 16 de multiplicadores
pequeños, además de recomendaciones DSP. La eliminación del elemento
`beat_payload_bytes2` es una optimización reportada del escritor; sus conteos
funcionales se comprueban en los bancos preservados.
No hay cajas negras. En síntesis faltan retardos de 2470 entradas y 842
salidas, además del contexto de reloj padre. No constituye cierre de placa.

El ruteo exploratorio termina con **38.627 LUT, 31.997 FF, 93 DSP** y las
mismas BRAM; 404 LUTRAM y 175 SRL incluidos. WNS **+0,068 ns**, WHS
**+0,010 ns**, cero endpoints setup/hold fallidos y 61.685 nets enrutables
contadas completas, sin errores de ruteo. El margen setup es estrecho.
La ruta crítica va del comparador del MAC de la lane 0 a CE de un
acumulador; no se cambió la aritmética ni se añadieron excepciones.

Se supone un buffer padre `BUFGCE_X0Y60` mediante `HD.CLK_SRC`; el reloj
todavía contiene un gap explícito. Las auditorías cotejan **1090 aliases
PARTIAL con la misma ruta completa**. Las ubicaciones de puertos y retardos
externos siguen pendientes: 2470 entradas y 841 salidas sin retardos en
el checkpoint optimizado. Los doce avisos `Timing 38-242` nombran seis
puertos de control/AXI dos veces; el reporte declara solo `clk` a 5 ns,
sin registros sin reloj, endpoints internos sin restricción ni loops.
Estos avisos de contexto se conservan, sin inventar clocks para suprimirlos.

La auditoría original acumulaba todas las cadenas de ruteo en una lista:
570,017 s y pico de 57.688,504 MB. La revisión que compara cada alias con
una única ruta guardada tarda 75,681 s y usa 4099,055 MB. Ambas cotejan
el mismo checkpoint; el conjunto de aliases, ruta canónica, reporte global
y contexto de puertos son idénticos. Se preservan las dos evidencias.

## Reproducción y respaldos

En Linux con Vivado cargado, usar rutas nuevas y las dos co-simulaciones
HLS previamente respaldadas como entradas:

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_sequencer.tcl \
  -tclargs sim NEW_CONTROL
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_HLS_RUN" "$SWIGLU_HLS_RUN" sim NEW_MEMORY
# Fixture real recuperado del respaldo de la tanda Brandon
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_HLS_RUN" "$SWIGLU_HLS_RUN" sim NEW_BRANDON "$FFN_FIXTURE"
python tools/check_ntrpu_ffn_memory_fixture.py "$FFN_FIXTURE" \
  --log NEW_BRANDON/seed-1.log
```

Cambiar `sim` por `synth` produce un checkpoint independiente. Los lanzadores
copian las fuentes HLS; no regeneran ni alteran las tandas anteriores.
Evidencia local en `artifacts/ntrpu-board-vivado/<run>/`, con archivo,
fuentes, logs, `result.md`, revisión y `VERIFIED.json`:

- `20261003T085310Z-V7-ffn-sequencer-unit` y `...-synth`.
- `20261003T085902Z-V7-ffn-memory-synth`.
- `20261003T090406Z-V7-ffn-memory-unit`.
- `20261003T090959Z-V7-ffn-memory-route`.
- `20261003T092050Z-V7-ffn-memory-route-audit` y
  `20261003T092954Z-V7-ffn-memory-route-audit-stream`.
- `20261003T093914Z-V7-ffn-memory-brandon`: 638,627 s; archivo de
  94.775.294 bytes, 320 archivos verificados, SHA-256
  `b99577576149e2cc62e6ce01449178a631cfb7f2a0636e6acc8b572f7497023d`.

Siguiente: ampliar las dimensiones de la cadena con salida FP32 integrada.
Estas FFN aisladas no son inferencia Brandon/Llama ni rendimiento
físico. DMA, DDR, runtime de buffers, pesos Meta autorizados y restricciones
de placa conservan las dependencias de [la cola](board/VIVADO_QUEUE.md).
