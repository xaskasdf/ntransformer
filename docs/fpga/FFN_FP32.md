# FFN AXI con salida FP32

Actualizado el 2026-10-03 UTC. La raíz `ntrpu_ffn_memory` incorpora el
[conversor comprobado](FP32_OUTPUT.md) al escritor de resultados. El flujo
sigue siendo x → gate → up → SwiGLU/Q8 → down; la última matriz escribe
FP32 directamente. Los resultados previos con salida binary64 se conservan
en [FFN_MEMORY](FFN_MEMORY.md).

## Formato y publicación

`OUTPUT_FP32=1` es el valor predeterminado de la raíz. Gate/up siguen
escribiendo binary64. El controlador captura un formato por comando hijo;
no se permite cambiarlo durante una matriz. Los componentes genéricos de
matriz/escritor conservan `FP32_CAPABLE=0` por defecto; la raíz habilita
esa capacidad y selecciona FP32 exclusivamente para down.

El destino 7 mantiene alineación de 64 bytes y requiere
`ceil(D/16)*64` bytes accesibles, frente a `ceil(D/8)*64` en binary64.
Se almacenan `D*4` bytes little-endian, 16 valores por beat, con máscaras
de cuatro bytes para la cola. Los grupos de 32 filas avanzan 128 bytes.
Se comprobaron los spans, solapamientos y conteos en cada nivel:
controlador FFN, matriz, GEMV y escritor.

El escritor conserva todos los resultados del grupo en un buffer privado.
El commit puede llegar junto con el último resultado FP64, pero su ready
espera que se vacíe el conversor. Un contador de elementos pendientes,
independiente del contador de filas inválidas, permite drenar incluso un
productor que envíe más de 63 resultados. NaN, infinito o magnitud superior
a FLT_MAX producen BAD_RESULT=5 y prohíben AW/W de ese grupo.
El primer error se conserva; la raíz devuelve CHILD_ERROR=6 cuando el hijo
falla. Las escrituras de grupos anteriores permanecen privadas.

Cancelación antes del commit drena el productor; después del commit se
termina el grupo comprometido y se esperan las respuestas B. Un timeout
no abandona un handshake AXI ofrecido. Solo completion raíz correcta
autoriza publicar. Reset debe coordinarse con el peer AXI y descartar
intermediarios y destino, igual que en el contrato anterior.

## Validaciones de integración

- Escritor mixto: 57 casos por semilla, 171 RTL y 171 sobre netlist de la
  primera integración. Comprueba todos los bytes y máscaras, cruces de
  página, errores B/BID, timeout, descriptor sobrescrito, commit adelantado,
  cancelación y tres resets. Incluye subnormales, empates, ±FLT_MAX y su
  vecino binary64 exterior; ese vecino es rechazado en FP32 y admitido
  en binary64. Cada semilla verifica 2608 bytes escritos.
- Regresión del escritor binary64: 126 casos; controlador raíz: 498 casos
  en cada formato. Las formas Llama de este banco son contratos de
  protocolo, no cálculo numérico a esas dimensiones.
- Primera cadena FP32: 63 casos con MAC/SwiGLU reales. Sus 71.040 bytes
  físicos coinciden con la referencia; 71.232 bytes W incluyen 192
  aceptados antes de AW y descartados por reset. Cuatro fases con
  error R/B, cancelación y reset por semilla, más recuperación.

La versión inicial pasó funcionalmente, pero el escritor infirió dos
puertos de escritura y utilizó 4744 LUT/2867 FF. Una optimización intermedia
colocó por error la escritura en `always_comb`: el banco falló en el primer
resultado y síntesis infirió 32 latches. Sus cuatro tandas `104133Z` se
conservan como **rechazadas**, incluida una síntesis aislada que salió con
código 0. Solo la síntesis FFN de esa revisión inválida fue detenida por
su PID/ruta verificados; no se interrumpieron validaciones anteriores.

La revisión corregida utiliza un único puerto en `always_ff` y pasa
171 casos RTL y otros 171 sobre su netlist, además de 63 casos de cadena
FP32 y 63 de regresión binary64. La cadena FP32 conserva exactamente los
71.040 bytes físicos de la primera integración. La regresión binary64
verifica 74.176 bytes. Cada revisión conserva su propia fuente y respaldo.

## Recursos de la revisión corregida

| Síntesis OOC, reloj 5 ns | LUT | FF | DSP | RAMB36 / RAMB18 | WNS / WHS |
| --- | ---: | ---: | ---: | ---: | --- |
| Escritor FP32/binary64 | 1988 | 2861 | 0 | 0 / 0 | +2,578 / +0,066 ns |
| Cadena FFN FP32 | 40.407 | 34.617 | 93 | 39 / 3 | +0,959 / +0,039 ns |

El escritor reduce LUT un 58,1% frente a la primera integración; la cadena
baja de 44.703 a 40.407 LUT. El buffer del escritor continúa en registros:
no se recuperó inferencia de RAM. No se infieren latches. La cadena mantiene
168 LUTRAM y 244 SRL en otros bloques. Faltan retardos de 2470 entradas y
845 salidas; estos márgenes de síntesis no son timing de placa.

## Implementación y reloj

El ruteo exploratorio del DCP corregido usa **40.080 LUT, 34.427 FF,
93 DSP, 39 RAMB36 y 3 RAMB18**. A 5 ns cumple setup/hold internos:
WNS **+0,164 ns**, WHS **+0,010 ns**, cero endpoints fallidos.
Las 65.492 nets enrutables están completas y no hay errores de ruteo.
El camino más ajustado va de `fill_count_reg[8]` a la dirección de la BRAM
del banco de pesos; no termina en el conversor FP32.

La auditoría conserva **un gap explícito de reloj** y 1051 segmentos
jerárquicos PARTIAL con exactamente la misma ruta. `HD.CLK_SRC=BUFGCE_X0Y60`
es una suposición del contexto OOC, sin driver del padre ni ubicaciones de
puertos. Siguen faltando 2470 retardos de entrada y 845 de salida.
Los avisos de contexto nombran `cancel_valid` y `job_valid`; no se añadieron
clocks ni excepciones para silenciarlos. CDC omite puertos no restringidos.
Se comprobaron cero latches LDCE/LDPE también en el checkpoint ruteado.

DRC/metodología conservan recomendaciones DSP, reset y temporales, además
de RTSTAT-10 por el reloj. Durante el ruteo se informó congestión; el
reporte final anterior determina los márgenes. Esto valida la implementación
interna bajo ese contexto, **no cierre de placa**. Hace falta integrar reloj
real, interfaces registradas y retardos externos, y repetir implementación.

## Pesos reales de Brandon

Tres cadenas de capa 0, `D=256,H=720`, escriben **42.048 bytes físicos
exactos**: 4320 valores gate/up binary64, 69 registros ABI1 y 768 valores
down FP32. La referencia independiente reconstruye cada byte y comprueba
que los FP32 almacenados coinciden bit a bit con la FFN Q8 nativa.

| Muestra | Ciclos | Error relativo L2 frente a FFN FP32 |
| --- | ---: | ---: |
| 1 | 307.247 | 1,149197% |
| 2 | 307.382 | 1,369425% |
| 3 | 307.387 | 1,503418% |

Cada comando lee 638.720 bytes y escribe 14.016: 418 AR, 9980 R, 82 AW,
219 W y 82 B. Down ocupa 1024 bytes y 16 beats menos que binary64;
el total escrito baja un 6,81%. Los ciclos siguen en el mismo orden que
la referencia anterior: esta reducción no demuestra aceleración.
Las pausas de AR/AW/W/completion están activas.

Se reutilizan los 37 archivos del [fixture anterior](FFN_MEMORY.md),
manifiesto SHA-256 `bf01bf1a0aeb709d0da6b14bb0856b0395b03f490bf976224c37a01f0e0be33a`.
Pesos: revisión `ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`; x sintética
normalizada, compartida por gate/up. No son activaciones de un prompt
ni inferencia completa. Ocho alteraciones deliberadas de la evidencia,
incluidos un byte y una máscara, fueron rechazadas.

## Procedencia

La revisión corregida de simulación, síntesis y replay usa la fuente
`source-20261003T104423Z`, SHA-256
`4611cdeef61daa6af971153198d1bdadabf19e831bef4b270f4e75c1a830ac28`.
Los 18 RTL coinciden entre esas ejecuciones. El DCP FFN sintetizado tiene
SHA-256 `60acdc5f37e452e2198b33770c991c7f081c951f7fa7a090340e970c19382843`.
Tandas en `artifacts/ntrpu-board-vivado/`:

- `20261003T104452Z-V7-fp32-writer-ram-r2-unit`.
- `20261003T104647Z-V7-fp32-writer-ram-r2-synth` y
  `20261003T104951Z-V7-fp32-writer-ram-r2-netlist`.
- `20261003T104647Z-V7-ffn-fp32-memory-ram-r2-unit` y `...-synth`.
- `20261003T104647Z-V7-ffn-binary64-format-regression`.
- `20261003T104647Z-V7-ffn-fp32-brandon`.
- `20261003T105639Z-V7-ffn-fp32-memory-route` y
  `20261003T110717Z-V7-ffn-fp32-memory-route-audit`.

DCP ruteado SHA-256:
`a4ae7a2f5c9e04d3a23ed7bc47d5547b8509dcf7bdde6d0687404d66a626d512`.
Ruteo: 590,806 s; auditoría de solo lectura: 76,982 s y pico 4113,297 MB.
La auditoría usa una copia posterior de scripts/documentación; verifica
el mismo DCP de entrada y no modifica sus constraints.

El nombre histórico `ram-r2` identifica el intento de optimización;
no afirma que el escritor infiera RAM. Cada respaldo contiene fuentes,
invocación, logs, revisión numérica y hashes en `VERIFIED.json`.

## Reproducción

Usar carpetas nuevas y los dos RTL HLS respaldados como entradas:

```bash
vivado -mode batch -source tools/ntrpu_board_fp32_writer.tcl \
  -tclargs sim NEW_WRITER
vivado -mode batch -source tools/ntrpu_board_ffn_sequencer.tcl \
  -tclargs sim_fp32 NEW_CONTROL
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_HLS_RUN" "$SWIGLU_HLS_RUN" sim_fp32 NEW_MEMORY
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_HLS_RUN" "$SWIGLU_HLS_RUN" sim_fp32 NEW_BRANDON "$FFN_FIXTURE"
python tools/check_ntrpu_ffn_memory_fixture.py "$FFN_FIXTURE" \
  --log NEW_BRANDON/seed-1.log --fp32
```

`synth_fp32` sintetiza la raíz FP32. Los modos históricos `sim`/`synth`
seleccionan explícitamente binary64 para reproducir el contrato anterior.
El sintetizador vigente rechaza almacenamiento transparente inesperado.
V2/V3, DMA host, DDR real, runtime, dimensiones numéricas Llama y cierre de
placa conservan sus dependencias en [la cola](board/VIVADO_QUEUE.md).
La ampliación numérica en curso se sigue en [FFN_LARGE](FFN_LARGE.md).
