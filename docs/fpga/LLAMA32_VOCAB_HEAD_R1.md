# Control de la salida de vocabulario de Llama 3.2 1B

Fecha: 2026-10-16. Estado: **protocolo nativo aprobado, interfaces compiladas;
síntesis y protocolo XSim encolados**. La aritmética completa de esta
composición RTL todavía no se ejecutó.

Actualización 13:46 UTC: el [banco aritmético completo](LLAMA32_VOCAB_ARITHMETIC_R1.md)
quedó encolado después de DDR R2, con los pesos de las 128.256 filas,
compilación de interfaces y quince controles de revisión Windows/Linux
respaldados. Esta tanda de síntesis/protocolo conserva su turno y sus fuentes.

## Contrato de memoria y secuenciación

`ntrpu_llama32_vocab_control_r1.sv` consume una terminación retenida de la
normalización final, con secuencia y época de 64 bits. Mantiene la propiedad
de sus 4.096 bytes ABI1 hasta la aceptación pública y el drenaje del hijo.
`ntrpu_llama32_vocab_memory_r1.sv` lo conecta al motor matricial original,
con salida FP32 y dos MAC reales.

Las tres regiones retenidas contienen activaciones, pesos compactados y
logits. Sus extensiones necesarias son 4.096, 279.085.056 y 513.024 bytes.
Se comprueban alineación de 64 bytes, límites mediante sumas de 65 bits y
separación de toda la reserva de escritura respecto de ambas reservas de
lectura antes de emitir el primer trabajo. Los pesos GGUF deben convertirse
al formato compacto; este controlador no interpreta GGUF.

Cada inferencia emite 16 trabajos: quince de 8.192 filas y uno de 5.376,
con 64 bloques K de 32 elementos por fila. Los desplazamientos por segmento
son 17.825.792 bytes de pesos y 32.768 de salida. Sólo publica los 128.256
logits después de terminar todos los segmentos sin errores.

Cada orden reserva 4.096 etiquetas privadas; cada segmento usa un intervalo
de 256. El último inicio permitido es `0xfffffffffffff000`; el agotamiento
es permanente hasta un reset coordinado. Cancelación, respuestas ajenas,
timeout y cambios tardíos de identidad conservan la retención y revocan el
resultado. Un reset exige quiescencia de origen, motor hijo y AXI.

## Evidencia nativa y límites

- Seis bancos Icarus: **679 trabajos**, tres semillas, cruce de época de
  etiquetas y límites de agotamiento/alineación.
- **35 variantes RTL defectuosas** detectadas por aserciones semánticas;
  **91 transcripciones alteradas** rechazadas por un revisor independiente.
- Cancelación y fallos en los 16 segmentos, ofertas retenidas, drenaje,
  desbordamientos y revocación tardía incluidos.
- Compilación de la jerarquía de 15 fuentes aprobada; utiliza exclusivamente
  la firma de un MAC para comprobar interfaces y nunca ejecuta ese sustituto.

La primera tanda se conservó como diagnóstico: el banco tenía una dirección
base incorrecta y un caso de reserva insuficiente con holgura inadvertida.
La revisión R2 corrigió sólo el banco y sus ejecutores; el RTL R1 no cambió.
Ambas tandas y la compilación tienen respaldo y restauración en D: y NFS.
El [recorrido numérico completo en CPU](LLAMA32_FORWARD_NATIVE_R1.md) es
evidencia separada; no sustituye una ejecución aritmética del wrapper RTL.

## Turno Vivado y revisión

Coordinador `vocab-memory-vivado-r1-coordinator.json`, PID inicial 14796,
después de `attention-ffn-quad-vivado-r1-coordinator.json`. Preserva todos
los turnos anteriores, incluido BRAM → actividad DDR. Una ejecución, dos
CPU y 3.600 segundos máximos, sin reintento automático.

El paquete incorpora 22 fuentes para síntesis OOC VU5P a 5 ns, ocho Verilog
MAC generados y cuatro Tcl de IP. Se exigen un controlador, un secuenciador,
un motor paralelo y dos MAC. El banco XSim separado repetirá los 679 trabajos
con respuestas sintéticas del hijo. Recursos y timing OOC son mediciones
de síntesis; quedan pendientes route, aritmética completa y acoplamiento
con normalización final y secuenciación de las 16 capas.

Entradas y resultados se encuentran en `build/cloud-setup/20261002-board/`:

| Evidencia | Identificador / SHA-256 |
| --- | --- |
| Revisión nativa | `vocab-control-native-r2-review.json` |
| Resultados nativos R2 | `fccc1c891e15e9d8bc3421da26db68dff84e3fbb640b8dfce06436d3894a8d7b` |
| Compilación de interfaces | `6c91ee58c283021fec076c7b4c3cbcb110634b4353185f0b210426bfc5b4bada` |
| Paquete Vivado de 50 archivos | `9e431ceea47c83ca0d09d8be60dd8807b8b59e2011ef559b55be16419af7e5c6` |

El respaldo de traspaso se registra en
`artifacts/ntrpu-board-vivado/vocab-head-handoff-20261016-r1/VERIFIED.json`.
Su manifiesto identifica las dependencias originales y la restauración.
