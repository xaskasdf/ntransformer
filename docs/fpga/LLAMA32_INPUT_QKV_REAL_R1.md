# Normalización y Q/K/V conectados por AXI — R1

Estado 2026-10-15 23:43 UTC: **interfaces compiladas; ejecución aritmética
conjunta sellada y encolada**. No hay todavía resultados RTL de esta tanda.
Las cantidades siguientes describen cobertura prevista.

Actualización 2026-10-16 06:13 UTC: el [control de retención hacia RoPE/atención](LLAMA32_QKV_ROPE_HANDOFF_R1.md)
pasó 1.044 trabajos de protocolo nativos y 25 mutantes. Está encolado para
síntesis/XSim. Su conexión a este productor y la validación aritmética del
camino completo siguen pendientes; no cambia el alcance de esta tanda.

## Camino de datos

[`ntrpu_input_qkv_real_r1.sv`](../../fpga/rtl/ntrpu_input_qkv_real_r1.sv)
conecta [normalización real](LLAMA32_INPUT_NORM_REAL_R1.md), el
[control de traspaso](LLAMA32_INPUT_QKV_HANDOFF_R1.md) y el
[productor Q/K/V](LLAMA32_QKV_PRODUCER_R1.md). Conserva sin cambios los
núcleos HLS originales, los controladores y sus protocolos de retención.

Un único puerto AXI de 512 bits atiende primero a normalización: lee entrada
FP32 y pesos, calcula RMSNorm/Q8 y escribe los 4.096 bytes ABI1. Después,
Q/K/V usa ese mismo puerto para leer ABI1 y los pesos Q8_0 y escribir las
tres proyecciones FP32. El selector permanece en Q/K/V hasta que terminen
el ACK público y el drenaje de ambos hijos. Las respuestas del bus se
habilitan únicamente para su propietario.

El llamador conserva las nueve asignaciones hasta el retiro final. Un reset
debe coordinar ambos hijos y el receptor AXI; este envoltorio no implementa
recuperación ante un reset aislado del bus externo. Los campos de identidad
del comando son una instantánea, no un enlace a un productor externo vivo.

## Banco preparado

El banco R2 usa cuatro embeddings originales de Llama 3.2 1B Instruct y los
pesos de normalización/Q/K/V de capa 0. ABI1 comienza envenenado. Sólo las
escrituras aceptadas, con AW/W completos y respuesta B, actualizan su memoria;
Q/K/V lee esa memoria. El ABI1 de referencia sirve únicamente como oráculo.

Los doce trabajos incluyen cuatro ejecuciones completas aceptables, una
revocación tardía, un rechazo de límites antes de cualquier tráfico, errores
de lectura y aritmética en normalización, cancelación durante RMSNorm y
cancelación en cada fase Q/K/V, con una ejecución completa final de recuperación.
El error de lectura distingue los 65 beats drenados en AXI de los tres
entregados correctamente a la RAM del cargador.

Cobertura prevista: 4.096 palabras ABI1 calculadas y comparadas exactamente,
otras 512 producidas durante normalización cancelada y descartadas,
19.968 resultados FP64 de matrices y 112.640 bytes escritos por AXI,
incluido trabajo finalmente descartado. Los resultados públicos aceptados
sumarían 49.152 bytes Q/K/V. No son medidas obtenidas todavía.

## Verificaciones realizadas

- Icarus compiló las 26 fuentes de la comprobación de interfaces. Incluyen
  dos firmas HLS sustitutas que **no se ejecutaron**. Permanecen los mismos
  19 avisos de enumerados del motor de matrices anterior, sin nuevos avisos
  de puertos. La ejecución real requiere XSim y las IP vendor.
- El revisor recalcula las cuatro normalizaciones con la referencia
  independiente y reconcilia escrituras HLS, AW/W/B y cada lectura ABI1
  de Q/K/V. Conserva la revisión numérica y temporal completa de las matrices.
- Un banco explícitamente sintético de 766.366 eventos pasó la revisión y
  rechazó 85 corrupciones. La lectura del log serializado reprodujo sus
  métricas y rechazó tres formatos inválidos. Esto comprueba el revisor,
  no el hardware.
- La tanda de compilación se restauró completamente desde su respaldo:
  `20261015T233121Z-V7-input-qkv-real-compile-r1`, SHA-256
  `fd41fcfa16bdf34f883a531fd5f04b5db6fe5707af8b9a10c3907ca228f0e89b`.

El empaquetador R3 reconoce los atributos AMD ante `module` y los dos módulos
del archivo DSP de multiplicación. Comprobó 36 módulos sin colisiones en
35 archivos HLS. Los preparadores anteriores se conservaron; estas
correcciones de inspección no cambiaron RTL ni aritmética.

## Ejecución y respaldo

Paquete `20261015T234214Z-input-qkv-real-vivado-r1`: 215 archivos,
59 fuentes de compilación, once IP y 89.447.190 bytes comprimidos.
SHA-256 `d4efac3e27ac422352a31de56d5110bbb23e196c2f78c0e8ab301d867742e1ad`.
Copias D:/NFS y restauración local verificadas. Coordinador PID 7880:
espera aprobación/restauración del traspaso; además exige aprobación
aritmética y restauración de **ambos productores**. Un diagnóstico Q/K/V
no satisface esta dependencia.

Usa dos CPU y un máximo de ocho horas para síntesis OOC a 5 ns y XSim,
sin reintento automático. La revisión contrasta fuentes realmente compiladas,
bytes, tiempos, recursos, ausencia de cajas negras, latches y lazos.
Resultados o diagnóstico se archivan y restauran antes de liberar el slot.
Traspaso: `artifacts/ntrpu-board-vivado/input-qkv-real-queue-20261015-r1/`.

La [actividad DDR](board/POWER_ACTIVITY_R1.md) conserva su turno anterior,
todavía sin SAIF. RoPE/KV conectado, inferencia multicapa, timing ruteado,
memoria física y consumo de la placa siguen pendientes.
