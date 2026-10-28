# Traspaso local tras perder acceso a Vivado

Fecha de corte: 2026-10-16 UTC. El usuario informa que ya no hay acceso a la
VM con Vivado. No se intentó una conexión nueva; por ello, este documento no
afirma que sus procesos remotos hayan terminado ni que el host siga activo.
Los PID remotos guardados en coordinadores son históricos, no una observación
del estado actual.

## Resultados conservados

- **DDR R1:** dos SAIF revisados, cada uno de 20,48 µs y 577.280 señales. La
  captura de 585 archivos se restauró localmente; SHA-256
  `203dd7b2de98042c6775b5e3f2a8e9b342d9caeb1f6c90070ae4f06bb93a28ff`.
  El análisis offline de estados de puerto usa el DCP archivado y su stub; en
  ambos SAIF encontró 3 entradas, 43 salidas y 108 pines `inout`. Los 108
  `inout` tienen tiempo `TZ` no nulo y ningún puerto registra `TX`. Esto aún
  no demuestra correspondencia con redes sintetizadas ni comportamiento de
  placa. Véanse [resultados R1](POWER_ACTIVITY_RESULTS_R1.md) y
  [revisión SAIF/DCP](POWER_SAIF_MAPPING_R1.md).
- **Aritmética de normalización real:** `20261016T145841Z-V7-input-norm-real-vivado-r3`
  pasó revisión como `PASS_INPUT_NORM_REAL_R1_XSIM_SYNTH_ARITHMETIC`; la
  síntesis OOC del top `ntrpu_input_norm_real_r1` en VU5P registró 12.137 LUT,
  11.434 registros, 15,5 BRAM y 11 DSP, con WNS de setup +0,978 ns y WHS de
  hold +0,039 ns a 5 ns. Son cifras de síntesis; el timing físico no está
  calificado y faltan restricciones de I/O.
  revisión SHA-256 `4da1eb110327f8c19c97926218764bbc807d92b08684468e18e4b04999dcebd7`
  y archivo de resultados SHA-256
  `9c3235b3c400e5cfdbcce9c69aad7d90f5d9a34db587513dc5df40b4bf299509`
  están restaurados localmente. No equivale a una capa Llama completa.
- **Normalización/QKV:** `20261016T153221Z-V7-input-qkv-real-vivado-r3` quedó
  como diagnóstico y respaldado, no aprobado. El revisor informa simulador
  fallido, marcador de exportación ausente y fallo de simulación independiente;
  no produjo métricas y `RVALID` con backpressure sigue sin calificar. Su
  paquete terminal local tiene SHA-256
  `b12556cb325ad99921a8930af08654e051734885115fe8cabcf803ee6544ba5e`.

## Cola y bloqueos

Los coordinadores locales de DDR R2 y de las etapas posteriores quedaron en
`stopped_requires_inspection`; no se lanzó una nueva tanda Vivado para ellos.
DDR R2 depende de `vocab-memory`, cuyo ancestro `attention-ffn-quad` se detuvo
porque el experimento de timing previo requiere inspección. El análisis de
correspondencia SAIF R1 quedó después del controlador de token; ese controlador
se detuvo porque la aritmética final de vocabulario requiere inspección. La
entrada compartida `queued_work` conserva el orden, pero no significa que un
dueño esté ejecutándose.

Antes de reanudar, recuperar acceso y almacenamiento persistente, comprobar
si quedan procesos o resultados remotos y contrastarlos con los respaldos
locales. Luego revisar los coordinadores desde sus dependencias bloqueadas y
continuar el orden registrado; no relanzar ni borrar por PID remoto copiado
de una instantánea antigua.

## Potencia y límites

No hay una nueva estimación de consumo: el diagnóstico inicial de 3,015 W no
es válido para dimensionar la tarjeta, R2 no se ejecutó y el paquete de
correspondencia no llamó `report_power`. La asociación SAIF/DCP, X/Z,
representatividad, modelo de memoria, bibliotecas y condiciones térmicas
siguen abiertos. Ninguno de estos resultados demuestra funcionamiento en
hardware físico.

Los estados y hashes de detalle están bajo
`build/cloud-setup/20261002-board/`; los respaldos recuperados viven en
`D:/AI/ntransformer/artifacts/`. La secuencia de diseño V0–V9 permanece en
[la cola Vivado](VIVADO_QUEUE.md).
