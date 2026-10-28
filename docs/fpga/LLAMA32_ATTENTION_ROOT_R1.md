# Wrapper de atención completa: revisión R1

Estado al 2026-10-13, 19:20 UTC: **guard de mantenimiento probado; banco
del wrapper compilado y encolado; integración aritmética pendiente**.

## Propiedad y metadata

`fpga/rtl/ntrpu_attention_root_r1.sv` conecta el motor por cabeza, el
secuenciador de 32 cabezas y `ntrpu_attention_root_guard_r1`. El generador
`tools/build_ntrpu_attention_root_wrapper_r1.py` clasifica los 102 puertos
del motor y rechaza cambios de interfaz no contemplados.

La admisión tiene prioridad bind → append → comando raíz. Una solicitud de
mantenimiento pendiente también impide admitir una raíz mientras su peer
aplica backpressure. Desde la aceptación de bind/append hasta consumir su
finalización, ninguna nueva raíz puede entrar. El bloqueo del secuenciador
impide todo bind/append durante carga Q, cómputo, copia y resultado retenido.

El guard mantiene 16 prefijos confirmados. El host no proporciona el prefijo
que autoriza atención: se deriva de la finalización del append aceptado,
con capa/posición e identidad retenidas, estado exitoso, 4.096 bytes y
`committed_tokens == position + 1`. Una respuesta retenida todavía no publica
metadata. Un append fallido conserva el prefijo anterior; una respuesta
inconsistente invalida la autoridad hasta un bind nuevo exitoso.

Un bind aceptado borra todos los prefijos. Cambiar `expected_epoch` revoca
la autoridad y volver al valor anterior no la restaura. La metadata inválida
produce un error de admisión en el secuenciador. Un reset compartido borra
propiedad y autoridad; requiere quiesce coordinado del motor/AXI.

## Evidencia disponible

- Guard nativo: tanda `20261013T183157Z-V7-attention-root-guard-native-r1`.
  Tres semillas, 288 snapshots, 4.896 comprobaciones por capa y 12 resets.
  Archivo D:/NFS: `42cac588a0e8131d3648c3c061e69693f73f1f62be63a3e095c3ba1407610cdd`.
- Tres mutantes compilados fallaron como se esperaba: ignorar bloqueo raíz,
  consumir metadata antes del handshake y restaurar autoridad tras rollback
  de epoch. El revisor rechazó ocho corrupciones por suite.
  Revisión: `60c29462a4861ab93191f13ffc1f5078773ab253ee0ec5902cd2ca4b0364d98a`.
- Wrapper: compilación Icarus `20261013T183952Z-V7-attention-root-interface-compile-r1`.
  Archivo: `da9455926846d408d1d23bd5d7371c41d2ea75781f0a231edb5e1f8947bbc0b9`.
  Usa declaración de interfaz HLS; **no ejecuta aritmética ni califica comportamiento**.
- Guard Vivado: propietario `queue_llama32_attention_root_guard_vivado_r3.py`,
  detrás de secuenciador R4 y motor R7. Tres suites XSim y síntesis OOC a 5 ns,
  con revisión y respaldo antes de liberar el slot. Las fuentes del guard
  ya están congeladas. El texto de alcance del runner heredó «140»; el banco
  y el revisor exigen 96 snapshots por cada una de tres suites, 288 en total.

## Banco completo y dependencias

`tb_ntrpu_attention_root_r2.sv` cubre 23 escenarios, 426 consultas hijas y
24.576 resultados públicos, con las cuatro muestras de 32 cabezas de
`attention-output-r1`. La referencia contiene 8.192 FP32 distintos en orden
`head*64 + lane`; el generador verificó su igualdad con los resultados por
cabeza de `attention-axi-r1` y la entrada de proyección O.

El banco intenta bind/append concurrentes durante carga, cálculo, copia y
resultado retenido. Incluye capa sin prefijo confirmado, identidad Q errónea,
índice Q duplicado, cancelación durante carga y cálculo, error AXI, revocación
de epoch durante copia y tras publicar, stalls y cuatro fases de reset
coordinado. No modela reset unilateral con tráfico AXI sin drenar.

Cada copia se compara con la referencia; la salida concatenada debe coincidir
bit a bit con lo copiado y estar a ≤1 ULP de la referencia. El revisor exige
trazas de propiedad, datos AXI, estado interno y 24 controles de corrupción.
Los controles aún deben ejecutarse sobre la traza XSim real.

La compilación Icarus de interfaz pasó en
`20261013T185818Z-V7-attention-root-interface-compile-r3`, archivo
`070f160bfda0d8a629ed8e67d1652e3627f9f6b8dc3d805e6b2c6c907003b536`.
La primera revisión usaba `break`, no soportado por Icarus; su diagnóstico
se preservó. Esta compilación no ejecuta ni califica aritmética.

La cola vigente es **motor R7 → secuenciador R4 → guard R3 → wrapper R2**.
El wrapper usa `queue_llama32_attention_root_vivado_r2.py`; verifica archivos
terminales, revisiones y hashes de hardware de los tres padres antes de tomar
el slot compartido. El paquete R1 anterior nunca se lanzó: dependía del motor
R6, que terminó con diagnóstico. Se conservan ambas revisiones.

Fuentes R2: 158 miembros, respaldo D:/NFS en
`20261013T191938Z-llama32-attention-root-vivado-r2`; SHA-256
`10abd430b1adbb9c001eae84ac50290c5b447087149b6051bd7d0647f1ad54cf`.
Paquete: `836dbbd874adeec5c9841a155d3461a2edf0e7f9d796d698ee5b230877d67a12`.

Después se conecta cuantización ABI1 → proyección O → residual → entrada
FFN. DDR física, timing ruteado e inferencia completa siguen pendientes.
