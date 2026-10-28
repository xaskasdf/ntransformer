# RoPE original dentro de captura y despacho Q/K/V

Estado 2026-10-16 04:25 UTC: **banco compilado y paquete encolado;
aritmética integrada todavía no ejecutada**. Continúa el
[controlador de composición](LLAMA32_ROPE_QKV_DISPATCH_R1.md) sin cambiar
su RTL ni los bloques que ya tienen tandas selladas.

## Recorrido y referencias

El banco [`tb_ntrpu_rope_qkv_real_r1.sv`](../../fpga/tb/tb_ntrpu_rope_qkv_real_r1.sv)
conecta el HLS RoPE original, cargador AXI, dueño del resultado, captura,
despacho, append KV, guardia y secuenciador de cabezas. Sólo la memoria AXI
y la aritmética de atención son pares sintéticos. La respuesta de cabeza
transforma su consulta recibida; no representa atención numérica.

Las entradas y los resultados esperados están en archivos distintos.
Se conservan los 39 vectores originales de 40 cabezas: posiciones 0–131.071,
ceros con signo, subnormales, extremos, coeficientes inválidos, desbordamiento
y seis proyecciones Q/K nativas de pesos Q8, capas 0 y 15. Las activaciones
de esas proyecciones siguen siendo sintéticas; el banco no ejecuta GEMV.
Los nueve casos de cantidad de cabezas variable quedan en la prueba aislada.

El revisor reconstruye el origen desde los beats AXI aceptados y calcula
RoPE con el oráculo independiente FP64/FP32 y comprobación racional.
Contrasta cada escritura HLS válida y su llegada a captura, KV, consultas
de cabezas y lectura pública. No acepta una copia identidad en lugar de
una rotación no trivial. Las salidas revocadas no se califican numéricamente.

## Prefijos reales y criterios

El plan tiene **82 trabajos previstos**, no 82 resultados aprobados:

- 39 vectores originales y seis transacciones para poblar prefijos 0/1
  antes de probar posiciones 1/2. No se fuerzan contadores de caché.
- 31 escenarios de error/cancelación/reset y seis recuperaciones después
  de reset. Las cancelaciones abarcan cálculo, traspaso, captura y despacho.
- Se esperan 76 respuestas, 54 errores previstos, seis resets, dos
  revocaciones y 45.056 palabras públicas comprobadas.

En posiciones superiores a 2 se comprueban aritmética y captura, pero se
exige rechazo del despacho por prefijo incompleto. Cuatro de las seis
proyecciones Q8 tienen esta limitación; las otras dos deben recorrer el
append y despacho completos en posición 1, con prefijo poblado de verdad.
Un error de prefijo puede prevalecer en el estado público: el banco y el
revisor comprueban también los estados propios de RoPE y captura.

La reutilización del prefijo debe seguir un append previo confirmado por
B y por la guardia, con la misma identidad/capa/capacidad. Aún queda probar
reutilización de una misma caché al alternar capas; los cambios de capa de
este banco comienzan con un bind nuevo.

## Preparación verificada y cola

La compilación de 15 fuentes con Icarus usa la firma HLS original y un stub
que falla si se ejecuta. **No se ejecutó ese stub**, ni está en el paquete
Vivado. Tanda `20261016T041148Z-V7-rope-qkv-real-compile-r1`, TAR SHA-256
`a2889a58f6746ba54de6df75c89dc6e765104e15987b7f9b93d4e78b2d26115b`;
fuentes y resultados restaurados. Revisión de compilación:
`rope-qkv-real-compile-r1-review.json`, SHA-256
`8135d0a7efaecef60f99d137271bb6f5ccb1726d5124448e6ef5c48e720d4a20`.

El revisor R3 pasó 51 casos sintéticos positivos y 101 corrupciones.
Se conserva el diagnóstico R1 de sombreado de una variable y las revisiones
sucesoras. Esta prueba del revisor tampoco acredita ejecución RTL.
Su certificado es `rope-qkv-real-checker-r3-selftest.json`, SHA-256
`c0d29f072036ddfd4f99ec47329914a625bb8f9b10dc0779808c890d082a61da`.

Paquete `20261016T042255Z-rope-qkv-real-vivado-r1`: 76 archivos, 28 fuentes
compiladas, cuatro IP originales y fixture restaurado dentro del paquete.
ZIP SHA-256 `0717fd15229cfb4106e6c35feee2c68093cf0c9a4e103d3f52b17e1c77692566`.
Coordinador PID 21452, `rope-qkv-real-vivado-r1-coordinator.json`, espera
la síntesis exacta anterior revisada y restaurada, además del slot vacío.

La tanda hará sólo XSim, una semilla, dos CPU y límite total de 60 minutos;
recreará las cuatro IP sin repetir síntesis. Exige coincidencia de las 24
fuentes de hardware con el predecesor, inventario compilado sin sustitutos,
revisión numérica completa y 114 controles negativos sobre las trazas reales.
Un fallo o timeout se respalda y restaura antes de liberar el slot; no hay
reintento automático. DDR R1 mantiene su turno tras BRAM, sin SAIF nuevo.
