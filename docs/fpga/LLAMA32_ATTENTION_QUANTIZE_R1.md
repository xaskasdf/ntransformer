# Atención FP32 → activación ABI1: R1

Estado al 2026-10-13, 19:31 UTC: **implementado, referencia y C++ local
PASS; C/RTL encolado**. Es el bloque numérico necesario entre la atención
concatenada y el motor GEMV de proyección O. Aún no conecta ambos motores.

## Interfaz y publicación

`fpga/hls/quantize_abi1_r1/ntrpu_quantize_abi1_r1.cpp` recibe memoria local
de 2.048 palabras FP32 y entrega hasta 512 palabras de 64 bits, con control
`ap_ctrl_hs`. Admite longitudes de 32 a 2.048, múltiplos de 32. Reutiliza
sin cambios el cuantizador `ntrpu_quantize_q8_0`, llamándolo por bloque.

Cada registro ABI1 ocupa 64 bytes: 32 enteros INT8, escala FP64 y 24 bytes
cero, byte menos significativo primero. Escala `max(abs(x))/127`, redondeo
al entero más cercano con empates alejándose de cero, dominio −127…127.
Se preservan subnormales FP32 mediante la conversión por campos IEEE existente.

Los registros se ensamblan en memoria privada. La copia a la salida empieza
solo después de validar todo el vector; errores de forma o valores no finitos
dejan intacta la salida. El propietario RTL deberá retener identidad, memoria
y resultado hasta finalizar, cancelar o drenar el comando según corresponda.

## Pruebas preparadas y obtenidas

El fixture externo `attention-quantize-r1` contiene 27 casos, 16 exitosos:
cinco longitudes, ceros con signo, subnormales, máximos FP32, datos inválidos
fuera de la longitud, vecinos de empates, cuatro vectores reales completos
de atención de capa 0, formas inválidas, NaN/Inf tardíos y precedencia de error.
Los vectores reales y sus registros esperados coinciden con `attention-output-r1`.

- C++ local, MSVC 19.51 con `/fp:strict`: 27 casos PASS; 5.984 palabras,
  748 registros y 47.872 bytes exactos. Comprueba canarios, entrada inmutable,
  cola de salida y conservación completa ante error.
- Referencia independiente con `Fraction`: mismos bytes; diez corrupciones
  numéricas rechazadas. Alterar un bit de la referencia hace fallar el ejecutable.
- Revisor de transacciones: diez corrupciones rechazadas. Revisor de entradas:
  cuatro controles adicionales para datos, orden de muestras, longitud e
  inicialización de salida. Estos últimos son pruebas de los revisores,
  **no resultados de una simulación RTL**.

La tanda HLS exigirá CSIM, síntesis, cosimulación Verilog, estados C/RTL,
55.296 palabras de entrada y todas las salidas exitosas. Medirá ciclos por
transacción y recursos estimados. Los límites de serialización de trazas
binarias y cualquier puerto de error inactivo deben permanecer explícitos.

## Cola y respaldo

Propietario: `queue_llama32_attention_quantize_hls_r1.py`, detrás de wrapper
R2, guard R3, secuenciador R4 y motor R7. Espera PASS y respaldo del wrapper,
revalida hashes, ocupa el slot compartido y archiva también los diagnósticos.
Límite: 5.400 segundos, dos CPU, VU5P `xcvu5p-flva2104-2-i`, reloj HLS 5 ns.

Paquete de 36 miembros: `20261013T193041Z-llama32-attention-quantize-hls-r1`.
Fuente D:/NFS: `f12aab66846eb493d0eff368b57d7177752309c3f383a91a07feb7cded4d6e36`.
Paquete: `fee138046068865503a8797a1214c49dad4b23f2caa0bfab62babf4e8ab773fa`.
Fixture: `121d09f22261000eeacb9359c82cdf09f98e29e3106e2338f5bbd51f588c1767`.

## Integración que falta

El [controlador de propiedad R1](LLAMA32_ATTENTION_QUANTIZE_OWNER_R1.md) ya
pasó su banco nativo y está encolado para XSim/OOC. Conserva directamente la
salida raíz como fuente síncrona, sin duplicar sus 8 KiB. Tras inspeccionar
los puertos RTL generados, conectar cuantización, escritura de 64 registros
ABI1 y GEMV O de 2048×2048. Verificar
los 2.048 resultados FP32 contra la referencia, con stalls, error AXI,
cancelación, revocación y reset coordinado, sin publicar resultados parciales.

La raíz usa identidad de 64 bits; los motores GEMV existentes usan 32 bits.
El controlador de composición deberá retener y comprobar la identidad completa
y definir explícitamente las etiquetas internas. Truncar sequence/epoch no
es una adaptación válida. Residual, normalización FFN, capa integrada, DDR
física y timing ruteado siguen abiertos.
