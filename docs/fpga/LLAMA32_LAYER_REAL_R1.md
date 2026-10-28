# Capa completa Llama 3.2 1B: banco aritmético R1

Fecha: 2026-10-16 09:52 UTC. **Preparado y encolado; aritmética conjunta aún sin ejecutar.**

## Alcance y continuidad de datos

El [banco R2](../../fpga/tb/tb_ntrpu_llama32_layer_real_r2.sv) instancia la
[capa completa R1](LLAMA32_LAYER_R1.md) y sus ocho núcleos HLS originales:
RMSNorm/Q8, QKV, RoPE, atención, cuantización/O, post residual/RMSNorm,
SwiGLU y residual final. Es un vector de capa 0, token 0 y posición 0:
2.048 dimensiones y FFN de 8.192. No representa las 16 capas ni generación.

Las referencias archivadas de entrada y salida coinciden byte por byte en
atención, entrada de FFN y residual. No hay recuantización intermedia añadida.
El paquete conserva 64.618.496 bytes de pesos Q8 y un manifiesto con seis
padres identificados por hash. Manifiesto de los vectores:
`045f310a0b85c016ebee24cdb78c00d4132d4cf932b9ea2829e319959ddd1ba1`.

Las 13 regiones escribibles y KV empiezan inválidas. Sólo los datos W
aceptados y confirmados por B llegan a la memoria leída por el siguiente
bloque. Las referencias esperadas se usan para comparar, nunca para servir
lecturas de resultados. Hay cinco maestros AXI, límites de página de 4 KiB,
contrapresión y comprobación de identidad, WLAST, permisos y drenaje.

La cobertura prevista es 1.072.452 beats de lectura y 3.264 de escritura
de 64 bytes. El revisor exige cada dirección del plan, los datos producidos,
2.560 palabras RoPE, 32 cabezas, 3.328 palabras de estado de atención y los
8.192 bytes del resultado final. Atención se contrasta además con la
referencia Decimal de 90 dígitos; estado con tolerancia escalada 2e-12 y
salida hasta 1 ULP. La igualdad con el vector entregado a O también es exacta.

## Comprobaciones previas y revisiones

- Dos compilaciones Icarus comprobaron 84 archivos y conexiones, usando
  ocho interfaces HLS sin ejecutar y dos adaptaciones ANSI heredadas.
  Persisten 22 advertencias de enumeraciones implícitas. Vivado usa originales.
- El servicio AXI extraído sin cambios del banco pasó un caso válido:
  W antes de AW, acceso entre maestros y respuesta retenida. Rechazó siete
  accesos ilegales por aserciones explícitas, sin aceptar timeouts.
- El revisor y sus 33 controles de corrupción pasaron en Windows y Python
  3.10 Linux sobre 104.354 eventos sintéticos. Esto no es ejecución RTL.

R1 suponía 2.048 peticiones RAM consecutivas. R2 permite las repeticiones
legítimas del HLS y verifica cada retorno, su entrada efectiva al cuantizador
y cobertura de las 2.048 direcciones. El primer paquete detectó una API de
hash no disponible en Python 3.10; el cargador R2 conserva los mismos bytes
y hashes con una lectura por bloques. Los controles R4 fuerzan a cero el
contador de esperas alterado, por lo que sirven también con contadores
mayores que uno. Se conservaron los paquetes anteriores y sus diagnósticos.

## Cola y criterios de aceptación

Paquete `20261016T094855Z-llama32-layer-real-vivado-r3`, coordinador PID 10788, después de la
síntesis/protocolo exactos de la capa. Sus 239 fuentes contienen 75 RTL,
163 Verilog HLS y el banco; 38 scripts IP vendor. No contiene sustitutos
aritméticos en la compilación Vivado. ZIP SHA-256:
`48782995e74cbb7f988189ecd0a3d4556276bc9a4c7978228c8dce63d3f58767`.

El límite es 43.200 segundos y dos CPU, sin reintento automático. Esta
tanda reutiliza la síntesis anterior; no duplica ese coste. Antes de lanzar,
exige el resultado anterior revisado, archivado y restaurado, además del
slot libre. La captura DDR sigue después de BRAM, por delante de este trabajo.

La aprobación requiere resultado final, revisión numérica independiente,
comprobación de fuentes/IP realmente compiladas y retención hasta ACK.
También un fallo o timeout se archiva en NFS, se copia a D: y se restaura
antes de liberar el slot. No basta el código de salida del simulador.

Revisión nativa: `llama32-layer-real-native-r1-review.json` en
`build/cloud-setup/20261002-board/`, SHA-256
`98e1f071d5e118d14099e95ec8fb95acffea3ea3379d40f9fa67dc63bc2e39e1`.
Respaldo del traspaso: `artifacts/ntrpu-board-vivado/llama32-layer-real-handoff-20261016-r1/`.

Pendientes: aritmética de esta composición, modelo completo/logits, arbitraje
DDR externo y cierre físico. **Sin SAIF, nueva potencia ni timing ruteado.**
