# Reutilización de caché KV entre capas y tokens

Estado 2026-10-16 04:52 UTC: **protocolo aprobado en Icarus; XSim encolado**.
El [banco R1](../../fpga/tb/tb_ntrpu_qkv_layer_reuse_r1.sv) amplía la
[captura/despacho](LLAMA32_QKV_CAPTURE_DISPATCH_R1.md) sin cambiar RTL de
producción. La selección del prefijo ya usa la capa entrante antes de admitir
el comando. Esta prueba verifica su reutilización sin volver a vincular la
asignación al cambiar de capa.

## Escenarios y observaciones

El [plan de 58 trabajos](../../tools/ntrpu_qkv_layer_plan_r1.py) recorre las
16 capas, con capacidad de tres tokens por capa y distintos órdenes de
ejecución. Incluye:

- Rechazo de un token duplicado, un salto de posición y un cuarto token.
- Error en la última respuesta B; trabajo intermedio en otra capa y reintento
  de la misma posición. El error no publica el token ni invalida otras capas.
- Cancelación de un resultado ya leído, conservando el commit KV confirmado.
- Rechazo de capas 16 y `0x10000000`, sin truncarlas a una capa válida.
- Nueva vinculación cambiando el bit 40 de secuencia y época; los prefijos
  se reinician, aunque los bytes físicos anteriores permanezcan en memoria.
- Rechazo de una fuente V con época incorrecta y recuperación posterior.

Sólo hay dos vinculaciones por semilla. El banco observa, sin forzarlos,
los 16 prefijos de la guardia y del motor. Conserva la memoria AXI entre
trabajos y comprueba todos los tokens publicados después de cada drenaje.
Las escrituras parciales de un append fallido permanecen fuera del prefijo
visible hasta que un reintento completo tenga éxito.

## Evidencia y límites

Las semillas 1, 7 y 42 completaron **174 trabajos**, 153 resultados correctos,
21 rechazos esperados y tres revocaciones retenidas. Se verificaron 29.568
tramas, 4.896 cabezas y **313.344 palabras públicas**. Cada semilla realizó
88.064 comprobaciones acumuladas de tramas históricas; incluyen revisiones
repetidas del mismo token.

El [revisor R2](../../tools/check_ntrpu_qkv_layer_reuse_r2.py) reconstruye
la memoria desde AW/W/B aceptados y contrasta ambos conjuntos de prefijos
con los commits confirmados. Rechazó **864 corrupciones de traza**. Para
evitar recalcular transporte idéntico en esos controles, reutiliza su
revisión únicamente cuando todos los eventos coinciden; reconstruye el
historial de memoria y metadatos en cada control.

Seis variantes RTL compiladas fallaron por aserciones explícitas: selección
de la capa anterior, alias de prefijos en guardia o motor, alias de posición
de escritura, invalidación de otras capas tras error y conservación indebida
de prefijos al vincular. No se contaron timeouts como detecciones.

RoPE, la memoria AXI y la aritmética de atención son pares sintéticos.
Esto **no valida inferencia, DDR física, consumo ni rendimiento de tokens**.
El banco serializa comandos; el arbitraje global entre usuarios de caché
sigue siendo responsabilidad de la integración.

## Respaldos y siguiente ejecución

El revisor R1 original también terminó después del respaldo inicial. Sus
tres suites y 864 controles coinciden exactamente con R2; no se repitió RTL.
Constancia `qkv-layer-checker-r1-r2-agreement.json`, SHA-256
`939d1b5a08576e6d3e3ac7429b1c942c0edccff8abf60a8f208989d1747d91f0`.
La nueva [unión RoPE/atención](LLAMA32_ROPE_ATTENTION_R1.md) se agrega detrás
de esta prueba, sin cambiar su turno.

Fuentes y resultados restaurados en D: y NFS:

| Tanda | SHA-256 del TAR |
| --- | --- |
| `20261016T043810Z-V7-qkv-layer-reuse-native-r1` | `b588305e2e3e6050bfc02e42aef7cfc236a9d9924ebcc5a75b996e0bf3d70b15` |
| `20261016T044204Z-V7-qkv-layer-reuse-mutants-r1` | `73246c5ffa446570f53017100ad5409a6d63aaa6aec8db31dcce5c4c737c5e3f` |

Revisión: `build/cloud-setup/20261002-board/qkv-layer-reuse-native-r1-review.json`,
SHA-256 `d413cec8d366e302f834b333bf2faed9d586e8b549e1c7d17284afcd2aefb823`.

Paquete `20261016T045027Z-qkv-layer-reuse-vivado-r1`: 40 archivos, ocho
fuentes compilables, ZIP SHA-256
`4719eab01f82f1a4bc95f1ec0d412c03aea2d79c5ea1becf1799863c494d38b4`.
Coordinador PID 26856, después de RoPE aritmético integrado; sólo XSim,
semilla 1, dos CPU, límite total de 30 minutos y sin reintento automático.
Exige revisión y restauración del predecesor. Comprueba que el hardware
coincida con el paquete de captura/despacho cuya síntesis ya está en cola.

Se mantiene atención/FFN → BRAM → actividad DDR y todos sus sucesores.
La actividad DDR R1 continúa pendiente, **sin SAIF ni potencia nueva**.
