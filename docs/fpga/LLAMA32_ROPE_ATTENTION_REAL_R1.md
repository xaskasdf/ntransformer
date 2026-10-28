# Banco aritmético integrado RoPE → KV → atención

Estado 2026-10-16 05:40 UTC: **compilación preliminar aprobada y XSim
encolado; todavía no se ha ejecutado la aritmética conjunta**.
El [banco R1](../../fpga/tb/tb_ntrpu_rope_attention_real_r1.sv) instancia
el [wrapper de producción](LLAMA32_ROPE_ATTENTION_R1.md), con sus 25
módulos RTL y ambos núcleos HLS originales. No modifica el hardware.

## Entradas y alcance

El [generador de datos](../../tools/ntrpu_rope_attention_fixture_r1.py)
reconstruye Q/K/V de la primera capa desde cuatro embeddings reales del
GGUF Q8_0 de Llama 3.2 1B Instruct: tokens `0`, `42`, `128000`, `128255`.
Usa la normalización y cuantización ABI1 del proyecto. Los cálculos de
proyección se realizan en el preparador; no son GEMV ejecutado por este banco.
Posiciones `0…3`, coeficientes calculados con Decimal90, RoPE contrastado
con aritmética racional y 128 referencias independientes de cabezas de atención.

La memoria AXI externa empieza sin datos válidos. Sólo los W realmente
aceptados y confirmados por B exitoso alimentan las lecturas KV posteriores.
No se precarga KV con resultados esperados ni se fuerzan metadatos internos.
Los tres maestros tienen esperas deterministas; las bases superan 32 bits
y algunos accesos cruzan páginas. El descriptor del host cambia después
de la admisión para comprobar su retención.

| Trabajo | Estímulo | Resultado previsto |
| --- | --- | --- |
| 1–2 | Posiciones 0 y 1 | Prefijos 1 y 2; 32 cabezas por comando |
| 3–4 | Posición duplicada 1 y salto a 3 | Rechazo sin append ni atención |
| 5 | Posición 2, error en el último B | Error; prefijo lógico permanece en 2 |
| 6–7 | Reintento de 2 y posición 3 | Prefijos 3 y 4 |
| 8 | Nueva vinculación y posición 0 | Prefijo 1 bajo nueva identidad |

Cobertura prevista: ocho ejecuciones RoPE, 20.480 palabras RoPE exactas,
160 cabezas de atención, 16.640 palabras de estado privado, 10.240 salidas
públicas, 384 escrituras y 2.816 lecturas de tramas KV de 64 bytes.
Las respuestas públicas retenidas excluyen una vinculación concurrente.

## Revisión independiente

El [revisor](../../tools/check_ntrpu_rope_attention_real_r1.py) reconstruye
direcciones, AW/W/B, memoria persistente y prefijos. Conecta las respuestas
AXI con las lecturas HLS, los Q/K/V efectivos, los estados y las salidas.
Comprueba máximos FP64 y query almacenada exactos, estado softmax contra
Decimal90 con tolerancia escalada `2e-12`, salida FP32 de hasta una ULP y
publicación pública idéntica a la salida HLS observada.

Una traza **sintética** completa pasó el revisor y 168 corrupciones fueron
rechazadas. Esto sólo prueba el revisor. No demuestra ejecución RTL. Las
lecturas HLS redundantes son válidas; no se afirma detectar la eliminación
de cualquier observación redundante.

## Evidencia, ejecución y recuperación

Compilación Icarus de 28 fuentes, incluidos dos stubs HLS no ejecutados:
`20261016T052505Z-V7-rope-attention-real-compile-r1`. TAR SHA-256
`cec1fc44940245cd64b291641bfc91847c6eca814d2140d5da3d1514e7ee5435`.
Fuentes y resultados restaurados; revisión SHA-256
`9357909a9368d0f130f86c772116729eb3fc79e36ae92351feab12ea2a2ea77b`.
Los stubs quedan excluidos de Vivado.

Paquete `20261016T053857Z-rope-attention-real-vivado-r1`: 113 archivos,
67 fuentes compilables —25 RTL, 41 Verilog HLS originales y el banco—,
diez IP vendor y `glbl`. ZIP SHA-256
`18e8b86d79627fe2e68ab8ed6288cc2b8d1f3ef195a9eaf8ff74d0f3fba31f1e`.
Fixture SHA-256
`68e824d0c874aefe7fb100a1a2e8e3b8c90ee1be15b21f5bf4b9488c209a4bb4`.

Coordinador PID 18736, al final de la cola, después de
`rope-attention-synth-r1-coordinator.json`. Exige síntesis aprobada, respaldo
restaurado, identidad de las 66 fuentes de hardware y slot libre. Ejecutará
sólo simulación, con dos CPU y límite total de 60 minutos, sin reintento
automático. Revisa las fuentes compiladas, el log y sus 168 controles;
archiva y restaura éxito o diagnóstico antes de liberar el slot.

El handoff documental y la traza sintética se conservan en
`artifacts/ntrpu-board-vivado/rope-attention-real-handoff-20261016-r1/`.
Las copias del paquete ejecutable y compilación están en D: y NFS.
Quedan pendientes XSim, integración de proyección O/FFN y ejecución del
modelo completo. No califica DDR físico, cierre temporal ni potencia.
La captura DDR R1 mantiene su turno después de BRAM, sin SAIF nuevo.

Actualización 2026-10-16 06:13 UTC: el [traspaso del productor Q/K/V](LLAMA32_QKV_ROPE_HANDOFF_R1.md)
ya tiene control RTL probado con pares sintéticos. Su síntesis/XSim se añadió
después de esta prueba, PID 2936. Este banco aritmético conserva el mismo
paquete y aún espera su turno; la composición con normalización y GEMV real
requiere un envoltorio posterior.
