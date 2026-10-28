# Traspaso de normalización a Q/K/V — R1

Estado 2026-10-15 23:23 UTC: **control nativo aprobado; síntesis/XSim encolados**.
Este banco usa respuestas sintéticas de ambos hijos. Todavía no conecta
la aritmética ni el transporte AXI de los productores reales.

## Propiedad y admisión

[`ntrpu_input_qkv_handoff_r2.sv`](../../fpga/rtl/ntrpu_input_qkv_handoff_r2.sv)
captura identidad de 64 bits y nueve asignaciones antes de iniciar trabajo:

| Región | Bytes útiles |
| --- | ---: |
| Entrada FP32 y pesos de normalización, cada una | 8.192 |
| Salida ABI1 de normalización | 4.096 |
| Pesos Q | 4.456.448 |
| Pesos K y V, cada una | 1.114.112 |
| Salida Q | 8.192 |
| Salidas K y V, cada una | 2.048 |

Verifica alineación de 64 bytes, límites con sumas de 65 bits y ausencia
de solapamiento de las cuatro asignaciones de escritura con las demás.
Dos asignaciones de lectura pueden compartir memoria. El llamador retiene
las nueve asignaciones hasta el ACK público y el drenaje final; los campos
`command_source_*` son una instantánea del llamador, no un productor externo
con un protocolo de retención implementado por este bloque.

Sólo una normalización correcta, con 256 beats leídos y 64 registros
escritos, habilita Q/K/V. Su respuesta permanece retenida mientras Q/K/V
consume ABI1. La publicación exige tres matrices completas, 12.288 bytes,
direcciones y tamaños exactos y ambos hijos todavía válidos. Los errores,
la cancelación, cambios de época y cambios de descriptor invalidan el
resultado; una revocación tardía permanece aunque desaparezca su causa.

Todo ACK público se consume una sola vez. Si la respuesta de un hijo tiene
identidad ajena, el controlador continúa drenando hasta recuperar la
identidad o aplicar un reset coordinado. No libera esa respuesta ajena ni
admite otro comando. `qkv_bus_owned` conserva la propiedad durante el drenaje
para la futura selección del bus. El watchdog solicita cancelación, pero
no finge que una operación pendiente ya terminó.

## Evidencia nativa

El banco R6 pasó **822 trabajos** en tres semillas, con **417 respuestas
públicas aceptadas**. Incluye 63 fallos de rango/alineación, 32 solapamientos
de escritura y 40 escenarios de identidad, errores, cancelación, espera,
revocación, respuesta ajena y reset, cada uno seguido por recuperación.
También comprueba solapamientos permitidos de lectura y envenena los campos
del comando después de capturarlos.

El revisor reconstruye los nueve descriptores, las etapas, los diagnósticos
y el orden de ACK/drenaje. Rechazó **174 alteraciones de trazas**. Los **28
mutantes RTL compilados** fallaron por aserciones específicas; no se aceptó
un timeout como detección. La duración por semilla fue 15.954–15.981 ciclos
de este banco sintético, sin significado como latencia de inferencia.

Se conservaron y restauraron siete tandas en D: y NFS. R1 corrigió su
sintaxis ANSI mediante RTL R2; el RTL posterior permanece idéntico. Los
bancos siguientes corrigieron una aserción de READY sin VALID y una
asignación compuesta a un elemento empaquetado que Icarus no aplicaba.
La revisión R4 dejó una traza antes/después que demuestra el estímulo
ausente; R6 aísla la alineación de las comprobaciones de contención.

Revisión: `build/cloud-setup/20261002-board/input-qkv-handoff-native-r1-review.json`.
Tanda aprobada: `20261015T231700Z-V7-input-qkv-handoff-native-r6`, SHA-256
`2e0c75c2416d3bf7fe30fbeeb4f0a11d153dc2ca583f9b7542fb475208d0ca7d`.
Mutantes: `20261015T231912Z-V7-input-qkv-handoff-mutants-r1`, SHA-256
`431ea8aefd764e29985ecb7bcf6991e1f4b714f19d6b6cef48617d757b8ee435`.

## Cola y siguiente integración

Paquete `20261015T232200Z-input-qkv-handoff-vivado-r1`: 46 archivos,
SHA-256 `3da429a0aaeb81b7b3a7b2b9ee63e81b5bd840c765fda7ba0d4ea83322f3fe07`,
con copia D:/NFS y restauración verificadas. Coordinador PID 26112: exige
aprobación y restauración de [normalización real](LLAMA32_INPUT_NORM_REAL_R1.md),
además del slot vacío. Ejecutará síntesis OOC a 5 ns y tres semillas XSim;
dos CPU, límite de 30 minutos y ningún reintento automático. Archiva y
restaura tanto resultados como diagnósticos antes de liberar el slot.

Traspaso respaldado en
`artifacts/ntrpu-board-vivado/input-qkv-handoff-queue-20261015-r1/`.
La FFN activa y la reserva [DDR R1](board/POWER_ACTIVITY_R1.md) se conservan.
La [integración real por AXI](LLAMA32_INPUT_QKV_REAL_R1.md) está implementada,
comprobada en interfaces y encolada después de esta tanda desde las 23:43 UTC.
Su ejecución aritmética y la comprobación de los bytes consumidos siguen
pendientes. Estas pruebas no acreditan SAIF, potencia, timing físico ni
inferencia completa.
