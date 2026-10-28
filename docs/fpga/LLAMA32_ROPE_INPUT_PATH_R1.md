# Entrada RoPE: cargador AXI y propietario conectados

Estado 2026-10-16: **protocolo aprobado en Icarus; síntesis/XSim encolados**.
El [camino R2](../../fpga/rtl/ntrpu_rope_input_path_r2.sv) conecta el lector
AXI, el [cargador Q/K/fases](LLAMA32_ROPE_LOADER_R1.md) y el propietario R2.
El banco utiliza un núcleo HLS sintético: esta evidencia no acredita el
cálculo RoPE, inferencia Llama ni funcionamiento sobre DDR físico.

## Contrato y propiedad

Cada comando carga Q (8.192 bytes), K (2.048 bytes) y coeficientes
(256 bytes). Conserva secuencia y época de 64 bits, y posición de 32 bits.
El propietario recibe las dos RAM sólo después de completar la carga;
la respuesta pública expone 2.560 palabras FP32 únicamente tras finalizar
el núcleo y validar su salida completa. Una respuesta ajena no recibe ACK.

Cancelación y cambios de época revocan el resultado. Se drenan las
transferencias y el núcleo antes de liberar sus memorias. La respuesta se
retiene hasta ACK; direcciones inválidas y CE inactivo no exponen datos.
El llamador mantiene las reservas originales Q/K/fases y la propiedad de V
hasta la liberación acordada con el productor. El ACK interno al cargador
no autoriza reutilizar V antes del append KV. Un peer que nunca termina
requiere recuperación/reset coordinado de todos los participantes.

## Fallo reproducido y corrección

Al cancelar exactamente entre la carga y la aceptación del propietario,
R1 observaba una fuente revocada y devolvía `BAD_PEER=11` en vez de
`CANCELLED=7`. El propietario R2 comprueba cancelación antes de rechazar
`source_result_valid=0`, conservando la precedencia de identidad, posición
y errores previos. Es una sola rama nueva; la revisión verificó esa diferencia.

Se conserva la tanda fallida `20261016T002845Z-V7-rope-input-path-native-r2`.
El mutante `handoff_cancel_lost` elimina la corrección y reproduce
`expected=7 actual=11` en el escenario 68. El propietario R1 y su paquete
ya encolado permanecen intactos; esta composición valida R2 por separado.

## Evidencia y límites

El [banco R3](../../fpga/tb/tb_ntrpu_rope_input_path_r3.sv) ejecutó tres
semillas: **195 casos, 183 respuestas, 15 resets y 122.880 palabras**.
Por semilla hubo 215 peticiones AR y 5.426 beats R, con pausas, errores,
cancelaciones durante lectura/escritura HLS, revocación tardía y recuperación.
El revisor reconstruye la salida sintética desde los bytes AXI aceptados.
Rechazó 99 alteraciones de traza y diez mutantes RTL mediante aserciones.

La primera tanda de mutantes contenía una alteración redundante de ready
y otra que sólo causaba timeout. Ambas se conservaron como diagnóstico;
la tanda R2 inyecta cambios observables y exige fallos explícitos.
Los cinco intentos tienen archivo y fuentes restaurados en D:, con copias
verificadas en NFS. Revisión: `rope-input-path-native-r1-review.json` bajo
`build/cloud-setup/20261002-board/`, SHA-256
`e8a738b95fc85cb6b03f15471e4d51ec557080fe45c7b9070f9b3620a9b09f72`.

| Tanda aprobada | SHA-256 del TAR |
| --- | --- |
| `20261016T003057Z-V7-rope-input-path-native-r3` | `766234dda9254bfd79869435fd5e86e3cf89a713ef0362a5473c20bb7d886d6b` |
| `20261016T004023Z-V7-rope-input-path-mutants-r2` | `b0c4f0335e5ad215d182d047ff6a437ec49d9a05b29ef918f7348fac4e4688e5` |

## Turno Vivado y siguiente integración

Paquete `20261016T004329Z-rope-input-path-vivado-r1`: 50 archivos, cinco
fuentes compilables, ZIP SHA-256
`ac7b3e53287a84f59347b4793dbbab337a367bd165fae41c377f9b719a4ae30f`.
El coordinador PID 4840 espera aprobación/restauración del cargador RoPE
y slot libre. Ejecutará síntesis OOC a 5 ns y tres semillas XSim, con dos
CPU, límite de 30 minutos y sin reintento automático. Cada resultado,
incluidos fallos, se archivará y restaurará antes de liberar el slot.
Respaldo del traspaso: `artifacts/ntrpu-board-vivado/rope-input-path-queue-20261016-r1/`.

Se recuperaron los 14 Verilog y cuatro generadores de IP del RoPE HLS
original. El adaptador de sus veinte puertos pasó 61 controles de contrato;
el [banco aritmético conjunto](LLAMA32_ROPE_INPUT_REAL_R1.md) ya compiló sus
interfaces y quedó encolado con las IP originales y la referencia numérica.
Su ejecución sigue pendiente; después faltan el publicador de Q rotado y
el append K/V. La interfaz de atención
ya acepta Q en streaming; su conexión y la retención de V siguen pendientes.
