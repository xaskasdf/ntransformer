# Banco aritmético de atención, FFN y residual

## Resultado real, 2026-10-16 10:12 UTC

**Atención → O → post-O → FFN → residual aprobados en XSim**, un vector
completo de capa 0/muestra 1, con Q/K/V externos. Tanda `20261016T022914Z-V7-attention-ffn-real-vivado-r4`:
22.282.102 ciclos, resultado final de 8.192 bytes idéntico a la referencia.
FFN/residual leyó 56.770.560 bytes y escribió 163.840; consumió datos reales
ABI1/residuales previamente confirmados por B. El revisor rechazó 82
alteraciones. Se conservaron las 139 condiciones del banco original.

SHA-256 del vector final:
`e431a82afd68b25ebc2c6883f0efbbba697d04770a8b030ff49a48c2b9b04123`.
Revisión: `20261016T022914Z-V7-attention-ffn-real-vivado-r4-independent-attention-ffn-real-review-r1.json`, SHA-256 `3f69571b1392443b7e51558dddc98a64bee61ff2949052e6608d61abcc6cd4bf`.
TAR D:/NFS: `2338013f5c1ad22a8e736308e5d9417c567e6c54a62479600c2287b6c2c73f35`; restaurados 765 archivos
de resultados y 311 fuentes, todos verificados.

La síntesis exacta anterior se reutilizó por hash: 120.523 LUT, 106.453
registros, 159,5 BRAM, 238 DSP y WNS −10,467 ns a 5 ns OOC. **No acredita
200 MHz físicos, QKV calculados por esta tanda, todas las capas o generación.**
La [variante de cancelación](LLAMA32_POST_FFN_CANCEL_R1.md) se encoló aparte
para medir ese problema de temporización. BRAM conserva su siguiente turno;
DDR viene después. Todavía no hay SAIF ni una nueva estimación de consumo.

## Reanudación, 2026-10-16 02:31 UTC

Está activa `20261016T022914Z-V7-attention-ffn-real-vivado-r4`, PID 27592.
Usa el [banco R6](../../fpga/tb/tb_ntrpu_llama32_attention_ffn_real_r6.sv):
las **139 condiciones y mensajes originales** se conservan, con comprobaciones
directas mediante macro. Una transformación inversa verificó el cuerpo
completo contra R5. Hardware, HLS, generadores IP y fixtures no cambiaron.
Se reutiliza la síntesis exacta respaldada; sólo se repite simulación.
El resultado aritmético integrado continúa pendiente.

Observación parcial 02:36 UTC: se completaron las 32 cabezas de atención
y la traza llegó al ciclo 117.668 sin fatal. Esto confirma que el replay
superó el fallo de arranque, pero no aprueba O/post-O/FFN/residual.
La observación está copiada y verificada en NFS:
`attention-macro-live-check-r2-20261016T023628Z.json`, SHA-256
`d6064dbdd540a7e9b554e25ca25949ed1997b8eae09f3919ef2cf8044f83e886`.

El [microbanco](../../fpga/tb/tb_ntrpu_assertion_guard_probe_r1.sv)
reprodujo en XSim 2026.1 la llamada a una función `void` condicional con
`ce=we=0`, en ciclo 6. La tarea equivalente respetó el guard y detectó
el dato incorrecto. La variante macro R3 pasó doce casos entre XSim e
Icarus, incluidos cuatro rechazos deliberados en el ciclo y enable
esperados. El intento Icarus original abortó durante elaboración; R2
expuso además sustitución del nombre de parámetro dentro del texto de
`$display`. R3 cambia ese nombre sólo en el microbanco. Ambos diagnósticos
permanecen; no se usan como aprobaciones.

Se restauraron las cinco tandas: integración fallida, repetición completa
de arranque y tres microbancos. TAR aprobado del microbanco R3:
`84425911c6d9d8e04c29f28ef53f671eaf711bf1b4f24622c517e7e2548ba922`.
Fuente del replay completo:
`a2be8c5639fe656d20a1b3aa3c52ef3ed4ea1c1f56ff00839bb3f353f59c3e60`.
La revisión integrada conserva 82 controles de traza; el gate BRAM
también exige esos 82, actualizando su conteo anterior de 78.

Quince coordinadores sucesores retoman el mismo orden. Recuperación
restaurada en `artifacts/ntrpu-board-vivado/attention-macro-queue-recovery-20261016-r1/`,
SHA-256 `0c1b86d0558d49777525e9a01103f81199ac702ad210a3fced8aaf7987c21d92`.
No se modificaron las tandas anteriores. La captura DDR sigue después
de BRAM y todavía no tiene SAIF ni una nueva estimación de consumo.

## Diagnóstico de arranque, 2026-10-16 02:03 UTC

La tanda `20261016T012306Z-V7-attention-ffn-real-vivado-r2` terminó
con error antes del primer trabajo: `Actual post residual arithmetic`,
ciclo 6, 27,5 ns. Completó síntesis, pero **no valida la integración
aritmética**. Se conserva el DCP
`b74554e7f05edc6795e163ee2bd38ce846d0e9117461e0c0a71495b68f864d53`.

El archivo de resultados tiene SHA-256
`54c5dee36a0ba5cdf25ca3602b1bfaca46d0f2fd994208c02652f732cc91c28f`.
Se restauraron y verificaron los 775 archivos de resultados y 308 fuentes
en `artifacts/ntrpu-board-vivado/20261016T012306Z-V7-attention-ffn-real-vivado-r2/`;
las copias NFS y D: están verificadas. Los catorce sucesores se detuvieron
sin lanzar herramientas, incluida actividad DDR. Se conserva su orden.

Está ejecutándose `20261016T020310Z-V7-attention-startup-probe-r2`,
coordinador PID 15640: repetición de arranque con observadores pasivos,
aserciones originales y tope de 200 ns simulados / veinte minutos reales.
No repite síntesis. Conserva las fuentes de hardware originales y registra
reset, FSM, enables y aceptación por el propietario post-atención.
El ZIP fuente tiene SHA-256
`21c042497b3c5832f60729c04c14ffe99db1050872e2874bed8c3a5e93abca3e`.
Su resultado será diagnóstico y se respaldará antes de decidir una corrección.

La revisión independiente de la síntesis conservada mide 120.523 LUT,
106.453 registros, 159,5 BRAM y 238 DSP. A 5 ns, WNS es **−10,467 ns**
y WHS +0,030 ns, con cero cajas negras, latches o bucles detectados.
Son estimaciones OOC sin placement/routing: no certifican 200 MHz.
La ruta crítica tiene 66 niveles entre `held_command_epoch_reg[9]` y
`handoff/completion_status_reg[0]`. Tras resolver la función, revisar
el encadenamiento combinacional de identidad/estado entre propietarios.
Evidencia: `build/cloud-setup/20261002-board/attention-ffn-retained-synth-r1-review.json`;
esta revisión no modifica el fallo aritmético ni habilita sus sucesores.

## Preparación original

Estado al 2026-10-15 20:06 UTC: **paquete real sellado y encolado; ejecución
aritmética integrada pendiente**. Sigue al
[handoff post-O](LLAMA32_POST_FFN_HANDOFF_R1.md), con revisión y respaldo
de cada dependencia antes de ocupar el slot principal. Coordinador PID
9668; nueve coordinadores vivos, un solo slot principal.

## Contrato

El [banco R5](../../fpga/tb/tb_ntrpu_llama32_attention_ffn_real_r5.sv)
instancia el [wrapper R4](../../fpga/rtl/ntrpu_llama32_attention_ffn_r4.sv).
Procesa un vector completo: capa 0, muestra 1, token 0 y posición 0,
dimensión 2048, FFN de 8192 y las cuatro matrices Q8 originales completas
(57.933.824 bytes de pesos). Q y K/V siguen siendo entradas externas;
el banco no ejecuta sus proyecciones ni constituye inferencia completa.

La memoria materializa atención, O, post-O, gate, up, SwiGLU, down y
residual final mediante escrituras reales confirmadas por B. La entrada
FFN ABI1 y el residual empiezan sin producir: leerlos antes de su B es
un error. Los resultados de referencia sólo permiten comparar salidas.
Las 16 asignaciones son disjuntas y conservan guardas de memoria. El banco
comprueba cruces de página, backpressure, identidades de 64 bits, cambios
de propietario con AXI drenado y ACK conjunto de toda la jerarquía.

## Verificaciones disponibles

- GGUF local: 1.321.082.528 bytes; SHA-256
  `3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
- Emparejamiento de fixtures: ABI1 de 4096 bytes y residual de 8192 bytes
  idénticos entre productor y consumidor, sin recuantizar la entrada.
  Las 2048 sumas residuales de referencia coinciden mediante aritmética
  entera FP32 con redondeo explícito.
- Elaboraciones vendor conservadas: bancos R2, R3 y R5. Cada una
  compila 71 fuentes: 56 RTL, nueve HLS residuales reales, cinco firmas
  aritméticas y el banco. No ejecutan XSIM ni síntesis.
- El [revisor de la etapa anterior a la FFN](../../tools/check_ntrpu_attention_ffn_upstream_r3.py)
  pasó contra el primer trabajo de la traza vendor real de atención/post-O
  y rechazó 70 alteraciones. Esto valida el revisor de esa etapa; no
  acredita una ejecución del nuevo banco integrado.
- El [revisor integrado](../../tools/check_ntrpu_attention_ffn_real_r1.py)
  correlaciona `APREAL_B` con las 32.896 lecturas vivas de ABI1/residual,
  comprueba los cinco planes de tráfico completos, resultados exactos y
  ACK conjunto. Rechazó 78 alteraciones de una traza explícitamente
  sintética; esta prueba no acredita ejecución RTL.

R4 intentó unificar las marcas de ciclo antes del flanco de reloj, pero
Vivado rechazó el orden de declaración de `cycles`. R5 conserva ese
contador determinista y mueve su declaración antes del generador. El
intento fallido permanece respaldado; R5 pasó análisis y elaboración.

| Evidencia local, `build/cloud-setup/20261002-board/` | SHA-256 |
| --- | --- |
| `attention-ffn-fixture-pair-r2-review.json` | `158ebec0f1ae91c774dfb8579334870d7282d08d753d3143dadc6492f48a8d8a` |
| `attention-ffn-bank-compile-r2-review.json` | `6702efb8f80868925deb7fc0d8254343a521c4c4be32e26208fe0f341c04be1f` |
| `attention-ffn-upstream-checker-r3-selftest.json` | `b66d72fa4a14b0fee23009596ad1b15eae4634e73cf5da944d419c6991115faa` |
| Banco R3 | `00cb137c956e019a462e8589876d8e08df003e757514f8f5ec5c583cbcb3bda3` |
| Archivo vendor R2, `20261015T194512Z` | `ef06acd9a0b5424d2dca0a949747ec4ee73d6f5c6e1b26e61722db465288ff8c` |
| Banco R5 | `0ed8570242aac2bec0b5fe99518343aef27412ea2764efe2d0d0d7c8662c8365` |
| `attention-ffn-bank-compile-r3-review.json` | `34a19e68678cb5da768c7225f268428276b08c27bc2db9806b1f30ee25c8394a` |
| `attention-ffn-real-checker-r1-selftest.json` | `cc57674b62a16c3751ba4179837185b8f28a2dc9f60c39ceb28918c7d0abdf0d` |
| Fuente real, `20261015T200407Z` | `7aa5053ed553288e0f8bbff3c2f0a630634d75ee04731fb28a0aee7be08c2cb4` |
| Paquete real R1 | `dc357b14d7918ac8c400006fd1c5321498205474458b75517a2f2e62f6aa30f0` |

## Ejecución y criterios de aprobación

El paquete contiene 308 archivos: 179 fuentes originales compiladas
(56 RTL, 122 HLS y el banco), 27 generadores IP y los fixtures completos.
Se esperan otras 28 fuentes de simulación vendor, incluido `glbl.v`.
No contiene firmas ni peers sustitutos en la jerarquía compilada.

El coordinador exige post-O handoff R1 revisado y respaldado, además de
la evidencia aritmética previa de atención/post-O y FFN/residual. Ejecuta
síntesis OOC a 5 ns y una muestra completa XSIM, con dos CPU y límite de
ocho horas. Debe reconciliar 56.770.560 bytes leídos por FFN/residual,
163.840 escritos y 8192 finales; también verifica la traza upstream,
los hashes de todas las fuentes compiladas y las IP generadas, ausencia
de cajas negras/latches/bucles y métricas de síntesis con identidad de top.
El estado sólo será aprobado después del respaldo D:/NFS verificado.

La FFN R4 y sus siete sucesores anteriores siguen intactos. También
permanecen pendientes timing físico, DIMM exacto, DDR4/PCIe, todas las
capas/logits/tokens y múltiples tarjetas.
