# Integración de atención, FFN y residual

Actualizado el 2026-10-15, 19:27 UTC. **Wrapper R4 aprobado en control
nativo e interfaces elaboradas por Vivado**. La aritmética integrada
todavía no se ha ejecutado; su banco real está en preparación.

## Alcance y memoria

[`ntrpu_llama32_attention_ffn_r4.sv`](../../fpga/rtl/ntrpu_llama32_attention_ffn_r4.sv)
conecta los módulos reales `ntrpu_attention_post_path_r2`,
`ntrpu_post_ffn_handoff_r2` y `ntrpu_llama32_ffn_residual_r3`.
Q y K/V comprometidos se proporcionan externamente. No incluye todavía
proyecciones Q/K/V, todas las capas, logits ni generación de tokens.

Antes de permitir escrituras, captura y comprueba 16 asignaciones:
caché KV; pesos, activación y salida O; entrada, pesos, residual y ABI1
post-O; tres matrices FFN, gate, up, ABI1 intermedio, salida FFN y destino
final. Valida extensiones, alineación, desbordamiento y los 120 pares de
regiones completas. El contrato exige subasignaciones disjuntas, también
para matrices de sólo lectura; regiones contiguas son válidas. La caché
conserva además la validación de layout realizada por su motor al bind.

La identidad y los límites de KV proceden del productor, no de regiones
aportadas posteriormente. Se bloquean bind, append, datos, commit,
cancelación de append y su ACK durante preflight y ejecución. Q sólo
llega a atención cuando ésta pertenece al trabajo activo.

El mux conserva un propietario para los cinco canales AXI. La FFN recibe
ABI1 post-O de 4.096 bytes y el residual de 8.192 bytes; publica un único
vector final de 8.192 bytes. Ambos productores permanecen retenidos hasta
el ACK público. Una identidad ajena impide ese ACK; al restaurarla, el
resultado sigue revocado y se puede liberar la propiedad sin atascarse.

## Banco nativo y revisión

El [banco R4](../../fpga/tb/tb_ntrpu_llama32_attention_ffn_control_r4.sv)
compila el wrapper y handoff reales con dos peers explícitos de protocolo.
Cada peer hace dos lecturas y una escritura de un beat. La FFN de prueba
consume el dato simbólico confirmado por B de su productor; no ejecuta
las matrices ni pretende representar su volumen o latencia.

| Medida, semillas 1/7/42 | Resultado |
| --- | --- |
| Trabajos / completions | 3.102 / 3.087 |
| Solapamientos prohibidos / contigüidades válidas | 720 / 360 |
| Casos de extensión inválida | 288 |
| Descriptores ofrecidos / aceptados sin revocación | 1.959 / 1.920 |
| Lecturas / escrituras de beats de 64 bytes | 8.064 / 4.032 |
| Alteraciones de trazas rechazadas / mutantes RTL detectados | 222 / 34 |
| Ciclos por banco | 174.379 / 174.397 / 174.414 |

El [revisor independiente R2](../../tools/check_ntrpu_attention_ffn_control_r2.py)
comprueba regiones, identidades, payloads simbólicos, confirmación B,
cambio de propietario, revocación y ACK. Los 34 mutantes fallaron mediante
aserciones directas; las esperas agotadas no cuentan como detección.

La revisión R2 del wrapper tenía un driver indebido sobre la entrada
`completion_ready`. R3 lo quitó; R4 corrigió además la publicación
prematura de `completion_valid`, preservando estado e identidad antes de
anunciar la respuesta. Todos los intentos permanecen respaldados.

## Evidencia y siguiente banco real

| Evidencia | SHA-256 |
| --- | --- |
| RTL R4 | `7635abe23a1d47408299e64b399f0410917652a593efcddd061fb5be5c70c46b` |
| Banco R4 | `f181e77753267a9a2b69d36993c59eda508d0d5f2c54d879f5b3eb8576dfff9d` |
| Nativo `20261015T192005Z` | `0fc2c88d586ef97086780cd9c2974d890e04b0897a459b4401395f5f8cebd383` |
| Mutantes `20261015T192151Z` | `5676c97db775e743d6f06b736dfcc2eb155ed154f4e441fa13cb021dafee1090` |
| Revisión nativa | `bcf8a56649026526176f05624555268a3d9f95b00cb481f74916b84291c0d419` |
| Compilación Vivado `20261015T192359Z` | `b28e300e23b0ca04eb17ac3dea8a8d6daaa2d2166d581bed26469b27cb9c66c2` |
| Revisión de compilación | `212e01e96fe46a48bc5da0ca712c8d1568e610f437d80b228adb971c01433bb1` |

Vivado analizó 70 fuentes y elaboró el top: 56 RTL del proyecto, nueve
HLS residuales reales y cinco firmas aritméticas sin implementación.
No ejecutó XSim ni síntesis. Se unificó el lector AXI compartido usando
la versión de atención con puertos `input wire`; el resto de su texto
lógico es idéntico al lector de la FFN.

El inventario real contiene **178 fuentes: 56 RTL y 122 HLS**, con hashes
de sus paquetes originales y sin peers ni firmas sustitutas. A las 20:06
UTC, el [banco integrado R5](LLAMA32_ATTENTION_FFN_REAL_R1.md) está
encolado detrás del [handoff post-O](LLAMA32_POST_FFN_HANDOFF_R1.md), con
179 fuentes originales, 27 generadores IP y fixtures emparejados byte
a byte. Su revisor upstream pasó contra una traza real y 70 alteraciones;
el revisor integrado rechazó 78 alteraciones de una traza sintética.
El preflight de R5 sólo elaboró firmas aritméticas. No hay todavía
ejecución aritmética integrada. Permanecen pendientes
timing físico, DIMM exacto, DDR4/PCIe, inferencia completa y múltiples tarjetas.
