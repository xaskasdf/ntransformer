# Retención del productor Q/K/V durante RoPE y atención

Estado 2026-10-16 06:13 UTC: **control RTL aprobado en Icarus;
síntesis y XSim encolados**. Los productores y consumidores del banco son
pares sintéticos de protocolo. Esta tanda no ejecuta aritmética ni AXI.

## Contrato de integración

El [wrapper de entrada completo](LLAMA32_INPUT_ATTENTION_R1.md) ya instancia
este controlador con ambos extremos reales. Tiene compilación de toda la
jerarquía y validación de protocolo con pares sintéticos; todavía no hay
ejecución aritmética conjunta.

La [consulta independiente KV](LLAMA32_CACHE_QUERY_R1.md) es necesaria para
la composición: antes de admitir el hijo, el prefijo publicado por el
wrapper R1 puede corresponder a la capa anterior. El padre debe seleccionar
la capa retenida en el puerto de consulta y reservar la caché durante la
producción y el drenaje. Los bancos sintéticos de este documento no
acreditan ese cableado entre productores reales.

El [controlador R1](../../fpga/rtl/ntrpu_qkv_rope_handoff_r1.sv) recibe la
respuesta retenida del [productor de normalización/Q/K/V](LLAMA32_INPUT_QKV_REAL_R1.md)
y ofrece un comando al [frente RoPE/atención](LLAMA32_ROPE_ATTENTION_R1.md).
No inicia el productor. El padre debe capturar sus nueve asignaciones y
la asignación de coeficientes desde el comando real y mantenerlas hasta
el drenaje final. Estos descriptores son un contrato del padre, no una
autenticación de memoria.

La admisión captura secuencia/época de 64 bits, capa, posición, lease y
diez regiones: entrada, pesos de normalización, ABI1, tres matrices Q8_0,
tres salidas FP32 y coeficientes RoPE. Valida alineación, contención y
sumas de 65 bits; limita capa, posición, capacidad y suma de leases.
Comprueba la publicación completa Q/K/V: tres matrices, 12.288 bytes y
direcciones/tamaños Q=8.192, K/V=2.048. Los coeficientes ocupan 256 bytes.

Las cuatro asignaciones de escritura no pueden solaparse con otra región.
Las lecturas pueden compartir región; ninguna de las diez asignaciones
puede solaparse con la asignación KV. El padre debe proporcionar
subasignaciones acotadas, no la totalidad de DDR como región de cada buffer.
La reserva `cache_exclusive` cubre desde la admisión hasta liberar ambos
hijos. La futura composición debe usarla para excluir nuevas vinculaciones.

Orden de retiro: **ACK público → ACK del consumidor → consumidor inactivo
→ ACK del productor → productor inactivo**. Una identidad ajena no se
confirma; espera restauración de identidad o reset coordinado. El watchdog
solicita cancelación y espera la respuesta real del consumidor.

Cambios de época, publicación fuente, metadatos de caché o respuesta hija
invalidan el resultado. La revocación tardía permanece aunque desaparezca
su causa; bloquea tanto la lectura de RAM como su validez pública. Se
conserva el certificado KV observado para diagnóstico. Un certificado de
una respuesta malformada (`BAD_COMPLETION`) no acredita un commit válido.
Reset debe coordinar productor, consumidor y sus interfaces de memoria.

## Validación nativa

El [banco R6](../../fpga/tb/tb_ntrpu_qkv_rope_handoff_r6.sv) pasó tres
semillas: **1.044 trabajos**, 612 comandos al consumidor, 1.038 ACK públicos
y del productor, 606 ACK del consumidor, nueve resets y 10.656 palabras
sintéticas de lectura. Cada escenario tiene una recuperación normal.
Incluye 70 variantes de rango/alineación, diez solapamientos con KV,
combinaciones de alias de escritura, identidades con bits altos, cambios
del descriptor del host, cancelaciones, espera, respuestas ajenas,
revocación transitoria y drenajes prolongados.

El [revisor](../../tools/check_ntrpu_qkv_rope_handoff_r1.py) contrasta cada
descriptor pretendido con el valor observado en la frontera del DUT y
reconstruye órdenes, estados, ACK y lectura pública. Rechazó **321
corrupciones de traza**. Las **25 variantes RTL defectuosas** compiladas
fallaron por aserciones explícitas; no se contó una expiración de tiempo
como detección. Los ciclos del banco no son latencias de inferencia.

Dos diagnósticos aislaron limitaciones de arrays en Icarus: salidas
conectadas a variables aparecían como `X`, y una segunda escritura con
índice variable no llegaba al puerto. R2/R4 registran valores internos y
externos diferentes. R5 usa nets de salida y drivers empaquetados con
conexiones de índice fijo; R6 agrega rechazo inmediato de comandos/ACK
ilegítimos y sondeos de lectura revocados. **El RTL R1 no cambió entre las
siete tandas**. Todas tienen fuentes y resultados restaurados en D: y NFS.

## Evidencia y cola

Revisión: `build/cloud-setup/20261002-board/qkv-rope-handoff-native-r1-review.json`,
SHA-256 `825bf38b0e0bf84aa668b3e27fd88109b76ec88810fe54f90b9338e71ce63f5b`.

| Tanda | SHA-256 del TAR |
| --- | --- |
| `20261016T060637Z-V7-qkv-rope-handoff-native-r6` | `77274465e6a693cd3f9279510a92af28eba5a702af12dd28e739fb3541d6291c` |
| `20261016T060743Z-V7-qkv-rope-handoff-mutants-r1` | `2bd3286172af9aa7f7816ba31810283818d706063171fae873b89464f0953559` |

Paquete `20261016T061142Z-qkv-rope-handoff-vivado-r1`: 46 archivos, RTL y
banco como únicas dos fuentes del proyecto, más `glbl` vendor.
ZIP SHA-256 `ecffd2322e31e9027993088e81c4edd40ad9627f021642dfe11d9b73a1751f06`.
Coordinador PID 2936, después de `rope-attention-real-vivado-r1` aprobado
y restaurado. Dos CPU, 30 minutos totales, síntesis VU5P a 5 ns OOC y tres
semillas XSim. Sin reintento automático; resultado o diagnóstico se
respalda y restaura antes de liberar el slot.

Handoff: `artifacts/ntrpu-board-vivado/qkv-rope-handoff-20261016-r1/`.
El wrapper de entrada ya conecta ambos extremos reales; falta ejecutar y
comprobar su aritmética completa desde el embedding hasta atención. O/FFN, modelo completo,
DDR físico y timing ruteado siguen pendientes. DDR R1 conserva su turno
después de BRAM; esta tanda no produce SAIF ni consumo nuevo.
