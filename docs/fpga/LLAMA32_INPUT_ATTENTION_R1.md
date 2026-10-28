# Composición desde la entrada hasta atención

Actualización 2026-10-16 07:56 UTC: el [banco aritmético completo](LLAMA32_INPUT_ATTENTION_REAL_R1.md)
ya compila y está encolado después de esta síntesis/protocolo. Usa los
cuatro HLS originales y memoria alimentada por escrituras reales.
Todavía no tiene resultado aritmético conjunto.

Estado 2026-10-16 07:06 UTC: **wrapper real cableado y compilado;
protocolo aprobado en Icarus; síntesis y XSim encolados**. El banco nativo
ejecuta el nuevo propietario y el controlador de retención reales con
productor y consumidor sintéticos. Todavía no ejecuta su aritmética conjunta.

## Camino de datos y propiedad

[`ntrpu_input_attention_r1`](../../fpga/rtl/ntrpu_input_attention_r1.sv)
instancia el productor real RMSNorm/Q8→Q/K/V, el controlador de retención
Q/K/V→RoPE y el frente real RoPE→append KV→32 cabezas de atención. El
[generador](../../tools/build_ntrpu_input_attention_r1.py) comprueba los
hashes de sus tres padres y las anchuras y direcciones de 178 puertos
públicos. Ningún padre archivado fue modificado.

El comando captura identidad de 64 bits, capa, posición, lease, origen y
diez asignaciones. **Antes de admitir el productor**, revisa las diez
regiones y las cuatro regiones de escritura: alineación de 64 bytes,
contención, sumas de 65 bits, exclusión de KV y ausencia de alias entre
escrituras y cualquier otra región. Permite alias entre lecturas.
Comprueba capacidad, posición siguiente, coeficientes y rango del lease.

La consulta de prefijos selecciona la capa retenida, independientemente
del último comando interno. La reserva cubre desde la admisión hasta
drenar ambos extremos. Un bind tiene prioridad antes de la admisión y
queda bloqueado durante toda la propiedad, incluida producción y retiro.

Orden normal: entrada → productor → publicación retenida → lease →
RoPE/atención → respuesta pública → ACK público → ACK del lease → ACK de
atención → atención inactiva → ACK del productor → productor inactivo.
Los rechazos previos no arrancan hijos. Una cancelación o timeout con un
hijo activo solicita su cancelación y espera su respuesta real; una
identidad ajena no recibe ACK. El reset exige detener toda la jerarquía
y sus pares AXI.

Cambios de época, caché o publicaciones invalidan el resultado; una causa
transitoria no restablece su validez. Un fallo tardío conserva el estado
de respuesta ya presentado, marca revocación y bloquea lecturas. Los
certificados KV observados se conservan para diagnóstico, incluso si una
respuesta malformada impide acreditarlos como commit.

Hay cuatro maestros AXI de 512 bits: productor, lecturas RoPE, lecturas V
y KV. La integración necesita memoria compartida coherente con los ACK de
escritura; el arbitraje físico de DDR queda fuera de este wrapper. Los
descriptores son un contrato del padre, no autenticación de memoria.

## Evidencia nativa

El [banco R3](../../fpga/tb/tb_ntrpu_input_attention_r3.sv) pasó tres semillas:
**894 trabajos, 519 productores, 501 consumidores, 894 ACK públicos y
9.984 palabras sintéticas**. Incluye 70 errores de rango, diez conflictos
con KV, 40 combinaciones de alias, identidad con bits altos, rechazo
inmediato, cancelaciones en distintas fases, watchdogs, identidades
ajenas restauradas, cuatro fallos tardíos y drenaje prolongado. El host
cambia sus descriptores tras admitirlos; los hijos deben recibir los
valores capturados.

El [revisor R3](../../tools/check_ntrpu_input_attention_r3.py) verificó el
plan completo, registros de frontera, direcciones, identidades, estados,
consultas y orden de retiro. Rechazó **150 corrupciones de traza**. Las
**16 variantes defectuosas del RTL** compiladas fallaron por aserciones
semánticas; no se aceptó un timeout como detección.

La jerarquía real compiló con 49 fuentes RTL y cuatro sustitutos HLS sin
ejecutar. El lector AXI común usa la copia con entradas `wire` explícitas;
se comprobó que sólo difiere en esos calificadores. No hay nuevas métricas
de inferencia, potencia, DDR físico ni timing ruteado.

R1 del banco no aplicaba `++` sobre un elemento empaquetado; R2 encontró
una limitación de Icarus en el índice variable seguido de part-select.
R3 usa asignaciones explícitas y máscaras. El RTL del propietario no
cambió. También se conservaron dos diagnósticos del revisor: respuesta
inmediata en la misma arista y colisión de números de caso con un banco
anterior. Las correcciones del revisor no repitieron la simulación.

Revisión: `build/cloud-setup/20261002-board/input-attention-native-r1-review.json`,
SHA-256 `08694df77911a135fa55979ec7c636f3ff7d1e1303f512a53c58cd5f5149f25e`.
Cuatro tandas tienen fuentes/resultados restaurados en D: y copias
verificadas en NFS:

| Tanda | SHA-256 del TAR |
| --- | --- |
| `20261016T064840Z-V7-input-attention-native-r1` | `e184a224e9cd5e8fed5d6a616b4a9be38d610eef94da9a9c0333ffeab168af8e` |
| `20261016T064950Z-V7-input-attention-native-r2` | `8ff76ca5a6fbac589496b72d5ef425ff01c086bd1916057b2cba90533bf988fc` |
| `20261016T065134Z-V7-input-attention-native-r3` | `cba6dff7aaf96576625e6bf9045dd501af0e755cd4b4b056f7d44088e6be285f` |
| `20261016T065409Z-V7-input-attention-mutants-r1` | `a80d07631e9cb0076257809cb7cd17d2a5c55f9304a37bd60cc6a57566cabc13` |

## Turno Vivado y siguiente prueba

Paquete `20261016T070401Z-input-attention-vivado-r1`: 193 archivos,
**125 fuentes de hardware y 21 IP originales**, cuatro fuentes separadas
para el banco de protocolo. ZIP SHA-256
`fd4eb2bbc840d7c74fe442b8c4eec389ef42beeed3ba27c9620d8f0393608c37`.
Coordinador PID 13140, después de consulta KV aprobada y restaurada.
Dos CPU, 60 minutos totales, síntesis VU5P a 5 ns OOC y tres semillas XSim
en un segundo proyecto con pares sintéticos. Se comprobarán identidad del
diseño, fuentes copiadas, presencia de los cuatro tipos de núcleo, recursos,
timing de síntesis y trazas. Resultado o diagnóstico se respalda y restaura
antes de liberar el slot; no hay reintento automático.

Handoff: `artifacts/ntrpu-board-vivado/input-attention-handoff-20261016-r1/`.
Falta ejecutar los cuatro HLS juntos con pesos reales y memoria poblada
por escrituras aceptadas. Después siguen O/FFN, capas completas y generación.
DDR R1 mantiene su turno después de BRAM; **sin SAIF ni consumo nuevo**.
