# RoPE, captura y despacho Q/K/V R1

Estado 2026-10-16 04:04 UTC: **protocolo aprobado en Icarus; síntesis con
HLS original encolada; aritmética de esta composición aún no simulada**.

## Contrato de hardware

[`ntrpu_rope_qkv_dispatch_r1.sv`](../../fpga/rtl/ntrpu_rope_qkv_dispatch_r1.sv)
conecta el cargador/dueño RoPE real con
[`ntrpu_qkv_capture_dispatch_r1`](LLAMA32_QKV_CAPTURE_DISPATCH_R1.md).
Sus 155 puertos conservan las interfaces existentes. Retiene doce campos
del descriptor para que el host pueda cambiar sus entradas después de la
admisión. Espera el resultado RoPE, ofrece captura y mantiene cancelación
entre etapas. No duplica las memorias de resultados.

`cache_exclusive` permanece activo desde la admisión hasta terminar el
drenaje de ambos hijos, incluida la respuesta pública retenida. Es una
obligación del llamador excluir otros usuarios; no constituye un árbitro
global. Q/K/coeficientes y V tienen puertos AXI separados. Las asignaciones
de origen deben permanecer inmutables y el reset debe coordinar todos los
hijos y sus pares AXI. El despachador valida caché/lease después de capturar;
una solicitud inválida puede leer entradas privadas antes de ser rechazada.

## Evidencia de protocolo

El banco R3 usa cargador, dueño de resultado, captura, despacho, append,
guardia y secuenciador reales. Sustituye únicamente el enlace HLS por un
modelo de identidad síncrono y la cabeza aritmética por un par sintético.
**No calcula RoPE ni atención.** El modelo obtiene datos del cargador, no
de la referencia de salida. Ninguno de estos pares se sintetizará.

Tres semillas, 35 casos cada una: 105 trabajos, 87 respuestas, 69 errores
esperados, 18 resets, seis revocaciones retenidas, 5.721 tramas y 36.864
palabras públicas verificadas. Incluye dos tokens consecutivos, capa 15,
lease máximo, cambios de época, errores AXI/NaN y cancelaciones durante
lectura, cálculo, captura, append y entrega. Una cancelación de un ciclo
entre RoPE y captura debe conservarse como estado 7.

El revisor independiente reconstruye lecturas aceptadas, escrituras del
modelo, captura, escritura KV, consultas de cabezas y lectura pública:
324 controles de corrupción rechazados. Detectó once variantes RTL por
aserción, sin contar timeouts. Una duodécima variante, que elimina sólo
la comprobación `rope_busy` del drenaje final, sobrevivió: el drenaje
interno ya termina después del dueño RoPE en estos 35 casos. Su traza
también pasó el revisor; esto no demuestra equivalencia general.

Cinco tandas y sus fuentes están restauradas en D: y respaldadas en NFS:
dos intentos iniciales del banco, ejecución aprobada y dos tandas de
mutaciones. Se conservaron los errores de generación y del modelo RAM;
el RTL de producción no cambió durante estas correcciones.

Revisión: `build/cloud-setup/20261002-board/rope-qkv-dispatch-native-r1-review.json`,
SHA-256 `cc254887bcdcc4f4c77a7e8141cba4385a39631eb25927cfa28429909ed2dc67`.
RTL: `4a5029c13278d6b98548314b8ae26dab55d2b2be7927285f16789ecdee2bec0e`.
Tanda aprobada: `20261016T035100Z-V7-rope-qkv-dispatch-native-r3`, TAR
`3bf3b4017e1d5b32397f1e3fc9cae0e7a4b8b2e925a347d900d15e0c42cd4158`.

## Síntesis en cola y siguiente validación

`rope-qkv-dispatch-synth-r1-coordinator.json`, PID 24044, espera la revisión
y restauración de captura/despacho R1 y el único slot vacío. Paquete
`20261016T040210Z-rope-qkv-dispatch-synth-r1`: 62 archivos, 24 fuentes RTL
y cuatro IP vendor; ZIP SHA-256
`8d84c9536954810a8fed9bc8bdcc6f66539e1a261330b93a9c2e681d754b4125`.
Restaura las catorce fuentes HLS originales del paquete RoPE ya archivado.

La tanda hará **sólo síntesis OOC**, 5 ns, `xcvu5p-flva2104-2-i`, dos CPU,
30 minutos, sin reintento automático. Comprueba identidad, inventario de
fuentes, ausencia de black boxes/latches/loops, BRAM y DSP presentes.
Registra WNS/WHS sin convertir una estimación OOC en cierre físico.
Append y la raíz de atención siguen fuera del top sintetizado.

El [banco aritmético sucesor](LLAMA32_ROPE_QKV_REAL_R1.md) ya conecta los
vectores originales y su oráculo independiente: compiló y quedó encolado
después de esta síntesis, sin el modelo de identidad. Su aritmética aún
no se ha ejecutado. También faltan la cabeza aritmética
real integrada, arbitraje global, temporización física y DDR de tarjeta.
La nueva síntesis se añadió al final: DDR R1 conserva su turno tras BRAM,
**sin SAIF ni estimación nueva de consumo**.
