# Recorrido AXI de normalización de entrada — R1

Estado 2026-10-15 22:47 UTC: **protocolo nativo aprobado; Vivado encolado**.
La [conexión del HLS real](LLAMA32_INPUT_NORM_REAL_R1.md) está preparada y
encolada como sucesora; todavía no se ha ejecutado en este recorrido.

## Contrato y composición

[`ntrpu_input_norm_path_r1.sv`](../../fpga/rtl/ntrpu_input_norm_path_r1.sv)
compone el cargador residual existente como lector de dos planos FP32,
el [propietario de normalización](LLAMA32_INPUT_NORM_OWNER_R1.md) y el
escritor ABI1 de atención. Lee 8.192 bytes de entrada y 8.192 de pesos;
publica 4.096 bytes mediante 64 escrituras AXI de un beat y sus respuestas B.

Captura identidad y tres descriptores. Antes de enviar comandos a los hijos,
comprueba alineación de 64 bytes, límites con sumas de 65 bits y ausencia
de solapamiento entre la **asignación completa** de salida y ambas fuentes.
Las regiones de sólo lectura pueden compartir memoria.

`command_source_*` es una identidad capturada, no un ACK de un productor vivo.
El llamador conserva entrada y pesos inmutables hasta el ACK público final.
Cancelación, cambio de época y fallos invalidan el resultado, drenan el
tráfico aceptado y conservan la respuesta. Cada VALID/READY público se
consume una vez; una respuesta ajena puede retener el drenaje hasta que
vuelva su identidad o se aplique reset coordinado con los hijos y el bus.
Las escrituras parciales siguen privadas.

## Evidencia ejecutada

El banco R1 usa cuatro entradas reales de Llama 3.2 1B. Su peer HLS lee las
RAM cargadas y reproduce referencias ABI1 fijadas por hash; **no calcula
RMSNorm**. Tres semillas pasaron 198 trabajos: 189 finalizaciones, 114
ofertas correctas y 105 resultados aceptados, equivalentes a 430.080 bytes.
Las restantes ofertas correctas se revocan o reinician antes de aceptarlas.

Cubre errores AXI/HLS, cancelaciones, watchdog, reinicios con tráfico activo,
retención, revocación tardía y regiones inválidas. El revisor independiente
comprueba bytes R/B, orden, identidad y aceptación; rechazó 66 corrupciones.
Los 14 mutantes RTL compilados fallaron por aserciones específicas.
391.536–391.595 ciclos por semilla son tiempos del banco con replay, no
latencia de normalización ni rendimiento físico.

## Cola y recuperación

Paquete `20261015T224608Z-input-norm-path-vivado-r1`: 51 archivos, siete
fuentes compilables, síntesis OOC a 5 ns y tres semillas XSim. Coordinador
PID 16564: espera aprobación y restauración del propietario de norma,
además del slot vacío; dos CPU, 30 minutos, sin reintento automático.
La captura DDR R1 conserva su posición anterior.

Las dos tandas nativas están restauradas y verificadas en
`D:/AI/ntransformer/artifacts/ntrpu-input-norm-path-native/`, con copia NFS.
Revisión: `build/cloud-setup/20261002-board/input-norm-path-native-r1-review.json`.
SHA-256 del paquete Vivado:
`2e44bd5309826fdc834f5423a8444980233156e21af532a077b6567bbcfa5f6a`.
Traspaso respaldado en `artifacts/ntrpu-board-vivado/input-norm-path-queue-20261015-r1/`.

Pendiente: unir el núcleo HLS original, comprobar sus bytes calculados sobre
AXI y conectar el ABI1 escrito a Q/K/V. Tampoco hay timing ruteado nuevo.
