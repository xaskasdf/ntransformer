# RoPE original conectado al cargador AXI

Estado 2026-10-16: **interfaces compiladas; ejecución numérica encolada**.
El [envoltorio R1](../../fpga/rtl/ntrpu_rope_input_real_r1.sv) conecta el
[camino de entrada](LLAMA32_ROPE_INPUT_PATH_R1.md) al RTL HLS original
mediante su adaptador de veinte puertos. La aritmética conjunta todavía
no se ejecutó: no hay nuevos resultados de recursos, latencia ni timing.

## Entradas y alcance numérico

Se conserva el núcleo de `20261013T154001Z-V7-llama32-rope-hls-r1`, cuya
revisión independiente aprobó el cálculo aislado. El paquete incluye los
14 Verilog originales y cuatro generadores de IP, cotejados contra su TAR
archivado; no contiene un modelo de cálculo que sustituya esas IP.

El [fixture derivado](../../tools/ntrpu_rope_input_fixture_r1.py) conserva
el binario original y sus hashes. Selecciona sus **39 casos de 40 cabezas**:
posiciones 0–131.071, ceros con signo, subnormales, extremos, overflow,
coeficientes inválidos y seis vectores Q/K procedentes de proyecciones con
pesos Q8 reales de las capas 0 y 15. Esas proyecciones partieron de
activaciones sintéticas; no constituyen inferencia completa.

Los nueve casos con número de cabezas distinto de 40 permanecen cubiertos
por el banco aislado, fuera de esta interfaz de dimensiones fijas. Los
valores esperados sólo se usan para comparar; el bus lee otro arreglo de
entradas. El revisor reconstruye los operandos aceptados por AXI y calcula
la referencia FP64/FP32 con comprobación racional independiente.

## Transporte, errores y recuperación

El [banco](../../fpga/tb/tb_ntrpu_rope_input_real_r1.sv) añade quince casos
de cancelación, revocación, rango de lectura y reset, cada uno seguido
por una operación correcta. Incluye cancelación durante lecturas y
escrituras HLS y exactamente en el traspaso cargador→propietario.

El contrato integrado distingue el origen del error: posición fuera de
rango devuelve 1; posiciones discordantes, 2; datos no finitos detectados
durante la carga, 4. Coeficientes finitos inválidos y overflow aritmético
devuelven estado público 5 con el diagnóstico HLS 3 o 5. La precedencia
puede diferir del núcleo aislado porque el cargador valida antes de arrancarlo.

Cobertura **prevista**, aún sin ejecución RTL conjunta: 69 casos, 65
respuestas, cinco resets, dos revocaciones, 107.648 palabras HLS comparadas
y 107.520 palabras leídas de la RAM pública. Se requieren coincidencia
exacta de bits, identidades completas, drenaje antes de liberar memorias
y resultados retenidos hasta ACK. Las medidas de ciclos y pausas vendrán
de la traza real; los ciclos de la traza sintética no son rendimiento.

## Preflight y turno Vivado

Icarus compiló ocho fuentes con una declaración de la firma HLS que falla
si se ejecuta; esa declaración no forma parte del paquete Vivado.
Tanda `20261016T005032Z-V7-rope-input-real-compile-r1`, TAR SHA-256
`07d193902483fa94ec1f3e2fcb26574c100be20de8e34154ce3161a66dcb3f38`.
Fuentes y resultados están restaurados y respaldados en D:/NFS.

El revisor pasó una traza sintética y rechazó 43 corrupciones; el contrato
de puertos rechazó otras 61. Son comprobaciones de preparación, no prueba
de aritmética ejecutada. Los registros están bajo
`build/cloud-setup/20261002-board/rope-input-real-*-review.json` y
`rope-input-real-checker-r1-selftest.json`.

Paquete `20261016T005604Z-rope-input-real-vivado-r1`: 70 archivos, 21
fuentes compilables y cuatro IP. ZIP SHA-256:
`e1d47c2d6532941630bc600267b4a534f25a151a1711d4ea6831ab10a6957d86`.
El coordinador PID 24812 espera aprobación/restauración del protocolo
RoPE y slot libre. Ejecutará OOC a 5 ns y XSim con dos CPU y una hora de
límite, sin reintento automático; archivará/restaurará cualquier resultado
antes de liberar el slot. Traspaso:
`artifacts/ntrpu-board-vivado/rope-input-real-queue-20261016-r1/`.

Siguen pendientes la ejecución anterior, el publicador de Q rotado, el
append K/V con V retenido, la atención completa, DDR físico y timing enrutado.
