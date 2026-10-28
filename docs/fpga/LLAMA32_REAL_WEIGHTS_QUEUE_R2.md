# Llama Q8 real: recuperación de la cola R2

Fecha: 2026-10-11 UTC. Los [pesos auditados](LLAMA32_Q8_REAL_WEIGHTS_R1.md)
y la [referencia FFN completa](LLAMA32_FULL_FFN_REAL_R1.md) permanecen intactos.

## Estado numérico

La FFN de capa 0 conserva los 53.477.376 bytes Q8 originales de gate/up/down,
con D=2048 y H=8192. Las referencias NumPy coinciden bit por bit con los
kernels C++ tanto con stdlib como con las bibliotecas matemáticas AMD:
86.016 valores entre tres entradas sintéticas, además de cuantización ABI.
Los ensayos de compilación fallidos también quedaron respaldados.
Esta comprobación es nativa; el PASS RTL sigue pendiente.

## Dependencias y recuperación

Los controladores R1 se detuvieron antes de lanzar modelos cuando la revisión
PCIe E/D encontró un nombre de advertencia incorrecto. La
[revisión E/D R2](board/PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_FULL_RESULTS_R2.md)
resolvió esa discrepancia contra las fuentes y resultados archivados.

Los controladores `queue_llama32_q8_gemv_r2.py` y
`queue_llama32_full_ffn_r2.py` usan coordinadores y configuraciones nuevos.
Conservan los registros R1, fuentes empaquetadas, fixtures, hashes, HLS,
testbenches, revisores numéricos y límites. Sus configuraciones vinculan por
SHA-256 los controladores originales, el estado detenido y la nueva dependencia.
El código de revisión numérica y lanzamiento de Vivado no cambia.

1. Cierre de síntesis, replay y diagnóstico E/D, seguido de `mask_rows`, en R6.
2. GEMV real: 28 trabajos por semilla, semillas 1/7/42; 2688 salidas bit exactas.
   Límite 2400 s y dos hilos.
3. FFN real completa: una entrada, tres matrices completas; 155.648 bytes
   escritos exactos y 56.754.176 bytes leídos. Límite 14.400 s y dos hilos.

Cada etapa espera revisión y respaldo del resultado anterior. Un fallo detiene
las dependencias; no se relanza una tanda por un timeout del observador.
Las revisiones/configuraciones R2 se guardan como suplemento en D: y NFS;
no se altera ninguno de los ZIP fuente originales.

Observación de arranque, 23:37:33 UTC: GEMV espera `mask_rows` con PID local
109696; la FFN espera GEMV con PID 221484. Ambos controladores están vivos
y sin salida de error. Configuraciones R2:
`cddc4237656c9c59162bbfc6dab1ab61193833391a93b24b6adab0e6cc15ee23`
y `3c9ab7a9d7023d1baca7cf902c902b03c659cb594962df32595309b5247d6753`.
La auditoría `43d56e502b931dea1b7748335c295a0e8bcdb96fd45d571923ae475c9a01af20`
confirma la derivación exacta y que todas las funciones originales conservan
el mismo AST; también verifica fuentes contra ambos ZIP sellados.

## Pendiente

Faltan ejecución y PASS RTL de los dos bancos. Las activaciones son sintéticas:
ni estos ensayos ni el PASS nativo prueban logits, tokens o inferencia completa.
RMSNorm, RoPE, atención/KV e integración física PCIe/DDR siguen abiertos.
