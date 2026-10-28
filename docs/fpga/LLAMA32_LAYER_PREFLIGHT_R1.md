# Admisión de memoria para una capa completa

Estado 2026-10-16 08:38 UTC: **RTL de validación aprobado en Icarus;
síntesis y protocolo XSim encolados**. Todavía no está conectado a un
wrapper que ejecute entrada → atención → O → FFN → residual.

## Propósito y composición prevista

La integración reutilizará `ntrpu_input_attention_r1`, su resultado FP32
retenido y `ntrpu_attention_output_path_r1`, seguidos por
`ntrpu_post_attention_real_r2`, `ntrpu_post_ffn_handoff_r2` y
`ntrpu_llama32_ffn_residual_r3`. Así conserva una sola autoridad de caché
KV en el bloque de entrada/atención. No combina dos raíces de atención.

El [validador](../../fpga/rtl/ntrpu_llama32_layer_preflight_r1.sv) revisa
también las asignaciones de las etapas finales **antes de permitir las
primeras escrituras de entrada/QKV**. Las validaciones locales de cada
hijo siguen siendo necesarias. El arbitraje AXI y la retención entre esas
etapas todavía deben integrarse y probarse en el wrapper completo.

## Contrato de asignaciones

Orden definido en [ntrpu_llama32_layer_layout_r1.py](../../tools/ntrpu_llama32_layer_layout_r1.py).
`command_addr`, `command_base` y `command_bytes` tienen índices 0–23:

| Índices | Contenido | Tamaño mínimo, bytes | Escritura |
| --- | --- | --- | --- |
| 0, 1 | Entrada FP32, pesos de RMSNorm inicial | 8.192 cada uno | No |
| 2 | ABI1 inicial | 4.096 | Sí |
| 3, 4, 5 | Pesos Q, K, V | 4.456.448; 1.114.112; 1.114.112 | No |
| 6, 7, 8 | Proyecciones Q, K, V | 8.192; 2.048; 2.048 | Sí |
| 9 | Coeficientes RoPE de la posición | 256 | No |
| 10 | Pesos O | 4.456.448 | No |
| 11, 12 | ABI1 de atención, proyección O | 4.096; 8.192 | Sí |
| 13 | Pesos de RMSNorm posterior | 8.192 | No |
| 14, 15 | Residual posterior a O, ABI1 de FFN | 8.192; 4.096 | Sí |
| 16, 17, 18 | Pesos gate, up, down | 17.825.792 cada uno | No |
| 19, 20 | Gate y up FP64 | 65.536 cada uno | Sí |
| 21, 22, 23 | ABI1 SwiGLU, down FP32, residual final | 16.384; 8.192; 8.192 | Sí |

La entrada 0 también alimentará el residual posterior a O; es la misma
asignación, no una segunda región que pueda divergir. KV se valida por
separado: región completa, capacidad y `capacity × 65.536` bytes.

Se exige alineación de 64 bytes y contención sin desbordamiento, usando
sumas de 65 bits. Ninguna asignación puede solaparse con KV. Se permiten
solapamientos entre lecturas; si alguno de los dos destinos es escribible,
se rechaza el solapamiento de **las asignaciones completas**.

## Admisión, reserva y revocación

El padre debe reservar caché y asignaciones antes de admitir el comando y
mantener esa reserva durante toda la capa. Este módulo no arbitra memoria
ni adquiere una reserva física. Captura los 24 descriptores y la identidad;
los cambios posteriores del host no alteran la validación. La consulta de
prefijo selecciona la capa capturada, no el siguiente comando del host.

Comprueba identidades completas de 64 bits, fuente válida, capa menor que
16, capacidad 1–131.072, posición igual al prefijo y posición de coeficientes
coincidente. El lease admite 32 identificadores sin desbordar 64 bits.

El padre sólo puede admitir trabajo si `completion_valid && result_valid`.
El ACK libera el validador; **no libera la reserva de memoria del padre**.
Cancelación, cambio de época o alteración del caché invalidan la publicación.
Un fallo tardío conserva el estado ya presentado y deja una revocación
persistente: leer únicamente `completion_status == 0` sería incorrecto.

## Evidencia y pendientes

El banco R2 pasó **581 casos × tres semillas = 1.743 trabajos**. Por semilla
recorre los 276 pares: 221 solapamientos con escritura rechazados y 55
alias entre lecturas permitidos. Incluye las 24 regiones contra KV,
alineación, límites, desbordamientos, bits altos de identidad, cambios
transitorios en distintas fases y revocación tardía antes del ACK.
El máximo observado es **301 ciclos de validación**, sin contar espera de ACK.

El revisor rechazó 75 trazas alteradas. Las 18 variantes defectuosas del RTL
compilaron y fallaron por aserciones semánticas. El banco R1 inicial y el
R2 reforzado permanecen archivados; R2 añade la comprobación combinacional
de revocación antes de la arista de ACK. El RTL validado no cambió.

Tres tandas con fuentes/resultados restaurados en D: y copias NFS:

| Tanda | SHA-256 del TAR |
| --- | --- |
| `20261016T083017Z-V7-layer-preflight-native-r1` | `e4c84d77964367cce38908cc0c710f447f2c6565fb959f7d83718f55e6cacb38` |
| `20261016T083306Z-V7-layer-preflight-native-r2` | `5bc51e9040f10847a5baa2070d553135533c3c4dfa59efbbcf8365da7f77e7f2` |
| `20261016T083346Z-V7-layer-preflight-mutants-r1` | `226cfd7ad95826f06f7155afa2f3123d7ccee858fccc061bce500ef617208d63` |

Revisión: `build/cloud-setup/20261002-board/layer-preflight-native-r1-review.json`,
SHA-256 `629d51f1a362840d02c64f6cec9d2e26fa8f239a9a0add46cd500483480bfbf6`.
Paquete Vivado `20261016T083714Z-layer-preflight-vivado-r1`: 36 archivos,
ZIP `9ae980f08e4f4b1965fecdc763fef8786285d679d122f439852df37b9ecbaea0`.
Coordinador PID 25120, tras la revisión y respaldo de entrada/atención
aritmética; dos CPU, 30 minutos, sin reintento automático.

Son pruebas del RTL de admisión con direcciones sintéticas dispersas de
64 bits; no constituyen un mapa físico de DDR ni ejecutan aritmética,
tráfico AXI o inferencia completa. Recursos y timing de este módulo siguen
pendientes de Vivado. La integración posterior deberá probar el consumo
del permiso, la conservación de reservas y el orden completo de retiro.

El contrato, fuentes, revisiones y referencias a las tres tandas se conservan
en `artifacts/ntrpu-board-vivado/layer-preflight-handoff-20261016-r1/`, con
copias D:/NFS y restauración verificada.
