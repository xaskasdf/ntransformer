# Cuantización: propiedad de la respuesta al liberarla (R2)

Actualizado 20261013T204319Z. Esta revisión sustituye al controlador R1 y a sus
tres coordinadores pendientes; conserva sus fuentes, paquetes y evidencias.

## Fallo reproducido

R1 capturaba la propiedad al aceptar el trabajo y activaba READY al terminar
HLS, sin volver a comparar la identidad. Cambiar el bit 40 de la secuencia
durante la lectura provoca que reconozca una respuesta ajena. El banco nuevo,
compilado contra el RTL R1 intacto, falla en trabajo 62/modo 31:
`Foreign source acknowledged`. Compilación 0, simulación 1.

Reproducción `20261013T203642Z-V7-attention-quantize-owner-release-repro-r1`; archivo D:/NFS verificado:
`17f8db071fae8135a62ccd4744a441c56b23e0117eec5f710800e5b5ea82fe29`.

## Corrección y contrato

`fpga/rtl/ntrpu_attention_quantize_owner_r2.sv` exige secuencia **y** época de
64 bits coincidentes con la identidad capturada antes de reconocer la fuente.
Un resultado fallido del propietario correcto todavía puede drenarse.

Una identidad ajena se retiene sin ACK ni publicación. Si vuelve la identidad
original, la fuente se libera y el error BAD_PEER permanece. Si nunca vuelve,
se necesita reset coordinado de fuente, controlador y núcleo HLS; un timeout
no autoriza liberar trabajo ajeno. El resultado privado permanece oculto.

## Validación nativa independiente

- 73 trabajos por semilla (1, 7, 42): **219 trabajos**, 192 completions,
  **59.904 palabras** públicas de 64 bits y 30 resets.
- Seis escenarios adicionales: cambios de secuencia/época durante cómputo
  o liberación; retención permanente/reset y restauración temporal. Cada fallo
  va seguido de un trabajo normal de recuperación.
- Cada respuesta ajena se mantiene al menos 32 ciclos tras acabar HLS.
  El registro ACK incluye la identidad completa, comprobada independientemente.
- Nueve mutantes RTL compilados y detectados, incluidos READY sin protección
  y comparaciones truncadas de secuencia o época. 26 corrupciones de traza
  rechazadas por semilla (78 en total).

Banco positivo `20261013T203728Z-V7-attention-quantize-owner-native-r2`; archivo:
`ab5417011e464edc36bde0022b1db28b88b225c0335aa2ecf9b59bca573b59d0`.
Revisión `llama32-attention-quantize-owner-native-r2-review.json` SHA-256 `8846cde1712573ea014467a2db11ba327369dad3107bc45ea34f660b50377e82`.
Los archivos completos, logs, invocaciones y fuentes están en
`D:/AI/ntransformer/artifacts/ntrpu-attention-quantize-owner-native/` y NFS.

El núcleo de este banco es un **emulador de protocolo**, sin aritmética de
cuantización. La composición con RTL HLS generado, escritor AXI y proyección
real sigue pendiente; estos resultados no acreditan inferencia completa.

## Cola y conservación de pruebas

Se detuvieron solamente los tres coordinadores locales verificados en espera
(PID 120220, 72128 y 165168). El wrapper raíz R2 y el cuantizador HLS R1 se
conservaron. La retirada queda registrada en
`attention-owner-r1-tail-retirement-20261013T203726Z.json` y su recibo `-stopped`.
Los registros antiguos son históricos y sus procesos ya no están activos.

La nueva cadena es raíz R2 → HLS R1 → propietario R2 → ruteo R2 → proyección R2.
Ruteo/proyección conservan sus RTL, banco y criterios anteriores; cambia su
coordinador y la dependencia sellada. Auditoría vigente:
`audit_attention_composition_active_r8.py`. Cada etapa conserva revisión y
respaldo antes de liberar la única ranura Vivado/Vitis.

| Coordinador | Estado al documentar | SHA-256 del paquete |
|---|---|---|
| quantize-owner R2 | `waiting_for_attention_quantize_hls_review_and_backup` | `8526eac2e756d65abdd42b4b3d1f69f45754d6c9bd99dc8d87901ccae81fb289` |
| engine-route R2 | `waiting_for_attention_quantize_owner_review_and_backup` | `4158e0e9dbdb16d93461f33f1ae5180e2f1b78001b5fd937e15cc610fc976c10` |
| projection-control R2 | `waiting_for_attention_engine_route_backup` | `ad223367540df90c4423c14d7e7e7577dafa99c64dab0d6f12057bb569974272` |

## Modelo conservado

Se comprobó nuevamente el archivo local del
[GGUF Q8_0 de Unsloth](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf):
1.321.082.528 bytes, SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
Ruta `D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/`.
No se volvió a descargar ni se alteraron los fixtures existentes.
