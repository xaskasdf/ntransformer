# Propiedad del resultado de atención durante la cuantización

Estado 2026-10-13, 19:51 UTC: **protocolo RTL nativo validado; XSim y
síntesis OOC encolados**. No se ha integrado todavía el RTL generado del
cuantizador ni el wrapper raíz. El banco usa un emulador de protocolo con
palabras identificables, sin aritmética de cuantización.

## Contrato del controlador

`fpga/rtl/ntrpu_attention_quantize_owner_r1.sv` retiene el resultado completo
de 2.048 FP32 de la raíz y lo sirve directamente al cuantizador mediante
lecturas síncronas. Evita otra copia de entrada de 8 KiB. La salida privada
contiene 512 palabras de 64 bits: 64 registros ABI1, en total 4 KiB; su
inferencia en BRAM debe confirmarse en síntesis.

La identidad conserva los **64 bits** de secuencia y época. Un comando con
identidad ajena no consume el resultado de otro propietario. Se publica la
salida solo después de recibir las 512 escrituras ordenadas, finalización
HLS válida y reconocimiento de liberación de la fuente. Se comprueban el
dominio INT8, escala FP64 y padding del ABI1.

Cancelación, cambio de época, error de fuente o timeout invalidan la salida.
El controlador espera la finalización del núcleo antes de liberar la fuente;
un núcleo detenido permanentemente requiere reset coordinado. La revocación
permanece aunque vuelva la época anterior. El reset lógico oculta la RAM sin
borrar su contenido y requiere reiniciar o detener también raíz y núcleo HLS.

## Evidencia nativa independiente

Tanda `20261013T194134Z-V7-attention-quantize-owner-native-r1`, semillas
1, 7 y 42: **183 trabajos, 168 completions, 50.688 palabras públicas y
18 resets**. Cada semilla incluye errores de identidad en bits altos,
cancelaciones, revocación tardía, pérdida de respuesta, desorden/pérdida de
escrituras, formato inválido, watchdog y cinco fases de reset, seguidos de
recuperación satisfactoria.

`tools/check_ntrpu_attention_quantize_owner_r1.py` reconstruye orden,
identidades, estados y bytes esperados; rechaza 20 alteraciones de trazas
por semilla. Seis mutaciones RTL reales compilaron y fallaron en los puntos
esperados: truncamiento a 32 bits, liberación anticipada, revocación reversible,
padding ignorado, palabra ausente y respuesta ausente. Son evidencia de
sensibilidad del banco, no seis fallas del diseño base.

Respaldo local en `D:/AI/ntransformer/artifacts/ntrpu-attention-quantize-owner-native/`,
con archivos y hashes verificados también en NFS:

- Archivo base SHA-256: `8ea91867b56be1ee016d84202e255638f6691a2f448994fd00e695f090a57760`.
- Revisión independiente: `442ef420cad0c577861466a3e72f39a83a487e389fe2449d11073f4fe2673635`.
- Cada mutación conserva su fuente, log, compilación y certificado propios.

## Cola y siguiente integración

El paquete `20261013T195105Z-llama32-attention-quantize-owner-vivado-r1`
contiene 16 miembros y está verificado en D:/NFS. Su SHA-256 es
`e59ffb4c96e75636d36c50ca3cf93511304279d4edcbc2799dfce795c1f84046`;
la fuente ZIP tiene SHA-256
`b0c4e17b178370a0d4c3593defeb44892b58b32db96352e9285232148719cb65`.

El coordinador espera al [cuantizador HLS](LLAMA32_ATTENTION_QUANTIZE_R1.md)
con 27 transacciones C/RTL revisadas y respaldadas; conserva todos sus
antecesores. Después ejecutará tres semillas XSim y síntesis OOC sobre
`xcvu5p-flva2104-2-i`, reloj de 5 ns, dos CPU y límite de 1.200 segundos.
Exige comparación independiente, fuentes compiladas verificadas, BRAM
inferida, ausencia de black boxes/latches y respaldo antes de liberar el slot.

Después corresponde inspeccionar los puertos generados y simular
raíz → cuantizador → escritor AXI → proyección O. El GEMV existente usa
identidad de 32 bits: requiere un mapeo de identificadores internos explícito,
manteniendo la identidad raíz completa. El protocolo aislado no acredita
esa integración, throughput por token, DDR física ni timing ruteado.
