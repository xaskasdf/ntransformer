# Cargador AXI de entradas post-O — R1

**Estado 2026-10-15:** síntesis y XSim R3 aprobados y respaldados en D:/NFS.
Tres semillas: 135 trabajos, 129.024 palabras RAM y 81 controles de traza.
Síntesis OOC a 5 ns: 1.405 LUT, 2.020 FF, 6 BRAM y 0 DSP;
WNS +1,362 ns, WHS +0,055 ns, sin cajas negras, latches ni lazos.
Son estimaciones de síntesis; faltan ruteo y restricciones externas.

Ejecución `20261015T081949Z-V7-llama32-post-attention-loader-vivado-r3`;
archivo `0387b886aff0cee4c18107ce749ef5188b6d23336d8ace9b7c2d88bb3b649729`.
La [unión HLS real](LLAMA32_POST_ATTENTION_REAL_R1.md) se valida aparte.

## Memorias e identidad

`fpga/rtl/ntrpu_post_attention_loader_r1.sv` carga tres regiones FP32 de
8.192 bytes: entrada residual preservada, salida O confirmada y pesos de
RMSNorm de FFN. Usa el lector AXI real y tres RAM privadas de 2.048 × 32 bits.
El llamador debe mantener las regiones inmutables hasta completar la carga;
las comprobaciones de límites no autentican descriptores DMA.

Antes de emitir AR verifica secuencia y época completas de 64 bits, validez
del productor, alineación de 64 bytes y límites de las tres regiones con
aritmética de 65 bits. Cada plano requiere 128 beats; las ráfagas tienen
hasta 64 beats y no cruzan páginas de 4 KiB. Un registro de 512 bits permite
serializar sus 16 palabras hacia RAM, aplicando backpressure al lector.
NaN e infinito se rechazan antes de almacenar el beat.

Los tres puertos públicos de RAM son síncronos y de latencia de un ciclo.
Solo una respuesta correcta, retenida y no revocada expone sus datos.
`completion_ready` libera las RAM: el futuro propietario debe retenerlo
mientras HLS las consume. Cancelación o cambio de época revocan la lectura
inmediatamente; restaurar la época no recupera una respuesta revocada.

| Estado | Significado |
| --- | --- |
| 0 | Carga correcta |
| 1 / 2 | Región inválida / identidad inválida |
| 3 / 4 | Transporte AXI fallido / datos no finitos o formato inválido |
| 7 / 8 / 11 | Cancelación / watchdog / productor inválido |

Cancelación, época y watchdog quedan registrados. Se drena la ráfaga ya
emitida antes de terminar; no se abandona una solicitud AXI por timeout.
Si el dispositivo nunca responde, hace falta reset coordinado de ambos
extremos. Las pruebas de reset reinician también el modelo del bus.

## Evidencia nativa

El banco `tb_ntrpu_post_attention_loader_r1.sv` usa semillas 1, 7 y 42:
135 trabajos, 129 respuestas, nueve resets y seis revocaciones. Verifica
129.024 palabras RAM contra los beats realmente aceptados por AXI, con
permutaciones de direcciones distintas para cada plano.

Cubre límites y overflow, diferencias de identidad en los bits altos,
RRESP/RID/RLAST, valores no finitos en cada plano, cancelación con AR
retenido, durante R y durante la serialización, timeout, revocación de
respuestas retenidas y recuperación. El verificador R2 rechaza 81 controles
de traza. Once mutantes compilados del RTL fallan en aserciones del banco;
se comprueba por separado que compilaron correctamente.

- Base: `20261014T045608Z-V7-post-attention-loader-native-r1`;
  archivo `6089e745a9d6b628fa00f8e9eeb73ca1b5c14122d8af109241883f32c9bac4d1`.
- Mutantes: `20261014T050049Z-V7-post-attention-loader-mutants-r1`;
  archivo `143751c8a2f9192a0e3ab8352e31ea49242dd9116cee30510d2d52dcb6a7da9e`.
- Revisión: `llama32-post-attention-loader-native-r1-review.json`;
  SHA-256 `eb7db5a872cfb00f3f5fe07325a3f85693f7d6ebb029ac3e2079b7c4db96d4ba`.

Los tres bancos completos requieren 104.248, 104.260 y 104.245 ciclos
Icarus. Incluyen fallos y lectura de resultados; no miden rendimiento DDR.
Fuentes, resultados y manifiestos están verificados en D: y NFS.

## Cola Vivado y siguiente unión

El [propietario de las dos RAM de salida](LLAMA32_POST_ATTENTION_OWNER_R1.md)
ya pasó pruebas nativas y Vivado de protocolo. La unión con el HLS generado
y la publicación AXI del par está en ejecución; los bloques aislados no
prueban esa integración.

El coordinador `llama32-post-attention-loader-vivado-r2-coordinator.json`
espera C/RTL aprobado y respaldado de post-atención HLS R2. Conserva
inalterados RTL, banco y verificador nativos. Sintetiza primero para VU5P,
5 ns; exige cero cajas negras/lazos y ejecuta después las tres semillas
XSim. Presupuesto: CPU0/1 y 1.800 segundos. El resultado y sus diagnósticos
se respaldan antes de liberar el turno.

- Fuentes: `d144e89910fc0960aeba00542c7df1ee6c05b9c965047c7095134ca4b3fa3060`.
- Paquete: `3fee6b5b8cb155ffdfe800ae6184c414a9874af543dec31278733be570c9a1ca`.

La siguiente unión debe retener la respuesta del cargador durante HLS,
comprobar sus puertos generados reales, almacenar ambos resultados y
publicar el par residual/ABI1 únicamente después de confirmar todas sus
escrituras AXI. Debe consumir la memoria O producida por el banco integrado,
no sustituirla por un fixture en el consumidor. FFN y la segunda suma
residual permanecen pendientes. Véanse [post-O](LLAMA32_POST_ATTENTION_R1.md)
y la [cola recuperada](LLAMA32_PROJECTED_ROOT_TAG_RECOVERY_R3.md).
