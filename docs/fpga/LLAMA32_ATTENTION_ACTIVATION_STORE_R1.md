# Publicación AXI de activaciones de atención (R1)

Estado 20261013T210631Z: **RTL nativo y revisión independiente aprobados;
XSim/síntesis encolados**. Se usa el escritor RTL real y un BFM de memoria,
con cuatro vectores ABI1 reales de atención de Llama 3.2 1B.

## Contrato

`fpga/rtl/ntrpu_attention_activation_store_r1.sv` lee una fuente síncrona
de 512 palabras de 64 bits y forma 64 registros ABI1 de 64 bytes.
Cada registro contiene 32 INT8 en [−127,127], escala FP64 no negativa
y 24 bytes cero. Se rechazan −128, escala fuera de rango y padding no nulo.

Antes de leer o escribir, verifica el rango completo de 4.096 bytes:
alineación de 64 bytes, límites de región y desbordamiento mediante sumas
de 65 bits. Captura dirección, región, secuencia y época del comando.
El escritor `ntrpu_activation_writer` ejecuta AW/W independientes y espera
B por cada registro. No se agrega una segunda copia completa de la activación.

La fuente permanece retenida hasta drenar el escritor. El ACK exige que
secuencia y época completas de 64 bits todavía coincidan. Una respuesta
ajena persistente necesita reset coordinado; no se reconoce por timeout.
El contador privado reserva 64 etiquetas internas por trabajo, cruza el
límite de 32 bits sin truncar la identidad padre y rechaza agotamiento.

`result_valid` autoriza la región únicamente después de 64 escrituras
terminadas sin error y del ACK de la fuente. Las escrituras parciales pueden
existir en memoria, pero nunca autorizan consumo. Cancelación, error, timeout
o cambio de época revocan la publicación; los AW/W ofrecidos se conservan
hasta su respuesta. La revocación tardía permanece aunque vuelva la época
anterior. `completion_records` cuenta respuestas del escritor sin error,
no garantiza publicación ni ausencia de otras escrituras parciales.

Reset debe detener/reiniciar también fuente y memoria AXI: no es una
operación para recuperar unilateralmente créditos de transacciones pendientes.

## Pruebas y evidencia

Tanda `20261013T205423Z-V7-attention-activation-store-native-r1`; tres semillas (1, 7, 42) con 70 trabajos cada una,
más cuatro trabajos de agotamiento de etiquetas:

- **214 escenarios**, 65.170 lecturas de fuente, 8.132 AW, 8.132 W y
  8.126 respuestas B. La diferencia proviene de resets con AW/W pendientes.
- 196 completions, 193 ACK de fuente y 23 resets coordinados.
- 113 completions con región válida: **462.848 bytes** comprobados byte a byte.
  También se ejercitan revocaciones tardías y reset de una completion retenida.
- Diez mutantes RTL compilados y detectados: identidad truncada, ACK temprano,
  revocación reversible, orden de empaquetado, rango corto, error B ignorado,
  avance de etiquetas, identidad de fuente truncada, publicación prematura y
  agotamiento ignorado. Incluye una mutación del escritor real.
- El verificador R2 reconstruye lecturas, etiquetas, AW/W/B y publicaciones;
  rechaza 123 corrupciones de traza (32 por semilla, 27 en agotamiento).

Los datos provienen del fixture `attention-output-r1`, ya contrastado con
los kernels nativos y la referencia numérica. Su manifest SHA-256 es
`73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e`. El archivo de esta tanda es
`b2598c7b22f6c85be120cf9d36693e321bd5132c3c1d3f8bc0f09f78e69339af`; revisión SHA-256 `a247ad99eb7bb7fae32f220f39ea25276126c6579798a9af0f42f63cfb688f80`.
Fuentes y logs positivos/negativos están verificados en
`D:/AI/ntransformer/artifacts/ntrpu-attention-activation-store-native/` y NFS.

## Cola y límites

Paquete `20261013T210449Z-llama32-attention-activation-store-vivado-r1`, SHA-256 `9e70fd11162915e2d5685f80d708aa6e2b6734c988b5dda057565dbbaf904a11`;
fuente ZIP `f8e00b690b22d1073e5511b534aea052ad93bf6b4e6f6b8c7bd2421e044955b8`. Depende de la revisión y respaldo
del controlador de proyección R3. XSim repite cuatro bancos; síntesis usa
VU5P, 5 ns, parámetros de producción, dos CPU y límite de 1.200 s.

La fuente de este banco es un modelo síncrono. Falta conectar el RTL HLS
del cuantizador con su propietario, este escritor y el GEMV O real; después
residual y FFN. Tampoco acredita DDR física, ruteo o inferencia completa.
La auditoría vigente de la cola de 21 etapas es
`audit_attention_composition_active_r10.py`.
