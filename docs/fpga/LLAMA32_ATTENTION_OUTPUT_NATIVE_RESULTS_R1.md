# Llama 3.2 1B: atención → O → residual → entrada FFN

Estado al 2026-10-13: **referencia independiente y composición nativa PASS**.
No constituye una capa RTL integrada ni inferencia completa.

## Datos y contrato numérico

Se usan cuatro prefijos de la capa 0, tokens `[0, 42, 128000, 128255]`,
del GGUF Q8_0 de Unsloth ya auditado. Cada muestra concatena 32 cabezas
de 64 valores FP32, con índice `head * 64 + lane`. Los valores proceden
de la ejecución nativa de atención con matemática AMD, previamente revisada.

La proyección usa los 4.456.448 bytes originales de `blk.0.attn_output.weight`,
matriz 2048 × 2048, reordenados en 128 tiles. El orden es grupo de 32 filas,
tile de 32 bloques K, bloque K y fila; cada bloque Q8_0 ocupa 34 bytes.
El desplazamiento inicial es 4.032 bytes y cada tile ocupa 34.816 bytes.

La activación usa ABI1: 32 enteros INT8, escala FP64 y 24 bytes cero.
Los productos se acumulan en FP64 en orden K, sin FMA. O se redondea a
FP32 antes de sumar el embedding. RMSNorm usa `blk.0.ffn_norm.weight` y
epsilon FP32 `0x3727c5ac`, y entrega la siguiente activación ABI1.

## Validación obtenida

- Referencia independiente con enteros, `Fraction` y `Decimal` de 90 dígitos:
  131.072 bloques de pesos verificados y 13 corrupciones semánticas rechazadas.
- Kernels C++ reales de cuantización, MAC Q8, conversión, residual y RMSNorm:
  524.288 llamadas MAC; 8.192 resultados O FP64, O FP32 y residual, respectivamente.
- Los 20 archivos producidos coinciden byte a byte con la referencia:
  16.384 bytes ABI1 de atención y 16.384 bytes ABI1 para la FFN.
- Un cambio de un bit en la referencia O provoca salida 1 y el diagnóstico esperado.

La ejecución positiva tardó 0,222 s en CPU; **no es rendimiento de FPGA**.
Se compiló con GCC C++14, `-fno-fast-math`, `-ffp-contract=off` y matemática AMD.
Las dependencias de cabeceras y bibliotecas quedaron identificadas por SHA-256.

## Evidencia y siguiente paso

Tanda `20261013T174026Z-V7-attention-output-native-amd-r1`, respaldada en
`D:/AI/ntransformer/artifacts/ntrpu-attention-output/` y NFS.

- Fuente y fixture: `c644fc9e1852432f0c349ead32f2f2b76ea821fca4017e9b620df2be9b741b4d`.
- Resultados: `817866f2ba5af906914b9c911f26278a0ebf199424d532fda93a52803f1cd071`.
- Revisión: `8cdbf9564b52bef0db57f817a4099161426446bf4953674e98472d789da9bfdf`.

Falta integrar un secuenciador de 32 cabezas, publicar los 2.048 resultados
atómicamente y ejecutar esta proyección mediante el motor GEMV RTL.
Después corresponde alimentar la FFN con estas activaciones reales.
