# FFN completa de Llama con pesos Q8 reales

Actualización: la FFN RTL completa aprobó posteriormente la tanda
`20261012T010512Z`, con 20.332.664 ciclos. Consultar
[revisión y métricas terminales](LLAMA32_FULL_FFN_METRICS_RESULTS_R1.md).
También pasó la [continuación nativa desde post-O](LLAMA32_POST_FFN_R1.md)
con cuatro entradas ABI1 reales del prefijo y segunda suma residual.
El resto de este documento conserva la preparación histórica.

Fecha: 2026-10-11 UTC. Capa 0 del
[artefacto Unsloth fijado](LLAMA32_Q8_REAL_WEIGHTS_R1.md), D=2048, H=8192.
La referencia CPU pasó; la validación RTL completa todavía está pendiente.

## Referencia numérica y kernels nativos

Se conservaron los **53.477.376 bytes originales** de gate/up/down:
8192×2048, 8192×2048 y 2048×8192. Tres vectores FP32 sintéticos de amplitud
0,25/1/4 ejercitan todas las filas y bloques; no son estados de una inferencia.

NumPy produce cuantización ABI1, GEMV con acumulación binary64 en orden de
bloque, conversiones FP32, SwiGLU, nueva cuantización y down FP32.
El ejecutable C++ invoca los kernels existentes de MAC, cuantización,
SwiGLU y SwiGLU/Q8. Por backend comprueba bit por bit **86.016 valores**
de gate/up/activación/down/final, las activaciones ABI y 4.718.592 llamadas
al MAC entre los tres casos. Alterar un bit de la referencia gate provoca
el fallo esperado antes del PASS.

- `20261011T230413Z-V7-llama32-ffn-native-r1`: stdlib PASS; AMD no compiló
  sus cabeceras bajo C++20. Archivo `7f91e64d83773684c3ae76da8326042f6d91db707b21bfe62fa5ba36be4c3cab`.
- `20261011T230740Z-V7-llama32-ffn-hls-math-native-r2`: la unidad AMD
  compiló con C++14, como `run_hls.tcl`; falló el enlace `-lgmp`.
  Archivo `beea0752c007ce2d58b02490bbbdcf194d8009f50bd3ae910003680a50704288`.
- `20261011T231220Z-V7-llama32-ffn-hls-math-native-r3`: **AMD math PASS**
  usando las bibliotecas `libhlsmc++-GCC46`/`libhlsm-GCC46` del flujo
  C-sim existente. Driver y otros kernels siguen en C++20. Archivo
  `6bad20903f0da973211dff017899a50a172e9b7b4e16dee9ebe854187f84b7e6`;
  revisión `9e1f7caa45d36b80fc9870afce7429f78c3a54ed1d2cbe2e2dfde78696008de5`.

Las tres tandas están respaldadas en D: y NFS, incluidos fallos, fuentes,
referencias, compilación, binarios y hashes de dependencias. Se corrigió
mediante `COUNTS_ERRATUM.json` el contador descriptivo del manifiesto inicial
(36.864→28.672 valores por caso); los archivos numéricos no cambiaron.
Esto es ejecución nativa, **no** una invocación Vitis csim/cosim ni RTL.
Los tiempos CPU observados no describen rendimiento de la FPGA.

## Banco RTL completo preparado

Se selecciona el caso 2 (RMS de entrada ≈1,159) y se empaquetan las tres
matrices completas en layout 2, tiles 32×1024. Una decodificación
independiente recupera todos los bytes originales. Manifiesto:
`cf5e8e07c88038ccd0ceb53abcb24a0378870196f4a411b18a1b815b4a133470`.

`tb_ntrpu_llama32_full_ffn_r1.sv` conserva el RTL del dispositivo y los
scoreboards anteriores. Añade memoria de prueba para los pesos reales;
conserva imágenes inmutables, pausas, guardas, drenaje y completion.
El revisor comprueba 16.480 AR, 56.754.176 bytes R mediante plan/checksums
y los 155.648 bytes W exactos, incluidos 2048 valores down FP32.
Rechazó 14 corrupciones de evidencia antes de ejecutar Vivado.

La cola conserva primero PCIe/E/D, `mask_rows` y las 28 proyecciones reales.
La FFN completa depende del cierre aprobado y respaldado de esas
proyecciones. Su ejecución tendrá límite de 14.400 s y respaldo de cualquier
resultado. El antecedente sintético completo tardó 11.160 s; no se promete
esa duración con pesos densos reales. Desactivar depuración reduce memoria,
pero el ensayo anterior no demostró aceleración.

Faltan el PASS RTL real, RMSNorm, RoPE, atención/KV, logits y tokens, además
de integración física PCIe/DDR. La FFN aislada no aprueba inferencia completa.
