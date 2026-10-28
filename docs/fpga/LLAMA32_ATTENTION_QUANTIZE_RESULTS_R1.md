# Cuantización de atención: C/RTL y propietario aprobados (R1)

Estado 20261013T213049Z: dos tandas terminadas, revisadas y respaldadas en D: y NFS.
La integración cuantizador → propietario → escritor AXI se valida aparte.

## Aritmética del RTL generado

Tanda `20261013T205505Z-V7-llama32-attention-quantize-hls-r2`, Vivado/Vitis 2026.1, VU5P, reloj de simulación 5 ns.
El núcleo `ntrpu_quantize_abi1_r1` admite de 32 a 2.048 FP32, en múltiplos
de 32, y produce registros ABI1: 32 INT8, escala FP64 y 24 bytes cero.
La salida se conserva privada hasta comprobar todos los bloques.

- **27 transacciones C/RTL**, 16 válidas y 11 errores de forma o no finitud.
- **5.984 palabras de 64 bits / 47.872 bytes exactos**, contrastados con
  referencia racional independiente y las salidas reales de C y RTL.
- Cuatro vectores de atención de capa 0 de Llama 3.2 1B Instruct Q8_0:
  cada uno tiene 2.048 entradas FP32 y 4.096 bytes ABI1 de salida.
- Cada vector real tarda **101.571 ciclos de latencia**, intervalo 101.572;
  a 5 ns son **507,855 µs simulados**. No es latencia de inferencia completa.
- Se verificaron los marcos de entrada realmente usados y se rechazaron
  10 corrupciones numéricas, 10 de trazas C/RTL y 4 de vinculación de entrada.

El informe estático contiene límites de bucle exagerados: no se utiliza
su latencia máxima como medición. Las cifras anteriores proceden de
`result.transaction.rpt`, asociado a cada caso. Las memorias serializadas
de la cosimulación son datos de dos estados; no demuestran ausencia de X/Z.

Estimación HLS: **4.849 LUT, 2.042 FF, 7 BRAM18K y 3 DSP**, período estimado
3,647 ns. Son estimaciones HLS, sin enrutado ni presupuesto de E/S.

## Propietario RTL

Tanda `20261013T211351Z-V7-llama32-attention-quantize-owner-vivado-r3`: **219 escenarios XSim**, 192 completions,
59.904 palabras publicadas y 30 resets. Se rechazaron 78 corrupciones de
traza. El banco utiliza un sustituto de protocolo HLS, sin aritmética.
La revisión R2 retiene la fuente hasta terminar HLS y exige que la identidad
completa coincida antes de reconocerla; una respuesta ajena no recibe ACK.

Síntesis OOC: **509 LUT, 202 FF, 1 BRAM36**, sin DSP, cajas negras o latches;
WNS +2,288 ns, WHS +0,066 ns con reloj de 5 ns. No acredita timing físico.

## Evidencia y continuidad

| Tanda | Archivo SHA-256 | Revisión SHA-256 |
|---|---|---|
| HLS | `4e7f971cc5ead4aad1a46dade6d233724de9fa8c1d957b1404c7814cdfc6c333` | `d7df5a09a7cf692586c268cb525cdabf943237dc5018f4a22a16a38c6332215f` |
| Propietario | `fba9a12c7332ca82d73a6e315d58636c660822eaa27b4b7825586d0a69edf500` | `a0f35a90890f4ff6c8309fe7da3b9cea34a7cc9e208aa6239740f7b9166414cf` |

Los 18 archivos Verilog y cuatro Tcl de IP se extrajeron del archivo HLS
terminal verificado. Los 15 puertos, direcciones y anchos coinciden con el
adaptador directo. El registro `llama32-attention-quantize-generated-terminal-r1.json`
vincula estos archivos a sus pruebas y respaldos.

La [integración AXI R1](LLAMA32_ATTENTION_QUANTIZE_STORE_R1.md) usará esos
archivos, sin repetir la cosimulación aislada aprobada.
