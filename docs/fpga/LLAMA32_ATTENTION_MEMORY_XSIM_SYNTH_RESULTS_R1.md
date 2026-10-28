# Contexto y memoria de atención: XSim y síntesis OOC

Ambas tandas pasaron revisión independiente y tienen archivo verificado
en D: y NFS. Son resultados del VU5P `xcvu5p-flva2104-2-i`, reloj de 5 ns.

| Top completo | LUT | FF | Tiles BRAM | DSP | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contexto R2 | 2.200 | 2.782 | 0 | 0 | +1,631 | +0,093 |
| Memoria R1 | 6.159 | 7.256 | 23,5 | 3 | +0,662 | +0,046 |

Los recursos de memoria ya incluyen los bloques anteriores; no deben sumarse
a los del camino KV. El contexto se implementó sin BRAM. WNS/WHS son
estimaciones de síntesis, con entradas/salidas todavía sin presupuesto de
integración; no son timing después de ruteo. Cero cajas negras, latches,
loops, clocks ausentes y endpoints internos sin restricción.

Contexto: `20261013T181704Z-V7-llama32-attention-context-vivado-r2`.
Tres semillas, 69 escenarios y cuatro resets por suite; 4.672 palabras Q,
1.976 de continuación y 2.240 de resultados comparadas por suite.
Archivo `97bd8377044f0982ec9d2c6eda643b5929ea802d9b264a7c052eb7cbbeeb0eca`;
revisión `9c76ecf9d30880f4902077a0e9c5bdb0df7880ac0f536855bf4d68ae1dffb161`.

Memoria: `20261013T182035Z-V7-llama32-attention-memory-vivado-r1`.
Cuatro suites, 61 consultas, 139 appends y cuatro resets por suite;
80.384 palabras K/V, 3.648 Q, 624 de continuación y 2.560 de resultados
comparadas por suite. Incluye watchdog corto y de producción, errores,
cancelación/revocación y recuperación. Cada suite rechaza 25 corrupciones.
Archivo `e3849e8bb2c5b1de24046133f30345b6dfebecb2d6b186402f4342b531ecac27`;
revisión `3fe4521dfac6d2a2469288114d9bc91e4fb5f3ca0bd350860b11c5850db406f0`.

Los peers comprueban protocolo y contenido. La aritmética HLS se valida
por separado y su integración con estos bloques está en recuperación;
véase [lecturas especulativas](LLAMA32_ATTENTION_PREFETCH_RECOVERY_R1.md).
