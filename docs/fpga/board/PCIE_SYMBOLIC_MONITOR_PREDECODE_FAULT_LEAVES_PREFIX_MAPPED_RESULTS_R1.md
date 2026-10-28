# Replay propio de la netlist por prefijo

## Resultado

La tanda 967, `20261011T133146Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-prefix-binary-r1`,
terminó con exit 0 en 1.525,835 s. La revisión independiente aprobó las
174.436 filas binarias nuevas de 16 bancos y los 447 bits públicos.
Verificó diez sondas de quiet con sus restauraciones, seis controles
negativos reales y el rechazo de 34 corrupciones de evidencia.

El replay utilizó la netlist propia de la síntesis 966; su SHA-256 es
`44ad804bc5259de5bc69ac1c4c44f2dc296d5ed40d421325729278fe8c216898`.
Las 183.058 filas RTL con X/Z de la tanda 965 constituyen una validación
separada. Este resultado binario no incorpora SDF ni acredita timing
físico o funcionamiento PCIe en una placa.

## Respaldo y trazabilidad

Archivo terminal de 94.829.855 bytes y 306 archivos, verificado en D: y
NFS, con fuentes congeladas y revisión posterior:

| Artefacto | SHA-256 |
| --- | --- |
| Archivo terminal | `6bddb3dc094d7ed597ce5e39876ecf919f12de2acf16534e71db4c28570aa65a` |
| Fuentes | `596828d3604ff8b772608a2c4ea129e57a99345c55e23969dfbfc04fb92c243a` |
| Evidencia | `edc06b855dfaf47b1ff7d4370943b22f7183ca05cd70f80d84fec10fea4f936c` |
| Revisión independiente | `2d6be3d099056e3dc963db08d891770feb08ce5eb6f022591bb825efcd667a0b` |

La carpeta local es `artifacts/ntrpu-board-vivado/<tanda>/`, respaldada
por `D:/AI/ntransformer/artifacts/ntrpu-board-vivado/backups/`.

## Decisión

El WNS de síntesis sigue en −2,546 ns a 4 ns; no se inicia otro ruteo.
La siguiente variante aborda el comparador de contexto identificado
formalmente mediante una [bandera del último evento](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_R1.md).
Debe superar su banco aislado antes de integrarse y medirse.
