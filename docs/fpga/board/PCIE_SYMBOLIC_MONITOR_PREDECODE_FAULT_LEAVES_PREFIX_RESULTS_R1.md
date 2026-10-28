# Integración RTL del contexto por prefijo

## Resultado observado

La tanda 965, `20261011T130303Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-prefix-r1`,
terminó con salida **0**, en **726,254 s**. La revisión independiente aprobó
`PASS_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_R1`. El archivo terminal
y las fuentes se verificaron en D: y NFS a las 13:18 UTC del 2026-10-11.

| Comprobación | Resultado |
| --- | ---: |
| Filas RTL nuevas, incluidos X/Z | 183.058 |
| Filas reutilizadas de otra variante | 0 |
| Bits públicos comparados | 447 |
| Observaciones antes / después del flanco | 183.138 / 183.138 |
| Sondas de quiet | 31, con 62 muestras de prueba/restauración |
| Bancos drenados sin cancelar tráfico original | 6 |
| Controles negativos rechazados | 6 |
| Corrupciones de evidencia rechazadas | 34 |

Se verificaron 24 fuentes SV compiladas por cada uno de los 22 casos.
Los 44 avisos corresponden a dos diagnósticos ya conocidos, repetidos por
caso: declaración posterior de `rq_held` y entorno del compilador XSIM.
Los controles negativos fallaron en sus primeras discrepancias previstas.

## Cambio validado y alcance

La validación tardía reconstruye el contexto al entrar en cada lane desde
los SOP/EOP anteriores y conserva las contribuciones de fallo por lane.
La integración mantiene las referencias originales, los estímulos, la
capacidad y las cinco etapas de reloj. Comprueba el estado público en
ambas fases y los contadores de tráfico original de las seis familias
drenadas. Previamente aprobó el [banco aislado de 47.734 muestras](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_RESULTS_R1.md).

La aprobación habilita medir la síntesis propia, ya encolada, y después
reproducir su netlist. Todavía no demuestra reducción de LUT, mejora de
timing ni calificación física, PCIe nativa o inferencia en placa.

## Evidencia

Archivo terminal: **46.988.892 bytes**, 1.531 archivos; SHA-256
`405ae5298d359bfc99804d6047b179f31f34559e8d572d10896b54672aa60779`.
Fuentes: `source-20261011T130137Z.zip`, SHA-256
`1a518a5ddc66df82a5d2668744e98dc4e5a74c7c3e28a3a165594f0c9daf2aa5`.
Revisión independiente SHA-256:
`fa50ff1c97829b5e41ff0683e6d17d5784060f1f5f1bdd81dcbbfddb150d9557`.
