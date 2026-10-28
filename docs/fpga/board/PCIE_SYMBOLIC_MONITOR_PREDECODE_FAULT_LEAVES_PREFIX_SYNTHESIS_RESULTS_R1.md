# Síntesis medida del contexto por prefijo

## Resultado y comparación

La tanda 966, `20261011T132012Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-prefix-synth-r1`,
terminó con salida **0**, en **455,903 s**. La revisión independiente y la
auditoría del grafo del DCP aprobaron su evidencia; el resultado y las
fuentes están verificados en D: y NFS.

Las tandas comparadas usan VU5P `xcvu5p-flva2104-2-i`, los mismos builds de
Vivado 2026.1, período de 4 ns, E/S de 0 ns, síntesis OOC,
`RuntimeOptimized` y jerarquía `rebuilt`.

| Medida | Referencia 951 | Por hojas 962 | Prefijo 966 |
| --- | ---: | ---: | ---: |
| LUT | 104.980 | 105.147 | 105.840 |
| FF | 27.542 | 27.542 | 27.534 |
| CARRY8 | 2.066 | 2.066 | 2.122 |
| WNS estimado | −3,584 ns | −3,746 ns | −2,546 ns |
| WHS estimado | −0,075 ns | −0,075 ns | −0,075 ns |
| TNS estimado | −2.178,434 ns | −2.176,026 ns | −2.174,826 ns |
| Endpoints que incumplen setup / hold | 1.415 / 2.177 | 1.415 / 2.177 | 1.415 / 2.177 |
| Niveles en la peor ruta de setup | 31 | 31 | 24 |

Frente a 951, el prefijo añade 860 LUT y 56 CARRY8, elimina ocho FF y mejora
el peor slack **1,038 ns**. No infiere RAM, DSP ni latches. Los endpoints
totales pasan de 70.903 a 70.887. La mejora pequeña del TNS y el número de
endpoints que siguen fallando muestran que el cierre global sigue pendiente.

## Ruta crítica y decisión

La peor ruta parte de `B.rc_user[79]` y termina en `R.fault_latched[10]`:
6,528 ns estimados, con 1,550 ns de lógica y 4,978 ns de interconexión.
Tiene 22 LUT y dos CARRY8. El [análisis de sus conexiones e INIT](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_CRITICAL_LUTS_R1.md)
comprueba que esta ruta ya no tiene las cadenas largas de factores AND/OR
de la variante anterior. No atribuye todavía las dos funciones CARRY8 a
sentencias RTL concretas.

Se completa el replay propio ya lanzado, tanda `20261011T133146Z`, antes de
promover el candidato. No se programa otro ruteo con este setup incumplido.
El [borrador de límites registrados en B](PCIE_SYMBOLIC_MONITOR_PREDECODE_REGISTERED_BOUND_DRAFT_R1.md)
queda sin calificar: la nueva ruta debe guiar sus pruebas y la selección del
siguiente cambio, sin presumir que mover esa suma resuelve ambos tramos.

## Evidencia y límites

Archivo terminal: **38.120.027 bytes**, 180 archivos; SHA-256
`c48a699d1c966e57aed39e64fef30b23e771440ed7aa9c2d5b7b955f9092a74f`.
Revisión SHA-256: `e3ff3d1901906c11e22ba11a82ba808107316941b25af6f8c4383acef0313892`.
Comparación SHA-256: `3ca0447f316c60f7fc2733e50c16e5082708ccf3357e7b92d668c20eda00f38c`.
La revisión incluye preflight de 529 filas, constraints e inventario; el
grafo es acíclico y no añade excepciones de timing. Son estimaciones de
síntesis, no timing ruteado, frecuencia física ni rendimiento de inferencia.
