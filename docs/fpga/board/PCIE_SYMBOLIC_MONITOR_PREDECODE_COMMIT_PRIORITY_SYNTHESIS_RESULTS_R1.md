# Síntesis RC por prioridad: menos área, peor setup estimado

La tanda 979 `20261011T165516Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-synth-r1`
terminó en **459,674 s**, con revisión independiente y respaldo en D: y
NFS. La comparación valida dispositivo, reloj, E/S, estrategia y builds
idénticos a la síntesis padre 970.

## Medidas comparables

| Medida | Padre 970 | Prioridad RC 979 | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 105.518 | 91.660 | −13.858 |
| FF | 27.534 | 27.532 | −2 |
| CARRY8 | 2.104 | 568 | −1.536 |
| WNS, ns | −2,768 | −3,704 | −0,936 |
| TNS, ns | −2.175,048 | −1.931,080 | +243,968 |
| Endpoints con fallo de setup | 1.415 | 1.456 | +41 |
| WHS, ns | −0,075 | −0,075 | 0 |
| THS, ns | −146,906 | −148,941 | −2,035 |

Los totales de endpoints pasan de 70.887 a 70.907. El ahorro de LUT es
aproximadamente 13,1 %. La menor suma de violaciones no compensa que el
peor setup empeore. Son estimaciones OOC sin colocación/ruteo; no se
atribuye una mejora de frecuencia ni de inferencia.

## Ruta y revisión

La peor ruta está ahora dentro de C:
`allocated_tags_reg[51]/C` → `allocated_tags_reg[188]/D`. Tiene 32 niveles
(28 LUT y cuatro MUXF7/F8), 7,686 ns de datos: 1,611 ns lógicos y
6,075 ns de interconexión estimada. Pasa por ownership, liberación RC,
admisión RQ y escritura del bitmap. Los nombres de nets orientan el
análisis; no constituyen por sí solos una prueba de equivalencia RTL.

El preflight aprobó 529 filas y 31 probes de quietud. Se verificaron las
26 unidades de simulación, las 14 fuentes sintetizadas, restricciones sin
excepciones y doce categorías de cobertura de timing sin incidencias.
No hay latches, BRAM, URAM ni DSP. El grafo del DCP contiene 160.251
instancias y 223.364 nets, es acíclico y rechaza ocho controles del auditor.
Las advertencias emitidas se revisaron individualmente; los límites de
cien avisos de igualdad y puertos no permiten afirmar nada sobre los
mensajes suprimidos por Vivado.

## Decisión y trazabilidad

Se conserva como candidato de menor área, sin promover su timing. Se
requiere el [replay propio](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_MAPPED_R1.md)
y la [comparación por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_STAGE_TIMING_R1.md),
ambos encolados. La ruta de C sugiere estudiar la decodificación temprana
del tag RQ para el bitmap; todavía no existe ni se califica ese cambio RTL.

Archivo SHA-256 `7e64e757a2e02da57e15731518f2a0df6404d5d54b7f9a9f2ba0b5bf91c06e2b`.
Revisión `71aa3dd0ebc71a23cb1af817f5d6dce8bbca23d043816e9abaf13c1d90cd3339`.
Comparación `b239ed2dde8ef6f15fe37d39f0efe9ae19c86eab4e7dbc99698ad901a9a4e25a`.
Los registros correspondientes están en `build/cloud-setup/20261002-board/`.
