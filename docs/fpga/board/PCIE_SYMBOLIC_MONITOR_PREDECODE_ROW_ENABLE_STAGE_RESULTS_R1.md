# E/D: resultados temporales por etapas

Revisión del 2026-10-12, 00:23 UTC. Los DCP sintetizados de E/D y RQ984
se reabrieron con el mismo Tcl, sin optimizar, colocar ni rutear.
El resultado está respaldado localmente y en NFS.

## Comparación

Slack setup del peor camino de cada grupo, en ns; una diferencia positiva
favorece E/D. Son estimaciones de síntesis, no timing físico.

| Grupo | RQ984 | E/D | Diferencia |
| --- | ---: | ---: | ---: |
| C→C | −2,758 | −2,773 | −0,015 |
| C→E | −1,183 | −1,428 | −0,245 |
| E→D | 0,441 | 0,735 | +0,294 |
| D→D | 0,416 | 0,710 | +0,294 |
| A→B | 2,283 | 2,283 | 0,000 |
| B→R | −2,768 | −2,768 | 0,000 |

E→D reduce el peor retardo de datos de 3,541 a 3,247 ns y los niveles
lógicos de 16 a 15. Los peores endpoints son distintos: fila 100 en RQ,
fila 102 en E/D. La diferencia compara extremos de cada grupo; no sigue
un mismo bit físico. Hold E→D permanece en 0,077 ns.

La ruta crítica E/D sigue desde `c_stage/allocated_tags_reg[51]` hasta
`c_stage/read_range_high_reg[0]`: 6,756 ns de datos, 28 niveles;
1,679 ns de lógica y 5,077 ns de interconexión **estimada**.
C→E añade un nivel (19→20). B→R no cambia.

## Recursos y decisión

Los 588 FF adicionales se distribuyen en C (+22), E (+549) y D (+17);
A/B/R/TOP conservan sus conteos. E/D usa 97.050 LUT y 28.148 FF,
frente a 92.913 y 27.560 de RQ. El WNS global sintetizado pasa de
−2,768 a −2,773 ns; no confundirlo con el delta C→C de la tabla.

No se promueve E/D por esta mejora local. Se conserva la comparación
física encolada, necesaria para evaluar la interconexión real, y RQ
continúa como referencia. La prueba de máscaras fijas aborda las rutas
de entrada por separado, sin combinar variantes no calificadas.

## Evidencia y límites

Tanda: `20261012T001602Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-stage-timing-r1`.
Se verificaron siete grupos disjuntos con todos sus FDRE, 49 pares
ordenados setup/hold y hasta veinte caminos por par. Diez corrupciones
de evidencia fueron rechazadas. Las advertencias siguen registradas
como diagnóstico, sin calificación individual.

Revisión SHA-256: `7a249ea3897fa35eb14c40989edd090756204ec84d1f10864f27acd02144d72d`.

Archivo SHA-256: `0fecdcaa41b6659aa740550b12e26e5cb385a7bc358701f5b406ce4d03668d87`.

El JSON `…-independent-row-enable-stage-timing-review-r1.json` en
`build/cloud-setup/20261002-board/` conserva endpoints, retardos, DCP
y referencias exactas. No aprueba timing físico, PCIe, DDR ni placa.
