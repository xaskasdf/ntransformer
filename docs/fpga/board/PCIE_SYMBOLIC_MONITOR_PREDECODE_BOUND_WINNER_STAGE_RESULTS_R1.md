# B: comparación temporal por etapas

Fecha: 2026-10-11 UTC. Tanda `20261011T224409Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-winner-stage-timing-r1`.
Revisión `REVIEWED_BOUND_WINNER_PAIRED_STAGE_TIMING_R1`, respaldada en D: y NFS.

## Resultados

Los DCP propios de B (`20261011T203802Z`) y RQ (`20261011T180537Z`) se
abrieron sin optimizar, colocar ni rutear: mismo dispositivo, 4 ns y E/S
de 0 ns. Se revisaron siete grupos, 49 pares ordenados y 98 estados
setup/hold por diseño, con diez corrupciones de evidencia rechazadas.

| Camino | RQ, ns | B, ns | Cambio |
| --- | ---: | ---: | ---: |
| C→C setup | −2,758 | −2,758 | 0 |
| A→B setup | +2,283 | +1,447 | −0,836 |
| B→R setup | −2,768 | −1,156 | +1,612 |
| C→TOP setup | +3,261 | +3,276 | +0,015 |
| C→TOP hold | +0,298 | +0,292 | −0,006 |

El peor slack de los demás pares comparables no cambia. B alivia B→R,
pero lo mantiene negativo y consume margen de A→B. C→C determina ahora
el WNS global de −2,758 ns: mejora global de solo 0,010 ns. La síntesis
añade 266 LUT, seis FF y cinco CARRY8 frente a RQ.

No se promueve B ni se combina con E/C a partir de este resultado. Falta
una comparación física controlada; las estimaciones de síntesis no
establecen frecuencia realizable. La matriz tampoco cubre por sí sola
los caminos desde/hacia E/S.

## Procedencia

- DCP B: `6f6b937d6aecde7923fc1943fbe0e0019196cfebfa8994897e1ada87ad259002`.
- DCP RQ: `924f3c41cedf89364e30201552777ac748b6fa66d087d70f679be7f883a59631`.
- Revisión: `5caaaff36cd585b21b38758c8540d01ce3f6fddc65006bfc41cfddbbc63f80be`.
- Archivo de tanda: `149f482c1f56cd73008063baf83b91611d92bbba3a295d31a5af17ced4ac0e12`.
- Snapshot: `917ed63dcc181409f96de19de399c774734b712f0e0434c6163b523a70fcb1e2`.

Las advertencias permanecen como diagnóstico y no quedan individualmente
calificadas por esta revisión. No hay aprobación de timing físico ni
promoción arquitectónica.
