# C: diagnóstico temporal y regresión de entrada

Fecha: 2026-10-11 UTC. Tanda `20261011T225347Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-admission-count-stage-timing-r1`.
Estado `REVIEWED_ADMISSION_COUNT_PAIRED_STAGE_TIMING_R1`, con respaldo D:/NFS.

## Resultado medido

Se compararon los DCP propios C (`20261011T215539Z`) y RQ
(`20261011T180537Z`), sin optimización ni implementación adicional: siete
grupos, 49 pares y 98 estados setup/hold por miembro, con diez controles
de corrupción de evidencia. Se conservaron dispositivo, período 4 ns y
retardos externos 0 ns.

| Camino | RQ, ns | C, ns | Cambio |
| --- | ---: | ---: | ---: |
| C→C setup | −2,758 | −2,530 | +0,228 |
| C→E setup | −1,183 | −1,184 | −0,001 |
| E→D setup | +0,441 | +0,622 | +0,181 |
| D→D setup | +0,416 | +0,597 | +0,181 |
| D→D hold | +0,103 | +0,111 | +0,008 |
| D→A hold | +0,091 | +0,087 | −0,004 |

Los peores slacks de los demás pares comparables no cambian. Estas mejoras
parciales no compensan la [regresión global de síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_SYNTHESIS_RESULTS_R1.md):
WNS **−3,857 ns**, frente a −2,768 ns de RQ. El camino crítico sale de
`rc_user[78]` y llega a `c_stage/registered_ops_reg[eops][0][mask][2]/D`;
no aparece en una matriz limitada a registro→registro.

Se mantiene RQ como referencia. No se promueve ni combina la variante C.
Las advertencias de este diagnóstico no están calificadas individualmente;
no hay aprobación de timing físico, frecuencia ni placa.

## Evidencia

- Revisión: `1cb38585fab03d21d26ce6452605d9fb6a460adc0a8429d5636c80f503b6ce42`.
- Archivo: `c1ff91d8ff64ce1072040923c9a6a70687dbe484364b9fceb9debc3a33756bd0`.
- Snapshot: `97ace7fde8ba4d9377f430c3f99502e2abd69a15961809076bbdb08450fee9a3`.
- DCP C: `b36e5ff1b906be78531ad635fa3b729a230037f5cc7f3abc88f808548ba6d733`.
