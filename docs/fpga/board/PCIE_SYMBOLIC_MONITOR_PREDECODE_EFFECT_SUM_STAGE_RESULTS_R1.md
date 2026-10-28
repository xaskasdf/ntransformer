# E: comparación temporal por etapas

Fecha: 2026-10-11 UTC. Tanda `20261011T223443Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-effect-sum-stage-timing-r1`.
Revisión `REVIEWED_EFFECT_SUM_PAIRED_STAGE_TIMING_R1`, respaldada en D: y NFS.

## Comparación controlada

Se abrieron los DCP propios de E (`20261011T193631Z`) y RQ
(`20261011T180537Z`), sin optimizar, colocar ni rutear. Mismo dispositivo,
periodo de 4 ns y retardos externos de 0 ns. Se revisaron siete grupos,
49 pares ordenados y 98 estados setup/hold por diseño; diez corrupciones
de evidencia fueron rechazadas.

| Camino | RQ, ns | E, ns | Cambio |
| --- | ---: | ---: | ---: |
| C→C | −2,758 | −2,758 | 0 |
| C→E | −1,183 | +1,340 | +2,523 |
| E→D | +0,441 | +0,441 | 0 |
| D→D | +0,416 | +0,416 | 0 |
| D→A | +1,504 | +1,504 | 0 |
| A→B | +2,283 | +2,283 | 0 |
| B→R | −2,768 | −2,768 | 0 |
| R→R | +2,732 | +2,732 | 0 |

E mejora el peor camino registrado C→E, pero no cambia el WNS global de
−2,768 ns. Su síntesis propia ya había reducido 269 LUT y 88 endpoints
setup fallidos, con dos FF adicionales. No se deduce aceleración de reloj,
cierre temporal ni mejora del cuello físico E→D de ensayos anteriores.
La matriz registro→registro no reemplaza los informes globales de E/S.

## Evidencia

- DCP E: `badce4f9e0fc233b61a52bda2cf9ced855be461869f3ea2f6c876d05f4d731d0`.
- DCP RQ: `924f3c41cedf89364e30201552777ac748b6fa66d087d70f679be7f883a59631`.
- Revisión: `5c1ba77eb3051e6568782201bb828697536d1cc48acba590357c07b1bbfc4b41`.
- Archivo de tanda: `1ccdfa6cecec3c81592343b97d34982f9d7cdb8b6433a61d03a88d4a9a00c28c`.
- Snapshot: `49bc98c52650945839e679b0e15a7bfd1827553274575f608047c99d74b2f74d`.

Las advertencias se conservan como diagnóstico; esta revisión no las
califica individualmente ni autoriza promoción arquitectónica. Continúan
B/C y los ensayos E/D con enables por fila, según sus dependencias.
