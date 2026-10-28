# Diagnóstico por etapas: prioridad RC frente a 970

## Resultado y alcance

La tanda 981 `20261011T173407Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-stage-timing-r1`
reabrió los DCP sintetizados 979 y 970 sin modificarlos. La revisión
comprobó siete grupos completos y disjuntos de registros, 49 pares
ordenados con estado setup/hold y hasta veinte rutas por reporte.
Detectó diez corrupciones de evidencia y conservó ambos reportes globales,
incluidas las rutas de E/S.

Los reportes son coherentes con sus DCP; sus advertencias permanecen
como diagnóstico, sin calificación individual. **No autorizan promoción
de arquitectura ni cierre físico.** El parser R2 conserva los niveles
cero sin absorber la línea siguiente.

## Setup entre registros

Peor slack por par, en ns; objetivo de 4 ns, sin colocación ni ruteo:

| Par | Referencia 970 | Prioridad RC 979 | Diferencia |
| --- | ---: | ---: | ---: |
| C→C | −2,407 | −3,704 | −1,297 |
| C→E | −1,184 | −1,183 | +0,001 |
| C→TOP | +3,242 | +3,251 | +0,009 |
| E→D | +0,742 | +0,735 | −0,007 |
| D→D | +0,717 | +0,710 | −0,007 |
| D→A | +1,504 | +1,504 | 0 |
| A→B | +2,283 | +2,283 | 0 |
| B→R | −2,768 | −2,768 | 0 |
| R→R | +2,732 | +2,732 | 0 |

Los otros pares no tienen rutas en estos DCP; no equivalen a trayectos
con slack positivo. C contiene 3.558 FDRE frente a 3.518; E contiene
5.075 frente a 5.117. El total neto cae dos FF. Son cambios en el mapeo,
no evidencia de haber eliminado exactamente dos bits arquitectónicos.

## Interpretación y siguiente experimento

La peor ruta C→C cambia de `allocated_tags[51]` hacia `commit_slot[422]`
a `allocated_tags[51]` hacia `allocated_tags[188]`. El candidato tiene
32 niveles y 7,686 ns de retardo, de los que 6,075 ns son interconexión
estimada (79,04 %). No comparar las dos cifras como si fueran el mismo
endpoint.

Se mantiene el experimento separado de igualdad temprana de tags RQ,
con pruebas enfocadas, regresión completa, síntesis propia y replay
condicionales. B→R y C→E continúan abiertos; el diagnóstico físico E→D
anterior sigue siendo relevante y no queda resuelto por estas estimaciones.

## Respaldo

Archivo de 45.883.788 bytes y 236 archivos, verificado en D: y NFS:
`bb0bacbcf32ebd08d73db3555a7245fb3cbbd0154d185d1193c6458918d298cf`.
Revisión `-independent-commit-priority-stage-timing-review-r1.json`:
`7ddc0171558d63ca7dfeb3a2648014a2fdb419d5ed2e52c7857378e4b638dee0`.
Los hashes de ambos DCP, recursos, fuentes y restricciones están en
`bindings` dentro de esa revisión.
