# Bitmap RQ: comparación por etapas, tanda 987

## Resultado medido

`20261011T184938Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-stage-timing-r1`
reabrió los DCP propios 984 y 979, sin optimizarlos ni rutearlos.
Terminó en **315,862 s** y quedó revisada y respaldada a las 18:57 UTC.
Se verificaron siete grupos disjuntos de registros y los 49 pares
ordenados, cada uno con estado setup/hold y hasta veinte rutas.
El revisor rechazó diez corrupciones de informes.

| Ruta | Referencia 979, ns | Bitmap RQ 984, ns | Cambio, ns |
| --- | ---: | ---: | ---: |
| C→C | −3,704 | −2,758 | +0,946 |
| C→E | −1,183 | −1,183 | 0 |
| E→D | +0,735 | +0,441 | −0,294 |
| D→D | +0,710 | +0,416 | −0,294 |
| D→A | +1,504 | +1,504 | 0 |
| A→B | +2,283 | +2,283 | 0 |
| B→R | −2,768 | −2,768 | 0 |
| R→R | +2,732 | +2,732 | 0 |

Son márgenes setup estimados a 4 ns. Las rutas de E/S permanecen en
el informe global, fuera de esta partición de registros.

## Atribución y decisión

La peor ruta C→C cambia de `allocated_tags[51]`→`allocated_tags[188]`
a `allocated_tags[51]`→`read_range_high[0]`: baja de 32 a 28 niveles.
El retardo del nuevo camino es 6,741 ns, con 5,062 ns de interconexión
estimada. Sigue incumpliendo y queda solo 0,010 ns por encima de B→R.

C→E conserva 19 niveles, siete CARRY8 y el camino desde
`registered_ops.prefix_counts[0]` al reset de `out_e.updates[0].summary[10]`.
B→R conserva 27 niveles y tres CARRY8. Se mantienen los experimentos
de suma E y límites B; aún será necesario trabajar sobre C.
No corresponde repetir ruteo antes de medir estas modificaciones.

El mapeo pasa de 3.558 a 3.554 FF en C y de 5.075 a 5.107 en E;
D/A/B/R/TOP permanecen en 15.928/1.296/1.210/447/18. El aumento neto
de 28 FF corresponde al mapeo, no a una nueva etapa RTL.

## Evidencia y límites

Archivo: **43.508.504 bytes, 236 archivos**, verificados en D: y NFS,
SHA-256 `2bcefe296513dde9dd15cb58f010d82891882fc1a4bc05b0bb7eee3d7adf1c69`.
Revisión `-independent-rq-bitmap-stage-timing-review-r1.json`, SHA-256
`6ef61a85e775042c601ab78f70e364a4226321841724f693a096075fcf788abe`.
Snapshot `source-20261011T184827Z.zip`, SHA-256
`4a06f39e24fe413c0227c7fb3488175eaf0f0b5cacf8c505088d720a3e933a59`.

Cada DCP emitió `Timing 38-242` por ausencia de `HD.CLK_SRC` en OOC.
Los avisos se conservan como diagnóstico; estos informes no acreditan
timing físico, skew de reloj ni funcionamiento en placa.
