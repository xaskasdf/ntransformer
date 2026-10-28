# Sumas E: síntesis propia, tanda 991

## Resultado medido

`20261011T193631Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-effect-sum-synth-r1`
terminó en **424,918 s**. La revisión estructural, de restricciones y
advertencias, junto con la comparación independiente, quedaron aprobadas
y respaldadas. La referencia es RQ 984; únicamente cambia E.

| Métrica | RQ 984 | Sumas E 991 | Cambio |
| --- | ---: | ---: | ---: |
| LUT | 92.913 | 92.644 | −269 |
| FF | 27.560 | 27.562 | +2 |
| CARRY8 | 568 | 568 | 0 |
| WNS estimado, ns | −2,768 | −2,768 | 0 |
| TNS estimado, ns | −1.472,519 | −1.395,176 | +77,343 |
| Endpoints setup incumplidos | 1.439 | 1.351 | −88 |
| WHS estimado, ns | −0,075 | −0,075 | 0 |
| Endpoints hold incumplidos | 2.068 | 2.029 | −39 |

El total de endpoints cambia de 71.157 a 71.215. No se infieren rutas
individuales mejoradas a partir de los totales. No hay latches, BRAM,
URAM ni DSP en este monitor. Los doce controles de cobertura temporal
no detectan faltantes; el diseño carece de excepciones de timing.

## Ruta crítica y decisión

La peor ruta sigue siendo `delivery_stage/out_b…rc_user[64]` →
`retirement_i/fault_latched[10]`: 27 niveles, tres CARRY8 y 6,750 ns,
de los cuales 5,190 ns son interconexión estimada. Este cambio reduce
recursos, pero no resuelve el cuello de botella global.

La comparación confirmó la misma parte, builds de Vivado, reloj de 4 ns,
retardos E/S cero y estrategia `RuntimeOptimized/rebuilt`. El preflight
repitió 529 filas, 31 perturbaciones y 26 fuentes compiladas.
Sigue el replay de la netlist propia, sin iniciar otro ruteo de E.
Los límites B y el conteo C permanecen experimentos separados.

## Evidencia

Archivo de 184 miembros y 35.169.886 bytes, verificado en D: y NFS:
`5f31e1f55da86dfdd9150f14c73d1097918852d64e817135fc6493f5b9329b72`.
Revisión independiente:
`90af1edca3701e9eaf43f47f5a1219e6338188a8212244cf63288ae8ef66f814`.
Comparación:
`f21b36dede5d3ee6b3946935076b104b2f74565e0b693a1955740c0d9a8fa6aa`.
Netlist funcional propia:
`570bd5eecaf7afe159ce276234d66cf02c7ce0c841c216bddaad137f98f85687`.

Resultados y certificado: `artifacts/ntrpu-board-vivado/`, carpeta de la
tanda. Revisiones: `build/cloud-setup/20261002-board/`, mismo prefijo.
Son recursos de síntesis y estimaciones anteriores al ruteo, sin medición
de throughput ni calificación física.
