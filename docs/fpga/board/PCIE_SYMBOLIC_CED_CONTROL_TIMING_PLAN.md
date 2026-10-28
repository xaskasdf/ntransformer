# C/E/D: siguiente investigación del control

## Evidencia de partida

La [síntesis C/E/D R1](PCIE_SYMBOLIC_CED_R1_SYNTHESIS.md) mide WNS de
−3,212 ns en el conjunto a 4 ns, sin colocación ni ruteo. El censo local
`pcie-symbolic-ced-r1-path-census.json` coteja los cuatro informes con el
archivo verificado `20261009T133419Z-V4-ffn-supervised-pcie-symbolic-ced-r1-synth`.
Su SHA-256 es `215a15fbecd93457c8d18e55d2793fa23bb3b1ea559b0c2c5dde23f98fc4b204`.

Los veinte caminos de setup publicados para el conjunto parten de
`c_stage/allocated_tags_reg`. Sus destinos son:

| Destino | Caminos publicados | Peor WNS |
| --- | ---: | ---: |
| `read_range_high_reg` | 2 | −3,212 ns |
| `outstanding_reads_reg` | 8 | −3,178 ns |
| `allocated_tags_reg` | 10 | −2,999 ns |

Este es un censo de los caminos publicados, no de todos los endpoints.
El peor camino tiene 30 niveles y 7,195 ns de retardo de datos; 5,285 ns
corresponden a ruteo estimado. El camino hacia el bitmap tiene 29 niveles
y 6,981 ns. Ambos atraviesan selección RC, liberaciones y la admisión RQ,
incluidas dos cadenas CARRY8 anteriores al resultado de admisión.

## Hipótesis que conviene comprobar

Optimizar únicamente la suma del contador de lecturas dejaría abierto el
camino hacia el bitmap. La dependencia compartida está en
`rq_admission_flags`: la primera admisión modifica `current_count` antes
de comprobar la capacidad de la segunda.

Se propone evaluar los predicados de capacidad antes de esa dependencia.
Para el contador registrado `t` y el número de liberaciones RC `r`:

- Primera admisión: `t - r >= 256` equivale a `t >= 256 + r`.
- Tras una primera admisión: `t - r + 1 >= 256` equivale a `t >= 255 + r`.

Los umbrales pueden precalcularse para cada valor de `r`; la llegada de
las liberaciones seleccionaría predicados ya calculados. La segunda
decisión usaría `previous_valid` para elegir el umbral. En una primera
revisión se conservaría la aritmética original que devuelve el contador,
para aislar el efecto de retirar esa aritmética de la decisión de admisión.
La mejora de recursos o temporización está sin demostrar.

## Condiciones y validación

La implementación y promoción quedan detrás del replay propio C/E/D R1.
No se alteran sus fuentes congeladas ni la tanda en curso. La revisión
candidata debe conservar dos ciclos adicionales, RQ2/RC4, 256 tags,
257 lecturas, la prioridad RC antes de RQ y los errores posteriores a fallo.

La equivalencia debe cubrir los 512 valores del contador de nueve bits,
liberaciones simultáneas, capacidad 255/256, falta de capacidad, tags
repetidos, posiciones malformadas y el comportamiento procedural con X/Z.
Un valor desconocido de contador o liberaciones debe conservar el resultado
desconocido de la comparación original; no se permite convertirlo en una
decisión conocida mediante simplificación de bits altos.

Después se exigen los bancos RTL completos, comparación de captura C,
retiro y reset, síntesis propia C y conjunta, auditoría del circuito y
replay de su netlist. Se compararán los mismos informes de 4 ns, I/O cero
y sin excepciones. D y el monitor R12 mantienen sus referencias actuales;
aplicación numérica, retiro de bytes y cierre físico siguen abiertos.
