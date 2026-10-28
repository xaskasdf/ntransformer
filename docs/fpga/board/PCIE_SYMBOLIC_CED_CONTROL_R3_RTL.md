# Control C/E/D R3: equivalencia RTL

La candidata R3 pasó su prueba de capacidad y los trece bancos completos
el 2026-10-09. **No tiene todavía síntesis, netlist propia ni cierre físico
aprobados.** La versión C/E/D R2 conserva su evidencia previa por separado.

## Cambio de diseño

`ntrpu_pcie_symbolic_control_r3.sv` calcula comparaciones constantes de
capacidad desde el contador registrado de nueve bits. El número de
liberaciones RC selecciona el límite; una admisión anterior selecciona
el umbral de la segunda petición. Se conserva la aritmética original que
devuelve el contador, la prioridad RC antes de RQ, la resolución de tags
duplicados, la latencia y toda la capacidad RQ2/RC4.

Las comparaciones usan los nueve bits completos. La selección desconocida
conserva X; no se sustituye la comparación por un bit alto que podría
ocultar X/Z en bits bajos. E y D permanecen idénticas a R2.

## Prueba focalizada y controles

- **16.777.216** combinaciones: todos los valores de cuatro estados de los
  nueve bits del contador y tres bits de liberaciones, incluso 5–7.
- **2.097.152** comparaciones del resultado completo de admisión de 19 bits:
  512 contadores, 16 máscaras de liberación, cuatro termómetros SOP,
  16 pares de punteros y cuatro patrones de propiedad/tipo.
- **972** casos suplementarios con X/Z en contadores, liberaciones, tags,
  punteros, tipos de petición y bitmap.

Las funciones se extraen de los archivos RTL exactos de R2/R3 y su
transformación se verifica. Tres mutaciones del código de capacidad
(límite adelantado, omisión del incremento de la primera admisión y pérdida
de X en bits bajos) se compilan como funciones derivadas y se rechazan.
Estos controles focalizados no se presentan como simulaciones completas de
módulos mutados. Las cinco mutaciones previas de las etapas E/D/C conservan
su evidencia histórica; esta tanda vuelve a comprobar el conjunto R3 real.

## Bancos completos de cuatro estados

| Banco | Filas | Orígenes retirados |
| --- | ---: | ---: |
| main | 22,339 | 22,069 |
| ownership | 46,356 | 30,870 |
| rq | 8,850 | 7,892 |
| prefix | 544 | 306 |
| lanes | 984 | 618 |
| context | 1,047 | 977 |
| directed | 217 | 155 |
| drained_main | 22,473 | 22,203 |
| drained_ownership | 54,098 | 38,612 |
| drained_rq | 9,328 | 8,370 |
| drained_prefix | 662 | 424 |
| drained_lanes | 1,166 | 800 |
| drained_context | 1,081 | 1,011 |

Son **169.145 filas**: 80.120 originales, 88.808 con vaciado y 217 dirigidas.
No se filtran filas malformadas, posteriores a fallo o con X/Z. Los bancos
con vaciado no cancelan tráfico original. Se conservan las comprobaciones
de captura/control C, 2.498 bits externos, 256 entradas descriptivas,
contexto retenido por valor, reset y origen contra S1/R7/modelo.
El revisor rechazó once corrupciones de evidencia y cotejó las fuentes
compiladas, advertencias, manifiestos, logs y contabilidad por origen.

## Respaldo y siguiente medida

- Tanda: `20261009T144128Z-V4-ffn-supervised-pcie-symbolic-ced-control-r3-rtl`.
- Archivo en D: y NFS: SHA-256 `a9b83e4c30b60e4a7a6ccc30b1a275348defd3359e9cf7e4fe61f889d882c07f`.
- Fuentes: SHA-256 `5d23fd39e950d0bb3dfaeb38b8f306ffd73c3986f365442e2c2136c903474fde`.
- Revisión: `20261009T144128Z-V4-ffn-supervised-pcie-symbolic-ced-control-r3-rtl-independent-symbolic-ced-control-r3-review.json`, SHA-256 `1df08b3fccc56ff99d9dfcad05d605ad4754476b0e672dfc4d888447841eff41`.

Sigue sintetizar C R3 y el conjunto con el mismo criterio OOC de 4 ns,
I/O cero y sin excepciones, auditar sus DCP y ejecutar su propia netlist.
La referencia conjunta anterior mide WNS −3,212 ns; la mejora de R3
todavía no está demostrada. El monitor completo R12, aplicación numérica,
retiro de bytes, CDC/reset, DDR de 512 GiB, pesos reales y placa siguen
abiertos según [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
