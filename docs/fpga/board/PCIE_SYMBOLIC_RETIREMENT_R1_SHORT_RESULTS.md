# R1: integración inicial de retiro aprobada

Prueba corta del 2026-10-09: **23243 filas en tres bancos**. R recibe B,
calcula el candidato de bytes de 23 bits y realimenta solo los 21 bajos.
Registra control y fallos del mismo origen; conserva el fallo de rango
visible y los fallos anteriores. Solo reset frío cancela las etapas.

## Qué se comparó

Una historia independiente de cinco posiciones guarda el estado de R7
posterior a cada flanco. Se comparan 410 bits de estado público, el control
completo de 391 bits, tres bits de guardas, validez, secuencia y obligación
pendiente. Los campos solapados se comprueban por separado. Se mantienen
todas las comprobaciones originales R7/modelo/S1/C/E/D/A/B/validación tardía.

| Banco | Filas | Retiradas | Canceladas por frío | Retiros con bytes no nulos |
| --- | ---: | ---: | ---: | ---: |
| `directed` | 217 | 114 | 71 | 73 |
| `controls` | 352 | 150 | 142 | 24 |
| `drained_main` | 22674 | 22203 | 335 | 18354 |

Cada banco incluye cinco flancos finales. El principal también añade
cinco ociosos antes de cada reset frío: sus 335 cancelaciones corresponden
a registros de vaciado, sin cancelar tráfico original. Los otros bancos
conservan sus cancelaciones originales y se contabilizan expresamente.

La revisión verificó las diecinueve fuentes SV compiladas en cada caso y
rechazó trece corrupciones de evidencia. Solo aparecieron las tres
advertencias XSIM y las tres de declaración de `rq_held` ya conocidas.
El resultado, fuentes y manifiestos están verificados en D: y NFS.

## Cobertura pendiente y siguiente tanda

El principal tuvo ocho fallos de rango de bytes; los tres bancos tuvieron
cero `sticky_hits`. Falta demostrar específicamente el retorno al rango
permitido con el bit 13 persistente y sin un fallo temprano de tags/lecturas
que lo enmascare. El [caso dirigido propuesto](PCIE_SYMBOLIC_RETIREMENT_RANGE_RETURN_PLAN_R1.md)
usa 256 tags, un pico de 1056760 bytes y retorno a 1048572.

Después faltan once bancos, 159286 filas, y los mutantes reales de R.
Los tres bancos aprobados podrán reutilizarse si las fuentes coinciden.
**Este resultado no aprueba la regresión completa ni quiet, síntesis,
netlist propia, monitor completo o cierre físico.**

- Tanda: `20261009T173830Z-V4-ffn-supervised-pcie-symbolic-retirement-smoke-r1`.
- Archivo D:/NFS: SHA-256 `5bb1909235f401024bd5bba392bdf8488303e8b63f28684ad748e621375469c3`.
- Revisión: `20261009T173830Z-V4-ffn-supervised-pcie-symbolic-retirement-smoke-r1-independent-symbolic-retirement-smoke-review.json`, SHA-256 `c122c8787901a9d21bea36baa937df4ad457576794549d7a1e688921a3f4b2c1`.
- Fuente: SHA-256 `dbceb547215d918222acbcd72c8c16b4a94f739f81ef29c856b78205b6809323`.
