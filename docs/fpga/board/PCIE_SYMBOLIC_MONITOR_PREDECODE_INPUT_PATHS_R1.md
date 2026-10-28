# Rutas de entrada en los reportes guardados RQ/E/B/C

## Evidencia

Se analizaron los veinte caminos setup publicados por cada síntesis,
comprobando el reporte contra su archivo local/NFS y revisión independiente.
Todos los endpoints corresponden a FDRE del inventario propio. Se verificaron
orden, reloj, período, retardos de entrada, niveles y suma de retardos;
veinte corrupciones deliberadas fueron rechazadas. No se ejecutó Vivado.

| Variante | Rutas desde puertos entre las 20 publicadas | Peor WNS, ns |
| --- | ---: | ---: |
| RQ 984 | 7 | −2,768 |
| E-sum 991 | 7 | −2,768 |
| B-bound/winner 995 | 9 | −2,758 |
| C-count 999 | 19 | −3,857 |

Las veinte rutas de E coinciden en origen, destino y slack con las de RQ.
B conserva diecisiete pares publicados con igual slack. C solo comparte
tres: B→retiro sigue en −2,768 ns; dos rutas `rc_user[78]`→máscaras de
updates mejoran de −2,700 a −2,506 ns. Otras máscaras pasan a dominar:
la peor llega a `eops[0].mask[2]` en −3,857 ns. También aparece
`rc_user[65]`→`fault_latched[2]` en −3,243 ns.

## Interpretación y siguiente diseño

La regresión no es uniforme. Una ruta ausente de estas listas **no equivale
a una ruta que cumple timing**; son conjuntos limitados con endpoints
distintos. Tampoco son porcentajes de incumplimiento del diseño completo.
La matriz por etapas entre registros debe acompañarse de estos caminos
de E/S. Se mantiene RQ como referencia mientras termina el replay C.

Un posible experimento posterior es formar cada bit de `header_prefix_mask`
con índices fijos, conservando las condiciones procedurales del selector RC.
La hipótesis debe probarse por separado sobre RQ; esta lectura de reportes
no demuestra que la escritura dinámica cause la regresión ni que cambiarla
mejore síntesis o ruteo.

Auditoría: `build/cloud-setup/20261002-board/pcie-symbolic-monitor-predecode-input-paths-r1.json`,
SHA-256 `98ea103c6acaa970a3cd512b592934c9e4b7d530ccccc47cd0c686d1f5fefefb`.
Contiene los ochenta caminos y las coincidencias exactas. No califica
timing físico, throughput ni placa.
