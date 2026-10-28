# Monitor PCIe R1: regresión RTL completa

El camino real C3/E2/D2/A1/B1/R1 pasó **183058 filas positivas** en
dieciséis bancos. Se ejecutaron 181960 filas nuevas y se reutilizaron
1098 del [filtro dirigido](PCIE_SYMBOLIC_MONITOR_R1_DIRECTED_RESULTS.md),
comprobando identidad de las diecinueve fuentes compiladas.

## Resultados por banco

| Banco | Filas | Orígenes retirados por R | Cancelados por frío |
| --- | ---: | ---: | ---: |
| `monitor` | 252 | 173 | 55 |
| `directed` | 217 | 114 | 71 |
| `controls` | 352 | 150 | 142 |
| `range_return` | 277 | 268 | 5 |
| `main` | 22339 | 21868 | 335 |
| `ownership` | 46356 | 19257 | 19355 |
| `rq` | 8850 | 7177 | 1193 |
| `prefix` | 544 | 129 | 295 |
| `lanes` | 984 | 345 | 455 |
| `context` | 1047 | 926 | 85 |
| `drained_main` | 22674 | 22203 | 335 |
| `drained_ownership` | 65711 | 38612 | 19355 |
| `drained_rq` | 10045 | 8370 | 1195 |
| `drained_prefix` | 839 | 424 | 295 |
| `drained_lanes` | 1439 | 800 | 455 |
| `drained_context` | 1132 | 1011 | 85 |

El monitor se compara con R7/S1 e historia independiente antes y después
del flanco: **183138 muestras por fase**, incluidos cinco vaciados por
banco. Se conservan los controles originales de tablas, contexto por
valor, estado, aritmética, fallos y secuencia. Los seis bancos con vaciado
cancelan **cero registros de tráfico original** en C/E/D, A, B y R.

La selección mantiene los catorce bancos originales con valores X/Z y
tráfico malformado; añade el retorno de rango y los testigos dirigidos.
Estos últimos incluyen 31 cambios entre flancos y 16 alteraciones reales
del contenedor rechazadas en las filas previstas. La revisión nueva
rechaza 23 corrupciones de cobertura/contabilidad y conserva las 23 del
filtro dirigido. Ninguna expectativa de quiet procede del candidato.

## Evidencia

- Tanda nueva: `20261009T193755Z-V4-ffn-supervised-pcie-symbolic-monitor-full-r1`.
- Archivo D:/NFS: SHA-256 `f6ea623fc66016a64324def45751487131b29a8ad08be76fc0b7299c01ce68e9`.
- Fuentes: SHA-256 `6c13f35e9950bc62b36ff0fe541a99682217340f62c3456288878129a7c089a4`.
- Revisión: `20261009T193755Z-V4-ffn-supervised-pcie-symbolic-monitor-full-r1-independent-symbolic-monitor-full-review.json`, SHA-256 `78e0c5464b59f56cabfb0afe802039ea6450d1e406b3abca171d6c050b737f4b`.

La copia usa el restaurador de cuatro conexiones, probado tanto desde
cero como desde un prefijo parcial. Conserva comprobación del prefijo,
hash del archivo completo, todos los miembros y ZIP de fuentes.
Los tres diagnósticos anteriores del banco permanecen respaldados.

## Límites y siguiente medición

Este resultado acredita el comportamiento RTL del observador pasivo y
sus condiciones de inventario vacío. No acredita colas ocultas PCIe,
recepción posted, CDC nativo, reset físico, DDR de 512 GiB ni placa.
La temporización sigue abierta: C/E/D aislado obtuvo WNS −2,546 ns y R,
−3,096 ns. No hay una medida física nueva en esta tanda.

Ejecutar el [plan de síntesis conjunta](PCIE_SYMBOLIC_MONITOR_SYNTHESIS_PLAN_R1.md)
y reproducir después su propia netlist. Se preservan caudal, causalidad,
contexto por valor, cancelación solo por frío y publicación alineada
durante cualquier optimización posterior.
