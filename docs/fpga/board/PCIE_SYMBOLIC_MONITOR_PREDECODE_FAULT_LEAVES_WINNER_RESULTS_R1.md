# Integración aprobada de la bandera de contexto

## Resultado

La tanda 969 `20261011T141336Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-winner-r1`
terminó con exit 0 en 724,311 s. La revisión independiente aprobó las
**183.058 filas nuevas**, sin reutilizar positivos de otra variante.
Comparó los 447 bits públicos con 183.138 muestras antes y después
del flanco, 31 sondas de quiet y 62 muestras de sonda/restauración.

Los seis controles de entrada alterada produjeron las divergencias
previstas y el revisor rechazó 34 corrupciones de evidencia. Los seis
bancos drenados mantuvieron cero cancelaciones de tráfico original.
Se conservaron referencia, oráculos, capacidad y latencia de cinco
ciclos desde C hasta retiro.

## Artefactos verificados

Archivo terminal de 47.007.057 bytes respaldado en D: y NFS, junto con
fuentes congeladas y revisión:

| Artefacto | SHA-256 |
| --- | --- |
| Archivo terminal | `75bf6a6c428787e51ca68798773af9a61a2cbd4765534b2888939c251923ee83` |
| Fuentes | `2ff39f063c3c887aa948813f4844ec21d360c33c1397de86a5deb4aa44ce8085` |
| Evidencia | `0bb63b1d79f987a7cb255dbcf2ed507a4b411d3e24ac3813544e66ec099f8801` |
| Revisión independiente | `7d9527189d47cedcc68f9f91838218e17abc19b0aea147e25e76295c8fba8027` |

La revisión enlaza el [ensayo aislado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_RESULTS_R1.md)
y comprueba las 24 fuentes SV compiladas en cada caso integrado.
La corrección del nombre heredado de esa revisión se preserva en la
[recuperación R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_QUEUE_R2.md).

## Siguiente paso

La síntesis propia arrancó como `20261011T143058Z`; el replay binario
de su netlist espera revisión y respaldo. Después se leerán los DCP
del padre y del candidato para [comparar las fronteras del pipeline](PCIE_SYMBOLIC_MONITOR_PREDECODE_STAGE_TIMING_PLAN_R1.md).
Este PASS RTL no demuestra todavía mejora temporal, timing físico ni
funcionamiento de la placa.
