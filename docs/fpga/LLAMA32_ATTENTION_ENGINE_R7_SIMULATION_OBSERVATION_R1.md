# Motor de atención R7: observación de simulación terminada

Estado 2026-10-13, 19:56 UTC: **traza completa XSim revisada con éxito**.
Esta observación de solo lectura no sustituye la revisión terminal de
síntesis, fuentes compiladas y respaldo que realiza el coordinador existente.
No se lanzó otra simulación ni se interrumpió la tanda.

## Cobertura y resultados

Tanda `20261013T191941Z-V7-llama32-attention-engine-vivado-r1`, watchdog
65.536, semilla 1. La integración utiliza aritmética RTL generada por Vitis
y memoria/AXI simulados.

| Medida | Resultado observado |
| --- | --- |
| Casos de referencia / consultas | 143 / 155 |
| Completions / palabras públicas FP32 | 152 / 9.600 |
| Tiles revisados numéricamente | 155 |
| Error máximo de salida | 0 ULP |
| Error escalado máximo de estado | 3,599 × 10⁻¹⁶ |
| Resets / vinculaciones / append | 4 / 19 / 756 |
| Tráfico escrito / leído en AXI BFM | 3.096.576 / 561.152 bytes |
| Ciclos del banco completo | 2.489.451 |
| Alteraciones de evidencia rechazadas | 42 |

Los ciclos corresponden al banco con fallas, pausas, resets y recuperación;
no son latencia por token ni una medición física de ancho de banda.

## Correcciones ejercitadas

La traza incluye 151 tiles iniciales terminados, aislamiento de lecturas
especulativas K/V en 155 tiles y cuatro continuaciones que aíslan el índice
104 del estado. Pasaron las consultas 135 y siguientes, incluido el punto
que detenía la revisión anterior. Los tres diagnósticos previos permanecen
archivados; sus ejecuciones parciales no se convierten retrospectivamente
en pruebas aprobadas.

## Evidencia y límites

Revisor: `tools/check_ntrpu_attention_engine_r4.py`. Registro local:
`build/cloud-setup/20261002-board/20261013T191941Z-V7-llama32-attention-engine-vivado-r1-simulation-observation-r2-20261013T195642Z.json`.
SHA-256: `ca1cd889bae649fa1d66f95e596eb80ae105e914614aadb48dc074b90104b4e4`.

La primera transferencia sin compresión agotó su límite de observación;
la segunda transportó la misma traza comprimida y verificó su hash. No se
reinició Vivado. El coordinador R7 sigue siendo propietario de la revisión
terminal y del respaldo completo. El secuenciador de 32 cabezas debe esperar
esa aprobación; DDR física, wrapper completo, proyección O y timing ruteado
no están acreditados por esta observación.
