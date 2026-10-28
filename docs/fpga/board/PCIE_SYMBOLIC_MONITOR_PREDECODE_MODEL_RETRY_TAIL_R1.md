# Cola dependiente recuperada tras el diagnóstico FFN

Fecha: 2026-10-12 UTC. Los ocho coordinadores anteriores se detuvieron
al fallar el banco de memoria de la FFN real. Ninguno inició su trabajo
Vivado ni creó un owner de hardware. Sus registros y archivos se conservan.

La preparación `pcie-symbolic-monitor-predecode-model-retry-tail-r1-preparation.json`
en `build/cloud-setup/20261002-board/` fija los reemplazos, hashes y estados
anteriores. SHA-256:
`99f7349118f4eca5c9c029910555ef961fe73a0de89a342174117ea88f108910`.
El nuevo origen es `llama32-full-ffn-r4-coordinator.json`, descrito en
[recuperación FFN](../LLAMA32_FULL_FFN_RETRY_R1.md).

| Orden | Coordinador nuevo | Prerrequisito |
| --- | --- | --- |
| 1 | `matched-clock-r2` | FFN R4 revisada y respaldada |
| 2 | `matched-clock-binary-r2` | Ambos diagnósticos de reloj |
| 3 | `matched-physical-r2` | Ambos replays optimizados |
| 4 | `matched-physical-review-r3` | Implementación y readback de ambos candidatos |
| 5 | `matched-route-binary-r2` | Revisiones físicas y comparación |
| 6 | `mask-rows-full-r2` | Ambos replays ruteados y selector aislado |
| 7 | `mask-rows-synth-r3` | Regresión integrada de máscaras |
| 8 | `mask-rows-binary-r2` | Síntesis propia revisada |

Todos llevan el prefijo `pcie-symbolic-monitor-predecode-` y el sufijo
`-coordinator.json`. Los runners y owners de hardware conservan su identidad
R1: todavía no se ejecutaron y sus revisores esperan esas rutas.
Se corrige además una referencia anticipada del coordinador de síntesis
al inexistente owner R2; debe leer el owner R1 que crea su runner.

Los dieciséis helpers derivados mantienen las mismas funciones Python.
Cambian rutas de coordinación y guardas de recuperación; permanecen los
RTL, bancos, oráculos, bibliotecas, restricciones, límites y criterios de
los documentos originales. Cada etapa exige PASS/revisión y respaldo antes
de habilitar la siguiente. Los diagnósticos físicos siguen sin acreditar
cierre temporal ni funcionalidad con SDF.
