# Síntesis medida de la bandera de contexto

## Resultado y comparación

La tanda 970, `20261011T143058Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-winner-synth-r1`,
terminó con salida **0** en **467,291 s**. La revisión independiente y la
auditoría del grafo aprobaron la evidencia; fuentes y resultado están
verificados en D: y NFS. La comparación usa la síntesis del padre 966,
con iguales builds de Vivado 2026.1, dispositivo VU5P, período de 4 ns,
E/S de 0 ns, estrategia `RuntimeOptimized` y jerarquía `rebuilt`.

| Medida | Prefijo 966 | Bandera 970 | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 105.840 | 105.518 | −322 |
| FF | 27.534 | 27.534 | 0 |
| CARRY8 | 2.122 | 2.104 | −18 |
| WNS estimado | −2,546 ns | −2,768 ns | −0,222 ns |
| WHS estimado | −0,075 ns | −0,075 ns | 0 |
| TNS estimado | −2.174,826 ns | −2.175,048 ns | −0,222 ns |
| Endpoints que incumplen setup / hold | 1.415 / 2.177 | 1.415 / 2.177 | 0 / 0 |
| Niveles de la peor ruta | 24 | 27 | +3 |

No infiere RAM, DSP ni latches. Ambas variantes tienen 70.887 endpoints.
La reducción de recursos **no mejora el peor setup**. El prefijo 966 sigue
siendo la referencia con mejor WNS entre estas dos síntesis comparables.

## Ruta crítica y decisión

La nueva ruta va de `B.rc_user[64]` a `R.fault_latched[10]`: **6,750 ns**,
con 1,560 ns de lógica y 5,190 ns de interconexión estimada. Contiene
24 LUT y tres CARRY8. El análisis de las conexiones e INIT de la netlist
archivada verifica estos tramos:

| CARRY8 en `delivery_stage` | Arco atravesado |
| --- | --- |
| `fault_latched_reg[10]_i_4077` | `S[2] → O[2]` |
| `fault_latched_reg[10]_i_3629` | `S[1] → O[4]` |
| `fault_latched_reg[10]_i_3010` | `DI[5] → CO[6]` |

Hay una cadena directa de seis factores AND y otra de tres OR, según los
cofactores binarios de los pines críticos. Esta caracterización no atribuye
todavía las funciones completas a expresiones RTL ni demuestra equivalencia
X/Z. El analizador R2 reconoce los segmentos de vectores emitidos por Vivado;
su diagnóstico R1 quedó conservado sin modificar el resultado de síntesis.
El [análisis posterior de los tres conos](PCIE_SYMBOLIC_MONITOR_PREDECODE_WINNER_CONES_R1.md)
probó que las dos primeras salidas son cero para todas sus entradas binarias;
la tercera corresponde al límite superior del byte 3 del lane 6. Esto no
autoriza excepciones de timing ni una sustitución RTL que ignore X/Z.

Se completa el replay binario propio `20261011T144252Z` y después la
[comparación de timing por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_STAGE_TIMING_PLAN_R1.md).
No se promueve esta variante por timing ni se encola otro ruteo. La
[integración RTL aprobada](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_RESULTS_R1.md)
conserva su alcance funcional independiente.

## Evidencia y límites

Paquete terminal: **37.975.641 bytes**, SHA-256
`ea97da1a4e399d63d252f6a0c893187dd4782d9ecf9ff6aa5386c88609527114`.
Revisión: `882d82696893142e68ea3b25c18d5c50ba2ff113ed4bbc62ba6b7104d321f0c2`.
Comparación: `0a92b651af698714f9e7fcacb68a238ba7a8c441420e7017165b88b483f02cd3`.
Análisis de ruta: `pcie-symbolic-monitor-predecode-fault-leaves-winner-critical-luts-r2-analysis.json`,
SHA-256 `9d2142e0b0a2b72de38bd3c6ef68d441c6fa35d0d06b74a37c05ccccc1d71240`.
Registros en `build/cloud-setup/20261002-board/`. Son medidas de síntesis;
no acreditan timing físico, frecuencia de placa ni rendimiento de inferencia.
