# Ruta crítica mapeada del contexto por prefijo

## Evidencia observada

Se analizaron la netlist y el informe certificados de la tanda 966, sin
volver a ejecutar Vivado. La peor ruta `B.rc_user[79] → R.fault_latched[10]`
tiene **22 LUT y dos CARRY8**, con WNS estimado −2,546 ns a 4 ns.

| Propiedad de la peor ruta | Por hojas 962 | Prefijo 966 |
| --- | ---: | ---: |
| Niveles combinacionales | 31 | 24 |
| Mayor cadena directa de factores AND | 10 LUT | 1 LUT |
| Mayor cadena directa de factores OR | 10 LUT | 3 LUT |
| CARRY8 atravesados | 1 | 2 |

La clasificación se refiere al pin recorrido por esta ruta, no a todos
los conos del diseño. Las rutas empiezan en bits distintos; la tabla no
es una atribución causal de cada celda a una edición de RTL.

## Conexiones comprobadas

Los tramos entre LUT atraviesan realmente
`fault_latched_reg[10]_i_502.DI[1] → CO[6]` y
`fault_latched_reg[10]_i_3531.DI[5] → CO[6]`, ambos dentro de `delivery_stage`.
Las nets de salida se llaman `late_i/cp2379_in` y
`late_i/expected_enable2108_in`. Esos nombres orientan la investigación;
no demuestran por sí solos las funciones aritméticas originales.

Se verificaron todos los cofactores binarios de los 22 INIT y la continuidad
de conexiones, incluidas las dos interrupciones por CARRY8. LUT2 y LUT3
se cotejaron contra 64 y 2.048 casos binarios de las tablas UDP instaladas;
LUT4–6 usan sus expresiones indexadas. El análisis posterior demostró
que el primer cono calcula `last_eop >= last_sop` antes del lane 13,
para todas las asignaciones binarias de sus 32 FF de entrada.
La [prueba formal y el candidato de bandera](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_R1.md)
detallan el alcance. El [análisis del segundo cono](PCIE_SYMBOLIC_MONITOR_PREDECODE_PREFIX_PAYLOAD_CONE_R2.md)
demostró la comparación binaria del primer byte del lane 13 contra el
límite superior de payload, incluso sin el guard de payload activo.
La sensitización temporal de toda la ruta no se ha demostrado.

## Artefactos y siguiente paso

`analyze_pcie_symbolic_monitor_predecode_fault_leaves_prefix_critical_luts_r1.py`
en `build/cloud-setup/20261002-board/` enlaza archivo terminal, revisión,
informe, netlist y bibliotecas. Análisis SHA-256:
`e33a47095cb9d786b16721014c1becbd677c8175fdf31b0d628ed02cdbe60948`.

La recurrencia sobre `delivered` comprobada para la tanda 962 no se
transfiere a esta ruta. Se seleccionó un experimento sobre el predicado
de contexto identificado; su efecto en síntesis aún debe medirse.
Este análisis no demuestra equivalencia X/Z ni cierre físico.
