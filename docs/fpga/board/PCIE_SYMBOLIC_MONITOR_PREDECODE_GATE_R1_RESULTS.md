# Monitor PCIe predecode R1: filtro RTL aprobado

El candidato de predecodificación E/D pasó el filtro inicial de **9.948
filas originales** contra la referencia R1. Se compararon sus **447 bits
públicos** antes y después de cada flanco: 9.973 muestras por fase,
incluidos cinco ciclos de drenaje por banco. La simulación duró 811,755 s.

| Banco | Filas |
| --- | ---: |
| `monitor` | 252 |
| `directed` | 217 |
| `controls` | 352 |
| `range_return` | 277 |
| `rq` | 8.850 |

## Controles y revisión

Se conservaron los oráculos originales R7/S1, el seguimiento por origen,
las 31 sondas de quiet con X/Z y sus 31 restauraciones. Se rechazaron
**27 mutantes reales**: 16 de guards/pending, cinco de selección de filas
y seis de entradas. El revisor rechazó también 34 corrupciones de evidencia.
Los 64 avisos emitidos corresponden a los dos diagnósticos previstos por
cada una de las 32 simulaciones; no hubo errores ni avisos críticos.

La primera revisión rechazó un testigo mal especificado para `omit_raw_rc`.
Ese mutante elimina el guard RC y todo el término de actividad inmediata
de `pipeline_pending`. El comparador público detecta primero la sonda RQ
12: `pipeline_pending` pasa de 1 a 0, mientras el guard RQ conservado deja
`inventory_empty` en 0. La antigua prueba de quiet aislada llegaba a la
sonda RC 15 porque no comparaba pending entre flancos.

La revisión R2 exige ese primer testigo, la secuencia previa de sondas y
una diferencia exclusiva en el bit público 1. Rechazó seis corrupciones
adicionales de fila, fase, secuencia y valor. **No se cambiaron el RTL,
los vectores ni la simulación archivada.** El diagnóstico R1 se conserva.

## Evidencia y siguiente dependencia

- Tanda: `20261011T073744Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-gate-r1`.
- Archivo verificado en D:/NFS: `17b6abaa24e6471ad79d19a41f1f9123ffc6bcdfd5c2e45e1d11d17d5a1cfdd5`.
- Fuente: `eab2cfb34c3a257674988f3dd7adfd2234c8879ef196b19a4a6e5f561754f6b7`.
- Revisión R2: `e54c8112ab22dae5dc02e84015cc48f8af1df611a0e4f897400061d4739205e4`.

Este filtro habilita las **173.110 filas restantes** del [plan](PCIE_SYMBOLIC_MONITOR_PREDECODE_PLAN_R1.md).
La síntesis propia, replay de netlist, temporización y validación física
siguen pendientes; el filtro no demuestra una mejora de timing.
