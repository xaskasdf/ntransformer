# Recursos por jerarquía: RQ 984, sumas E 991 y límites B 995

## Fuente y unidad de medida

Se leyeron `monitor/primitives.tsv` y `utilization.rpt` de los tres archivos
de síntesis certificados. Sus hashes coinciden con las revisiones aprobadas.
Todos los nombres de celda se atribuyen una sola vez a C, E, D, A, B, R o
TOP; el total coincide con el inventario revisado.

**Primitivas LUT y CLB LUT ajustadas son cifras distintas.** Vivado avisa
que su utilización incorpora combinación de LUT. El intento R1 exigió
erróneamente igualdad y falló; R2 conserva el fallo y separa las cantidades.
El inventario no permite distribuir ese ajuste entre jerarquías ni afirmar
ocupación final tras colocación y ruteo.

| Síntesis | Primitivas LUT | CLB LUT ajustadas | Diferencia | FF | CARRY8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RQ 984 | 99.660 | 92.913 | 6.747 | 27.560 | 568 |
| Sumas E 991 | 99.003 | 92.644 | 6.359 | 27.562 | 568 |
| Límites B 995 | 99.871 | 93.179 | 6.692 | 27.566 | 573 |

## Distribución y cambios observados

| Grupo | Primitivas LUT RQ | FF RQ | CARRY8 RQ | Δ LUT E | Δ LUT B |
| --- | ---: | ---: | ---: | ---: | ---: |
| C | 14.747 | 3.554 | 251 | −657 | −5 |
| E | 20.367 | 5.107 | 16 | 0 | 0 |
| D | 58.504 | 15.928 | 90 | 0 | 0 |
| A | 1.492 | 1.296 | 0 | 0 | +86 |
| B | 4.392 | 1.210 | 197 | 0 | +119 |
| R | 158 | 447 | 14 | 0 | +11 |
| TOP | 0 | 18 | 0 | 0 | 0 |

El experimento E cambia un módulo RTL, pero su ahorro aparece en C tras
`rebuilt`: C pierde un FF y E gana tres. No corresponde atribuir el ahorro
global de 269 CLB LUT al bloque E aislado. B añade seis FF en B y cinco
CARRY8 repartidos entre A (+2) y B (+3).

D también conserva 22.678 MUXF7 y 11.319 MUXF8. Esta concentración apoya
medir el experimento de habilitaciones por fila, pero no demuestra que
vaya a reducir área ni demora. Se mantienen sus filtros RTL, replay propio
y comparación de todos los pares de etapas antes de decidir ruteo.
La jerarquía retenida describe el resultado de síntesis; no establece
causalidad ni costes de bloques sintetizados por separado.

## Reproducción

Ejecutado `build/cloud-setup/20261002-board/audit_pcie_symbolic_monitor_predecode_component_resources_r2.py`.
Tres inventarios aceptados y dieciocho alteraciones rechazadas: celda
omitida, repetida, vacía, cabecera incorrecta, primitiva desconocida y
jerarquía desconocida. Resultado `pcie-symbolic-monitor-predecode-component-resources-r2.json`,
SHA-256 `a0e57aeabad058149b5a3d0fe381f1d351c2b73e5f07c66b93bfc8393f1bd978`.
No se ejecutó otra síntesis ni se modificaron fuentes o restricciones.
