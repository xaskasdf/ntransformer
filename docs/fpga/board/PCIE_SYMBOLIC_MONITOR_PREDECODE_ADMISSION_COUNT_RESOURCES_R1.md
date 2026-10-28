# Distribución de recursos del contador C — R1

## Comparación archivada

Se contrastaron los inventarios y reportes de utilización RQ 984 y C 999
con sus archivos certificados en D: y NFS. Se reutilizó exactamente el
parser de la auditoría de recursos R2, cuya evidencia está identificada
por hash. Los totales coinciden con las revisiones de síntesis.

| Grupo conservado por Vivado | Cambio en primitivas LUT | Cambio en FF | Cambio en CARRY8 |
| --- | ---: | ---: | ---: |
| C | −192 | −18 | 0 |
| E | +812 | −28 | +2 |
| D | −1.052 | +17 | −2 |
| A/B/retiro/top | 0 | 0 | 0 |

La reducción total es de **29 FF**, distribuida entre tres etapas.
El inventario cambia 88 nombres de registros por 59 nombres nuevos.
En C desaparecen 44 nombres bajo `out_record.rc_data` y aparecen 26 bajo
`out_record.rq_spans`; en E cambian varias réplicas. D añade diecisiete
nombres `out_record_reg[valid]_rep*`. Esto describe las asignaciones y
réplicas de la netlist, no una correspondencia entre estados lógicos ni
una prueba de que se haya eliminado almacenamiento funcional.

## LUT: dos recuentos distintos

Las primitivas LUT bajan de 99.660 a 99.228, mientras que la utilización
CLB LUT ajustada por combinación **sube de 92.913 a 93.025**. La diferencia
entre ambos recuentos pasa de 6.747 a 6.203. El reporte del fabricante
declara ese ajuste; el inventario no permite repartirlo entre etapas.
Por ello, la disminución de primitivas en C o D no acredita ahorro físico
de esos bloques ni contradice las 112 CLB LUT adicionales del conjunto.

## Decisión y evidencia

Aunque solo cambió una asignación RTL en C, el mapeo resultante cambió
también en E y D. Se mantiene RQ como referencia. El replay propio y los
diagnósticos por etapas siguen siendo necesarios; esta auditoría no
prueba equivalencia binaria ni mejora de timing físico o throughput.

Resultado:
`build/cloud-setup/20261002-board/pcie-symbolic-monitor-predecode-admission-count-resources-r1.json`,
SHA-256 `5ebb65ede2fa7a94f1ad46b753018351e23799126b4f96d70e4060a7c6113cac`.
Contiene los inventarios por jerarquía, diferencias y listas exactas de
nombres de registros. No requirió otra ejecución de Vivado.
