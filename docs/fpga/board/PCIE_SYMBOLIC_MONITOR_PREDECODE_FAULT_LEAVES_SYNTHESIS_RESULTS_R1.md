# Síntesis medida de fallos por lane

## Resultado

Tanda 962, `20261011T121546Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-synth-r1`:
salida **0**, **469,358 s**, respaldo verificado en D: y NFS. La revisión
independiente y la auditoría del grafo del DCP aprobaron la evidencia de
síntesis; el timing sigue incumplido.

## Comparación con la referencia

Ambas ejecuciones usan VU5P `xcvu5p-flva2104-2-i`, Vivado 2026.1 y los mismos
builds, reloj de 4 ns, E/S de 0 ns, `RuntimeOptimized`, jerarquía `rebuilt`,
modo OOC y dos hilos. Los scripts, reportes y fuentes se enlazaron con sus
archivos certificados. La referencia es la tanda 951.

| Medida | Referencia | Fallos por lane | Cambio |
| --- | ---: | ---: | ---: |
| LUT | 104.980 | 105.147 | +167 |
| FF | 27.542 | 27.542 | 0 |
| CARRY8 | 2.066 | 2.066 | 0 |
| WNS estimado | −3,584 ns | −3,746 ns | −0,162 ns |
| WHS estimado | −0,075 ns | −0,075 ns | 0 |
| TNS estimado | −2.178,434 ns | −2.176,026 ns | +2,408 ns |
| Endpoints que incumplen setup / hold | 1.415 / 2.177 | 1.415 / 2.177 | 0 |
| Niveles en la peor ruta de setup | 31 | 31 | 0 |

No se infirieron latches, RAM ni DSP. La nueva peor ruta va de
`delivery_stage/out_b...rc_user[75]` a `retirement_i/fault_latched_reg[10]`;
la anterior partía de `rc_user[64]`. El retardo estimado nuevo es 7,728 ns:
1,595 ns de lógica y 6,133 ns de interconexión. No son mediciones posteriores
al ruteo ni una frecuencia demostrada en hardware.

## Decisión de diseño

Separar las hojas de fallos no redujo la profundidad y empeoró el peor setup.
La leve mejora del TNS no resuelve ese incumplimiento. Se conserva la variante
como experimento validado en RTL y se completa el replay ya encolado de su
propia netlist. No se inicia otro ruteo por este resultado.

La siguiente modificación aborda la propagación de contexto entre lanes:
el [experimento por prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_R1.md)
debe aprobar sus pruebas aisladas e integradas antes de medir su síntesis.

## Evidencia y límites

La revisión comprobó 529 filas de preflight, 14 unidades de síntesis,
constraints, inventario, ausencia de ciclos combinacionales y avisos
emitidos. Los diagnósticos de case-equality y puertos no usados alcanzan
el límite de 100 mensajes de Vivado; no se afirma haber enumerado los
suprimidos. La síntesis binaria no sustituye las pruebas RTL X/Z.

Archivo de 37.655.076 bytes, SHA-256
`836767a3f916b117bf995cf8c900c4bceba4a5e2ff637bf084c1689d6f7aa3dc`.
Revisión SHA-256 `3da0389ac72b7a5c7e7b260eb100097b07ae9c552df1099d7178cd7d48e32d01`.
Comparación SHA-256 `e943806ca80552f208f1441d45d8a0e38de06f8cab51a39e20f35738831fd902`.
