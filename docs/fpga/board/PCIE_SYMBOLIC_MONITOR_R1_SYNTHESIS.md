# Monitor PCIe R1: síntesis conjunta

Vivado 2026.1 sintetizó el camino real C3/E2/D2/A1/B1/R1 y sus condiciones
de inventario vacío después de la [regresión RTL completa](PCIE_SYMBOLIC_MONITOR_R1_FULL_RESULTS.md).
Parte `xcvu5p-flva2104-2-i`, OOC, `RuntimeOptimized`, jerarquía `rebuilt`,
dos hilos, reloj de 4 ns, retardos I/O cero y sin excepciones.
**Las cifras son estimaciones previas a colocación y ruteo.**

## Recursos y temporización

| Diseño | LUT | FF | CARRY8 | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monitor completo | 104259 | 26976 | 2066 | -3.584 | -0.075 |

No hay latches, BRAM, URAM ni DSP. El setup estimado
no cumple los 4 ns; el hold estimado
no cumple su restricción.
Esta medida sustituye sumar recursos o retardos de bloques aislados;
conserva sus resultados históricos para comparación.

```text
Slack (VIOLATED) :        -3.584ns  (required time - arrival time)
  Source:                 delivery_stage/out_b_reg[origin][record][rc_user][64]/C
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            retirement_i/fault_latched_reg[10]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        7.566ns  (logic 1.538ns (20.328%)  route 6.028ns (79.672%))
  Logic Levels:           31  (CARRY8=1 LUT4=3 LUT5=6 LUT6=21)
```

## Comprobaciones y límites

El banco previo repitió 529 filas: 252 del monitor y 277 de retorno de
rango. Reprodujo los contadores completos de la revisión RTL y los
31 cambios entre flancos. Las diecinueve fuentes compiladas coinciden
con el manifiesto; la síntesis utiliza catorce fuentes de producción.

Reportes, netlist, EDIF, DCP y XDC están vinculados al archivo respaldado.
El grafo del DCP es acíclico y coincide con el inventario de primitivas;
el auditor rechaza siete corrupciones y un ciclo inyectado. Las doce
categorías de `check_timing` no presentan incidencias; los reportes no
contienen excepciones de ruta ni DRC. Esto no significa cierre temporal.

Se revisaron individualmente los avisos emitidos. Los de igualdad de
cuatro estados y puertos sin carga alcanzan el límite de cien mensajes
de Vivado; la revisión no enumera las instancias suprimidas. La
equivalencia binaria de la netlist aún requiere su propia simulación.
Las pruebas X/Z permanecen separadas en RTL. CDC, reset nativo, recepción
posted, DDR de 512 GiB, ECAD y medidas en placa siguen abiertos.

## Próxima tanda y evidencia

Derivar el adaptador de los puertos realmente exportados y comparar
sus 447 bits públicos contra una referencia RTL con módulos renombrados.
La selección prevista conserva 174436 filas binarias en dieciséis bancos,
cinco vaciados por banco, comparaciones pre/post, diez sondas binarias
entre flancos y seis alteraciones reales de las entradas del candidato.
No denominar este paso aprobado hasta revisar su archivo terminal.

- Tanda: `20261009T195029Z-V4-ffn-supervised-pcie-symbolic-monitor-synth-r1`; duración Vivado 460.609 s.
- Archivo D:/NFS: SHA-256 `9eee155bfbf6cc1bc20e3e0b4b91535eb639a1db17bf0c6f383ed0cf275c9cec`.
- Fuentes: SHA-256 `6c13f35e9950bc62b36ff0fe541a99682217340f62c3456288878129a7c089a4`.
- Revisión: `20261009T195029Z-V4-ffn-supervised-pcie-symbolic-monitor-synth-r1-independent-synthesis-review.json`, SHA-256 `97f719cca42d21869aaae6f53bcd0fbd11ed93e647080cecd4508aa642ac00a9`.
