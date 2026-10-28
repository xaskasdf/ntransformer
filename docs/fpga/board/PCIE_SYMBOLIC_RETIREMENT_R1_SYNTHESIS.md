# R1: síntesis propia del retiro y validador tardío

Medidas del 2026-10-09 posteriores a la [regresión completa](PCIE_SYMBOLIC_RETIREMENT_R1_FULL_RESULTS.md).
Vivado 2026.1 sintetizó R1, su validador tardío y sus cuatro guardas de
cabecera en `xcvu5p-flva2104-2-i`: OOC, `RuntimeOptimized`, jerarquía
`rebuilt`, dos hilos, reloj de 4 ns, I/O cero y sin excepciones.
**Son estimaciones previas a colocación y ruteo.**

## Recursos y temporización

| Unidad | LUT | FF | CARRY8 | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R + validación tardía | 3199 | 465 | 153 | -3.096 | -0.075 |

No hay latches, BRAM, URAM ni DSP. El setup estimado
no cumple los 4 ns; el hold estimado
no cumple su restricción.
No se ha medido A/B/R conjuntamente ni se ha cerrado el monitor completo.
C/E/D R3 mantiene su déficit de setup de −2,546 ns.

## Evidencia revisada

El banco previo repitió las 277 filas dirigidas y reprodujo los mismos
testigos de origen, rango, persistencia y frío de la regresión completa.
Sus diecinueve fuentes SV coinciden con el manifiesto. Los reportes,
netlist funcional, EDIF, DCP y XDC están vinculados al archivo sellado.
El grafo del DCP es acíclico, coincide con el inventario de primitivas y
rechaza los controles de corrupción/ciclo. Las doce categorías de
`check_timing` tienen cero incidencias; no hay excepciones ni DRC reportados.

Las advertencias están registradas por línea y fuente en la revisión.
La conversión de igualdad de cuatro estados a lógica binaria durante
síntesis conserva separada la obligación de comprobar X/Z en RTL.
Se comprobaron 74 conversiones de igualdad y 100 avisos de campos sin
carga en los consumidores de cabecera/validación tardía. Estos campos
no son leídos allí; la publicación de control pertenece a R. El revisor
R2 normaliza las claves numéricas de Python a su representación JSON
para comparar contadores sin cambiar valores. Se conserva R1 y su
diagnóstico; no se repitió la síntesis.

```text
Slack (VIOLATED) :        -3.096ns  (required time - arrival time)
  Source:                 in_b[origin][record][rc_user][64]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            fault_latched_reg[10]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        7.106ns  (logic 1.473ns (20.729%)  route 5.633ns (79.271%))
  Logic Levels:           30  (CARRY8=1 LUT4=2 LUT5=2 LUT6=25)
  Input Delay:            0.000ns
```

## Siguiente tanda

Derivar el adaptador de los puertos realmente exportados y reproducir
**esta netlist propia** contra R1 y R7 por origen. La selección por
escenarios completos prevé 174184 filas binarias de las 182806 originales,
incluidas las 277 dirigidas; no se excluyen fallos binarios. Conservar
cinco flancos de vaciado, estabilización GSR y controles reales del candidato.
La generación del banco y su ejecución siguen pendientes.

Después, aplicar el [plan del monitor completo](PCIE_SYMBOLIC_FULL_MONITOR_PLAN_R1.md).
El camino crítico de validación de payload debe investigarse con esta
referencia funcional conservada. Distinguir el retardo de ruta estimado
de medidas colocadas/ruteadas antes de decidir una nueva partición.
Quiet, integración, medidas conjuntas, cierre físico y validaciones nativas
y de placa siguen abiertos.

- Tanda: `20261009T183059Z-V4-ffn-supervised-pcie-symbolic-retirement-synth-r1`; duración Vivado 138.818 s.
- Archivo D:/NFS: SHA-256 `574dda4ef32fbf4c1822527e65e57c71014114af001e7978fd7afe80a3a0ee46`.
- Fuentes: SHA-256 `f60329b0aaceb001adb8db2ef1350be4f6d5f92df351900070d99c6cd51f6f6f`.
- Revisión: `20261009T183059Z-V4-ffn-supervised-pcie-symbolic-retirement-synth-r1-independent-synthesis-review.json`, SHA-256 `256e529be54c9f7358a72dda3aed8f7596acb70a4a71cc25c1dd046c883f0e9c`.
