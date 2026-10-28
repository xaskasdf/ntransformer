# Control R3: síntesis propia y comparación con R2

Resultados del 2026-10-09 posteriores a la [prueba RTL R3](PCIE_SYMBOLIC_CED_CONTROL_R3_RTL.md).
Vivado 2026.1 sintetizó C R3 y el conjunto C/E/D R3 para
`xcvu5p-flva2104-2-i`, con `RuntimeOptimized`, jerarquía `rebuilt`, reloj
de 4 ns, I/O cero y sin excepciones. **Son estimaciones OOC sin colocación
ni ruteo; el replay de esta netlist sigue pendiente.**

## Medidas comparables

| Unidad | LUT | FF | CARRY8 | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control R2 | 24,002 | 4,448 | 1,677 | -3.361 | -0.075 |
| control R3 | 23,900 | 4,446 | 1,675 | -2.536 | -0.075 |
| combined R2 | 101,953 | 26,913 | 1,815 | -3.212 | -0.075 |
| combined R3 | 99,923 | 26,884 | 1,813 | -2.546 | -0.075 |

El cambio del conjunto respecto a R2 es **+0.666 ns
de WNS**, **-2,030 LUT**, **-29 FF** y
**-2 CARRY8**. La referencia y la candidata usan el
mismo criterio de síntesis. Estas cifras no certifican frecuencia en placa.
E y D conservan sus fuentes R2; el informe combinado mide el circuito real,
sin sumar las cifras de unidades aisladas.

## Revisión de circuitos y restricciones

Los DCP propios tienen grafos combinacionales acíclicos, primitivas
cotejadas con los informes y siete controles de corrupción/hierarquía más
un ciclo inyectado rechazados por unidad. No se infirieron latches,
BRAM, URAM o DSP. Las doce categorías de `check_timing` tienen cero
incidencias y no hay excepciones ocultas. Las advertencias exactas y los
límites de DRC/metodología OOC se conservan en la revisión independiente.

## Caminos críticos publicados

### control

```text
Slack (VIOLATED) :        -2.536ns  (required time - arrival time)
  Source:                 rc_user[76]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            registered_ops_reg[commit_slot][422]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        6.546ns  (logic 1.582ns (24.167%)  route 4.964ns (75.833%))
  Logic Levels:           27  (CARRY8=7 LUT2=1 LUT3=1 LUT4=5 LUT5=3 LUT6=10)
  Input Delay:            0.000ns
```
### combined

```text
Slack (VIOLATED) :        -2.546ns  (required time - arrival time)
  Source:                 rc_user[76]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            c_stage/registered_ops_reg[commit_slot][422]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        6.556ns  (logic 1.644ns (25.076%)  route 4.912ns (74.924%))
  Logic Levels:           27  (CARRY8=7 LUT2=2 LUT3=1 LUT4=4 LUT5=2 LUT6=11)
  Input Delay:            0.000ns
```

## Evidencia y siguiente paso

- Tanda: `20261009T145408Z-V4-ffn-supervised-pcie-symbolic-ced-control-r3-synth`; duración: 598.724 s.
- Archivo en D: y NFS: SHA-256 `2bc8642b4c55ba96e000a084dd63c7fc1330db8328fe530db9ff84e6e4bc64ae`.
- Fuentes: SHA-256 `910b19735c387be0c1e12deeee5b16fa97e22e6d9b3b263f0023b84cc08a31d3`.
- Revisión: `20261009T145408Z-V4-ffn-supervised-pcie-symbolic-ced-control-r3-synth-independent-synthesis-review.json`, SHA-256 `8b81ff288dad73d6fdc4c7d6035218c3242d20b5c45d65a6f6425b409a16adf9`.

Está preparado el adaptador de los 88 puertos físicos del circuito real:
2.499 bits, de los cuales 2.107 forman el registro descriptivo. Sigue el
replay de 205.493 filas binarias y cuatro controles negativos, con GSR
de 200 ns, muestreo de 1 ns y dos ciclos finales comparados. Las referencias
RTL R3 solo cambian de nombre; la netlist permanece intacta.

La aplicación numérica, validación tardía, retiro de bytes, monitor completo
R12, CDC/reset, DDR de 512 GiB, pesos reales y placa siguen pendientes.
Se mantienen todas las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
