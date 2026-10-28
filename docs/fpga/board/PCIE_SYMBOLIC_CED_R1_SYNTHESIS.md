# Frontend C/E/D: caracterización de síntesis

Resultados del 2026-10-09, posteriores a los [bancos RTL](PCIE_SYMBOLIC_CED_R1_RESULTS.md).
Se sintetizaron C, E, D y el conjunto con Vivado 2026.1,
`RuntimeOptimized`, `xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos I/O
cero y sin excepciones de temporización. Son estimaciones sin colocación
ni ruteo. **El diseño aún no tiene cierre físico ni replay propio aprobado.**

## Recursos y temporización

| Unidad | LUT | FF | CARRY8 | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 24,002 | 4,448 | 1,677 | -3.361 | -0.075 |
| effects | 1,523 | 3,900 | 112 | -0.876 | -0.075 |
| descriptions | 78,076 | 16,885 | 26 | +0.588 | -0.075 |
| combined | 101,953 | 26,913 | 1,815 | -3.212 | -0.075 |

No se infirieron latches, BRAM, URAM ni DSP. Las unidades independientes
exponen interfaces diferentes: no se deben sumar sus recursos como si
fueran el circuito final. La columna conjunta mide el conjunto real.
Frente al S1 anterior (WNS −6,566 ns con el mismo criterio OOC), el cambio
de WNS del conjunto es +3.354 ns; esto no demuestra una frecuencia
alcanzable en placa. Cualquier margen negativo sigue abierto.

## Revisión de evidencia

Los cuatro DCP tienen grafo combinacional acíclico y sus inventarios de
primitivas coinciden con los informes de Vivado. Cada unidad rechaza siete
controles de corrupción/hierarquía y un ciclo inyectado. Las doce categorías
de `check_timing` tienen cero incidencias; no hay endpoints internos sin
restricción ni excepciones ocultas. DRC no encontró incidencias dentro de
su alcance OOC, que no cubre todas las comprobaciones físicas.

Se revisaron las advertencias de declaraciones adelantadas de `rq_held` y
`rc_held`, comparaciones de cuatro estados convertidas a lógica binaria,
bits sin carga del selector, campos originales que D sustituye por su
contexto retenido o por comandos precompuestos de E, y eliminación de
registros duplicados de validez de consulta. Se cotejan los primeros cien
avisos de puertos sin carga y el límite de mensajes del compilador.
Sus líneas exactas están en la revisión. Las
advertencias de reloj/floorplan reflejan las limitaciones OOC. El replay
del circuito exacto sigue siendo obligatorio antes de aprobar su función.

## Caminos críticos medidos

### control

```text
Slack (VIOLATED) :        -3.361ns  (required time - arrival time)
  Source:                 allocated_tags_reg[51]/C
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            read_range_high_reg[0]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        7.344ns  (logic 1.881ns (25.613%)  route 5.463ns (74.387%))
  Logic Levels:           30  (CARRY8=4 LUT1=1 LUT2=5 LUT3=3 LUT4=2 LUT5=3 LUT6=8 MUXF7=2 MUXF8=2)
```
### effects

```text
Slack (VIOLATED) :        -0.876ns  (required time - arrival time)
  Source:                 in_c[ops][prefix_counts][3]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            out_e_reg[updates][0][summary][10]/R
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        4.787ns  (logic 1.444ns (30.165%)  route 3.343ns (69.835%))
  Logic Levels:           20  (CARRY8=8 LUT2=1 LUT4=6 LUT6=5)
  Input Delay:            0.000ns
```
### descriptions

```text
Slack (MET) :             0.588ns  (required time - arrival time)
  Source:                 symbolic_rows_reg[51][4]/C
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            symbolic_rows_reg[100][6]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        3.394ns  (logic 1.053ns (31.025%)  route 2.341ns (68.975%))
  Logic Levels:           14  (CARRY8=2 LUT2=1 LUT3=1 LUT4=1 LUT5=2 LUT6=4 MUXF7=2 MUXF8=1)
```
### combined

```text
Slack (VIOLATED) :        -3.212ns  (required time - arrival time)
  Source:                 c_stage/allocated_tags_reg[51]/C
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            c_stage/read_range_high_reg[0]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        7.195ns  (logic 1.910ns (26.546%)  route 5.285ns (73.454%))
  Logic Levels:           30  (CARRY8=4 LUT1=1 LUT2=3 LUT3=1 LUT4=2 LUT5=3 LUT6=12 MUXF7=2 MUXF8=2)
```

## Respaldo y siguiente tanda

- Tanda: `20261009T133419Z-V4-ffn-supervised-pcie-symbolic-ced-r1-synth`.
- Archivo local y remoto: SHA-256 `215a15fbecd93457c8d18e55d2793fa23bb3b1ea559b0c2c5dde23f98fc4b204`.
- Fuentes: SHA-256 `5a94dac0f50ad164754eafb53359ea6f1408605e111e5a087777d28da4ee5125`.
- Revisión independiente: `20261009T133419Z-V4-ffn-supervised-pcie-symbolic-ced-r1-synth-independent-synthesis-review.json`, SHA-256 `f3842d5e7d2e5313ed3187f0fa58c3700573dccccc4ce9def2f0fb6db7c8a4dd`.

Está preparado el [plan de replay](PCIE_SYMBOLIC_CED_R1_MAPPED_PLAN.md)
con 205.493 filas positivas, cuatro controles negativos y comparación de
los 2.499 bits externos. La netlist se conserva intacta; solo los módulos
RTL de referencia reciben otros nombres. Aún no se ha ejecutado esa tanda.

La aplicación numérica, validación tardía, retiro de bytes, monitor nativo,
CDC/reset, cuatro DDR hasta 512 GiB, pesos reales y placa continúan abiertos.
R12 sigue siendo la referencia del monitor completo; se conservan todas
las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
