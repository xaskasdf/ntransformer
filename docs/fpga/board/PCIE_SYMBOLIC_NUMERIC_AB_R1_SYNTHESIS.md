# A/B R1: síntesis propia y replay preparado

Medidas del 2026-10-09 posteriores a la [validación funcional](PCIE_SYMBOLIC_NUMERIC_AB_R1_RESULTS.md).
Vivado 2026.1 sintetizó A, B y su conexión literal para `xcvu5p-flva2104-2-i`:
OOC, `RuntimeOptimized`, jerarquía `rebuilt`, dos threads, reloj de 4 ns,
I/O cero y sin excepciones. **Son estimaciones sin colocación ni ruteo.**

## Recursos y temporización

| Unidad | LUT | FF | CARRY8 | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| apply | 2,253 | 2,723 | 90 | +1.730 | -0.075 |
| delivery | 1,686 | 2,892 | 9 | +2.366 | -0.075 |
| pair | 3,942 | 5,616 | 99 | +1.730 | -0.075 |

El conjunto se midió directamente. No tiene latches, BRAM, URAM ni DSP.
El setup estimado cumple 4 ns; el hold OOC es negativo y no se ha
demostrado frecuencia en placa. C/E/D R3 conserva su déficit de setup
de −2,546 ns: estas cifras A/B no cierran el monitor completo.

## Comprobaciones

Antes de sintetizar, el contenedor repitió los 266.681 pasos focalizados,
comparando 2.723 bits de A y 2.892 de B. Las ocho fuentes compiladas
coinciden con el manifiesto; doce corrupciones de evidencia fueron
rechazadas. Las tres netlists DCP tienen grafos combinacionales acíclicos,
inventarios de primitivas coincidentes y ocho controles estructurales
rechazados por unidad. Las doce categorías de `check_timing` tienen cero
incidencias; no hay excepciones de temporización.

Se revisaron las advertencias OOC y una de entorno XSIM. El primer
revisor de evidencia confundió el eco del Tcl con un marcador ejecutado;
R2 delimita por líneas completas. No se repitió ni modificó la síntesis.

## Caminos críticos

### apply

```text
Slack (MET) :             1.730ns  (required time - arrival time)
  Source:                 in_record[pre_rc_context][description][3]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            out_a_reg[context_value][24]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        2.280ns  (logic 0.770ns (33.772%)  route 1.510ns (66.228%))
  Logic Levels:           9  (CARRY8=3 LUT5=2 LUT6=4)
  Input Delay:            0.000ns
```
### delivery

```text
Slack (MET) :             2.366ns  (required time - arrival time)
  Source:                 in_a[record][eops][0][words][2]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            out_b_reg[eop_delivered][0][0]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        1.644ns  (logic 0.560ns (34.063%)  route 1.084ns (65.937%))
  Logic Levels:           6  (CARRY8=1 LUT4=1 LUT5=2 LUT6=2)
  Input Delay:            0.000ns
```
### pair

```text
Slack (MET) :             1.730ns  (required time - arrival time)
  Source:                 in_record[pre_rc_context][description][3]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            a_stage/out_a_reg[context_value][24]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        2.280ns  (logic 0.770ns (33.772%)  route 1.510ns (66.228%))
  Logic Levels:           9  (CARRY8=3 LUT5=2 LUT6=4)
  Input Delay:            0.000ns
```

## Evidencia y siguiente prueba

- Tanda: `20261009T162540Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-synth-r1`; duración 275.507 s.
- Archivo D:/NFS: SHA-256 `275e74ca1d956bedd49e020db7af0d1c13f2c415a82fda693099295a312a57e3`.
- Fuentes: SHA-256 `5e48ec9c153f56b8772cfcb55ebd41ec10be9bdad2322e8e168ac97c03808697`.
- Revisión: `20261009T162540Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-synth-r1-independent-synthesis-review.json`, SHA-256 `e51a4f569050e8cbb406d49dab3aafa4d7f1fd965addd4109ab2af6333df6ce4`.

El adaptador se derivó de los 89 puertos de entrada y 109 de salida
realmente exportados. Comparará todos los 2.892 bits B contra RTL
inmutable y las observaciones R7 originales: 169.601 filas binarias,
GSR de 200 ns, muestreo de 1 ns y cuatro flancos de vaciado.
Cinco controles alteran únicamente entradas/reset del candidato.
El replay propio sigue pendiente; la prueba de cuatro estados se conserva.

Validación tardía de 15 bits, retiro de 23 bits, monitor completo, CDC/reset,
DDR de 512 GiB, pesos reales y placa mantienen sus requisitos en
[VIVADO_QUEUE.md](VIVADO_QUEUE.md).
