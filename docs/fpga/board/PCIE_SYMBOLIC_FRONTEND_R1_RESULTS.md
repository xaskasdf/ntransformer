# Frontend simbólico S1 R1: resultados

Revisión del 2026-10-09. S1 ya está implementado y probado; **no es todavía
el monitor completo S1/S2/S3 ni una implementación física cerrada**.

## Frontera registrada

`fpga/rtl/ntrpu_pcie_symbolic_frontend_r1.sv` conserva control y tabla por
valor de 51 bits. Captura un registro de 2.107 bits con beat crudo, contexto
previo, secuencia, cuatro consultas y cuatro eventos EOP con lane original.
Los spans RQ conservan signo. No aplica remaining/dirección ni retira bytes;
`control_pending` debe combinarse con las obligaciones de las etapas futuras.

## Validación

Se ejecutaron **80.120 ciclos** en seis bancos, con **20.510.720 filas de tabla**
comparadas contra R7, también después de fallos. Consultas y deltas numéricos
usan R7; las 16 contribuciones por lane se cotejan con el modelo simbólico
revisado. Se comprueban captura al flanco, contexto, guarda de tags/lecturas,
reset frío y wrap. Cuatro fuentes defectuosas reales fueron rechazadas:
omitir EOP, recargar el tag reutilizado, alterar secuencia y retener el registro
tras reset frío.

La netlist propia pasó **79.709 ciclos binarios**, comparando **2.498 bits de
salida** contra el RTL exacto, con cuatro controles negativos. Se esperan
200 ns de GSR y se muestrea a 1 ns del flanco, sin SDF. En contexto se excluyen
cuatro escenarios X/Z completos (50 filas); siguen cubiertos por el banco RTL.

## Caracterización aislada

Vivado 2026.1, `xcvu5p-flva2104-2-i`, RuntimeOptimized OOC, período 4 ns,
I/O cero, sin excepciones ni placement/routing. El DCP pasó la auditoría
de conectividad combinacional y sus primitivas coinciden con los reportes.

| Métrica S1 | Resultado |
| --- | ---: |
| LUT | 74191 |
| FF | 18312 |
| CARRY8 | 1719 |
| WNS / TNS | -6.566 / -149827.656 ns |
| WHS / THS | -0.075 / -207.571 ns |
| Endpoints que fallan setup | 35938 / 53327 |

No hay latches, BRAM, URAM ni DSP. Estas cifras cubren solo S1; no son
comparables como reemplazo completo de R12. R12 conserva su estado anterior.

Camino peor reportado:

```text
Slack (VIOLATED) :        -6.566ns  (required time - arrival time)
  Source:                 rc_user[76]
                            (input port clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Destination:            symbolic_rows_reg[104][5]/D
                            (rising edge-triggered cell FDRE clocked by pcie_user_clk  {rise@0.000ns fall@2.000ns period=4.000ns})
  Path Group:             pcie_user_clk
  Path Type:              Setup (Max at Slow Process Corner)
  Requirement:            4.000ns  (pcie_user_clk rise@4.000ns - pcie_user_clk rise@0.000ns)
  Data Path Delay:        10.576ns  (logic 2.898ns (27.402%)  route 7.678ns (72.598%))
  Logic Levels:           41  (CARRY8=8 LUT1=1 LUT2=3 LUT3=4 LUT4=8 LUT5=4 LUT6=13)
  Input Delay:            0.000ns
```

## Evidencia y continuidad

Las tandas están certificadas en D: y NFS, con fuentes, scripts, logs y hashes.
El primer lanzador falló antes de simular por argumentos `set_property`;
su revisión R2 corrigió solo el transporte de opciones. Un aviso Vivado
12-3645 sobre agrupar `add_files` se revisó explícitamente; no era un fallo RTL.

| Tanda | Identificador | SHA-256 del archivo |
| --- | --- | --- |
| pcie_symbolic_frontend_r1_launch_diagnostic | `20261009T114329Z-V4-ffn-supervised-pcie-symbolic-frontend-r1-rtl` | `544dd62d394133fe9443632a256a8dac68b4fc14556aa4859432d0cfa36be151` |
| pcie_symbolic_frontend_r1_rtl | `20261009T114628Z-V4-ffn-supervised-pcie-symbolic-frontend-r1-rtl-r2` | `65b287d29343aacf0ccad8590db8e5fce010e3ae7ef04e16a3fa4cb97dff660e` |
| pcie_symbolic_frontend_r1_synth | `20261009T115531Z-V4-ffn-supervised-pcie-symbolic-frontend-r1-synth` | `27f0dc0b349df8155fe19866111280da8233d1b7325be74582bff8f0d62416fc` |
| pcie_symbolic_frontend_r1_mapped_adapter_diagnostic | `20261009T120750Z-V4-ffn-supervised-pcie-symbolic-frontend-r1-binary` | `0c258a030f733d9207e975642d2f738cc7123f9d8dcc1dc5bf3580f5b80eee0d` |
| pcie_symbolic_frontend_r1_mapped | `20261009T121610Z-V4-ffn-supervised-pcie-symbolic-frontend-r1-binary-r2` | `46ab1d14deb716767feccc9a169c667bb301b607f97773afc92c3e329a9657c5` |

Siguiente trabajo: Implement the C/E/D candidate in docs/fpga/board/PCIE_SYMBOLIC_FRONTEND_R2_PLAN.md: register raw selection, precompose state-independent effects, then select the own descriptor base and compose once per commit. Derive exact origin/cold cancellation and keep payload values independent of reused tags. Measure each boundary; no promised timing or depth. Preserve raw RQ2/RC4 and all capacities. The complete pipeline needs its own four-state banks, mapped replay and timing; preserve cold-only cancellation, post-fault state, wrap, retained payload values and immediate empty revocation. All native/CDC/reset/DDR512GiB/weights/board gates remain open.

Contrato: [partición candidata](PCIE_SYMBOLIC_PIPELINE_R1_CONTRACT.md).
Dependencias de placa: [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
