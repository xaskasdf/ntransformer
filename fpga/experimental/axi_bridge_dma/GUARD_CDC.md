# Revision de cruces y temporizacion del guardian

Este documento conserva la auditoria de V1. La variante aislada y sus nuevas
comprobaciones se documentan en [GUARD_V2.md](GUARD_V2.md).

Estado al 2026-10-11: **diagnostico OOC revisado, con CDC critico pendiente**.
Vivado 2026.1 sintetizo el guardian original para `xcvu47p-fsvh2892-2-e`.
El RTL conserva el SHA-256 de la prueba nativa de reloj
`d5b340ab788143cdca33e4cec760778722e76c783cb5e0e3ee63798df74d142d`.
No se modifica el top ni la cola principal; no se aprueba implementacion fisica.

## Hallazgos que siguen abiertos

`report_cdc -details -all_checks_per_endpoint -no_waiver` identifica dos
cruces de datos CDC-3 y dos resets sincronizados CDC-9. Tambien conserva un
**CDC-10 critico** entre `armed_reg/C` y `admit_a_reg[0]/CLR`: hay logica
combinacional antes del sincronizador. La [regla AMD CDC-10](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Understanding-the-Clock-Domain-Crossings-Report-Rules)
requiere revisar esa topologia; una restriccion temporal no elimina el hallazgo.

Las LUT que generan los controles de `admit_a` y `healthy_m` producen dos
alertas LUTAR-1. Quedan pendientes glitches, anchos minimos de pulso y
recovery/removal. Los cinco puertos de control entrantes y las ocho salidas
carecen de retardos de integracion. El reporte CDC no cubre completamente
esos puertos sin restricciones de I/O, segun [UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_cdc).
Los inventarios de pines si reconstruyen sus conos logicos, pero eso no
caracteriza el comportamiento fisico de las fuentes externas.

## Estructura y ensayo de restricciones

La sintesis reporta 22 LUT logicas y 36 FF. Se verifican siete cadenas de tres
registros: `cold_m`, `cold_a`, `healthy_m`, `admit_a`, `stop_a`, `echo_a` y
`echo_m`. Sus 21 celdas conservan tipo, INIT=0, ASYNC_REG, reloj, habilitacion,
continuidad Q-D y fuentes de control esperadas.

El ensayo reabre el DCP original y aplica restricciones solo en memoria:

- Dos `set_max_delay -datapath_only`: 4 ns de `request_reg/C` a
  `echo_a_reg[0]/D`; 10 ns de `echo_a_reg[2]/C` a `echo_m_reg[0]/D`.
- Nueve pares false-path desde C de `armed_reg`, `fault_reg` y
  `cold_m_reg[2]` hasta los tres CLR de `admit_a`.
- Tres pares desde `fault_reg/C` hasta los tres PRE de `stop_a`.

Son limites experimentales de los relojes 100/250 MHz. No se agregan grupos
de reloj, excepciones sobre entradas externas ni waivers. El DCP no se reescribe.
Se usan pines C como inicio valido: iniciar en Q segmento las rutas en un
intento descartado. Ver [UG903: segmentacion de rutas](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Path-Segmentation).

| Comprobacion independiente | Resultado |
| --- | --- |
| Rutas exportadas dentro del mismo reloj | 1488 conservadas con identicos margenes |
| Conexiones entre etapas | 14; 28 comprobaciones setup/hold iguales |
| Margen minimo entre etapas | setup +3,709 ns; hold +0,079 ns |
| WNS / WHS global del ensayo | +3,709 / +0,073 ns, estimados de sintesis |
| Excepciones nuevas / ignoradas / waivers | 14 / 0 / 0 |
| TIMING-6 y TIMING-7 | Desaparecen al eliminar relaciones de fase ficticias |
| CDC-10 / LUTAR-1 / puertos sin retardos | Permanecen: 1 / 2 / 13 |

Estos margenes no incluyen colocacion, ruteo, consumidores del guardian ni
la IP PCIe completa. El revisor rechaza once corrupciones de la evidencia,
incluyendo cadenas rotas, margen alterado, excepciones amplias y alertas ocultas.

## Reproduccion y evidencia

Copiar `audit_guard_cdc.tcl`, `scope_guard_cdc.tcl`, `run_guard_cdc.py`,
`review_guard_cdc.py` y el RTL del guardian a una carpeta privada de la VM.
Ejecutar con salidas nuevas bajo `ntrpu-side`:

```bash
python3 -B run_guard_cdc.py NUEVA_BASE --settings VIVADO_SETTINGS --execute-native
python3 -B run_guard_cdc.py NUEVO_ENSAYO --settings VIVADO_SETTINGS --execute-native \
  --checkpoint NUEVA_BASE/reports/guard_synth.dcp
python3 -B review_guard_cdc.py NUEVA_BASE/reports NUEVO_ENSAYO/reports NUEVO_REVIEW.json
```

El runner usa dos CPU, nice 10 y limite de 300 s por proceso propio.
La base `baseline-r1-20261011T052038Z` tardo 78,419 s y el ensayo
`scoped-r1-20261011T052708Z`, 60,816 s. Se conservan tambien dos intentos
fallidos de scripts: `redirect` no disponible y restricciones desde Q.
La suma de los cuatro procesos fue aproximadamente 202 s.

Todos los archivos quedan en `D:/AI/artifacts/ntrpu-axi-bridge-guard-cdc/`,
incluidos DCP, netlist, inventarios, reportes y fuentes de cada intento.
`20261011-guard-cdc-bundle.zip` y `20261011-backup-receipt.json` documentan el
respaldo restaurado y revisado. SHA del DCP inmutable:
`3486657f103ebd8a87cea31d205e983d8b3989e396b69b5d41923db036f42263`.

## Siguiente paso de diseno

Resolver o justificar los dos combinadores asincronos con sus condiciones
reales de cambio, conservando cierre sin reloj y contexto retenido. Cualquier
revision RTL debe repetir la matriz del guardian y los dos casos nativos de
reloj antes de reemplazar el candidato actual. Despues corresponde revisar
CDC/STA con fuentes y consumidores reales, restricciones de placa y ruteo.
El lector DDR/HBM, ACK entre tarjetas y fallo fisico GT/REFCLK siguen pendientes.
