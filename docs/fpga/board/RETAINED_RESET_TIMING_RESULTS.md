# Supervisor retenido: netlist y restricciones específicas

Resultados del 2026-10-06, siguiendo el
[plan de revisión](RETAINED_RESET_TIMING_PLAN.md). Se conserva el DCP
original sin modificaciones y no se añaden waivers.

## Simulación de las celdas sintetizadas

Los 13 escenarios originales pasan con reloj fuente de 5 y 37 ns.
Cada escenario abre exactamente una sesión. Los casos de fallo mantienen
quarantine después de recuperar clocks/calibración; los casos de arranque
con estado pendiente esperan su limpieza. Se usan únicamente el netlist,
el banco de puertos públicos original y las primitivas de simulación.
El manifiesto de compilación no incluye el RTL original del supervisor.

Tandas `20261006T135635Z-…-retained-start-netlist-fast-frozen` y
`20261006T135851Z-…-retained-start-netlist-slow-frozen`, donde `…` es
`V4-ffn-supervised`. Duraciones nativas: 58,801/58,855 s. Archivos SHA-256:

- Rápido: `3bfccec7589e55cd825d512be8f1fb73bf8dbfd2e739427aad0fd05df7eea830`.
- Lento: `53f6ab8512a220ea130a7fb766f6cfc67eeab08e6dbff183eebbf4163aa8e763`.

Es simulación funcional, sin SDF; no prueba metastabilidad ni timing físico.

## Inventario y ensayo de restricciones

`20261006T135426Z-V4-ffn-supervised-retained-start-timing-diagnostic-frozen`
identifica 45 rutas entre clocks: todas terminan en CLR de 15 cadenas
de tres FDCE, con INIT=0 y `ASYNC_REG`. Archivo:
`aa8acac514a3801808388afc90feacff462afbc60996520330ae92e9f06a6b36`.

`ntrpu_retained_start_reset_constraints.tcl` comprueba los nombres exactos,
las tres etapas, clocks y clears comunes, la entrada constante uno y las
conexiones Q→D. Solo después restringe esos 45 CLR. No excluye pares de
clocks ni los consumidores del reset sincronizado.

La tanda `20261006T140040Z-V4-ffn-supervised-retained-start-scoped-timing-frozen`
ensayó esas restricciones en memoria, sin escribir otro DCP. Su revisión
independiente comparó el XDC efectivo: exactamente 45 excepciones nuevas,
con las anteriores conservadas. Las 30 rutas D entre etapas mantienen
idénticos startpoint, endpoint y margen.

| Comprobación | Antes | Después |
| --- | ---: | ---: |
| `TIMING-6` / `TIMING-7` | 6 / 6 | 0 / 0 |
| Endpoints setup / hold | 254 / 254 | 209 / 209 |
| WNS / WHS | 4,092 / 0,077 ns | sin cambios |
| Endpoints internos sin restricción | 0 | 0 |
| Puertos sin retardos de entrada/salida | 11 / 11 | 11 / 11 |
| DRC de síntesis / waivers | 0 / 0 | 0 / 0 |

El archivo de esta prueba tiene SHA-256
`2e38356f41d3107bfe496898eeb84a2838e8b37e421536cc086d4fb32094886b`.
Los diagnósticos iniciales conservan sus avisos originales.

## Síntesis R2 integrada

`20261006T141645Z-V4-ffn-supervised-retained-start-synth-r2-frozen`
ejecutó `ntrpu_board_retained_start_synth_r2.tcl` y exportó un DCP nuevo.
La revisión independiente compara el XDC efectivo completo con el ensayo:
coinciden las restricciones, incluidos los 45 CLR. El RTL, todos los tipos
de celdas y sus INIT permanecen iguales a la síntesis original.

Recursos: **102 LUT / 149 FF**, sin RAM ni DSP. WNS/WHS/WPWS internos:
**4,092 / 0,077 / 2,225 ns**; 209 endpoints setup/hold y cero fallos.
No aparecen `TIMING-6/7`; quedan 22 `TIMING-18` por los 11 puertos de
entrada y 11 de salida sin retardos externos. DRC y waivers: cero.

- Archivo: `7bacc7c7e17f7090fb238d3f2357f6a94a6360831e2dde89041d39067e8bce02`.
- DCP: `7eed7b4c72bed8ee553621569e7e46c6f2c000c409770d7c06fa903f74030a1d`.
- Revisión: `e81c8ef8a5a182efba330183837d127f575adafb9af321b1136e380a17f4416b`.

La tanda `20261006T144238Z-V4-ffn-supervised-retained-start-dcp-r2-frozen`
reabrió ese DCP sin añadir excepciones ni reescribirlo. Su revisión compara
el XDC efectivo completo con la síntesis y las 30 rutas D con el ensayo:
coinciden endpoints, startpoints y márgenes; el mínimo entre etapas sigue
en 4,709 ns. El SHA-256 del DCP de entrada permanece intacto. Los informes
de timing, CDC, metodología, DRC y waivers conservan los resultados anteriores.

Archivo de auditoría:
`592fc184d1c3adef5d7577b98137547f38c7674d2bdb07cf11fec25bd2203992`.
Revisión:
`fe13dd689c9ee6011921b1c5f8569d6fef7864c05cd55589e66fb5196d7f85c8`.

`CDC-26` en el reset externo sigue
sin resolver ni eximir. Los retardos de placa, colocación/ruteo, integración
con consumidores y reset operativo continúan fuera de la aprobación.
