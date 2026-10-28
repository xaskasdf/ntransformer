# Correspondencia de las ventanas SAIF R1 con el DCP archivado

> Estado de cierre (2026-10-16 16:13:49 UTC): la fase Vivado se detuvo antes
> de lanzar la captura porque el controlador de token requiere inspeccionar
> su dependencia. El análisis de puertos que sigue es offline y no prueba
> asociación SAIF/DCP. Este estado reemplaza el estado de "encolada" indicado
> en la primera nota histórica. Véase el [traspaso por pérdida de acceso](VIVADO_ACCESS_LOSS_HANDOFF_20261016.md).

## Revisión offline de estados de puerto

El parser local registró 154 puertos escalares en el stub del DCP y 577.280
señales por SAIF. Para cada ventana, los 3 puertos de entrada y 43 de salida
no tienen campos `TX`, `TZ` ni `TB` no nulos; los 108 `inout` tienen `TZ` no
nulo y no tienen `TX` ni `TB`. En el SAIF PRBS0, 75.334 señales tienen `TX`
y 26.164 tienen `TZ`; en PRBS1 son 74.844 y 26.164. El análisis solo describe
estados en archivos archivados: todavía no vincula actividad a redes del DCP,
no califica el significado eléctrico de esos campos y no estima potencia.
Evidencia: `ddr-saif-port-states-r1-review.json`.

Fecha: 2026-10-16. **Encolada; todavía sin ejecución Vivado ni nueva
estimación de consumo.** Continúa la [captura R1](POWER_ACTIVITY_RESULTS_R1.md).

## Turno y entradas

Coordinador `ddr-saif-mapping-vivado-r1`, PID 28148, después de
`token-guard-vivado-r1` revisado y respaldado. Conserva las 86 entradas
previas y el turno de [DDR R2](POWER_ACTIVITY_R2.md). Máximo 1.200 segundos,
CPU 0–1, sin reintentos; revisión en CPU 2–3 y respaldo de cada resultado,
incluidos los diagnósticos.

Usa exclusivamente las dos ventanas PRBS de R1, 20,48 µs cada una, y el
DCP `20261002T125230Z-V3-ddr-route/output/route_attempt.dcp`. Los tres hashes
se comprobaron en D:/NFS. No vuelve a simular ni incorpora una captura
R2 futura como si ya existiera. El paquete fuente de veinte archivos está
restaurado; SHA-256
`b0c5fa60f854aa7ce04e87fc36051062b3e4164dacc222cf56d767452533e862`.

## Operación y evidencia esperada

Para cada ventana abre el DCP de nuevo y comprueba `example_top`, VU5P y
154 puertos. Exporta inventarios de redes, celdas y puertos; relojes,
estado del ruteo y actividad heredada. Limpia la actividad, importa un
único SAIF con prefijo `ntrpu_ddr_example_activity_tb/dut` y conserva el
informe de asociación, redes no asociadas y actividad final.

El [comando `read_saif`](https://docs.amd.com/r/2025.2-English/ug835-vivado-tcl-commands/read_saif)
permite retirar ese prefijo y exportar nombres no asociados.
[`report_switching_activity`](https://docs.amd.com/r/2021.2-English/ug835-vivado-tcl-commands/report_switching_activity)
registra la actividad de objetos explícitos. Las apariciones de etiquetas
de origen no equivalen a un porcentaje de cobertura: se conservarán los
informes nativos para revisar su formato y denominadores.

## Controles y límites

Catorce alteraciones fueron rechazadas en Windows/Linux y el arnés Tcl
confirmó aperturas independientes, limpieza e importación por ventana.
Son sustitutos de herramientas; no prueban que Vivado haya asociado redes.
El respaldo de controles está restaurado: SHA-256
`4e5ada648114c453eba20237a100f9dfda3ab5572b3f4d7d77fe6176edd4db5e`.

No ejecuta `report_power`. Una exportación correcta deja pendientes la
interpretación de asociaciones, X/Z, equivalencia de fuentes y bibliotecas,
representatividad y condiciones térmicas. La ocupación de puertos o sus
nombres coincidentes no acredita funcionamiento eléctrico del DIMM.
