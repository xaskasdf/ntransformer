# Revisión R1 de DIMM y entradas para estimar potencia

Fecha: 2026-10-15. Continúa la [base eléctrica](ELECTRICAL_BASELINE_R1.md).
Resultado: corrientes documentales de DRAM recuperadas y diagnóstico Vivado
preparado. La selección del DIMM y el presupuesto total permanecen abiertos.

Actualización de ejecución: el [diagnóstico posterior](POWER_PROBE_RESULTS_R1.md)
exportó 3,015 W on-chip con confianza baja, sin SAIF y con avisos de reset.
Conserva el alcance genérico; no cierra el presupuesto total ni la selección
del DIMM. Las secciones de preparación siguientes describen el estado previo.

## Decisión sobre memoria

Se conserva `MTA36ASF8G72PZ-3G2F1` como candidato del escenario de ocho
módulos. El addendum Micron Rev. F menciona esa referencia en la página 2
y contiene una tabla para **Die Revision F** en la página 10. Esto amplía
la evidencia eléctrica, sin completar SPD, código de proceso, RCD ni modelo
funcional de la unidad que se montará.

La consulta pública del 2026-10-15, 20:04 UTC, devolvió HTTP 200 con
`Invalid Partnumber` en el API usado por la [página SPD de Micron](https://www.micron.com/products/memory/dram-modules/rdimm/part-catalog/part-detail/spd-data/mta36asf8g72pz-3g2f1).
Su cuerpo coincide con el archivado anteriormente: SHA-256
`74f57811780d78623e1da3586a53fed2f0bd2f507fd185946550f2f3424d7d66`.
Se registra ausencia de datos por esa vía, no inexistencia del producto.

La etiqueta y revisión de módulos físicos siguen sin confirmar. Los
[SPD B1/B2](MEMORY_SPD_B1_B2.md) y la [alternativa Samsung 3DS](MEMORY_3DS_ALTERNATIVE.md)
conservan sus propios alcances; ninguno sustituye automáticamente a F1.
[MEMORY_CONFIG.json](MEMORY_CONFIG.json) continúa sin una configuración exacta
aprobada. Para cerrar V2 se necesitan los datos del módulo y proceso elegidos,
su soporte en MIG y las validaciones de [MEMORY_QUALIFICATION.md](MEMORY_QUALIFICATION.md).

## Corrientes recuperadas y cálculo parcial

Se comprobó visualmente la tabla 7 de la página 10 del
[addendum Micron Rev. F 08/2021](../../../artifacts/ntrpu-board-vivado/20261002T100234Z-V2-generic-geometry/20261002T100234Z-V2-generic-geometry/vendor-inputs/micron-rdimm-f1-addendum.pdf).
SHA-256: `e71fd83003c6857b31fed567a663c361113808817702510694f159c99800d8eb`.
Es una copia de documento del fabricante recuperada de un espejo de distribución.
La columna corresponde a **3200 MT/s**, con alcance general de **0–85 °C**.

Los valores agregan las DRAM de un módulo bajo la condición indicada;
no deben multiplicarse otra vez por 36 chips o por dos ranks.
Se calculó `P_parcial = 1,2 × IDD_A + 2,5 × IPP_A` usando tensiones nominales:

| Condición | IDD (mA) | IPP (mA) | Parcial VDD (W) | Parcial VPP (W) | Suma parcial (W) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACTIVATE/PRECHARGE, IDD0/IPP0 | 1.674 | 90 | 2,0088 | 0,2250 | 2,2338 |
| Active standby, IDD3N/IPP3N | 2.160 | 72 | 2,5920 | 0,1800 | 2,7720 |
| Refresh 1x, IDD5R/IPP5R | 1.908 | 108 | 2,2896 | 0,2700 | 2,5596 |
| Lectura intercalada entre bancos, IDD7/IPP7 | 5.544 | 396 | 6,6528 | 0,9900 | 7,6428 |

En standby todos los ranks están en la condición tabulada. En los otros
tres casos, la nota 1 combina un rank activo y el otro en IDD2P/IPP3N.
Son estados de especificación diferentes: no sumar las filas.

Estas cifras **no son potencia total del DIMM, pico de corriente, TDP ni
consumo de inferencia**. Faltan RCD, SPD/sensor, terminaciones y contribución
completa de E/S. La [ficha de familia Rev. A, página 19](../../../artifacts/ntrpu-board-vivado/20261002T100234Z-V2-generic-geometry/20261002T100234Z-V2-generic-geometry/vendor-inputs/micron-family-asf36c8gx72pz-revA.pdf)
distingue IDD, IPP e IDDQ y no proporciona IDDQ. Esa ficha anterior se usa
para interpretar el alcance, no para reemplazar parámetros F1.

No reducir los valores a la mitad para estimar 1600 MT/s. Tampoco
multiplicar una lectura sostenida por ocho: dos DIMM comparten cada canal,
y su actividad simultánea necesita un modelo de tráfico. Corriente de
arranque, transitorios, tensión máxima, temperatura y opciones de operación
requieren tratamiento adicional. R11/R12 tienen ahora referencias parciales;
el presupuesto definitivo de ambos sigue pendiente.

## Entrada FPGA verificada

Se identificó un checkpoint local enrutado del ejemplo de **un canal DDR
genérico** de 128 GiB, sin cómputo NTRPU ni PCIe integrado:

- Tanda: `20261002T125230Z-V3-ddr-route`.
- Parte/top: `xcvu5p-flva2104-2-i`, `example_top`.
- Recursos: 28.805 LUT, 34.962 registros CLB, 26,5 BRAM, 3 DSP y 154 IOB.
- Archivo: `artifacts/ntrpu-board-vivado/20261002T125230Z-V3-ddr-route/20261002T125230Z-V3-ddr-route/output/route_attempt.dcp`.
- SHA-256: `161d111ce97d471d461328e615c96d32232b6e6c101673de210cb75b282f205c`.

El DCP coincide byte a byte con su miembro del respaldo original. También
se verificó el hash del archivo TAR contra `VERIFIED.json`. Su alcance
genérico y restricciones pendientes se conservan en [PIN_PLANNING.md](PIN_PLANNING.md).
Ni multiplicar sus vatios por cuatro ni añadirle el presupuesto HBM
produciría una estimación válida de la tarjeta integrada.

## Diagnóstico preparado para Vivado

Paquete local: `build/board-electrical-study-20261015-r2/`.
Incluye `power-input-review.json`, la respuesta SPD, imagen de la tabla
revisada y `power_probe_r1.tcl`. El script exige sesión nueva, parte/top y
154 puertos; rechaza una salida existente. Abre el DCP y exporta clocks,
utilización y `report_power` con advisory y XML, sin guardar cambios al DCP.

Antes de ejecutarlo, comprobar el SHA-256 anterior del DCP y el del script
registrado en `power-input-review.json`. En una carpeta nueva de la máquina
Vivado, después de cargar su entorno:

```bash
vivado -mode batch -source /ruta/power_probe_r1.tcl \
  -tclargs /ruta/route_attempt.dcp /ruta/nueva/power-probe
```

Se comprobó la estructura de comandos Tcl, **no se ejecutó Vivado**. El
sondeo inicial usa actividad heredada y valores por defecto; es diagnóstico,
no una calificación de consumo. Conservar versiones, logs, condiciones
ambientales, confidence level, cobertura de actividad y corrientes por rail.
Referencia del comando: [UG835 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_power).

Para estimar operación: obtener actividad de ventanas separadas de reposo,
lectura sostenida y tráfico mixto, con calibración concluida; evaluar arranque
y reinicio aparte. Comprobar que enables y resets representan esas ventanas,
registrar anotación SAIF y definir condiciones térmicas respaldadas. Después
repetir sobre la integración real. El informe distingue potencia interna y
demanda de los rails: [UG907](https://docs.amd.com/r/en-US/ug907-vivado-power-analysis-optimization/Review-Your-Design-Power-Distribution).

La ejecución nativa debe incorporarse desde el hilo principal respetando su
disponibilidad de recursos. Este estudio local no modifica esa cola.

## Evidencia local y respaldo

El paquete `artifacts/ntrpu-board-electrical-20261015-r2/electrical-inputs-r2.zip`
conserva documentos, PDFs fuente, páginas revisadas, consulta SPD, cálculos
y script preparado. `BACKUP.json` registra hash, tamaño y restauración
verificada. El DCP de 26,8 MB permanece en su respaldo original; el paquete
nuevo conserva su ruta y SHA-256 para evitar duplicarlo. La restauración del
paquete documental no constituye una ejecución ni aprobación de Vivado.
