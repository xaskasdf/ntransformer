# Base eléctrica R1: componentes y presupuesto por rail

Fecha: 2026-10-15. Primera entrega del [plan físico](PHYSICAL_DESIGN_PLAN.md).
Avance del mismo día: [revisión de DIMM y entradas de potencia](MEMORY_POWER_REVIEW_R1.md),
con corrientes documentales de la revisión F. El [diagnóstico Vivado ejecutado](POWER_PROBE_RESULTS_R1.md)
aporta una estimación de confianza baja del ejemplo de un canal; el presupuesto
de diseño permanece pendiente.
Alcance: prototipo de banco DDR4 de cuatro canales, objetivo 512 GiB.
Los componentes siguen siendo candidatos; esta revisión no libera compras,
esquemático ni fabricación. El experimento VU47P/HBM tiene otro presupuesto.

## Matriz de componentes

| Bloque | Cantidad del escenario | Candidato o requisito | Estado y dato necesario |
| --- | ---: | --- | --- |
| FPGA | 1 | `xcvu5p-flva2104-2-i`; alternativa VU9P/FLGA2104 | VU5P usado en exploraciones; cerrar recursos, cuatro interfaces, velocidad, temperatura y suministro |
| Memoria base B | 8 módulos | `MTA36ASF8G72PZ-3G2F1`, 64 GiB, 2Rx4; dos por canal | Configuración exacta rechazada por catálogo instalado; faltan SPD completo, modelo, RCD y consumo de la revisión |
| Memoria alternativa A | 4 módulos, en otro perfil | `M393AAK40B41-CTC`, 128 GiB 3DS | Geometría generada; modelo/simulador y módulo sin calificar. Requiere rehacer población y presupuesto |
| Conectores DIMM | 8 para B; 4 para A | RDIMM DDR4, parte por seleccionar | Plano, altura instalada, apertura de pestillos, corriente por contacto, huella y modelo STEP |
| Interfaz host | 1 enlace | PCIe Gen3 x16; conector/adaptador de banco por seleccionar | Orden de lanes, referencia, PERST, pérdidas del canal y relación con alimentación host |
| Configuración | 1 flash prevista y 1 acceso JTAG | QSPI x4 y arranque convencional | Parte, capacidad, tensión, straps y recuperación; Tandem no pertenece al reparto actual |
| Relojes | Cantidad pendiente | Referencias DDR, PCIe y gestión según IP | Frecuencia, jitter, fanout, niveles y secuencia de cada salida |
| Alimentación | Cantidad pendiente | Reguladores, terminaciones y supervisión | Corrientes, tolerancias, rampa, secuencia y respuesta transitoria antes de elegir referencias |
| Supervisión y refrigeración | Cantidad pendiente | Sensores, medida de corriente, ventilación y disipador | Temperatura ambiente, potencia, puntos de medida, fallo de ventilación y acceso de banco |

Fuentes locales: [configuración exacta](MEMORY_CONFIG.json),
[calificación DDR](MEMORY_QUALIFICATION.md),
[alternativa 3DS](MEMORY_3DS_ALTERNATIVE.md) y
[coexistencia de interfaces](INTERFACE_COEXISTENCE.md).
El perfil genérico de 1600 MT/s no certifica el DIMM F1. Los SPD parciales
[B1/B2](MEMORY_SPD_B1_B2.md) son evidencia de otras revisiones y no completan F1.
Precio, disponibilidad y plazo de todos los componentes quedan pendientes de
cotización vigente; no se deducen del catálogo de Vivado.

## Dominios de alimentación

Base FPGA: VU5P provisional **−2I**. Tensiones nominales documentales en
[DS923, condiciones recomendadas](https://docs.amd.com/r/en-US/ds923-virtex-ultrascale-plus/Recommended-Operating-Conditions).
Cambiar grado obliga a revisar valores y límites. Las filas son cargas a
presupuestar: no representan un regulador independiente por fila.

| ID | Dominio/carga | Nominal de trabajo | Corriente continua/pico; potencia |
| --- | --- | --- | --- |
| R01 | VCCINT | 0,85 V | Pendientes |
| R02 | VCCBRAM + VCCINT_IO | 0,85 V; unión obligatoria | Pendientes |
| R03 | VCCAUX + VCCAUX_IO | 1,8 V; unión obligatoria | Pendientes |
| R04 | VCCO de bancos asignados a DDR | 1,2 V propuesto; comprobar estándar/IP | Pendientes por banco |
| R05 | VCCO_0, configuración | 1,8 V propuesto; coordinar QSPI/JTAG | Pendientes |
| R06 | MGTAVCC | 0,9 V | Pendientes por grupo GTY |
| R07 | MGTAVTT | 1,2 V | Pendientes por grupo GTY |
| R08 | MGTAVTTRCAL | 1,2 V | Pendientes; revisar conexión por columna |
| R09 | MGTVCCAUX | 1,8 V | Pendientes |
| R10 | VCCADC | 1,8 V | Pendientes |
| R11 | DDR VDD/VDDQ | 1,2 V, base DDR4 | Referencias parciales F/3200 en la revisión; presupuesto pendiente |
| R12 | DDR VPP | 2,5 V, base DDR4 | Referencias parciales F/3200 en la revisión; presupuesto pendiente |
| R13 | Terminación DDR VTT | 0,6 V nominal, siguiendo VDDQ/2 | Fuente y sumidero pendientes; topología por canal |
| R14 | SPD/RCD y auxiliares del módulo | Según ficha exacta | Pendientes; evitar duplicar consumo incluido en R11/R12 |
| R15 | Relojes, flash, gestión y adaptación de niveles | Según partes seleccionadas | Pendientes por carga |
| R16 | Ventilación y entrada de banco | Sin tensión seleccionada | Pendientes, incluida corriente de arranque |

Los nominales DDR son referencias de arquitectura: AMD documenta VDD de
1,2 V y VPP de 2,5 V en [tarjetas Alveo](https://docs.amd.com/r/2020.2-English/ug1393-vitis-application-acceleration/query),
y VTT de 0,6 V en [KCU1500](https://docs.amd.com/r/en-US/ug1260-kcu1500-data-center/DDR4-Component-Memory).
Sus corrientes y reguladores no se trasladan a ocho RDIMM. VTT requiere
evaluar entrega y absorción de corriente; VREF es una referencia, no una
salida de potencia intercambiable con VTT. Resolver VREF, VBATT y alimentación
de recursos no usados según parte y modos elegidos, incluyendo GTY no activos.

La secuencia base recomendada es VCCINT → VCCBRAM/VCCINT_IO → VCCAUX/VCCAUX_IO
→ VCCO; apagado en sentido inverso. DS923 permite ciertas agrupaciones con
tensiones compatibles. Incorporar también GTY, DIMM y fallos de alimentación
antes de fijar el secuenciador: [DS923, secuencias](https://docs.amd.com/r/en-US/ds923-virtex-ultrascale-plus/Power-On/Off-Power-Supply-Sequencing).
La igualdad de tensión nominal por sí sola no decide la unión de otros rails.

## Presupuesto pendiente de corrientes

### Referencias del diagnóstico ejecutado

Tanda `20261015T205842Z-V8-ddr-power-probe-r2`: 3,015 W on-chip,
1,349 W dinámicos y 1,666 W estáticos, sin SAIF. Se verificaron localmente
los hashes del TAR y de `REVIEW.json` contra su certificado. La revisión
conserva las corrientes XML; se relacionan aquí con los dominios existentes:

| Dominio | Referencia de corriente total del informe (A) | Correspondencia |
| --- | ---: | --- |
| R01 | 0,978074 | Vccint |
| R02 | 0,494535 | Vccint_io + Vccbram; suma de las dos filas XML |
| R03 | 0,880923 | Vccaux + Vccaux_io; suma de las dos filas XML |
| R04 | 0,123650 | Vcco12 agregado; sin desglose por banco |
| R10 | 0,016000 | Vccadc |

Son referencias del ejemplo bajo sus condiciones heredadas, **no corrientes
continuas o de pico aprobadas para diseñar reguladores**. El informe presenta
320 filas de resets inferidos activos en exceso, actividad incompleta y
modelos de potencia ausentes. Los ceros de otros rails no eliminan esas
cargas del presupuesto de la tarjeta. Vcco12 tampoco es la alimentación
externa R11 del DIMM.

Los supuestos térmicos de 25 °C, 250 LFM y disipador/PCB `medium` no describen
un conjunto mecánico aprobado. No extrapolar a cuatro canales, añadir HBM
ni aplicar un margen a 3,015 W para seleccionar la fuente. Faltan actividad
correspondiente tras calibración, condiciones térmicas justificadas, DIMM,
auxiliares y diseño integrado. El [traspaso](POWER_PROBE_HANDOFF_R1.md)
describe las ventanas y las diferencias entre banco y checkpoint.

### Datos necesarios para dimensionar

Para cada R01–R16 registrar: fuente/revisión, cantidad, actividad,
`I_cont_A`, `I_pico_A`, duración del pico, rampa, tolerancia DC, ripple,
`P_carga_W`, regulador, eficiencia y margen. Todos esos valores sin evidencia
permanecen **pendientes**, incluido el total de entrada y calor a evacuar.

En una carga DC: `P_carga_W = V_nom_V × I_cont_A`. Sumar ocho módulos solo
si la ficha o medida es por módulo; un informe de canal con dos módulos
se multiplica por cuatro. Un cambio de 1 W por módulo cambia en 8 W la
carga del escenario B. Esto es sensibilidad aritmética, no consumo estimado.

Construir el árbol real de conversión antes de sumar la entrada. Contar
cargas finales y pérdidas de cada etapa una vez: si VTT sale de DDR VDD,
su demanda se incorpora al regulador padre y no se suma nuevamente como
otra entrada independiente. Tratar pérdidas y corriente absorbida de VTT
con el modelo del regulador. Si la ficha da potencia total del módulo,
no volver a añadir DRAM/RCD ya incluidos.

Orden para completar el presupuesto:

1. Resolver DIMM/revisión y obtener consumo por rail, SPD, RCD y modelo.
2. Completar la estimación FPGA del candidato DDR con recursos, relojes,
   actividad y temperatura representativos; conservar herramienta y entradas.
   El diagnóstico inicial está ejecutado, pero no satisface este criterio.
3. Completar auxiliares, terminaciones y arranque; elegir conversores y fuente
   desde sus límites continuos/transitorios y pérdidas a esa carga.
4. Dimensionar disipador, ventilación y conectores con el presupuesto resultante;
   validar después con medidas del prototipo.

## Comprobación del inventario de bolas

Se releyeron los dos pinouts oficiales conservados en
`build/ntrpu-board-study-20261002/usaall.zip`: 2.104 bolas únicas por parte.
Los siguientes conteos coinciden en ambos candidatos; no prueban consumo
igual ni compatibilidad de sustitución:

| Alimentación | Bolas por encapsulado |
| --- | ---: |
| VCCINT | 78 |
| VCCBRAM / VCCINT_IO | 5 / 9 |
| VCCAUX / VCCAUX_IO | 10 / 18 |
| VCCO_0 / suma VCCO HP | 2 / 102 |
| MGTAVCC / MGTAVTT / MGTAVTTRCAL / MGTVCCAUX | 37 / 66 / 4 / 8 |
| VCCADC / VBATT | 1 / 1 |

Referencia y procedimiento: [inventario de encapsulados](VIVADO_HANDOFF.md).
SHA-256 del ZIP: `b6398572a94213a03961501ce96cbabba1266a0f5c8525c618ef6b62ba490de8`.
Archivos internos:

- `usaall/xcvu5pflva2104pkg.txt`: `d5eba67ef12efa30018f93db3b448421c4ea59136bafbf03d3d740186abf6e93`.
- `usaall/xcvu9pflga2104pkg.txt`: `c7ef35f2bc3286c36d5679c93b08658c95c0956e03c8a86b7a82189937d95f57`.

La [distribución mecánica R1](MECHANICAL_LAYOUT_R1.md) usa esta matriz.
Esta entrega fija dominios, fuentes y pendientes; el presupuesto numérico
no está cerrado mientras falten corrientes y arquitectura de conversión.
