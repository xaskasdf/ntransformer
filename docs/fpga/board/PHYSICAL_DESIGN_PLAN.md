# Plan de cierre FPGA y diseño físico de la tarjeta NTRPU

Fecha: 2026-10-15. Estado: planificación y evidencia parcial de implementación.
Este documento concreta el [roadmap de placa](../BOARD_DESIGN_ROADMAP.md).
La [cola Vivado](VIVADO_QUEUE.md) conserva sus dependencias y ejecuciones;
su [registro de estado](VIVADO_STATUS.md) sigue siendo la referencia operativa.

## Punto de partida y alcance

El cierre físico en Vivado verifica colocación, ruteo y tiempos dentro del
FPGA. La tarjeta requiere además componentes compatibles, esquemático,
PCB, alimentación, refrigeración y ensayos del hardware montado.

La rama experimental `xcvu47p-fsvh2892-2-e` con HBM tiene una mejora local
revisada sobre `ntrpu_axi_bridge_hbm_controlled`:

| Medida | Antes | Después |
| --- | ---: | ---: |
| Endpoints con setup negativo | 2.262 | 1.744 |
| WNS | −0,438 ns | −0,438 ns |
| TNS | −279,876 ns | −168,230 ns |
| Fallos de hold | 0 | 0 |

Se recuperan 518 endpoints sin introducir otros negativos. La revisión
conserva colocación, rutas y reloj. Esto es un ECO sobre un checkpoint;
no acredita resíntesis completa, tráfico PCIe/HBM en hardware ni cierre
de placa. La reapertura independiente de este candidato sigue pendiente
en la evidencia consultada. Persisten además 1.154 endpoints internos sin
restricción y dos entradas sin retardo declarado, cuyo alcance debe revisarse.

Evidencia local: carpeta
`D:/AI/artifacts/ntrpu-hbm-validated-physical-ch0-20261015/`, tanda
`board-validated-physical-ch0-inspect-20261015T192425Z`.
`REVIEW.json` conserva `physical_timing_qualified=false` y
`board_qualified=false`; `BACKUP.json` registra 368 archivos verificados
del archivo nativo. SHA-256 del checkpoint candidato:
`c8cd1f6ab222866d291e3713b70ec2382a5462eb9dd53d2c0f66779240a1914d`.
Este respaldo de tanda no equivale al cierre de la campaña completa.

El objetivo de **512 GiB DDR4** pertenece a otra configuración. Según la
[arquitectura](ARCHITECTURE.md), la primera placa será un prototipo de banco
con cuatro canales de 128 GiB. VU5P/FLVA2104 y VU9P/FLGA2104 son candidatos,
no partes aprobadas. El resultado HBM no califica esa memoria ni ese pinout.

## Orden de trabajo y criterios de avance

| Etapa | Entregable | Condición para cerrar la etapa |
| --- | --- | --- |
| 1. Arquitectura eléctrica | Matriz FPGA/encapsulado/DIMM, PCIe, arranque, potencia y coste | Componentes exactos y configuración de memoria sustentados por evidencia; dependencias V1–V4 resueltas para el candidato |
| 2. Interfaces y pines | Tabla `señal → puerto RTL → banco → bola → net PCB`, IP y XDC reproducibles | Coexistencia de cuatro canales, PCIe, relojes y reset comprobada; V5–V6 revisados |
| 3. Esquemático | Hojas eléctricas, BOM, símbolos y huellas comprobados | ERC revisado, tensiones y secuencias compatibles, correspondencia con el mapa de pines |
| 4. PCB y mecánica | Stackup, colocación, ruteo y conjunto 3D editable | Reglas eléctricas y de fabricación definidas; acceso, alturas y refrigeración comprobados |
| 5. Cierre integrado | Implementación del diseño objetivo y revisión conjunta FPGA/PCB | V7–V8, timing y cobertura revisados, CDC/reset justificados, integridad de señal y alimentación evaluadas |
| 6. Liberación a fabricación | Fuentes editables, Gerber/taladros, BOM y posiciones de montaje | Una misma revisión reconciliada, exportaciones inspeccionadas y capacidad del fabricante confirmada |
| 7. Puesta en marcha | Mediciones y pruebas reproducibles sobre la placa | V9: alimentación, configuración, memoria, PCIe/DMA y cómputo comprobados progresivamente |

Las etapas describen dependencias de cierre. Los borradores de esquemático,
presupuestos y mecánica pueden avanzar en paralelo; cada cambio de componente
o pin debe volver a comprobarse antes de fijar el PCB. AMD recomienda validar
E/S y relojes con información del diseño sintetizado e implementado:
[UG899, planificación de E/S](https://docs.amd.com/r/en-US/ug899-vivado-io-clock-planning/Netlist-I/O-Planning).

Para cerrar timing se exige ausencia de incumplimientos de setup, hold y
ancho de pulso en el alcance declarado, ruteo completo y restricciones
justificadas. Registrar márgenes, esquinas analizadas y señales excluidas;
un bloque aislado aprobado no cierra la integración.

## Primer paquete: componentes, presupuestos y distribución preliminar

Completar este paquete antes de elegir huellas definitivas o comprar componentes:

| Área | Base disponible | Información por cerrar |
| --- | --- | --- |
| FPGA y memoria | Cuatro canales x64 útil más ECC; dos RDIMM de 64 GiB por canal como hipótesis inicial | Parte, velocidad, temperatura, DIMM/RCD, modelo y configuración exactos; ver [calificación de memoria](MEMORY_QUALIFICATION.md) |
| Bancos y PCIe | Objetivo Gen3 x16, arranque JTAG/QSPI | El ejemplo DDR ocupa cuatro bancos; la reserva de tres no está validada. Resolver adaptación de PERST y coexistencia: [pin planning](PIN_PLANNING.md) |
| Potencia | Separar FPGA, memoria, terminaciones, relojes, supervisión y ventilación | Tensiones por rail, corriente continua/pico, eficiencia, secuencia, protecciones y puntos de medida |
| Mecánica | Prototipo de banco con acceso a DIMM y depuración | Contorno, conectores, separación entre módulos, extracción, anclajes, alturas y flujo de aire |
| Coste y montaje | BOM preliminar por bloque | Disponibilidad, cotizaciones, stackup fabricable, montaje BGA e inspección |

Presupuesto por rail: registrar carga, fuente del dato, tensión, corriente
continua/transitoria, eficiencia y margen elegido. Calcular
`P_entrada = suma(P_salida_rail / eficiencia_rail) + cargas_directas`, sin
duplicar consumos. Refinar FPGA con actividad representativa del diseño;
los valores desconocidos permanecen pendientes, no se contabilizan como cero.
Dimensionar disipación y alimentación a partir de esos datos:
[AMD, alimentación y térmica](https://docs.amd.com/r/en-US/ug583-ultrascale-pcb-design/15.-Power-Supplies-and-Thermal-Considerations).

El primer modelo mecánico contendrá volúmenes provisionales: FPGA/disipador,
grupos DIMM ligados a los bancos candidatos, alimentación, conexión PCIe y
depuración accesible. Registrar dimensiones y zonas libres desde planos de
componentes. Convertirlo en conjunto 3D detallado junto con el PCB, comprobando
interferencias, tolerancias y mantenimiento. Aún no hay dimensiones aprobadas.

## Validación y respaldo por etapa

Guardar revisión de fuentes, parte/IP/herramientas, comandos, restricciones,
logs completos, resultados y limitaciones. Cada implementación aceptada debe
reabrirse desde el checkpoint guardado y reproducir los criterios aplicables.
Copiar evidencia localmente, verificar hashes y comprobar su restauración
antes de retirar una copia de trabajo.

En el prototipo, avanzar desde rails y temperaturas a JTAG/configuración,
relojes, un canal DDR y después todos los canales; seguir con enumeración
PCIe/DMA e inferencia comparada. Cada población DIMM necesita su configuración
correspondiente. Las pruebas con dos tarjetas y memoria agregada son una
validación posterior. Aplicar las reglas de memoria, transceptores y PCB de
[AMD UG583](https://docs.amd.com/r/en-US/ug583-ultrascale-pcb-design).

Primera entrega R1 del 2026-10-15: [base eléctrica y matriz de componentes](ELECTRICAL_BASELINE_R1.md),
con dominios de alimentación, nominales, inventario de bolas y pendientes,
y [distribución mecánica preliminar](MECHANICAL_LAYOUT_R1.md), con zonas y
dimensiones documentadas. Corrientes, potencia total y contorno siguen abiertos.
El siguiente cierre requiere el DIMM exacto y la estimación de potencia del
candidato DDR antes de seleccionar reguladores, disipación y dimensiones.
El [traspaso del diagnóstico de potencia](POWER_PROBE_HANDOFF_R1.md) fija
entradas, comando, criterios de revisión y respaldo. Su [ejecución R2](POWER_PROBE_RESULTS_R1.md)
exportó 3,015 W on-chip para el ejemplo de un canal, con confianza baja y
sin actividad simulada; el respaldo está verificado. La estimación representativa
y el presupuesto de placa siguen pendientes, incluidos los supuestos térmicos.
Este registro documenta el orden; no lanza ni reordena tandas Vivado.
