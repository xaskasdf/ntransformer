# Auditoría CDC y observación del arranque HBM

Continuación: [implementación física y arranque nativo completado](PEER_RX_HBM_PHYSICAL.md).

## Resultado y alcance

Esta etapa continúa las [21 pruebas funcionales entre relojes](PEER_RX_HBM_CDC.md)
sin modificar el RTL acreditado, `AGENTS.md` ni la cola del agente principal.
Sintetiza la composición asíncrona sobre `xcvu47p-fsvh2892-2-e`, revisa sus
restricciones y observa el modelo HBM ya compilado. **No acredita todavía
colocación, ruteo, timing físico, tráfico HBM nativo ni PCIe entre tarjetas.**

## Cruces y restricciones

La síntesis `20261006T221147734848Z-synth` termina en 279,978 s. Vivado informa
**31 cruces internos informativos**: 19 sincronizadores de un bit y 12 cruces
de reset sincronizado. No hay advertencias CDC ni hallazgos críticos; el
reporte sin exenciones conserva esos mismos resultados. Los 21 registros
del supervisor mantienen `ASYNC_REG`.

La inspección distingue ese reconocimiento estructural del análisis
temporal: tres estados de `memory_clk` hacia `transport_clk` seguían
analizándose con una relación de fase inexistente. La restricción
`fpga/constraints/ntrpu_hbm_startup_cdc.xdc` aplica un límite de propagación
de un período receptor, **5 ns**, exclusivamente a las primeras etapas:

- Dos fallos de adaptador y la liberación de reset de memoria: tres pines D.
- Las dos entradas de disponibilidad HBM: dos pines D adicionales.

No se crean excepciones globales entre relojes. Los tramos posteriores
de los sincronizadores y las rutas funcionales de liberación de reset
conservan sus comprobaciones. Los convertidores mantienen sus restricciones
AMD. El driver verifica los cinco destinos, sus atributos y sus relojes.

La tanda `20261006T222446281217Z-constraints`, 82,696 s, confirma que esas
restricciones se aplican y que **todos los márgenes dentro de cada dominio
permanecen idénticos**. Se conserva el diagnóstico del primer intento:
Vivado rechazó control Tcl dentro de XDC y un `set_max_delay` sin origen.
Ese intento no se considera una validación aceptada.

Los dos estados externos se modelan con un reloj APB virtual de 100 MHz
y retardos de entrada de cero para esta auditoría. Es una hipótesis de
frontera, no una restricción de placa. Se inspeccionan sus rutas hasta los
sincronizadores por separado; no forman parte de los 31 cruces internos.

## Recursos y timing pendiente

| Medida | Resultado de síntesis |
| --- | ---: |
| LUT totales / LUTRAM incluidas | 12.109 / 1.160 |
| Registros | 25.621 |
| RAMB36 / RAMB18 | 14 / 2: 15 BRAM equivalentes |
| DSP / URAM / latches / cajas negras | 0 / 0 / 0 / 0 |
| Transporte, 5 ns: setup / hold | +1,471 / −0,010 ns |
| Memoria, 2,222 ns: setup / hold | −0,200 / −0,036 ns |

No hay violaciones DRC ni extremos internos sin reloj o restricción.
Quedan 1.073 entradas y 1.150 salidas sin retardos externos: la integración
de placa deberá definirlos. Los márgenes son estimaciones anteriores a
colocación; los valores negativos impiden afirmar cierre temporal.

La peor ruta de memoria va de `startup/memory_release_reg[2]` a
`memory[0].adapter/guard/w_done_reg[0]`, atraviesa nueve niveles lógicos y
estima 2,302 ns de datos. Incluye la propagación de disponibilidad por los
convertidores y la cola de escritura. Debe conservarse en el análisis;
no es una excepción CDC. El siguiente paso físico necesita distribución
global de ambos relojes, colocación, ruteo y revisión de esta ruta y del hold.

Esta medición cambia de parte y restricciones respecto a la síntesis
anterior; no permite atribuir toda la diferencia de recursos o margen
únicamente al convertidor de reloj.

## Observación del modelo nativo

El ejecutable y las tablas del modelo se copian y verifican por SHA-256
antes de cada observación. No se recompila la IP ni se fuerza ninguna
señal para omitir la inicialización. Se registra tiempo simulado, tiempo
real, resets y estados cada microsegundo simulado, mediante
[`current_time`](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/current_time)
y [`get_value`](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_value).

La primera observación llegó a **18 µs simulados en 265,376 s reales**.
Ambos resets estaban liberados desde la muestra de 5 µs, mientras
`apb_complete_0/1` seguían a cero. El primer microsegundo requirió unos
74 s; los incrementos posteriores variaron con la carga. No se debe usar
esa tasa como una predicción de la duración total de inicialización.

XSIM conserva la pareja generate/instancia como un solo componente:
`/tb_ntrpu_hbm_boot/dut/memory/inst/TWO_STACK.u_hbm_top`.
Usar dos componentes separados hacía que los sondeos internos devolvieran
`UNAVAILABLE`; eso no significaba que el hardware estuviera detenido.
El sondeo ahora comprueba la visibilidad antes de ejecutar y se detiene
rápidamente si faltan esos objetos.

La observación APB distingue el avance de `xpm_addra`, las direcciones de
configuración y los estados de polling. El muestreo es diagnóstico: puede
omitir handshakes entre muestras y no mide ancho de banda. Los datos APB
fuera de una transferencia válida pueden ser indeterminados.

La tanda con rutas corregidas, `20261006T222532749262Z-apb`, alcanza
**21 µs simulados en 264,433 s reales** y observa 26 señales internas.
Ambos stacks avanzan desde la dirección de ROM `0x008` a los 5 µs hasta
`0x194` a los 21 µs, con distintas direcciones de configuración APB.
No se observan `valid_window_fail` ni `PSLVERR` en esas muestras.
`apb_complete_0/1` siguen a cero: hay avance de configuración demostrado,
pero **no una inicialización terminada**. La próxima tanda nativa debe
conservar estos hitos y cubrir el fin de la secuencia antes de habilitar
escrituras de transporte; estos minutos son coste de simulación, no tiempo
de arranque medido en una FPGA física.

## Reproducción y evidencia

```bash
python3 -B tools/run_ntrpu_hbm_cdc_audit.py /sintesis/output \
  --simulation /cdc-aceptado/output --review /cdc-aceptado/REVIEW.json
python3 -B tools/check_ntrpu_hbm_audit.py /sintesis/output --review /sintesis/REVIEW.json
python3 -B tools/run_ntrpu_hbm_cdc_constraints.py /restricciones/output --prior /sintesis/output
python3 -B tools/check_ntrpu_hbm_audit.py /restricciones/output \
  --prior /sintesis/output --review /restricciones/REVIEW.json
python3 -B tools/run_ntrpu_hbm_boot_probe.py /observacion/output --apb \
  --donor /simulacion-precompilada/xsim --manifest /SNAPSHOT_INPUTS.json
```

Los procesos propios usan dos CPU y prioridad baja; la síntesis tiene un
límite de 900 s y las observaciones/revisiones, de 300 s. Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-hbm-audit\20261006T220852511934Z`.
Se conserva respaldo incremental por contenido en
`D:\AI\artifacts\ntrpu-content-store`, sin ZIP de transferencia retenidos.
