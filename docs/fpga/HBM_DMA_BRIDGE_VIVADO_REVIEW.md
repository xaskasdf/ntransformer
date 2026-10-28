# Puente HBM–DMA: revisión Vivado y CDC de control

Revisión aislada del [RTL conductual](HBM_DMA_BRIDGE_RTL.md) en Vivado 2026.1
build 6511674, para `xcvu47p-fsvh2892-2-e`. Tanda local:
`D:/AI/artifacts/ntrpu-hbm-dma-vivado/20261009T012200862312Z`.
Las fuentes del puente y la cola principal se conservan sin cambios.

## Resultado y alcance

La síntesis OOC termina en 244,635 segundos: **9.213 LUT y 13.216 FF**, sin
BRAM, URAM ni LUTRAM. Los seis FIFO emplean 5.272 FF para sus datos. Esto
confirma elaboración y síntesis, pero **no autoriza integrar este RTL como
un puente físicamente calificado**.

El informe CDC original, sin dispensas, contiene:

| Regla | Cantidad | Interpretación |
| --- | ---: | --- |
| CDC-1 | 2.358 | Cruces sin estructura reconocida; revisar datos, enables y consumidores del FIFO |
| CDC-10 | 2 | Lógica combinacional antes de los sincronizadores de parada |
| CDC-6 | 14 | Doce buses Gray y dos grupos de notificación de envío |
| CDC-15 | 226 | Estructuras controladas por enable; requieren revisar estabilidad y temporización |
| CDC-3 | 5 | Sincronizadores de un bit reconocidos |

Estas cantidades describen comprobaciones del informe, no fallos funcionales
independientes. La simulación previa de 48 escenarios sigue siendo evidencia
conductual; no resuelve por sí sola los avisos físicos.

## Restricciones parciales

`fpga/constraints/ntrpu_hbm_dma_bridge_controls_cdc.xdc` cubre únicamente
los controles cuyo origen está registrado: 12 grupos Gray y cinco grupos
de estado, 43 bits de primera etapa. El revisor comprueba cantidad, reloj,
conexión directa, `ASYNC_REG` y segunda etapa sin fanout adicional.

- Punteros: `set_max_delay -datapath_only` de un período de origen
  (5 ns lector, 4 ns DMA) y `set_bus_skew` de 4 ns por bus.
- Fallo DMA, generaciones de envío y ready: máximo de 4 ns hasta primera
  etapa. Las dos generaciones son notificaciones independientes; no se
  presupone coherencia de un contador binario de dos bits.
- No hay `set_clock_groups`, dispensas CDC ni exclusiones globales.
- Parada, datos FIFO, reset e interfaces externas siguen fuera de este
  conjunto parcial. No debe incorporarse como XDC completo de placa.

La elección sigue [UG903: señales asíncronas](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Constraining-Asynchronous-Signals)
y su [ejemplo de bus Gray](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/set_bus_skew-Example-Two).
Estos límites necesitan comprobarse nuevamente después de placement y routing.
La revisión `controls2` termina en 71,727 s: 17 excepciones aplicadas, 43
endpoints y ninguna excepción ignorada. Los 12 informes de skew son estimaciones
de síntesis, no medidas físicas. Los avisos CDC originales siguen presentes.

## Correcciones que preceden a XDMA

1. **Memoria FIFO:** Vivado emite `Synth 8-7137` y `8-4767`. El proceso de
   escritura incluye sensibilidad al reset asíncrono, impidiendo inferir RAM
   y generando una advertencia de posible discrepancia de simulación.
   Preparar una variante con escritura síncrona separada o FIFO XPM, conservar
   reset coordinado y repetir las pruebas de contrapresión, wrap y eventos.
2. **Parada:** registrar la combinación en el dominio del lector antes de
   cruzar, definir el reloj contractual de `stop` y probar su nueva latencia.
   [UG906](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Combinatorial-Logic)
   explica el riesgo de glitches antes del sincronizador.
3. **Datos y salida:** justificar la estabilidad del bus hasta el consumidor;
   revisar registro de salida y temporización de descriptor/AXIS con XDMA real.
4. **Cierre físico:** integrar presupuestos de entrada/salida, buffers de
   reloj, reset y ubicación; ejecutar implementación y revisar CDC sin
   dispensas automáticas. Después resolver el reloj XDMA 250/125 MHz y unir HBM.

## Evidencia y reproducción

`synth1` conserva fuentes, netlist, DCP e informes originales. La revisión
reabre una copia restaurada de ese DCP en un directorio independiente.
`controls1` conserva un fallo del verificador: contaba un puerto de jerarquía
como controlador adicional. `controls2` filtra controladores primitivos.

Los ejecutores limitan cada trabajo a dos CPU, prioridad `nice 10` y 900 s.
`FILES.json` verifica transferencias; `BACKUP_*.json` identifica manifiestos
del almacén incremental y restauraciones. `OUTCOME.json` contiene el cierre
de la revisión. Ninguna etapa lanza placement, routing ni simulaciones del
principal.

`tools/check_ntrpu_hbm_dma_vivado_review.py SYNTH_STAGE CONTROLS_STAGE` verifica
hashes, identidad del DCP, bindings, cobertura y avisos pendientes. Rechaza
cinco alteraciones de evidencia: binding omitido, reloj intercambiado,
presupuesto cambiado, exclusión global y advertencia crítica ocultada.
Su resultado es `STRUCTURAL_REVIEW_VERIFIED_NOT_QUALIFIED`.

Los relojes de 5/4 ns se añaden después de la síntesis inicial para inspección.
No se proporcionan presupuestos de los 7.663 inputs y 3.516 outputs, ni
`HD.CLK_SRC`. Los márgenes estimados y el DRC OOC sin hallazgos **no constituyen
cierre de timing o calificación de placa**.
