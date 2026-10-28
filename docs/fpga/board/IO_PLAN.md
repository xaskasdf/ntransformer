# Plan preliminar de bancos, PCIe y arranque

Estudio local del 2026-10-02 para el **prototipo de banco** de la [arquitectura NTRPU](ARCHITECTURE.md). Se han revisado los mapas oficiales de encapsulado; todavía faltan generación de IP, asignación de bolas, DRC e implementación en Vivado. Las reservas siguientes permiten preparar el esquemático, pero no liberarlo para fabricación.

## Reservas de cuatro interfaces DDR4

Cada fila reserva tres bancos HP completos para un canal x72, sin compartir bancos entre controladores. Es una hipótesis de planificación; MIG debe confirmar el reparto de datos, dirección, control y relojes para los DIMM exactos.

| Canal | XCVU5P / FLVA2104 | Zona física | XCVU9P / FLGA2104 | Zona física |
| --- | --- | --- | --- | --- |
| DDR0 | 44, 45, 46 | Izquierda, inferior | 40, 41, 42 | Izquierda, inferior |
| DDR1 | 50, 51, 52 | Izquierda, superior | 45, 46, 47 | Izquierda, central |
| DDR2 | 64, 65, 66 | Derecha, inferior | 64, 65, 66 | Derecha, central |
| DDR3 | 70, 71, 72 | Derecha, superior | 70, 71, 72 | Derecha, superior |
| Sin asignar | 47, 53, 67, 73 | Uno por grupo | 43, 48, 67, 73 | Uno por grupo |

Las zonas describen el dibujo, **no identificadores `SLR0/SLR1/...` de Vivado**. Transcripción visual de [UG575 v1.22](https://docs.amd.com/api/khub/documents/GnuVZDIoTcrlIuu8Aok5XQ/content), figura 1-111, página 152, y figura 1-119, página 160. El [manifiesto JSON](io_plan.json) registra grupos completos y huella SHA-256 del PDF consultado.

Como filtro inicial, cada interfaz debe permanecer en una columna y un SLR, sin saltar bancos. La pertenencia a un grupo correcto no comprueba sus byte lanes ni la colocación de dirección/control. [PG150: reglas de pines DDR4](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-Pin-Rules?contentId=vIJySmS_aQOK66n82tWyUg).

Los bancos sin asignar son margen de planificación: conservarlos hasta cerrar MIG y relojes. Después podrán alojar supervisión y señales auxiliares con tensiones compatibles; no tratarlos aún como conectores de expansión disponibles.

## PCIe Gen3 x16

| Encapsulado candidato | Bloque PCIe | Quad del lane 0 documentado | Reserva provisional de quads |
| --- | --- | --- | --- |
| VU5P / FLVA2104 | `PCIE4_X1Y0` | `GTY_Quad_227` | 224, 225, 226, 227 |
| VU9P / FLGA2104 | `PCIE4_X1Y2` | `GTY_Quad_227` | 224, 225, 226, 227 |

Son las selecciones x16 de esos encapsulados en [PG213, tablas 3 y 5](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Virtex-UltraScale-Devices-Available-GT-Quads). El quad 227 identifica el inicio elegido por la IP; no contiene por sí solo dieciséis lanes. Exportar el orden completo TX/RX y sus bolas desde la IP generada antes de conectar el símbolo ECAD. La reserva 224–227 se comprobará con ese resultado.

Mantener separado el reloj de referencia PCIe de las entradas de reloj DDR. Elegir conector/adaptador de banco y trayecto de señales después de verificar esa asignación; todavía no se ha seleccionado un cable PCIe ni un retimer.

## Banco 65: decisión de arranque

AMD indica que una IP de memoria no debe ocupar el banco 65 cuando se usa Tandem. DDR2 lo necesita en este reparto. Dentro del grupo 64–67, cualquier terna consecutiva incluye 65; quitarlo deja este plan de bancos exclusivos con espacio para tres canales. Es una limitación de **esta propuesta**, no una demostración de imposibilidad para cualquier arquitectura. [PG150: Tandem y memoria](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-Pin-Rules?contentId=vIJySmS_aQOK66n82tWyUg).

Propuesta para el banco de pruebas:

- Configuración inicial por JTAG y posterior arranque convencional desde QSPI x4.
- Reservar el banco dedicado 0 para configuración; usar el reloj interno de configuración. No reservar `EMCCLK`, datos SPI x8 ni otras señales de configuración en 65.
- Mantener Tandem desactivado al generar la IP. Si se vuelve requisito, rehacer el reparto o estudiar otro encapsulado antes del PCB.

UG570 sitúa JTAG y los cuatro datos SPI en el banco 0, mientras `EMCCLK` y datos adicionales ocupan 65. Comprobar flash, tensión, straps y resistencias con el número de parte definitivo. [Definiciones de configuración](https://docs.amd.com/r/en-US/ug570-ultrascale-configuration/Configuration-Pin-Definitions).

El arranque convencional exige resolver la enumeración PCIe: preparar y probar una secuencia compatible de alimentación, configuración, reloj host y reset antes de la enumeración. No asumir que reprogramar por JTAG hará reaparecer el endpoint automáticamente. Para el prototipo se estudiará configurar la placa antes de iniciar/reiniciar el host; el comportamiento de `PERST#` y del reloj debe quedar medido.

## Orden de puesta en marcha propuesto

| Etapa | Evidencia necesaria para avanzar |
| --- | --- |
| Alimentación y supervisión | Rails, secuencia, consumo y temperaturas dentro del presupuesto aprobado; puntos de medida accesibles |
| JTAG y configuración | Identificación del dispositivo, carga mínima, estado de configuración y recuperación repetibles |
| DDR0, un módulo | SPD identificado, IP correspondiente, calibración y patrones de lectura/escritura reproducibles |
| DDR0, dos módulos | IP regenerada para la población exacta; acceso al segundo slot, ranks y direcciones altas, sin alias |
| Cuatro canales | Calibración de cada canal, tráfico simultáneo, contadores de error y límites térmicos registrados |
| PCIe y DMA | Enumeración repetible, lectura de registros, transferencias con integridad y recuperación tras reset |
| Cómputo integrado | Resultados numéricos de modelos pequeños y mediciones de ancho de banda sostenido |

La población de un solo módulo no equivale automáticamente a la configuración MIG de dos slots. Conservar imágenes y parámetros distintos para cada prueba. Esta tabla define criterios futuros; ninguna etapa física está realizada.

## Comprobación local y siguiente entrega

[check_ntrpu_io_plan.py](../../../tools/check_ntrpu_io_plan.py) comprueba coherencia contra el inventario transcrito: bancos presentes, continuidad, columna/SLR, exclusividad, lista libre, configuración y selección PCIe. Solo lee JSON e imprime resultados; no invoca Vivado ni produce XDC.

```bash
python tools/check_ntrpu_io_plan.py
python tools/check_ntrpu_io_plan.py ruta/a/otro_plan.json
```

Salida 0 significa que pasan esos filtros documentales; 1 indica un conflicto y 2 un archivo inválido. La salida conserva siempre `preliminary_not_vivado_validated`. No valida capacidad DIMM, tensiones, byte lanes, relojes, timing ni coexistencia real de las IP.

Comprobación local de esta revisión: los dos perfiles pasan; diez modificaciones inválidas se rechazaron (banco no conectado, salto, cruce de SLR, cruce de columna, solapamiento, Tandem, EMCCLK, bloque PCIe, lista libre e ID de canal). Se verificaron los códigos de salida 0/1/2 y ocho enlaces locales entre este documento y la arquitectura.

La próxima entrega de ingeniería es un proyecto Vivado separado: parte completa, un MIG con la memoria exacta, ampliación a cuatro MIG y PCIe, y exportación de `puerto → banco → bola → net ECAD`. Guardar configuración de IP, XDC e informes de E/S/DRC/relojes/timing. Con esos datos podremos fijar símbolos, conectores, alimentación y colocación del prototipo.

Ya está preparado el [paquete de inventario y entrega a Vivado/ECAD](VIVADO_HANDOFF.md), con contraste de 2.104 bolas por candidato, SLR del archivo oficial, hojas de conexiones y script batch. Incluye la revisión pendiente del reset PCIe en AM17/AP17, dentro del banco 65 de DDR2.
