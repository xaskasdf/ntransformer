# Síntesis aislada del receptor multibuffer

Tanda del 2026-10-04. Ambas síntesis terminaron y ambas netlists pasaron la
simulación funcional. La variante con ocho reservas incumple el objetivo
interno de setup de 5 ns. Se mide el
[prototipo funcional](PEER_RX_MULTIBUFFER_RTL.md) sin modificar su RTL ni
integrarlo en la cola del diseño principal. La comparación utiliza dos
destinos, dos bancos propios por destino, reserva de cuatro u ocho bancos,
4 KiB por banco, 16 descriptores por destino y cuatro escrituras AXI
pendientes por puerto.

## Alcance de la medición

Vivado 2026.1, referencia `xcvu5p-flva2104-2-i`, síntesis fuera de contexto
y un reloj de 5 ns. No hay excepciones de timing ni delays externos.
El dispositivo sirve para comparar estas variantes; el resultado no
califica una tarjeta HBM, un endpoint PCIe ni la conexión entre NTRPU.

Se ejecutan perfiles secuenciales con dos CPU permitidas, prioridad
`nice 10` y dos threads de Vivado, desde copias independientes. No se
modifican los procesos, fuentes o restricciones del agente principal.

## Recursos y timing estimado

| Medición | Reserva 4: 8 bancos totales | Reserva 8: 12 bancos totales |
| --- | ---: | ---: |
| Payload útil | 32 KiB | 48 KiB |
| LUT / FF | 6883 / 9408 | 9314 / 9691 |
| RAMB36E2 / RAMB18E2 | 56 / 8 | 84 / 12 |
| Tiles BRAM equivalentes reportados | 60 | 90 |
| Fracción de los 1024 tiles del dispositivo | 5,86 % | 8,79 % |
| LUTRAM / URAM / DSP | 0 / 0 / 0 | 0 / 0 / 0 |
| WNS de setup, objetivo 5 ns | +0,365 ns | **−0,653 ns** |
| TNS / endpoints de setup incumplidos | 0 / 0 | −1816,166 ns / 5146 |
| Slack mínimo de hold estimado | +0,075 ns | +0,077 ns |

Aumentar la reserva añade 50 % de payload y BRAM, 35,3 % de LUT y 3,0 %
de FF. Los porcentajes de dispositivo no representan presupuesto libre
del diseño integrado. No hay latches, cajas negras, lazos combinacionales
ni endpoints internos sin reloj o sin restricción.

**Reserva ocho no está aprobada para 200 MHz.** Reserva cuatro alcanza la
restricción interna estimada, pero todavía no prueba cierre físico. No
se cambiaron el reloj ni las excepciones para ocultar el incumplimiento.

## Avisos y límites de los informes

La revisión conserva estos avisos, sin waivers:

- `SYNTH-6`: 64/96 primitivas RAM sin registro de salida adicional absorbido
  en el bloque. Evaluar ese pipeline y su latencia en una variante futura.
- `TIMING-18`: faltan delays externos. `check_timing` enumera 673 entradas
  y 1332/1336 salidas sin delay; el resumen de metodología muestra 1000
  incidencias por perfil. No se toma ese resumen como cobertura completa.
- `Timing 38-242`: falta `HD.CLK_SRC`; el reloj aún no tiene fuente física
  del top integrado para estimar su inserción y skew.
- `DRC 23-814`: los DRC de conectividad tienen alcance reducido por ser
  OOC. El ruledeck ejecutado informa cero hallazgos; no califica la placa.
- `Synth 8-3936`: recorte de un bit interno no conectado de `available`.
  `Synth 8-3917`: salidas AXI constantes intencionales, con 100 mensajes
  registrados. La simulación posterior comprueba los atributos públicos.
- `Netlist 29-101`: muchas primitivas en el top dificultan floorplanning.

No hubo errores ni avisos críticos de herramienta. La ruta peor tiene
26/28 niveles lógicos y 4,617/5,635 ns de retardo estimado respectivamente;
la parte denominada `route` en ese informe **no es ruteo realizado**.

## Geometría de memoria y siguiente optimización

Ambos perfiles infieren siete RAMB36E2 y una RAMB18E2 por banco lógico.
Ocho/doce bancos reservan 270/405 KiB físicos brutos, incluyendo bits de
paridad, para 32/48 KiB útiles: aproximadamente 11,85 % de aprovechamiento. No es
ocupación media de tráfico ni una medición de consumo eléctrico.

La causa geométrica es el ancho de 512 bits y la escasa profundidad de
cada array independiente. AMD documenta BRAM síncrona de hasta 512×72
bits en modo de una lectura y una escritura.
[UG573, geometría BRAM](https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources/Block-RAM-Summary).

Comparar posteriormente bancos lógicos dentro de arrays más profundos:
una opción es replicar el pool por lector y difundir cada escritura a
las copias. Puede reducir fragmentación manteniendo lecturas simultáneas,
pero duplica almacenamiento lógico y añade distribución de direcciones
y datos. Otra opción agrupa bancos por destino, con acceso explícitamente
arbitrado a la reserva. Son propuestas sin síntesis ni validación aún.

La ruta crítica observada pasa desde el puntero de envío hasta el
contador de generación del asignador. Antes de crecer, estudiar la
selección y actualización de préstamos y el coste de reutilización en
el mismo ciclo; conservar la detección de devoluciones obsoletas y las
reglas de generación. No reducir esas garantías para mejorar timing.

## Validación posterior a síntesis

Se simuló la netlist de ambos perfiles en XSIM, semilla 1, con el banco
existente adaptado a señales públicas y espera inicial de GSR de 200 ns.
La comparación conserva datos, orden AW/W/B/CPL, límites observables,
pausas, fallos y drenaje. Los préstamos internos y el pipeline conservan
su cobertura en RTL; no se reconstruyen desde nombres optimizados.

| Resultado | Reserva 4 | Reserva 8 |
| --- | ---: | ---: |
| Transferencias / bytes comparados | 81 / 80.832 | 85 / 81.088 |
| Ciclos de banco | 3869 | 4046 |
| Ciclos W simultáneos en ambos puertos | 253 | 252 |
| Máximo de escrituras pendientes por puerto | 4 / 4 | 4 / 4 |
| B con error / B de ID ajeno | 1 / 1 | 1 / 1 |
| Tiempo total de herramienta | 65,348 s | 77,689 s |

El verificador público rechazó un beat alterado y una finalización con
estado incorrecto en pruebas de control. El banco usa un reloj de 10 ns
y no hay SDF: **el PASS funcional no aprueba el timing ni certifica una
frecuencia física**. Los ciclos tampoco son una medición de caudal PCIe
o HBM. Faltan placement, routing, timing externo y la integración del
contrato de transporte con épocas, ventanas, framing y recuperación.

XSIM registró tres avisos por perfil: redefinición de `glbl`, aplicación
del generic `CHANNELS` a ese módulo sin tal parámetro, y presencia de
`LIBRARY_PATH`. Las dos definiciones de `glbl` coinciden al normalizar
espacios. Los perfiles elaborados y las trazas confirman 81/85 trabajos,
incluyendo la diferente reserva. Se conservan los avisos y logs de
compilación; no hubo errores de compilación, simulación o replay.

## Decisión para el siguiente diseño

Mantener reserva cuatro como referencia funcional y de coste. Primero
comparar una organización de RAM más profunda y una ruta de asignación
más corta, conservando generaciones y créditos; después repetir estas
pruebas y medir placement/routing con fronteras temporales definidas.
Reservar más bancos requiere beneficio medido de caudal o tolerancia a
pausas. Esta tanda no justifica agotar toda la memoria ni aprueba reserva
ocho a 200 MHz.

La [variante compacta posterior](PEER_RX_MULTIBUFFER_COMPACT.md) implementa
arrays más profundos y un asignador paralelo en módulos separados. Su
regresión incluye comparaciones por cambio y conserva esta tanda como
referencia de recursos y comportamiento.

## Evidencia y reproducción

Ejecutar desde una copia de fuentes, con el entorno Vivado cargado:

```bash
python3 -B tools/run_ntrpu_multibuffer_vivado.py /ruta/nueva/sintesis
python3 -B tools/run_ntrpu_multibuffer_vivado.py /ruta/nueva/netlist \
  --netlist-input /ruta/nueva/sintesis
```

Cada carpeta debe ser nueva. El límite actual es 900 s por perfil y se
termina únicamente el grupo de procesos creado por el ejecutor. Se
conservan fuentes y hashes, comandos, informes, DCP, netlist, logs y un ZIP
verificado. El primer intento agotó su límite anterior de 420 s tras
terminar síntesis; sus diagnósticos se conservan como intento incompleto.
La copia del ejecutor usada para estas netlists conserva el límite previo
de 420 s; ambas terminaron dentro de él.

La síntesis final tardó 444,821 s y 745,716 s por perfil, incluyendo
exportación. Sus entradas coincidieron por hash antes y después. El ZIP
descargado pasó SHA-256, CRC y todos los hashes del manifiesto; los informes
de cuatro reservas descargados anticipadamente coinciden con la entrega
final.

- [Revisión de síntesis](../../artifacts/ntrpu-multibuffer-vivado/20261004T055353350113Z-multibuffer-synth-r2/REVIEW.json).
- [Informes y netlists](../../artifacts/ntrpu-multibuffer-vivado/20261004T055353350113Z-multibuffer-synth-r2/output.zip).
- [Control del verificador público](../../artifacts/ntrpu-multibuffer-vivado/public-pin-checker-review.json).
- [Revisión de ambas netlists](../../artifacts/ntrpu-multibuffer-vivado/20261004T055118702171Z-multibuffer-netlist/REVIEW.json).
- [Evidencia completa de simulación](../../artifacts/ntrpu-multibuffer-vivado/20261004T055118702171Z-multibuffer-netlist/output.zip).
- [Respaldo consolidado local](../../artifacts/ntrpu-multibuffer-vivado/20261004-vivado-handoff.zip).

Ambos ZIP finales se descargaron y verificaron mediante SHA-256, CRC y
manifiesto de todos sus miembros. Las netlists simuladas coinciden byte a
byte con las exportadas por síntesis. El respaldo consolidado conserva
los tres intentos, entradas exactas, resultados, herramientas y documentos;
incluye su manifiesto SHA-256 y tiene verificación externa. No incluye
licencias ni instaladores. El RTL del prototipo, el asignador y el receptor
original permanecen sin cambios.
