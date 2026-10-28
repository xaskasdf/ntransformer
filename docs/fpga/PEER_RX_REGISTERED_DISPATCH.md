# R4 con cabeceras registradas

Estado: **protocolo validado; ruteo incompleto y 200 MHz no cualificados**.
Simulación/síntesis: `20261005T060047907307Z-registered`, Vivado 2026.1,
`xcvu5p-flva2104-2-i`. Variante aislada de
[R4 con reserva atómica](PEER_RX_RESERVED_DISPATCH.md).
Ruteo desde checkpoint: `20261005T063419059810Z-registered`.

## Cambio y contrato de admisión

El camino anterior leía la cabecera en RAM distribuida, validaba dirección
y longitud, elegía destino y habilitaba el descriptor receptor en un solo
ciclo. La nueva composición `fpga/rtl/ntrpu_registered_dispatch.sv` usa
una copia generada del despachador con un registro de cabecera por destino.
Cada registro conserva los 232 bits del descriptor, su validez y el
resultado de comprobar alineación, longitud y límite de página.

La entrada permanece contada en su FIFO hasta `reserve_take`: estos registros
no añaden capacidad de cola ni consumen bancos. Tras retirar una cabecera,
se invalida su copia y se carga la siguiente en un ciclo posterior. La
recarga suele solaparse con comando, fetch y llenado de la trama anterior;
una cola que estaba vacía incorpora un ciclo de preparación adicional.

**Solo se registra información del descriptor.** La oferta de bancos y
descriptores se consulta en el ciclo de asignación. El handshake real de
comando de R4 sigue siendo la reserva atómica; no se emplea una oferta
registrada que pudiera haber quedado obsoleta. La concesión, una vez
otorgada, conserva las reglas previas frente a stop y fallo.

Se mantienen ocho entradas por destino, dos bancos dedicados por destino,
cuatro bancos compartidos, 16 descriptores receptores por destino y hasta
cuatro escrituras pendientes de respuesta por destino. El pool sigue
teniendo **32 KiB lógicos**; no se aumentó su profundidad.

`tools/prepare_ntrpu_registered_dispatch.py` genera el despachador y el banco
de pruebas a partir de referencias preservadas, junto con diferencias y
hashes SHA-256. Las copias del receptor se generan mediante el preparador
de reserva atómica existente. No se modifica la cola principal.

## Evidencia funcional

Los ocho escenarios anteriores conservan sus trabajos, bytes y estados:
tráfico de 64 B/512 B/4 KiB, agotamiento de bancos, agotamiento de descriptores,
barrera, fallo externo, BRESP con error, stop antes del payload y W antes
de AW. Pasaron **87 publicaciones, 85 transferencias y 138.944 bytes**.
Hay 84 completions OKAY, una SLVERR inyectada y dos trabajos en cuarentena.

El revisor reconstruye 4.719 ciclos y comprueba la transferencia de cada
cabecera desde la FIFO al registro, su estabilidad y su retiro. Revisa
4.978 ciclos de retención acumulados entre los dos destinos, 85 selecciones
y dos reutilizaciones de banco en el mismo flanco. Los ocho controles
negativos previos y tres nuevos —descriptor alterado, validación falsa y
validez retirada— fueron rechazados.

`WORKLOAD_COMPARISON.json` comprueba la identidad de trabajos, metadatos,
payload y estados respecto a la versión anterior. Algunos tiempos cambian
porque las pausas del destino dependen del ciclo. Esta comparación demuestra
conservación funcional; no mide ganancia de ancho de banda.

## Coste y timing

Comparación de síntesis con el mismo reloj de 5 ns:

| Métrica | Reserva anterior | Cabeceras registradas |
| --- | ---: | ---: |
| LUT | 6.131 | 5.990 |
| Registros | 12.548 | 13.012 |
| BRAM equivalentes de 36 Kib | 15 | 15 |
| CARRY8 | 249 | 249 |
| WNS estimado | −0,810 ns | −0,264 ns |
| Extremos con incumplimiento de setup | 5.652 | 526 |

El incremento es de 464 registros, con 141 LUT menos en esta síntesis.
La estimación aún no cumple 200 MHz. Placement y optimización física
terminaron; el ruteo agotó su presupuesto sin producir checkpoint ruteado
ni informe final de timing. **La variante no sustituye a la referencia.**
En síntesis, el camino crítico ya no nace en la lectura de la cabecera:
parte del control de generaciones del asignador. La estimación después
de optimización física señala una cadena desde el estado de respuesta,
pasando por la oferta, selección y validación del comando receptor,
hasta la habilitación de otro descriptor. Ambas son estimaciones previas
al ruteo; no deben confundirse con su resultado final.

| Etapa | Setup global | Hold global | Interpretación |
| --- | ---: | ---: | --- |
| Placement guardado | −0,709 ns | −1,810 ns | Estimación con los presupuestos OOC |
| Optimización física guardada | −0,518 ns | −1,810 ns | Sigue habiendo incumplimiento de setup interno |
| Última estimación intermedia del ruteador | −8,978 ns | −1,810 ns | Sin ruteo final; no permite calcular una frecuencia validada |

El ruteador detectó congestión de nivel 5 y 7.836 pines con restricciones
ajustadas de setup/hold. Logró eliminar los solapamientos en su primera
iteración global, pero abrió otra para mejorar timing. Al detenerse quedaron
32 nodos con solapamientos registrados. Sus diagnósticos señalan, entre
otras, `return_generation[15]`, `[17]`, `[111]` y una red de generación
del banco 5. Los archivos `iter_*_CongestedCLBsAndNets.txt` conservan el detalle.

El contexto físico usa `HD.CLK_SRC=BUFGCE_X0Y60`, periodo de 5 ns,
presupuestos de entrada máximo/mínimo de 1,0/0,2 ns y de salida de
1,0/−0,1 ns, con incertidumbre setup/hold de 0,1/0,05 ns. Son supuestos
de integración OOC, no retardos medidos de placa. No se añaden excepciones
false-path ni multicycle para ocultar caminos.
Los puertos externos carecen de `HD.PARTPIN_LOCS`: Vivado advierte que
sus rutas de interfaz están incompletas y su timing no es preciso.
Los presupuestos anteriores no sustituyen la colocación del módulo padre.
El peor hold del checkpoint optimizado parte de `submit_desc[16]` hacia
la RAM de la FIFO, con 0 ns de recorrido de datos y 1,882 ns de latencia
de captura. **No es una medida de hold exclusivamente interno**: mezcla
una frontera externa idealizada con el reloj físico supuesto.

## Alcance y siguiente integración

Continuación aislada: [preparación de candidato y fronteras de evaluación](PEER_RX_CANDIDATE_DISPATCH.md).
Conserva esta referencia y separa la preparación del descriptor de la toma
real de recursos; registra pruebas adicionales y una comparación con un
montaje de reloj y registros de frontera explícitos.

La prueba funcional utiliza RTL, un único reloj, un productor de patrones
y un destino AXI de comportamiento. No acredita simulación funcional de
netlist ruteada, recuperación en caliente, DDR física, PCIe/P2P, HBM ni DMA
real. La autorización de direcciones y de época sigue aguas arriba.

Antes de otro ruteo largo, evaluar un padre con fronteras registradas y
contexto de reloj explícito, y distinguir sus caminos internos de los
puertos externos sin ubicación. No relajar el periodo ni añadir excepciones
para convertir esta tanda en un PASS. Con ese contexto, revisar la cadena
respuesta/oferta → selección → revalidación → asignación; una etapa adicional
debe registrar el candidato, manteniendo la toma real de recursos en el
handshake y reconsultando disponibilidad al comprometerlo.

Repetir pruebas de datos, errores, barrera y stop, y medir la congestión
antes de integrar CDC, capacity guards, supervisor de época y MIG. La
comparación de ancho de banda contra R4 ordenada sigue pendiente.

## Reproducción y evidencia local

```bash
source /ruta/Vivado/settings64.sh
cd /snapshot/inputs
python3 -B tools/run_ntrpu_registered_dispatch.py /snapshot/output --timeout 900
# Para simulación sin implementación: salida nueva y --simulation-only
cd /ruta/ntransformer
python3 -B tools/check_ntrpu_registered_dispatch.py /snapshot
# Si la implementación física falló después de completar el RTL:
# añadir --allow-incomplete-physical; el fallo físico queda explícito en REVIEW.json.
```

Si el límite se alcanza después de sintetizar, el checkpoint y la simulación
pueden recuperarse sin repetirlos. La reanudación comprueba que no cambió
ninguna fuente lógica y guarda checkpoints tras placement y optimización:

```bash
python3 -B tools/run_ntrpu_registered_dispatch.py /retry/output \
  --resume-synth /snapshot/output/suite --threads 4 --timeout 900
# Desde una optimización física ya guardada, ejecutar solo el ruteo:
python3 -B tools/run_ntrpu_registered_dispatch.py /route/output \
  --resume-synth /retry/output/suite --resume-stage physopt \
  --threads 8 --timeout 1800
```

Runner y revisor requieren salidas nuevas. El runner limita afinidad,
prioridad y duración, y archiva también los intentos que alcanzan el límite.
Esta evaluación conservó dos tandas de 900 s interrumpidas durante el ruteo;
la segunda dejó checkpoints de placement y optimización física. La última
usó únicamente el checkpoint optimizado, con ocho CPU, y se detuvo a los
1.800,007 s. Los tres límites sumaron unos 60 minutos de ejecución acotada.
Los registros conservan afinidad, prioridad, fuentes y motivos de terminación.
Evidencia local de esta tanda:
`D:\AI\artifacts\ntrpu-registered-dispatch\20261005T055807149734Z`.
Incluye los intentos, fuentes, revisores, informes, checkpoints, manifiestos
y `20261005-registered-dispatch-handoff.zip`, verificado mediante SHA-256,
CRC y hashes por archivo. La referencia previa y la cola principal se preservan.
