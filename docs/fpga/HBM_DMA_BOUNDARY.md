# Frontera registrada del puente HBM–DMA

Experimento aislado sobre la [revisión 2](HBM_DMA_BRIDGE_REV2.md), después de
la [primera implementación física](HBM_DMA_BRIDGE_REV2_PHYSICAL.md).
Tanda: `D:/AI/artifacts/ntrpu-hbm-dma-boundary/20261009T040422369542Z`.
Dispositivo `xcvu47p-fsvh2892-2-e`, Vivado 2026.1 build 6511674.

## Resultado seleccionado: timing pendiente

Estado **`BOUNDARY_ROUTED_TIMING_OPEN`**. `opt1` termina en 340,013 s y
`audit1` reabre su checkpoint sin optimizarlo en 165,758 s; ambos producen
los mismos resúmenes de timing y métricas.

| Dominio | Frecuencia | Peor setup | Peor hold |
| --- | --- | --- | --- |
| Lector | 200 MHz | −0,339 ns | +0,009 ns |
| DMA | 250 MHz | +0,744 ns | +0,010 ns |

Setup conserva 2.958 endpoints incumplidos y TNS −434,544 ns. Hold y ancho
de pulso no tienen incumplimientos; el margen mínimo de pulso es +1,468 ns.
Los mínimos de captura RAM, controles, link y skew Gray son respectivamente
+3,037, +3,453, +3,600 y +3,366 ns. Estos datos no autorizan promover el
banco como cerrado a 200 MHz.

Hay 21.805 nets ruteables completamente ruteados y cero errores de ruteo.
Las 11.375 conexiones hacia el núcleo están `ROUTED` (11.368) o `INTRASITE`
(7), sin puertos externos en esos nets. `RTSTAT-10` corresponde a las
3.713 salidas exteriores de captura sin carga ruteable: están fuera de la
medida. Permanecen visibles los catorce `CDC-6`, 1.318 `CDC-15` y cuatro
`CDC-26`; la topología y los límites físicos se comprueban por separado.

El peor camino va del registro de `submit_context[source_offset][3]` al
CE de `lanes[0].bridge/contexts_reg[1][870]`: 18 niveles lógicos, nueve
CARRY8 y 5,002 ns de datos (1,431 ns de lógica, 3,571 ns de ruteo).
La admisión/rango del lector es el siguiente candidato de diseño.
Conviene evaluar precomputación o una etapa de admisión en una revisión
aislada, manteniendo solicitud, lease, créditos, replay y parada coherentes.
Placement también puede influir; este resultado no prueba que añadir una
etapa sea la única solución. Antes de otro ensayo físico, esa revisión
necesita regresión funcional de aceptación/rechazo y backpressure.

## Alcance de la frontera

Se añaden 7.662 FDRE de lanzamiento y 3.713 FDRE de captura alrededor de
una copia del checkpoint. Comparten los dos BUFGCE con el puente: lector
a 200 MHz y DMA a 250 MHz. El RTL de la revisión 2 permanece intacto.
Los registros externos tienen CE alto, reset inactivo y `DONT_TOUCH`.

```mermaid
flowchart LR
  E[Estímulo externo] -. Exclusión de medida .-> L[Registros de lanzamiento]
  L --> P[Puente rev2]
  P --> C[Registros de captura]
  C -. Exclusión de medida .-> M[Observación externa]
```

Se temporizan Q→puente y puente→D. Las dos exclusiones externas terminan
exclusivamente en los D de lanzamiento o parten de los Q de captura.
En XDC, el origen de la exclusión de salida se expresa como celdas
secuenciales de captura; usar sus pines Q como startpoints es inválido.
No hay exclusiones globales entre relojes. Los presupuestos de puerto
solo describen el aparato exterior, que no se califica en este banco.

**Estos registros no constituyen un adaptador de protocolo.** Retardar
`ready`, `valid`, datos y controles arbitrariamente no preserva por sí
solo el handshake. Una integración funcional necesita colas o registros
elásticos, gestión de reset y pruebas con las IP HBM/XDMA reales.

`boundary.tsv` inventaría cada puerto, dominio, registro y conexión del
núcleo. `connectivity.tsv` verifica nuevamente todos los consumidores o
drivers originales tras insertar la frontera completa. Las optimizaciones
posteriores son las de Vivado; esta comprobación ECO no es equivalencia
formal ni simulación funcional del netlist implementado.

## Link y reset

`dma_link_up` nace ahora en un FDRE del dominio DMA. Su camino hasta
`link_s1_reg/D` conserva un máximo de 4 ns; la segunda etapa permanece
temporizada normalmente. El reset frío conserva su polaridad activa baja
y exactamente diez pines CLR de los sincronizadores de arranque. Esa excepción no
califica duración de pulso, recuperación externa ni quiescencia PCIe.

El reproductor `ntrpu_reset_link_probe.sv` contiene ocho registros y ningún
RAM. En `probe2/port_origin`, Vivado informa `CDC-26` sobre la entrada link
y `partial_input_delay=1`, aunque ambos retardos de entrada están definidos.
Al registrar la fuente en DMA, informa `CDC-3` con profundidad dos y
`partial_input_delay=0`. Añadir la exclusión exterior puerto→D mantiene
ese resultado. El reset pequeño obtiene dos `CDC-9`. Esto acota el síntoma
de link al modelo de origen; no demuestra que cualquier advertencia de
reset del puente completo sea inocua.

El primer intento `probe1` falló después de exportar el caso de puerto por
usar un comando inexistente. `probe2` reabre un checkpoint común anterior
a la restricción de link para comparar ambos casos sin restricciones
residuales. Se conservan ambos intentos y sus hashes.

`probe3` intenta la implementación física pequeña y alcanza su timeout de
240 s durante placement; no aporta un resultado ruteado. `prepare1` llega
a insertar la frontera completa, pero falla al usar Q como origen de la
exclusión. `prepare2` corrige ese selector, agrupa las conexiones comunes
y conserva un checkpoint anterior a las restricciones. Termina en
428,915 s y verifica los 11.375 registros y todas las conexiones originales.
En ese checkpoint preparado, el CDC completo tiene seis `CDC-3`, catorce
`CDC-6`, dos `CDC-9` y 1.318 `CDC-15`, sin `CDC-26`. Ya no hay retardos
parciales de entrada. El único puerto sin presupuesto es el reset frío
asíncrono, limitado a sus diez CLR. Esto aún no es evidencia de ruteo.

`route1` completa optimización y placement, pero alcanza 900 s durante
ruteo Default. Su checkpoint colocado, también conservado por separado
en `route1-recovery`, tiene WNS de setup +0,468 ns y hold −0,314 ns
(2.654 endpoints incumplidos). El informe de congestión identifica ocho
nets de contextos y eventos en nueve tiles CLB. `route2` reutiliza ese
checkpoint con cuatro CPU, límite de 1.200 s y `RuntimeOptimized`.
Esta directiva reduce el esfuerzo de ruteo, por lo que se exige medir el
resultado sin cambiar ninguna restricción; véase
[AMD UG835, route_design](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/route_design).

`route2` termina de rutear, pero el guard inicial rechaza la representación
optimizada del reset. `reset1` reabre el DCP y encuentra los mismos diez CLR
con conexión directa. El inversor externo fue absorbido en
`IS_CLR_INVERTED=1`, transformación descrita en
[AMD UG974, FDCE](https://docs.amd.com/r/2025.1-English/ug974-vivado-ultrascale-libraries/FDCE).
La revisión corregida exige exactamente los diez destinatarios, `FDCE`,
`ASYNC_REG`, INIT cero, dominio correcto y polaridad activa baja: inversor
LUT1 con CLR no invertido, o conexión directa con CLR invertido.

`probe4` aplica solo `opt_design` al reproductor sin RAM. Sus ocho CLR quedan
invertidos y el CDC cambia a un `CDC-3` y tres `CDC-26`, situados en los
resets de `dr`, `sr` y `link_s1`. En el puente hay cuatro, por incluir también
`ready_s1`. Se reproduce así el diagnóstico sobre registros de reset sin
LUTRAM; no se introduce waiver ni se ocultan las advertencias. La revisión
estructural no sustituye la calificación del reset externo.

`opt1` revalida la polaridad antes de optimizar la copia del DCP ruteado.
Ejecuta `phys_opt_design -directive Explore` y repara rutas conservando las
completadas. Los periodos siguen siendo 5 y 4 ns. El verificador conserva
el rechazo original de `route2` como parte de la procedencia; no lo convierte
en una tanda aprobada. La aceptación depende de la comprobación corregida
y de una reapertura independiente del resultado seleccionado.

## Evidencia y reproducción

`tools/ntrpu_hbm_dma_boundary.tcl` separa preparación, implementación y
reapertura mediante `evidence/mode.txt`. Guarda checkpoints después de
preparar, optimizar, colocar y rutear. El runner limita cada proceso a
dos CPU y prioridad reducida para las primeras tandas. El runner específico
`run_ntrpu_boundary_probe.py` permite presupuestos de una, dos o cuatro CPU;
la continuación utiliza cuatro. El timeout solo termina su grupo privado.

Se conservan el CDC completo y otro informe entre los dos relojes definidos;
el segundo no reemplaza el primero. Se verifican los 1.318 bits de captura
RAM, 45 controles originales, link registrado y 12 buses Gray. Véase
[AMD UG835, report_cdc](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_cdc)
para el alcance del análisis y sus filtros.

```text
python -B tools/check_ntrpu_hbm_dma_boundary.py \
  TANDA/prepare2 TANDA/opt1 TANDA/audit1 DONANTE/ports.tsv \
  --placement TANDA/route1 --baseline TANDA/route2 --reset-probe TANDA/probe4
```

La promoción exige setup, hold, ancho de pulso, máximos CDC y skew sin
violaciones, auditoría de restricciones y explicación de cada aviso
restante. Incluso al satisfacer ese criterio, la frontera solo demuestra
timing bajo este entorno; HBM/XDMA integrado y la tarjeta siguen pendientes.

El checkpoint seleccionado tiene SHA-256
`9338b5520db017175243744c32cffb5b7909830a1225d2e10417dadd56e85d10`.
`OUTCOME.json` contiene la revisión completa y sus ocho controles negativos;
`STATUS.json` identifica los intentos y el resultado seleccionado. Cada
tanda conserva fuentes, informes, resultado original y hashes, incluidos
los intentos fallidos. Los recibos `BACKUP_*.json` apuntan al almacén local
`D:/AI/artifacts/ntrpu-content-store`; las copias `restored-*` se verifican
por contenido. `handoff/` reúne el verificador y sus dependencias, el
inventario de puertos, la documentación y la procedencia para repetir la
revisión sobre esas restauraciones.

Los 150 archivos previamente protegidos se comparan por SHA-256 al cerrar
esta tanda. No se modifica el núcleo rev2 ni la cola de trabajo principal.
Las pruebas funcionales anteriores no se repiten aquí: no se ha añadido
un adaptador funcional ni se ha simulado el netlist implementado.

## Integración funcional posterior

Aplicar el [contrato HBM–XDMA](HBM_PCIE_BRIDGE_CONTRACT.md) en una variante
separada. Primero resolver los 250 MHz configurados frente a los 125 MHz
de metadatos del wrapper mediante el reloj efectivo del modelo nativo.
El banco actual de 4 ns no demuestra la frecuencia de esa IP.

Conectar ambos canales C2H, mantener juntos datos/contexto/`last` bajo
backpressure y conservar la correlación de descriptor, slot de writeback
y pulso de finalización. Probar lectura de errores y habilitación antes
de generar un evento DMA con estado OK. Los registros de medida no
implementan ninguna de esas funciones.

Después ejecutar tráfico PIPE con memoria destino y writeback comprobados,
unir el lector HBM nativo y validar resets coordinados, aislamiento y
obligaciones retenidas hasta CPL local, DMA atribuido y ACK válido.
El perfil serial y las asignaciones conjuntas de GT/pines quedan para
esa integración; ningún resultado de este banco autoriza fabricar la PCB.
