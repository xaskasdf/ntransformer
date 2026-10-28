# Implementación física de la admisión compartida

## Alcance

Esta tanda implementa la [composición compartida](PEER_READ_SHARED.md), con un
origen, una lease y cuatro reservas entre dos canales. Conserva sus fuentes
funcionales aprobadas y el lector F1/C1. No incorpora endpoint PCIe, HBM física,
DDR4 ni restricciones externas de placa.

Objetivo interno: `xcvu47p-fsvh2892-2-e`, reloj de transporte de 5,000 ns y reloj
de memoria de 2,222 ns, incertidumbre de setup de 0,100 ns y hold de 0,050 ns.
Dos BUFGCE en el wrapper físico. Se mantienen las restricciones específicas
para sincronizadores de estado y las de los IP CDC; no se excluyen dominios
enteros. Las E/S externas siguen sin calificar.

Síntesis y ruteo son tandas separadas. El segundo recibe el checkpoint auditado
del primero; no vuelve a generar ni sustituir el RTL. La síntesis usa cuatro
CPU; el ruteo usa cuatro u ocho, con prioridad baja y límites indicados por
tanda. Cada resultado se trae al almacén incremental y se verifica mediante
restauración.

## Estado de la ejecución

La síntesis `synth2` pasó en 597,859 segundos y su checkpoint se restauró y
verificó. Reporta 24.657 LUT, 25.561 registros y cero BRAM, latches y black boxes.
Se comprobaron los 21 registros de sincronización de arranque, los cinco
endpoints de estado y cero endpoints internos sin restricciones. CDC registra
13 casos CDC-3 y ocho CDC-9, todos informativos. Nueve alteraciones del reporte
fueron rechazadas por el auditor. El timing de síntesis no está calificado:
el peor setup estimado es −0,289 ns en memoria. Hay 6.077 entradas y 5.940
salidas sin retardos externos; la cobertura declarada es interna.

`route1` agotó sus 1.200 segundos durante la resolución de congestión local
de nivel 5; llegó de 25.403 a 14 solapamientos en el último registro, sin
checkpoint ruteado. Su salida `−15` corresponde a la terminación controlada
por el límite, no a una prueba de timing. Se respaldaron también los
checkpoints de optimización y colocación. `finish1` retomó la colocación
verificada con ocho CPU y límite de 1.800 segundos, conservando RTL y
restricciones. También agotó el límite, sin checkpoint ruteado. La última
iteración completa mostró setup intermedio de −0,228 ns; no es un informe
final ni califica hold o bus skew. Ambas tandas están archivadas y restauradas.
No se ejecutaron las etapas dependientes de netlist o HBM nativa.

Resultado `finish1`:
`f94469c307030f5e0d4961b863ca2d81375d26c9f1d09602f000878deff9a23a`.

El checkpoint sintetizado de origen es
`43897e847706f89966df14486e90c2c0412865edd2aed0ea04f633e7ffbdf397`.
Resultado de síntesis:
`f07ebd3b073ef19858da37d25a9308484feb9ce7a89d333dd9d317fe4a12a75d`.

`synth1` conserva un fallo de preparación
anterior a síntesis: al cargar un procedimiento Tcl, su cabecera recalculaba
la raíz fuera de la copia privada. El lanzador restablece ahora la ruta privada
antes de crear los IP. No se cambió RTL para corregirlo.

El banco público `public_rtl1` pasó contra el RTL en 61,332 segundos: seis casos,
27 transferencias satisfactorias, 46.848 bytes confirmados y cuatro resultados
fallidos que retienen sus reservas. La auditoría comprobó 2.760 muestras y
rechazó diez alteraciones de evidencia. Este paso valida el banco que se usará
con el netlist; **todavía no es una simulación del netlist**.

Resultado `public_rtl1`:
`358a6e517dab8dea8dc4b8a14421300fd7c2b4228ad6def9431cba62f733b7b7`.
Sus fuentes, evidencia y auditor ya se restauraron y verificaron.

## Localización de la congestión

`hotspot1` inspeccionó una copia lógica del checkpoint colocado, sin modificarlo.
En el rectángulo de tiles X=96–119, Y=174–236 indicado por el router aparecen
30.105 de las 61.063 instancias primitivas inventariadas. De ellas, 12.221 son
del guard, 7.143 del ledger 0 y 10.138 del ledger 1: aproximadamente el 98 %
del rectángulo. Son instancias de netlist, incluidas transformaciones de RAM;
no equivalen a LUT ni a porcentajes de ocupación del dispositivo.

El informe jerárquico de colocación aporta otra medida: guard de 9.073 LUT y
5.769 FF; ledgers de 4.552/4.273 LUT y 4.665 FF cada uno. El total colocado
es 24.362 LUT y 25.547 FF, antes del ruteo final. La inspección y dos controles
negativos están respaldados; resultado
`746d1a38e0230e0c23c07e4fe561faca8d795008f5c19b949dcf6199ea3fbcf0`.

## Candidata de metadatos compactos

`ntrpu_peer_read_admission_compact.sv` conserva cuatro reservas y la interfaz
completa. Guarda por reserva solamente secuencia, offset fuente, offset destino
y tamaño: 256 bits frente a 1.024. Reconstruye sesión, handles, generaciones y
época desde la lease fijada. Esta transformación depende de dos invariantes:
la admisión verifica esos campos y la configuración no cambia mientras haya
reservas activas. La retirada sigue comparando los 1.024 bits del contexto.
No añade buffers de payload ni permite más trabajo en vuelo.

Las pruebas aisladas `unit1` y `full1` pasaron, con auditoría independiente y
restauración del respaldo. Cubren los mismos 15 casos unitarios y seis casos
completos de la referencia: 27 retiros satisfactorios, 46.848 bytes confirmados
y cuatro fallos que retienen propiedad. Los 832/6.663 eventos SG/SF coinciden
exactamente con sus fixtures de referencia; esto no demuestra equivalencia
formal para todas las entradas. Los auditores rechazaron ocho y 15 controles
negativos, respectivamente.

La síntesis de la candidata pasó en 707,021 segundos, con los mismos relojes,
restricciones y configuraciones de IP. Conserva cero latches, black boxes,
endpoints internos sin restricciones y CDC no informativos. El checkpoint,
fuentes y nueve controles negativos se restauraron y verificaron.

| Recurso sintetizado | Referencia | Compacta | Reducción |
| --- | ---: | ---: | ---: |
| LUT | 24.657 | 23.152 | 1.505 (6,1 %) |
| Registros | 25.561 | 21.713 | 3.848 (15,1 %) |
| BRAM36 equivalente | 0 | 0 | 0 |

El ahorro de registros incluye optimizaciones derivadas; no es solamente la
resta de bits explícitos. El peor setup de síntesis sigue en −0,289 ns y no
califica timing físico. El ruteo compacto `route1`, con ocho CPU, agotó sus
1.200 segundos sin checkpoint final. La última iteración completa mostró
setup/hold intermedios de −0,927/−0,005 ns; esos valores no son una aprobación
ni un informe final. La candidata no sustituye el guard de referencia.

Resultado de ruteo incompleto, archivado y restaurado:
`97e0df9f76fb7e2bf3732e0f6d28513f5b03c3d1d3e5b352b309c94a733a9444`.

Resultado de síntesis compacta:
`aa417bf69f8e6698be553d112dc70d42a7f7535e3504c8158f6a539eed7033af`.
Checkpoint:
`95610d88377ccb11772b25368f72f78061d6baf26e5583ef1ceecae48a9a0965`.

El inventario compacto `hotspot1` se revisó y restauró. En el rectángulo
X=33–63, Y=63–113 que reúne las zonas de congestión observadas, cuenta 29.490
primitivas: 11.809 del guard y 8.346/8.944 de los ledgers. Este rectángulo es
distinto del de referencia; sus conteos no miden una mejora porcentual entre
colocaciones. El router todavía informa congestión de nivel 5.

La utilización jerárquica colocada del guard es 8.149 LUT/2.685 FF, y la de
los ledgers es 4.245/4.119 LUT y 4.280 FF por instancia. La compresión reduce
almacenamiento, pero queda lógica densa de validación y seguimiento; el
inventario por sí solo no identifica un camino crítico. El checkpoint colocado
`05ad976310e6e190e602f545ff41bacb49d5e1180726a69afa0c658475b02b42`
también está respaldado localmente, sin calificación de ruteo.

Evidencia compacta:
`D:/AI/artifacts/ntrpu-peer-shared-compact/20261008T013308790964Z`.
Resultados unitario/completo:
`3e284adcc873e42949d8b8ac1d0bce983548cc1bd1cf06a0a9bc28b2f56a1fbf` y
`cd0a4c776b7a8e6386ad8833b2203e87e7d8ed4e00759b97255196713539db3a`.

## Criterios de aceptación

- Síntesis: cero latches y black boxes; sincronizadores y cobertura interna
  completos; mismo RTL funcional y configuración de los tres IP de lectura.
- Ruteo: todas las redes ruteables completadas, cero errores de conexión,
  setup/hold/pulse width y ocho límites de bus skew satisfechos. Registrar
  utilización, caminos críticos y cualquier advertencia DRC; no crear waivers.
- Transacciones del netlist: DUT único, sin editar el netlist ni usar `force`;
  comprobar solo puertos públicos. Esperar 150 ns al inicio para GSR. Verificar
  reservas, rechazos, payload, DMA, ACK y retención de fallos. La prueba funcional
  no emplea SDF ni afirma equivalencia exacta por ciclo.
- HBM nativa: volver a vincular el nuevo RTL y los IP, escribir/leer patrones
  conocidos y comprobar ACK, reutilización y parada. La prueba de la composición
  anterior no se hereda.

## Evidencia y herramientas

Raíz de tandas:
`D:/AI/artifacts/ntrpu-peer-shared-physical/20261008T004616550798Z`.
Respaldos `peer-shared-physical-<tanda>` en
`D:/AI/artifacts/ntrpu-content-store`; conservar manifiestos y objetos juntos.

`run_ntrpu_peer_shared_physical.py` ejecuta las etapas `synth` y `route`;
`check_ntrpu_peer_shared_physical.py` distingue síntesis revisada de timing
interno cerrado. `prepare_ntrpu_peer_shared_routed.py` deriva el banco público;
`run_ntrpu_peer_shared_routed.py` y `check_ntrpu_peer_shared_routed.py` ejecutan y
revisan sus transacciones. Utilizar copias privadas y salidas nuevas, con
`repo` y la salida como carpetas hermanas.

La entrega consolidada `HANDOFF` conserva dos árboles preparados independientes
y once tandas, incluidas las fallidas. Su manifiesto es
`peer-shared-physical-20261008T004616550798Z-handoff`. Los checkpoints colocados
también tienen respaldos propios, recuperables sin la VM. El diseño compartido
todavía no tiene simulación de netlist, DRC de ruteo final ni HBM nativa aprobados;
los lanzadores de esas etapas están preparados, pero no se ejecutaron.

El próximo experimento físico debe abordar la congestión del guard y ledgers
con una hipótesis explícita de colocación o pipeline, conservando los relojes
y las pruebas de propiedad. El ahorro de área no demuestra cierre de timing.
Antes de ampliar el pool o afirmar escalado de HBM, completar también
[la prueba de localidad y mapa físico](PEER_HBM_LOCALITY.md).

El [experimento posterior de dispersión de colocación](PEER_READ_SHARED_PLACEMENT.md)
conserva esta síntesis y modifica únicamente la directiva de colocación.
