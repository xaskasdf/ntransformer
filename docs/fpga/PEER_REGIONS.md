# Dos regiones autorizadas y cambio de pseudocanal

Estado: **REGIONS_NATIVE_TRAFFIC_REVIEWED**. Simulación nativa y auditoría aprobadas;
once tandas conservadas en el almacén incremental, incluidos los intentos fallidos.

## Alcance y contrato

Variante experimental de [origen compartido con HBM](PEER_SHARED_NATIVE.md).
Conserva cuatro reservas globales, dos por canal. Añade un mapa de dos intervalos
lógicos hacia direcciones físicas de HBM; ambas regiones pueden estar autorizadas
al mismo tiempo. No amplía el pool de datos ni modifica el RTL de referencia.

Cada intervalo contiene base lógica, base física y longitud. Bases y longitud
deben estar alineadas a 4096 bytes. El mapa rechaza intervalos vacíos, solapamientos
lógicos o físicos, desbordamientos, direcciones fuera de 34 bits y generaciones
distintas de la autorización del origen. Los extremos exclusivos se calculan con
65 bits. Un hueco produce `MAP_MISS=8`, después de los errores previos de identidad,
permisos, replay, rango, publicación y tamaño; no consume el watermark.

La configuración queda fijada mientras existan reservas. Cada aceptación captura
la dirección física en su propia reserva; los cambios posteriores en las entradas
no cambian el comando aceptado. El contexto conserva su desplazamiento lógico.
Un mapa inválido no llega a los dos ledgers de salida: se rechaza la configuración
completa. DMA, ACK y aceptación del resultado siguen siendo necesarios para retirar
una reserva; los resultados fallidos mantienen su propiedad.

## Orden y concurrencia

El lector de referencia usa ID 0. Esta variante permite dos comandos simultáneos
del mismo pseudocanal por puerto. Un comando hacia otro pseudocanal espera las
finalizaciones locales pendientes. El otro puerto puede operar en un stack distinto
simultáneamente. El cambio de dominio no espera el ACK remoto, pero tampoco libera
por sí solo las reservas que lo esperan.

Es una restricción conservadora del experimento, no una afirmación de que todos los
accesos HBM necesiten serialización. La selección física usa los bits `[33:29]` del
perfil HBM congelado; no debe reutilizarse sin revisar otra geometría. AMD describe
la separación entre puerto de entrada y destino, latencias del switch y requisitos
de orden en [PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/System-Level-Considerations).
No se añade un reordenador de respuestas ni un planificador de localidad.

## Evidencia obtenida

Directorio: `D:/AI/artifacts/ntrpu-peer-regions/20261008T121521445611Z`.
Las tandas selladas se guardan en `ntrpu-content-store`; se restauran en una carpeta
privada, se repite el auditor y se elimina únicamente esa restauración temporal.

| Banco | Resultado revisado |
|---|---|
| Mapa y admisión (`unit3`) | 278 vectores, 8 escenarios, 35 solicitudes, 26 admisiones, 9 rechazos y 21 retiros correctos |
| Propiedad | Pico de 4 reservas; 12 configuraciones inválidas; retirada falsificada detectada; 2 retiradas fallidas retenidas |
| Evidencia del mapa | 24 direcciones físicas contrastadas con un modelo entero independiente; 12 alteraciones de traza rechazadas |
| Mutaciones RTL | Se detectaron autorización de huecos, truncamiento del bit de stack y uso de entradas vivas en lugar de la dirección capturada |
| Control de dominio | 32 dominios, 129 comandos y 129 finalizaciones; 288 ciclos de espera por cambio de dominio; 134 ciclos de contrapresión |
| Fallos del dominio | Finalización espuria detectada, reset comprobado, 32 aceptaciones simultáneas con finalización y 5 alteraciones de traza rechazadas |

Se conservan dos intentos iniciales fallidos: un identificador reservado de
SystemVerilog y salidas indeterminadas en la primera formulación procedural del
decodificador. La versión revisada usa expresiones combinacionales explícitas.
No se atribuye el segundo fallo a una causa interna del simulador no demostrada.

## Integración preparada

`prepare_ntrpu_peer_regions_full.py` crea una copia privada desde la tanda nativa
revisada. Para reproducir, usar esa copia `repo/` completa: no mezclar la variante
con versiones distintas del lector o de los convertidores presentes en otros
experimentos. La precarga escribe una sola vez 16 KiB en PC0 y 8 KiB en PC16. El mapa
inicial autoriza `[0,16384)` hacia PC0 y `[32768,40960)` hacia PC16; los huecos
permanecen prohibidos. Una segunda generación intercambia las bases físicas con
longitudes compatibles con la misma precarga.

La prueba combina cuatro reservas, destinos de ambos stacks, cambios de dominio
pendientes en cada puerto, contrapresión asimétrica, configuración inválida,
revocación, cambio de generación y stop global. La auditoría relaciona mapa,
comandos, bytes, finalizaciones locales, DMA, ACK y retiro.

Estado conductual: **REGIONS_NATIVE_PREFLIGHT_REVIEWED**. La tanda `preflight2`
terminó en 72,447 s: 16 retiros correctos, 35.200 bytes compartidos, 74 solicitudes
AR de la ruta compartida y dos reservas fallidas retenidas. Incluyendo las tres
ventanas AXI directas: 122 AR y 59.776 bytes leídos. Se verificaron 48 respuestas B
de precarga y se rechazaron 31 alteraciones de evidencia.

Ambos stacks estuvieron activos simultáneamente a través del mapa. El auditor
observó 200 ciclos de espera por cambio de dominio, pico de dos comandos por puerto
y 18 comandos totales. Los dos comandos locales retenidos tras stop no autorizan
una nueva configuración ni una liberación de propiedad.

El contador de esperas se revisó usando el estado anterior a cada arista. La
revisión congelada inicial contó 196: omitía las cuatro aristas que drenaban el
último comando del dominio anterior. La corrección del auditor se registra por
separado y no modifica RTL, simulaciones ni revisiones ya selladas.

`preflight1` también pasó. Se repitió después de corregir el nombre del marcador
de finalización del script nativo, antes de congelar las fuentes para compilar.
Estas pruebas usan memoria conductual; la comprobación con el modelo HBM de AMD
se completó después, como se detalla a continuación. La variante todavía no tiene
cierre físico.

## Resultado con el modelo HBM de AMD

La tanda `20261008T130010261412Z-traffic1` terminó en 1012,4 s, a los 61,570001 µs
simulados. Ambos stacks completaron la inicialización a los 48,415001 µs; la
precarga terminó a los 53,065001 µs. El auditor verificó los mismos recuentos de
memoria y propiedad del preflight: 35.200 bytes compartidos, 16 retiros correctos,
cuatro reservas de pico y dos reservas fallidas retenidas tras stop.

Las **773 transacciones de 23 flujos** coinciden en contenido y orden por flujo
entre preflight y modelo nativo, excluyendo timestamps y muestras de estado. Las
31 alteraciones de evidencia fueron rechazadas. La revisión adicional confirma
solapamiento de lecturas nativas entre AR aceptado y RLAST en stacks distintos,
además de la concurrencia de comandos del lector.

El contador corregido registra 253 ciclos de espera por cambio de dominio en
la simulación nativa, frente a 200 en el preflight. Las revisiones congeladas
contaban 249 y 196, respectivamente. `DOMAIN_METRICS_REVIEW.json` explica la
corrección y vincula el auditor actualizado con ambas trazas; no se reescribieron
las fuentes del simulador ni los resultados ya sellados.

Las ventanas AXI directas reprodujeron 190 ciclos activos para accesos locales,
208 para cruzados y 464 cuando ambos puertos leen PC0. Son ventanas cortas de
8 KiB conjuntos, no rendimiento sostenido del mapa. Las señales térmicas CATTRIP
permanecieron en X: no hay validación térmica.

- Resultado nativo: `eac5bd19b66464b99626c568a9bb85c7f3c89a799514853a9915b3abd654e3f1`.
- Respaldo nativo: `peer-regions-20261008T130010261412Z-traffic1`.
- Snapshot: `peer-regions-20261008T124346432160Z-compile1`; 15 archivos, 511.201.969 bytes.
- Entrega: `peer-regions-20261008T121521445611Z-handoff`.

El snapshot se restauró y verificó por contenido. `RUNTIME_MODES.json` conserva
los permisos necesarios para reconstruirlo en Linux; Vivado compatible sigue
siendo un requisito externo. La restauración en Windows no equivale a ejecutar
el snapshot en una segunda instalación.

## Coste y siguientes criterios

La declaración RTL añade nominalmente 722 bits de estado: mapa de 450 bits,
cuatro direcciones capturadas de 64 bits y dos controles de dominio de 8 bits.
No es un recuento de FF/LUT sintetizadas: faltan síntesis y ruteo. No añade BRAM
de payload. La lógica de límites y traducción también consume recursos y timing.

Antes de ampliar el pool: repetir implementación física con los clocks existentes
y conectar carga de pesos y publicación al mismo mapa. La integración conductual
y nativa ya está aprobada; quedan rendimiento sostenido y coste físico de esta
variante. Esta tanda no integra PCIe/BAR físico, coherencia entre tarjetas,
esquema eléctrico, PCB, alimentación ni validación térmica.

## Investigación para reducir las esperas

La configuración HBM congelada mantiene `REORDER_EN`, `REORDER_QUEUE_EN` y
`MAINTAIN_COHERENCY` activos en los 16 controladores, con ambos switches globales
habilitados. Esa coherencia interna no establece por sí sola el contrato de
asociación entre las respuestas de varios destinos y nuestras reservas.

Como alternativa posterior, [RAMA, PG310](https://docs.amd.com/r/en-US/pg310-rama/Functional-Description)
fragmenta transferencias, sustituye IDs y restaura el orden de las respuestas.
Su cola de reordenamiento puede usar BRAM o URAM; es un coste de almacenamiento
distinto de los metadatos de este mapa. La documentación
[UG1700 2026.1](https://docs.amd.com/r/en-US/ug1700-vitis-accelerated-data-center/Random-Access-and-the-RAMA-IP)
lo contempla para maestros con ID único o que cambia lentamente y acceso a varios
pseudocanales. Su beneficio para nuestro patrón aún no está medido.
El catálogo de la instalación 2026.1 contiene `rama_v1_1`; sólo se comprobó su
presencia, sin generar ni probar ese IP. `REORDER_RESEARCH.json` registra las
opciones del perfil HBM y las referencias consultadas.

[RAMA no comprueba rangos de direcciones](https://docs.amd.com/r/en-US/pg310-rama/Addressing).
Su interleaving opcional también cambia el mapa físico: no debe activarse sin
integrar esa traducción con autorización, carga de pesos y publicación. Comparar
en otra variante: agrupación por localidad con el control actual, IDs y reordenador
propios, o RAMA. Ninguna de estas alternativas se incorpora a la tanda presente.

Para medir rendimiento, ampliar las ventanas después de la prueba funcional.
[PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/HBM-Reordering-Options)
propone como punto de partida diez ciclos de refresh, aproximadamente 40 µs de
tráfico representativo. El tiempo de inicialización no cuenta como esa ventana.
Registrar distribución de latencias, ocupación, stalls y recursos; no elegir el
tamaño del pool a partir de los picos de ventanas cortas.
