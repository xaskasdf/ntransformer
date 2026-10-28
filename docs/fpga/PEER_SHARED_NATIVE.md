# Origen compartido y localidad con HBM nativo

## Alcance

Experimento aislado sobre el RTL de [context commit](PEER_READ_COMMIT.md), cuyo
checkpoint `f082b92b5c969b597d3c8bf0a2af7aa5e10cde52004cb8be98f850fd0fc222fa`
cerró timing interno. Se conservan sus fuentes RTL, clocks y configuraciones de
los convertidores. El cierre de timing no se transfiere al banco con el IP HBM.

Estado: **SHARED_NATIVE_TRAFFIC_REVIEWED**. Preflight, snapshot y tráfico nativo
aprobados; cinco tandas respaldadas y restauradas, incluidos dos intentos fallidos.
Los 41 archivos de referencia protegidos permanecieron iguales. Sin trabajos
propios activos al cierre; la cola del agente principal se conserva.

## Mapa y permisos

El perfil congelado utiliza HBM de 16 GB, dos stacks, direccionamiento global y
los puertos AXI_00/AXI_16. Se inicializan una sola vez 16 KiB desde dirección 0 y
8 KiB desde `0x2_0000_0000`. El patrón incluye los bits altos de la dirección:
truncar el bit del stack debe producir un error de datos.

Con direccionamiento global, el puerto de entrada y el destino físico son
decisiones distintas. AMD también exige esperar la respuesta de escritura
antes de un acceso dependiente: [PG276, System-Level Considerations](https://docs.amd.com/r/en-US/pg276-axi-hbm/System-Level-Considerations).
El auditor relaciona la precarga con las respuestas B y rechaza bytes escritos
dos veces, lecturas no inicializadas y publicación prematura.

La admisión compartida sigue teniendo una sola región contigua autorizada.
El banco cambia de región únicamente con cero reservas y generaciones nuevas;
no autoriza el hueco de 8 GiB ni introduce traducción de direcciones en el RTL.

## Matriz de pruebas

| Fase | Qué verifica |
|---|---|
| AXI directo local | AXI_00 hacia PC0 y AXI_16 hacia PC16 simultáneamente |
| AXI directo cruzado | Los mismos datos desde el otro puerto de entrada |
| AXI directo común | Ambos puertos leen PC0, sin duplicar su precarga |
| Cuatro reservas | Dos lecturas por canal, origen común y destinos distintos |
| Reutilización | Ocho transferencias de 64, 512, 576 y 4096 bytes; rechazo de replay |
| Revocación | Continúa lo aceptado, se rechaza trabajo nuevo y se cambia de generación |
| Región alta | Lecturas mediante ambos canales, conservando el bit 33 |
| Stop global | Dos contextos fallidos permanecen retenidos |

Los accesos directos pertenecen al banco y se drenan antes de habilitar la ruta
compartida. No demuestran todavía striping simultáneo entre stacks mediante la
admisión compartida.

## Reproducción y evidencia

Preparar una copia privada con `tools/prepare_ntrpu_peer_shared_native.py`,
seleccionando la tanda de ruteo revisada. Ejecutar el runner del mismo nombre con
`--stage preflight`, después `compile` y finalmente `traffic`, respetando sus
argumentos de procedencia y las revisiones requeridas. El auditor independiente
`tools/check_ntrpu_peer_shared_native.py` revisa la memoria y reutiliza el modelo
de propiedad y confirmaciones con el alcance explícito de este banco.

Directorio de esta ejecución:
`D:/AI/artifacts/ntrpu-peer-shared-native/20261008T112024417438Z`.
Cada tanda se guarda en el almacén incremental y se restaura para verificarla.
`RUNTIME_RECOVERY.json` conserva los modos de los 15 archivos del snapshot para
restaurarlos en Linux; la instalación compatible de Vivado sigue siendo un
requisito externo. La restauración local comprueba contenido y hashes, no una
reinstalación completa en otra máquina.

## Preflight revisado

La tanda `20261008T112442142316Z-preflight3` completó 16 retiros correctos,
35.200 bytes compartidos, pico de cuatro reservas y dos reservas fallidas
retenidas después del stop. El auditor verificó 48 respuestas B, 122 solicitudes
AR y 59.776 bytes nativos, incluidos 24.576 de la matriz AXI directa. Rechazó
12 alteraciones de la traza HBM y 15 de la traza de propiedad. Esta etapa usa un
stub; la ejecución posterior con el IP de AMD se detalla a continuación.

Se conservaron dos intentos fallidos: `preflight1` detectó una declaración
implícita de ancho incorrecto en el banco; `preflight2` terminó por SIGSEGV en
el análisis paralelo de bibliotecas XPM. Se corrigió el orden de declaraciones
y se limitó la preparación del proyecto a un hilo. Ninguno modificó el RTL
seleccionado ni se contabiliza como validación aprobada.

## Resultado con el modelo HBM de AMD

La tanda `20261008T114203833815Z-traffic` terminó en 1106,534 s de ejecución del
simulador, a los 64,435001 µs simulados. Ambos stacks completaron la inicialización
a los 48,415001 µs. El auditor confirmó los mismos recuentos de datos y propiedad
del preflight y rechazó las mismas 27 alteraciones. Las **752 transacciones de 21
flujos** coinciden en contenido y orden por flujo, excluyendo timestamps y muestras
de estado. Se conservaron dos reservas fallidas al terminar.

Cada ventana directa leyó 4096 bytes por puerto, con ocho solicitudes por puerto
y sin stalls observados en ARREADY/RREADY. Ciclos activos: desde el primer AR
aceptado hasta el último R aceptado, inclusive, con período de 2,222 ns.

| Acceso directo | Ciclos activos conjuntos | Bytes/ciclo conjuntos |
|---|---:|---:|
| Local: PC0 y PC16 | 190 | 43,12 |
| Cruzado entre stacks | 208 | 39,38 |
| Ambos puertos hacia PC0 | 464 | 17,66 |

La ventana común necesitó 2,44 veces los ciclos de la local. Es evidencia para
investigar la distribución de direcciones, no una medición sostenida ni una
predicción de velocidad del sistema: se ejecutó una ventana por caso, en orden
fijo, y la ruta directa evita la admisión compartida. No se calificó temperatura;
las señales CATTRIP del modelo permanecieron en X.

Identificadores de evidencia:

- Resultado nativo: `d420c4a33c9c2244a03bee435224c14773eaa6df457e8a9048afe982e599752d`.
- Configuración HBM: `d74a3e492748cf8a87100205e04bafe232fcdba09ae7e4a32105cdd02090539e`.
- Manifiesto del respaldo nativo: `b81f01bfcf22909c3c2285a6c567939744f52d47a0232568f95bd164095bc1b0`.
- Snapshot: `peer-shared-native-20261008T112824787299Z-compile`, 15 archivos de runtime.
- Entrega: `peer-shared-native-20261008T112024417438Z-handoff` en el almacén incremental.

## Límites

Las ventanas cortas registran ciclos activos, bytes y stalls por puerto: sirven
para comparar este patrón, no para publicar ancho de banda sostenido. Las
confirmaciones DMA/ACK son estímulos del banco. Quedan fuera PCIe/BAR real,
coherencia entre tarjetas, comportamiento térmico y validación eléctrica de la
placa. La traducción por fragmentos o múltiples leases necesita un experimento
posterior con autorización y generación fijadas hasta el retiro.

## Siguiente dependencia

Antes de aumentar el pool, permitir dos regiones autorizadas simultáneas con
un mapa explícito por fragmento: intervalo lógico, base física y generación
fijada durante la reserva. La escritura de pesos debe usar el mismo mapa que
la lectura. Rechazar huecos, traducciones ambiguas y cambios con operaciones
activas; verificar que los rechazos no consuman el watermark.

Intercalar destinos dentro de un mismo puerto y verificar orden, IDs y asociación
respuesta-contexto bajo backpressure antes de habilitar el mapa. Este banco drena
entre fases y mantiene un destino por puerto durante cada ventana; no cubre esa
mezcla. Revisar también la configuración de reordenamiento/coherencia del IP.

El criterio de salida sería repetir la matriz local/cruzada mediante la ruta
compartida, conservando aislamiento, DMA/ACK y retención ante fallo. El mapa
y el planificador aún no están implementados. No heredan el margen de timing
del checkpoint actual.
