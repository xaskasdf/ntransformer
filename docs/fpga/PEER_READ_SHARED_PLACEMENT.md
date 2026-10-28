# Dispersión de colocación del origen compartido

## Hipótesis y alcance

Experimento del 2026-10-07, hora de Chile. Parte de la síntesis compacta revisada
en [la tanda física anterior](PEER_READ_SHARED_PHYSICAL.md), checkpoint
`95610d88377ccb11772b25368f72f78061d6baf26e5583ef1ceecae48a9a0965`.
No cambia RTL, interfaz, relojes, incertidumbres, excepciones CDC ni parámetros
de los IP. La hipótesis es que distribuir la colocación reduzca la congestión
que impidió completar el ruteo anterior.

El único cambio de algoritmo en Tcl es:

```tcl
# Antes
place_design
# Candidata
place_design -directive AltSpreadLogic_high
```

AMD asocia esa directiva con su estrategia `Congestion_SpreadLogic_high` en
[UG904](https://docs.amd.com/r/2025.1-English/ug904-vivado-implementation/Directives-Used-by-opt_design-and-place_design-in-Implementation-Strategies).
Es un experimento: dispersar lógica también puede aumentar la longitud de
conexiones; el criterio sigue siendo el timing final, no el nombre de la estrategia.

## Diagnóstico previo

`inspect1` abrió el checkpoint colocado compacto sin modificarlo y exportó
80 caminos, utilización y fanout. Se restauró y verificó el respaldo. Resultado:
`398a7d34dadf5519cb02bb40e010825c2d262db54b6b449d9b56e3a7b4d4fa18`.

- Setup colocado estimado: +0,160 ns en transporte y +0,125 ns en memoria.
  Hold colocado: −0,168/−0,114 ns. No son resultados de ruteo final.
- El peor camino de transporte va de la secuencia de un contexto al enable de
  `saved_result` del ledger: 23 niveles, incluidos 15 CARRY8; demora de datos
  de 4,524 ns, con 3,444 ns de interconectado estimado.
- Hay controles de lease con fanout 1.157 y controles del ledger con fanout
  1.032. Los índices de escritura de frames alcanzan fanout 1.330. El reset
  sincronizado tiene 11.720 cargas y ya usa BUFGCE.

Estos datos justifican evaluar la colocación; no demuestran que un único net
cause la congestión. La salida de ayuda Tcl no quedó impresa en el primer
sondeo; se corrigió el auditor para no exigir ese texto ausente. Los informes
y hashes de entrada no cambiaron.

## Ejecución y aceptación

`spread1` terminó por timeout a los **1.200,134 segundos**, con retorno `-15`,
ocho CPU y prioridad baja. Seguía en `Post Placement Timing Optimization`;
no terminó la colocación ni comenzó el ruteo. Conservó el checkpoint de
entrada y `optimized.dcp`, pero **no produjo `placed.dcp` ni `routed.dcp`**.
El WNS intermedio de −0,722 ns procede de la inicialización de la optimización;
no permite comparar timing final con el candidato anterior.

Resultado: `e7ce10e773acdec263e9f161fd4297b52b6a345c4b78ccc0836a8897372a4084`.
Estado auditado: `SPREAD_FAILED_BATCH_ARCHIVED`, `qualified_timing=false`.
No se selecciona esta directiva: no aportó una mejora demostrada dentro del
presupuesto, aunque este límite no prueba que el diseño sea imposible de rutear.

`run_ntrpu_peer_shared_spread.py` genera el flujo privado;
`check_ntrpu_peer_shared_spread.py` verifica que, salvo la ruta inicial de esa
copia, solo cambie la línea de colocación. Esa comprobación pasó, junto con
los hashes de fuentes y del DCP de origen. Se descargaron 19.197.793 bytes sin
retener un ZIP suelto; los 135 archivos del respaldo se restauraron y el auditor
reprodujo el mismo resultado. El almacén añadió 15 objetos, 8.878.213 bytes
comprimidos; los demás objetos ya estaban presentes. No quedan procesos de
`spread1` en la VM y los diez archivos protegidos conservan sus hashes.

La aceptación física utiliza el auditor existente: todas las redes ruteables
completadas, setup/hold/pulse width y bus skew satisfechos, cobertura interna,
CDC y DRC revisados. Después corresponden la simulación de puertos públicos
del netlist y la revisión individual de avisos DRC. HBM nativa, E/S de placa
y PCIe permanecen fuera de esta prueba.

Evidencia local:
`D:/AI/artifacts/ntrpu-peer-placement/20261008T022807631298Z`.
Los manifiestos por tanda usan el prefijo `peer-shared-placement-` en el
almacén incremental `D:/AI/artifacts/ntrpu-content-store`.

## Siguiente alternativa: selección de contexto

El camino exportado atraviesa dos comparaciones en serie: `dma_index0`,
selección del contexto y `dma_match1`, antes del enable de `saved_result`.
La selección actual compara la identidad completa de 192 bits; la validación
posterior vuelve a comprobar identidad, destino y generación de ventana.

Una candidata acotada es seleccionar DMA/ACK por `sequence_id` de 64 bits,
como ya hacen los eventos de datos/CPL, y conservar íntegra la comprobación
posterior. La unicidad de secuencias activas es una precondición: la impone
el watermark de admisión. Una notificación con secuencia correcta pero
sesión u origen incorrectos debe seguir deteniendo el flujo sin liberar reservas.

Al cerrar esta tanda, la candidata todavía no estaba implementada ni medida.
Su implementación y evidencia posterior se registran en
[Selector de contexto DMA/ACK por secuencia](PEER_READ_SELECTOR.md).
Requiere pruebas con dos
contextos activos, identidades parcialmente coincidentes, generación o destino
incorrectos, duplicados, eventos anticipados y backpressure de resultados.
Debe conservar la latencia y la retención ante fallos. Después se comparan
las trazas completas y se repiten síntesis y cierre físico. Cambiar el ledger
invalida la herencia de la calificación nativa anterior: necesita un perfil
propio, sin actualizar los hashes del perfil antiguo para aparentar continuidad.
