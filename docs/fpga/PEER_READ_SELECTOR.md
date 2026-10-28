# Selector de contexto DMA/ACK por secuencia

## Cambio y alcance

Variante aislada del 2026-10-08 que responde al
[diagnóstico de colocación](PEER_READ_SHARED_PLACEMENT.md). En
`ntrpu_peer_read_lane_seq_selector.sv`, el índice candidato de DMA/ACK compara
solo `sequence_id` (64 bits), en lugar de la identidad completa (192 bits).
Las comprobaciones posteriores de identidad completa, destino, generación,
estado, longitud y bytes enviados permanecen intactas. No agrega ciclos,
buffers, excepciones de timing ni permisos de memoria.

La precondición es que dos contextos activos no compartan secuencia. El
watermark de admisión conserva esa propiedad; el reset también vacía los
contextos. Elegir un candidato por secuencia no autentica una notificación:
una sesión u origen incorrectos deben detener el flujo y retener reservas.

El módulo original permanece intacto. La preparación cambia el nombre del
módulo únicamente en la copia privada que utiliza el wrapper existente.
`SELECTOR_PROFILE.json` registra la derivación exacta y el RTL original.
`NATIVE_PROFILE.json` se conserva byte por byte como procedencia, sin atribuir
su calificación física o HBM nativa a la variante. El auditor admite exactamente
las dos sustituciones del selector; verifica todos los demás archivos heredados.

## Validación funcional

| Prueba | Evidencia revisada |
| --- | --- |
| Diferencial de puertos públicos | 48 casos: 4 de éxito y 44 de fallo; 1.328 ciclos comparados y 96 resultados |
| Coincidencia parcial de identidad | 4 casos donde los selectores internos difieren, con rechazo y retención equivalentes |
| Ledger existente | 21 casos, 29 rechazos, 18 escenarios de cuarentena y 98.816 bytes observados |
| Composición compartida | 6 casos, 27 retiros exitosos, 46.848 bytes exitosos y 4 reservas fallidas retenidas |
| Comparación con referencia compacta | 6.663 líneas `SG`/`SF` idénticas, incluidos tiempos y payloads |

El diferencial cubre ambos contextos y ambos eventos, sesiones/orígenes
incorrectos, destino y generación de ventana incorrectos, secuencias ausentes,
longitudes/estados incorrectos, eventos antes de emisión o payload, duplicados,
resultados demorados y pausas de DMA. La comparación incluye todas las salidas
públicas. Es evidencia de simulación dirigida, no equivalencia formal exhaustiva.

Los auditores rechazan cinco alteraciones de la traza diferencial, seis del
ledger y quince de la composición. Las mutaciones RTL que omiten la identidad
completa de DMA o ACK, o fijan incorrectamente el selector a cero, producen
`PUBLIC_OUTPUT_DIFFERENCE`. XSIM devuelve código
cero incluso ante esos `$fatal`; el runner exige el marcador final y ausencia
de errores, y registra correctamente las tres mutaciones como fallos esperados.

## Medición física

`synth1` terminó en 666,344 segundos sobre `xcvu47p-fsvh2892-2-e`, con ocho CPU.
Conserva los periodos de 5,000 ns y 2,222 ns y las incertidumbres de la referencia.

| Recurso de síntesis | Referencia compacta | Selector por secuencia | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 23.152 | 22.845 | −307 (−1,326 %) |
| Registros | 21.713 | 21.710 | −3 |
| BRAM | 0 | 0 | 0 |

Auditoría: cero latches, black boxes, bucles o endpoints internos sin restricción;
21 registros de sincronización y el mismo inventario CDC (13 `CDC-3`, 8 `CDC-9`,
informativos). Pasaron los nueve controles negativos del auditor físico.
El setup global estimado de síntesis sigue en −0,289 ns: **no hay aprobación
de timing**. Las 6.077 entradas y 5.684 salidas sin delays de E/S quedan fuera
del alcance de este bloque interno. Resultado de síntesis:
`da1ba2be3e11bdba6bb3ada9bb2cea7f526d53e2242a554b929a30ee5b93e1fb`.

`route1` terminó por timeout a los **1.200,083 segundos**, retorno `-15`, con
ocho CPU, prioridad baja y las directivas originales. Completó colocación,
pero no produjo `routed.dcp`. La estimación de congestión siguió en nivel 5;
la primera iteración bajó de 23.135 a tres nodos con solapamientos, sin acabar.
Los contadores de iteraciones diferentes no son una comparación de rendimiento.
Resultado: `28396b37b27a0ab2ffad75495f2dc0bce9721e5d97a36164edbd4feb1becc46b`.

Se preservaron `optimized.dcp`, `placed.dcp` y los informes de congestión.
La iteración 140 señaló cuatro nets `lanes[0].peer/result_context`; la 160,
`guard/lane_request[231]` y `[232]`. Son pistas sobre conexiones de metadatos,
no prueba de que esos dos nets expliquen todo el problema. La mejora física
confirmada es de recursos de síntesis; **el cierre de timing sigue pendiente**.

Solo el resultado final de setup, hold, pulse width, bus skew, cobertura, CDC
y DRC puede calificar el timing interno. No se ejecutó simulación de un netlist
ruteado de esta variante porque todavía no hay uno que cumpla ese criterio.
HBM nativa, localidad entre stacks, PCIe, E/S y placa requieren evidencia propia.

`inspect1` abrió el checkpoint colocado sin modificarlo y exportó 80 caminos.
El peor camino de transporte sigue pasando por `dma_index0`, `dma_match1` y
el enable de `saved_result`, cuya red tiene fanout 962. Pasó de 23 niveles
(15 CARRY8) a 22 (14 CARRY8); el retardo de datos estimado pasó de 4,524 a
4,425 ns. La parte lógica varió solo de 1,080 a 1,078 ns: la diferencia de
interconectado colocado domina y no constituye una mejora de frecuencia medida.

| Slack colocado estimado (ns) | Referencia compacta | Selector por secuencia |
| --- | ---: | ---: |
| Transporte, setup | +0,160 | +0,186 |
| Transporte, hold | −0,168 | −0,195 |
| Memoria, setup | +0,125 | +0,173 |
| Memoria, hold | −0,114 | −0,107 |

El hold de transporte empeoró en esa estimación. Ninguna columna es timing
final ruteado. El nuevo peor camino de memoria va del reset sincronizado a
un enable del retorno validado; el endpoint difiere del de la referencia.
Resultado de inspección:
`26bb4cecb0737a9ab347238e4627987421e7c72c0737bf7b9f71ff0b5faf23fa`.

## Siguiente experimento propuesto

Evaluar las condiciones completas de DMA/ACK para cada contexto en paralelo,
antes de combinar sus resultados. Así se intentaría evitar la dependencia
«seleccionar un contexto ancho y después validarlo», que sigue en el camino
crítico. Es una hipótesis pendiente: puede duplicar comparadores y aumentar
fanout o área. Debe mantener autenticación completa, prioridades de fallo,
retención y latencia observables, y repetir los bancos diferenciales y físicos.
No introducir un ciclo adicional ni modificar excepciones de timing como parte
de esa comparación. Los buses de metadatos congestionados deben seguir auditándose.

La implementación y mediciones de ese experimento se registran por separado en
[Validación DMA/ACK por contexto en paralelo](PEER_READ_PARALLEL.md).

## Evidencia y reproducción

Directorio local: `D:/AI/artifacts/ntrpu-peer-selector/20261008T030814794818Z`.
Cada tanda tiene copia congelada, `FILES.json`, `REVIEW.json` y recibo de
transferencia. Los manifiestos `peer-selector-*` del almacén incremental
`D:/AI/artifacts/ntrpu-content-store` se restauran y el auditor se ejecuta otra vez.
No se guardan ZIP duplicados. Conservar juntos `objects/` y `manifests/`.
Las nueve tandas (incluidos los tres mutantes) se respaldaron, restauraron y
revisaron. La entrega agrega documentos, procedencia y los recibos de cada tanda.

Herramientas: `prepare_ntrpu_peer_selector.py`, `run_ntrpu_peer_selector.py` y
`check_ntrpu_peer_selector.py`; el banco específico es
`fpga/tb/tb_ntrpu_peer_selector.sv`. La composición utiliza el runner compartido
existente y el nuevo auditor de procedencia. Los resultados anteriores no se
sobrescriben ni se vuelve a etiquetar una prueba nativa antigua como nueva.
