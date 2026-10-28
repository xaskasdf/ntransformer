# Frontera registrada entre conversión y cruce de reloj

Estudio C1 del 2026-10-07 sobre [F1 de dos frames](HBM_READ_FRAMES.md). **Se conserva F1 con un nuevo ruteo: +87 ps de setup y +7 ps de hold en memoria, sin aumentar recursos ni modificar RTL.** Las dos variantes con cola adicional se archivan como experimentos: la revisión 1 falla timing y la revisión 2 resulta inferior al control F1 optimizado. Se mantienen 200/450 MHz, incertidumbres nominales setup/hold de 0,100/0,050 ns, ocho créditos y 16 KiB de frames.

La evaluación posterior de esta cola dentro de la composición peer está en [PEER_READ_CUT.md](PEER_READ_CUT.md). Es otro diseño y conserva su propia evidencia; no cambia la selección F1 aislada registrada aquí.

## Ruta que motiva el cambio

El primer camino crítico de `physical2` de F1 conecta la cola de comandos del conversor de anchura del canal 1 con el write-enable de la RAM distribuida del cruce de reloj. Tiene seis niveles de lógica, 1,819 ns de retardo de datos —1,258 ns de ruteo— y una net final de 604 cargas. Su slack es +0,001 ns.

La hipótesis es dividir ese camino con almacenamiento registrado antes del cruce. AMD documenta el uso de [AXI Register Slice para cortar rutas críticas](https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/axi-register-slice.html). Este experimento reutiliza la cola RTL de dos slots ya presente en el adaptador; no incorpora esa IP ni presupone sus resultados de rendimiento.

## Cambio aislado

`prepare_ntrpu_read_cut.py` compone F1 en una copia privada y añade `return_cut` entre la salida de 512 bits del conversor y la entrada del cruce. Cada slot conserva los 512 bits, RID de cuatro bits, RRESP y RLAST: **519 bits × dos slots × dos canales**, incluidos 256 bytes adicionales de payload. Las reservas de frame y los créditos nativos no cambian.

`adapter_pending` incluye la cola nueva hasta vaciarse. Una parada o fallo no borra sus datos ni anula una respuesta ofrecida; solo el reset frío coordinado reinicia su ocupación. READY depende del contador registrado. Una cola llena vuelve a admitir en el ciclo posterior a retirar un elemento, por lo que la contrapresión puede introducir una burbuja.

## Validación y criterios

El banco F1 conserva doce escenarios, incluidos errores tardíos, fallos de protocolo, retención, cuarentena y stop. Un segundo perfil reduce el reloj de origen a 50 MHz manteniendo memoria a 450 MHz para aumentar la contrapresión; su caudal no se compara con el objetivo de 200/450 MHz.

El auditor adicional `check_ntrpu_read_cut.py` reconstruye por ciclo el contador, los punteros, cada entrada y la palabra de salida. Comprueba orden, payload y metadatos, conservación, estabilidad durante retención y `pending` con ocupación distinta de cero. Rechaza seis corrupciones de evidencia, además de los controles F1.

La regresión inicial a 200/450 MHz pasó doce casos: 2.576 entradas y salidas de la cola, sin elementos remanentes; alcanzó sus dos slots y observó 298 muestras llena y 512 retenida, agregadas entre canales. El perfil lento también pasó doce casos, con 14.346 muestras llena y 14.668 retenida. La ventana sintética de 64 KiB cambió de 4.080 a 4.110 ns. No se ejecutó tráfico HBM nativo de C1, al no seleccionar ninguna de sus variantes.

También pasaron el perfil común de 200 MHz, la preprueba nativa de cuatro casos y los veinte escenarios anteriores. Una copia RTL que omitía `return_cut` de `pending` completó el banco, pero el auditor independiente la rechazó: había 140 muestras con la cola ocupada y `pending=0`. El primer caso está registrado en la revisión de `pending_omitted`. Ese resultado es un control negativo, no una variante utilizable.

El criterio fue revisar timing, recursos, CDC, DRC y transacciones del netlist antes de conservar C1; solo un candidato seleccionado requeriría tráfico nativo con igual configuración y calentamiento. El estado térmico, las E/S externas, PCIe y la placa conservan sus criterios independientes.

## Revisión 1: timing abierto

El ruteo terminó en 690,420 s. Setup/hold de origen: +0,777/+0,011 ns; memoria: **−0,156/+0,010 ns**. Ocupa 7.384 LUT y 10.460 registros, frente a 6.399/8.386 de F1. Conserva cero latches, blackboxes y endpoints internos sin restricción; esto no compensa la violación de setup.

El camino limitante se trasladó al enable de `return_cut/slot1_reg`: seis niveles lógicos, 2,050 ns de datos y 515 cargas en su net final. El `VALID` largo del conversor sigue habilitando directamente los registros de payload. Insertar la cola no eliminó ese cono lógico.

La elaboración HBM nativa terminó en 485,978 s; no se ejecutó su tráfico ni se descargó el snapshot de ejecución porque la revisión quedó descartada por timing. Se respaldan fuentes, configuración, logs, manifiesto de runtime y revisión de elaboración. No se acredita regresión ruteada ni auditoría DRC detallada para esta revisión rechazada.

## Revisión 2: escritura del slot libre

`--speculative` selecciona `ntrpu_hbm_read_return_cut.sv`. La memoria de payload captura la entrada mientras su slot está libre, incluso sin VALID; el contador y el puntero de escritura avanzan exclusivamente en el handshake. Con un elemento ocupado, se escribe el otro slot. Con la cola llena, ninguno se escribe, incluso si ocurre una retirada en ese ciclo. La palabra retenida permanece intacta.

Así, VALID termina en la lógica pequeña de propiedad, y la habilitación ancha depende de estado registrado. No se añaden slots ni se cambian los criterios de aceptación. Es posible aumentar la conmutación al capturar datos inválidos en un slot libre; potencia y actividad eléctrica no están medidas.

Los bancos normal y lento pasaron y conservaron exactamente las secuencias registradas RR/RF/RW/RC de la revisión 1: 110.808 y 273.088 eventos, respectivamente. `C1_R2_TRACE_COMPARISON.json` registra hashes y comparación. Esa coincidencia está limitada a estos estímulos; no constituye equivalencia formal ni del netlist.

También pasaron reloj común, veinte escenarios anteriores y preprueba de cuatro escenarios. Los controles RTL de omitir `pending` y alterar un bit fueron rechazados; este último falló con `RETURN_DATA` en el primer caso, sin watchdog. La elaboración nativa pasó en 487,660 s, sin ejecutar tráfico ni descargar otro snapshot de runtime.

El primer ruteo de la revisión 2 cerró en 0 ps de setup al redondear y +5 ps de hold. Eliminó el camino hacia el enable ancho; el nuevo límite está en la selección de datos del conversor hacia el pin D del slot, con cuatro niveles y 1,982 ns. Tiene 7.348 LUT y 10.460 registros.

## Optimización física y control comparable

Se repitió la optimización sobre los checkpoints de C1 revisión 2 y F1, con una exigencia adicional de **100 ps de setup en memoria**. Durante `phys_opt_design` y `route_design`, la incertidumbre de memoria fue 0,200 ns. Después se restauró su valor original de 0,100 ns antes de generar el checkpoint y los informes nominales; hold permaneció en 0,050 ns y no cambiaron los periodos ni las excepciones CDC.

La técnica sigue la [recomendación AMD de añadir incertidumbre para buscar margen](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Overconstraining-the-Design). `check_ntrpu_read_cut_guard.py` verifica el orden del flujo, los informes con ambas incertidumbres y la diferencia exacta de 0,100 ns sobre el mismo ruteo. No se redujo la exigencia nominal respecto de F1.

| Implementación | Setup memoria | Hold memoria | LUT / registros |
| --- | ---: | ---: | ---: |
| F1 original | +1 ps | +2 ps | 6.399 / 8.386 |
| C1 revisión 1 | −156 ps | +10 ps | 7.384 / 10.460 |
| C1 revisión 2 inicial | 0 ps | +5 ps | 7.348 / 10.460 |
| C1 revisión 2 optimizada | +56 ps | +5 ps | 7.348 / 10.460 |
| **F1 optimizado, conservado** | **+87 ps** | **+7 ps** | **6.399 / 8.386** |

La optimización tardó 329,974 s para C1 y 269,251 s para F1. El objetivo adicional de 100 ps **no se alcanzó por completo**: sus slacks bajo esa exigencia son −44 y −13 ps. Ambos pasan las restricciones nominales. F1 obtiene mayor margen, menos recursos y conserva la ventana sintética de 4.080 ns; por eso se selecciona `refined_reference`, no `refined2`.

## Validación del checkpoint conservado

`routed_reference` pasó doce escenarios como único DUT ruteado, con 60 comandos, 56 CPL, 2.558 beats entregados y cuatro comandos en cuarentena. Su digest de transacciones coincide con F1 y rechaza 18 corrupciones de evidencia. C1 revisión 2 también pasó su regresión ruteada, pero eso no cambia la selección. No se declara equivalencia exacta por ciclo, formal ni con SDF.

F1 optimizado conserva setup/hold de origen +0,754/+0,015 ns, 13.519 nets ruteadas sin errores, cero endpoints internos sin restricción y 22 orígenes registrados de RREADY. CDC informa 11 CDC-3 y ocho CDC-9 informativos; las ocho restricciones de bus skew tienen margen mínimo +1,251 ns. `drc_reference` clasificó las siete advertencias, sin waivers: seis PDCN-1569 y RTSTAT-10 con 1.335 nets —1.233 puertos externos OOC sin carga interna y 102 sin cargas lógicas—. No es un DRC limpio.

`NATIVE_REUSE.json` vincula el checkpoint con el tráfico HBM nativo anterior de F1. Diez fuentes RTL sintetizables coinciden por SHA-256. En las tres configuraciones de lectura, todos los parámetros `CONFIG.*` y `IPDEF` coinciden; solo cambia `IP_FILE`, que identifica el directorio de generación. Se conserva la ventana nativa previamente medida de **14,711 GB/s**, con sus hashes, revisión y respaldo restaurado. No se presenta como una nueva ejecución nativa ni como rendimiento de placa. El snapshot F1 existente sigue disponible.

Esta tanda mejora el margen interno de 1 a 87 ps; no cierra el conjunto con hard HBM, PCIe, cómputo ni E/S de placa. CATTRIP continúa sin calificar. La siguiente composición debe repetir timing y comprobar el contrato peer, incluyendo permisos, épocas y recepción remota.

## Evidencia y recuperación

Raíz local: `D:/AI/artifacts/ntrpu-read-cut/20261007T190813803082Z/`. `BASELINE.json` identifica los resultados F1; cada tanda guarda `VARIANT.json`, fuentes privadas, revisión y hashes. Los recibos `BACKUP_*.json` apuntan al almacén deduplicado `D:/AI/artifacts/ntrpu-content-store/`. Las tandas previas se conservan intactas.

`PREVIEW_DIAGNOSTIC.json` registra un ensayo local del generador que rechazó un stub ya transformado. No lanzó simulación remota; las tandas posteriores usan el stub original.

La revisión 2 utiliza `D:/AI/artifacts/ntrpu-read-cut-r2/20261007T192556963183Z/`, sin sobrescribir la primera. `PROFILE_OVERRIDES.json` registra las modificaciones intencionales posteriores a la composición inicial: reloj lento y controles negativos. Los hashes de fuentes del resultado identifican los bytes realmente simulados.

La primera raíz contiene ocho tandas cerradas; la segunda, quince, incluidos controles negativos, ambas optimizaciones y los dos pares de auditorías ruteadas/DRC. `RESTORE_REPLAY.json` verifica restauración y repetición de sus auditores, también los suplementarios de la cola y de incertidumbre. Los ZIP de transferencia y las copias temporales de restauración no se retienen.

`COMPARISON.json` identifica el checkpoint seleccionado; `REMOTE_FINAL_AUDIT.json` comprueba fuentes y resultados remotos/locales y la ausencia de procesos de estas tandas al cierre. `HANDOFF/README.md` y `HANDOFF_BACKUP.json` permiten recuperar el conjunto desde el almacén por contenido. Las fuentes calificadas y la cola del principal permanecen intactas; no se requieren las nuevas colas C1 para reproducir el resultado conservado.
