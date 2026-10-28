# Caudal, latencia y evaluación física del transporte candidato

Estado: **rendimiento RTL y equivalencia funcional de netlist validados;
timing interno ruteado a 200 MHz. Integración PCIe/HBM pendiente**.
Continuación aislada de la [frontera elástica](PEER_RX_CANDIDATE_TRANSPORT.md).
Se conservan el receptor, asignador, núcleo candidato y envoltura de transporte.

## Carga reproducible

Quince casos combinan bloques de 64 B, 512 B y 4 KiB con cinco perfiles:
sin pausas, fetch habilitado un ciclo de cada ocho, productor habilitado
un ciclo de cada dos, ambas salidas W habilitadas un ciclo de cada dos y
destino 0 bloqueado hasta que el destino 1 termina. Cada caso publica
64 trabajos por destino con productores concurrentes e independientes.

La tanda `20261006T003423316750Z-perf` termina en 61,311 s con dos CPU,
prioridad reducida y límite de 900 s. El revisor reconstruye 70.258 ciclos,
1.920 trabajos y **2.990.080 bytes**: orden, propiedad, payload completo,
direcciones, respuestas y finalizaciones. Cuatro supresiones de eventos
son rechazadas como controles negativos.

Una primera ejecución falló por el banco: `disable` de un bloque nombrado
podía terminar otra activación concurrente de la misma tarea. Se sustituyó
por `return` local y se añadió detección de tags publicados por duplicado.
Se conserva la ejecución fallida; no se modificó el RTL para hacerla pasar.

## Caudal observado

Valores en **GB/s equivalentes a ciclos de 5 ns**. Son resultados del modelo
de comportamiento, no mediciones de una tarjeta PCIe/HBM. El intervalo de
lote comprende desde la primera publicación hasta la última finalización,
ambos ciclos incluidos; incorpora llenado y drenaje de las colas.

| Bloque | Sin pausas | Fetch 1/8 | Productor 1/2 | W 1/2 por destino |
| --- | ---: | ---: | ---: | ---: |
| 64 B | 1,5769 | 1,5769 | 1,5754 | 1,5754 |
| 512 B | 6,7493 | 6,3350 | 4,6168 | 6,7216 |
| 4 KiB | 11,4398 | 11,2835 | 6,0854 | 11,3605 |

La ventana central de escrituras, excluyendo el primer y último octavo de
beats, da 1,6147 / 6,8600 / 11,5499 GB/s sin pausas. Esa ventana se define
por eventos de beat y no elimina todos los efectos de fase; no se presenta
como un límite asintótico exacto.

La habilitación de fetch es periódica, no una latencia fija por solicitud.
Con bloques de 64 B coincide con la cadencia natural de ocho ciclos y no
introduce demora adicional. Para caracterizar dispositivos externos faltan
barridos de latencia por solicitud y fase, además de tráfico con variación temporal.

Con el destino 0 bloqueado, el destino 1 completa sus 64 trabajos antes
de liberarlo. Su caudal de lote es 1,3861 / 6,1134 / **10,3861 GB/s** para
los tres tamaños. Se ocupan los ocho bancos; no se pierden datos ni se
bloquea globalmente el receptor. El caudal del lote completo, después de
liberar el destino 0, no describe por sí solo ese período de bloqueo.

## Coste de las etapas e implicaciones

Sin pausas, el intervalo entre fetches es exactamente **8 / 15 / 71 ciclos**:
N + 7 para bloques de N beats. Durante el llenado entra un beat por ciclo;
la siguiente solicitud espera la transición entre bloques. La entrada única
de 512 bits tiene un techo aritmético de 12,8 GB/s a 200 MHz.

Desde una reserva comprometida hasta el fetch transcurren tres ciclos;
del fetch al primer beat de entrada, uno. Después del último beat de entrada,
la primera escritura externa aparece seis ciclos después. La respuesta B
del modelo llega cinco ciclos después del último W y la finalización externa
tres ciclos después de B. Estas latencias incluyen el modelo y las colas de
frontera; no representan latencia de DRAM física.

La latencia mínima publicación–finalización es 22 / 36 / 148 ciclos.
Con productores saturando las colas, el percentil 95 sube a 176 / 330 / 1.562
ciclos: incluye espera en cola y no debe confundirse con latencia de servicio.

En tráfico equilibrado sin pausas solo se ocupan dos bancos. Ampliar el pool
no resolvería el intervalo entre bloques observado. La prioridad propuesta
es solapar preparación/cierre de bloques conservando reserva atómica,
presupuesto de payload, stop y barreras. Los bancos adicionales son útiles
para absorber pausas y mantener independencia, como muestra el caso bloqueado.

Limitar W a medio ritmo en cada destino conserva casi todo el caudal agregado
porque las dos salidas pueden progresar en paralelo. Limitar el productor
único sí reduce el caudal de 4 KiB aproximadamente a la mitad.

## Montaje físico y límites

`ntrpu_transport_physical` conecta un BUFGCE explícito a la envoltura sin
añadir registros de muestreo a las interfaces. Conserva los handshakes.
Contexto de ensayo: `xcvu5p-flva2104-2-i`, período 5 ns, incertidumbres
setup/hold 0,100/0,050 ns, reloj en `BUFGCE_X0Y60`.

Las etapas de síntesis, placement, ruteo y simulación funcional de netlist
son independientes y verifican hashes de las fuentes y checkpoints previos.
No se introducen excepciones de timing ni se relajan avisos DRC. El cierre
interno no acredita temporización externa: faltan retardos de las interfaces,
placa, IP PCIe/HBM, CDC y coordinación real de recuperación.

La tanda `20261006T004648678113Z-route` termina en 381,876 s. Todos los
criterios internos pasan:

| Comprobación | Resultado |
| --- | ---: |
| Setup / hold / ancho de pulso | +0,144 / +0,006 / +1,958 ns |
| Redes ruteables / completamente ruteadas | 17.806 / 17.806 |
| Errores de ruteo, latches, bloques sin resolver | 0 |
| Endpoints internos sin restricciones de timing | 0 |
| LUT / FF / bloques BRAM | 6.755 / 16.029 / 15 |

La auditoría de `RTSTAT-10` identifica 1.551 redes con puertos externos
y sin cargas internas. Se conservan los avisos de puertos sin
`HD.PARTPIN_LOCS`, reloj de entrada sin `HD.CLK_SRC`, ubicación fija del
BUFGCE y alcance DRC fuera de contexto. Hay 1.069 entradas y 1.641 salidas
sin retardos externos definidos en el informe de timing. El resultado vale
para caminos internos bajo ese reloj común; no cierra la interfaz con la
placa/controlador ni permite extrapolar la frecuencia a otros dispositivos.

Checkpoint ruteado SHA-256:
`40e10e1052f62ac85b758e2e62bc6501aa6d76bd6dab9bfcd749fac1a9b8f3e0`.
Netlist funcional SHA-256:
`2db077727996b192ffd90fc401469c39fb7ab81f4b59c1f515e308843ca089e1`.

## Comparación funcional de la netlist

La tanda `20261006T005334119839Z-netlist` termina en 120,043 s. Compara
los 28 puertos de salida durante **5.084 ciclos**, con datos calificados
por VALID incluso cuando hay contrapresión. No añade latencia artificial:
el montaje físico solo incorpora el buffer de reloj. La referencia usa
nombres de módulos distintos mediante una transformación reversible.

Pasan los trece escenarios originales: 91 publicaciones, 88 transferencias
completas, tres cuarentenas esperadas y **127.040 bytes**, sin divergencias.
El revisor también compara las trazas de fetch, W y finalizaciones de los
puertos ruteados. Rechaza trece alteraciones de la referencia y tres
supresiones de eventos de la netlist.

El arranque espera 150 ns para respetar el reset global estándar de UNISIM.
Se verificó que la definición `glbl` incluida en la netlist coincide con
la estándar de esta instalación. Se conservan los avisos de declaración
posterior del contador del banco, redefinición de `glbl` y entorno del
compilador XSIM; no hubo errores ni advertencias críticas. La simulación
es funcional, sin SDF ni `force`; el timing se acredita mediante STA.

## Reproducción y evidencia

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_transport_study.py /perf/output --stage perf
python3 -B tools/check_ntrpu_transport_study.py /perf/output --review /perf/REVIEW.json
python3 -B tools/run_ntrpu_transport_study.py /synth/output --stage synth
python3 -B tools/run_ntrpu_transport_study.py /place/output --stage place --prior /synth/output
python3 -B tools/run_ntrpu_transport_study.py /route/output --stage route --prior /place/output
python3 -B tools/run_ntrpu_transport_study.py /netlist/output --stage netlist --prior /route/output
# Revisar cada etapa con check_ntrpu_transport_study.py antes de continuar.
```

Directorio local:
`C:\Users\xasko\artifacts\ntrpu-transport-study\20261006T003154926287Z`.
Cada etapa conserva fuentes, resultados, revisiones y manifiestos por archivo;
las transferencias ZIP se mantienen en memoria. Respaldo por contenido en
`D:\AI\artifacts\ntrpu-content-store`, con manifiestos por tanda y reutilización
de objetos idénticos, sin volver a copiar toda la historia.

Manifiestos verificados: `transport-study-perf-20261006`,
`transport-study-synth-20261006`, `transport-study-place-20261006`,
`transport-study-route-20261006` y `transport-study-netlist-20261006`.
El intento inicial del banco se conserva en
`transport-study-perf-initial-20261006`. El manifiesto
`transport-study-handoff-20261006` reúne documentación, fuentes de revisión
e índice de esos respaldos. Las tandas terminaron sin procesos propios activos.
