# GEMV Q8 con transporte AXI y acumulación entre tiles

Estado al 2026-10-03 UTC: **72 casos RTL de fallos/reset y 162 operaciones
con pesos reales de Brandon correctas**. Se compararon 5088 salidas de las
proyecciones gate/up/down. El bloque segmentado se sintetizó y se exploró
su colocación/ruteo aislado; faltan el reloj y las interfaces del sistema padre.

`fpga/rtl/ntrpu_q8_gemv_stream.sv` conecta el lector AXI de 512 bits, los
buffers dobles, el desempaquetador Q8 y el MAC HLS VU5P optimizado. Usa el RTL
del MAC respaldado en `20261002T195413Z-V7-q8-block-mac-bitcast`; no reemplaza
su aritmética por un modelo de comportamiento. Todos comparten reloj.

## Contrato de operación

1. `job_valid/ready` registra de 1 a 32 filas, de 1 a 256 bloques K de 32
   elementos (hasta 8192 columnas), `sequence`, `epoch` y un rango de pesos
   de 64 bits. Rechaza dimensiones inválidas, región vacía o desbordamiento.
2. Cargar las activaciones cuantizadas por `activation_valid/ready`, con
   índice, 32 INT8 y escala FP64. Una máscara nueva por operación impide usar
   memoria residual. Se exige la época registrada; el primer tile requiere
   todas las activaciones. Después de comenzar K no se permite modificarlas.
3. Cada comando de tile indica dirección, etiqueta, época, K inicial y de 1
   a 32 bloques K por fila. Los tiles llegan en orden K creciente. Su contenido
   es row-major **dentro del tile**, no una ventana de la matriz original sin
   reempaquetar. La dirección debe estar alineada a 64 bytes y dentro del
   rango registrado; el producto filas × bloques K debe ser múltiplo de 32
   para cumplir el empaquetado existente de 34 bytes por bloque. Tamaño,
   dirección final y comparación de rango se calculan en etapas registradas
   antes de habilitar el lector; ninguna lectura precede a esas validaciones.
4. El MAC consume cada bloque con su activación y acumulador de fila. Los
   acumuladores persisten entre tiles K. No se publican subtotales.
5. Después del último tile, solo el éxito de carga, decodificación y todos
   los MAC permite emitir `result_valid`: un FP64 por fila, con índice,
   `last`, secuencia y época. Tras consumir todas las filas se ofrece `done`.

Un tail de filas que no permita paquetes integrales requiere padding explícito
en el empaquetador, nunca reducción silenciosa de dimensiones. El rango se
registra al aceptar el job; todavía falta conectarlo al registro de buffers
y autorizaciones del runtime. Los pesos deben permanecer inmutables durante
la operación. No hay escritura DMA ni carga AXI de activaciones en este top.

## Errores, cancelación y reset

| Estado | Código |
| --- | ---: |
| Éxito | 0 |
| Comando, orden K, época de tile o rango inválido | 1 |
| Activación faltante, índice o época inválidos | 2 |
| Transporte AXI o confirmación de carga incorrectos | 3 |
| Decodificación o su confirmación incorrectas | 4 |
| Índice, etiqueta, LAST o conteo de bloques incorrectos | 5 |
| Error numérico o respuesta incompleta del MAC | 6 |
| Cancelación aceptada | 7 |
| Watchdog sin progreso | 8 |

El primer error queda retenido. La operación drena el transporte emitido y
espera el MAC en curso; después notifica fallo sin resultados. El watchdog
detecta falta de progreso, pero **no abandona transacciones AXI pendientes**:
si el peer nunca responde, se necesita reset coordinado externo. El reset
de este módulo requiere quiescer/resetear también ese peer.

`cancel_ready` delimita cuándo puede aceptarse una cancelación. Una vez
comenzada la publicación ya es falsa; un consumidor pausado conserva datos
y metadatos estables hasta su handshake. No hay rollback de filas publicadas.

## Banco y resultados

El banco tiene 22 escenarios de operación y dos de reset, con memoria AXI sintética construida
byte a byte y referencia escalar FP64 independiente. Incluye K=2048/8192,
32 filas, cruces de 4 KiB con direcciones superiores a 4 GiB, pausas, errores
AXI/escala tardíos, comandos inválidos sin lecturas, corrupción de etiquetas,
cancelación, timeout con drenaje, intentos de sobrescritura y recuperación.
Se comparan resultados con tolerancia relativa `1e-12` y piso absoluto `1e-12`.
Los resets reinician ambos extremos, primero con una ráfaga AXI pendiente y
después con el MAC en curso; se exige ausencia de resultados/confirmaciones
antiguos y éxito de una operación posterior. El watchdog del banco es de
96 ciclos; el valor predeterminado sintetizado es 65.536 ciclos.

```bash
vivado -mode batch -source tools/ntrpu_board_gemv_stream_sim.tcl \
  -tclargs RUTA_HLS_MAC_COMPLETADO NUEVA_CARPETA
```

Registrar comparación numérica, conteos, ciclos, avisos y respaldo por tanda.
Este banco no usa MIG, PCIe ni pesos de Llama; la integración física mantiene
las dependencias de [VIVADO_QUEUE](board/VIVADO_QUEUE.md).

### Historial revisable

- `20261003T013858Z-V7-gemv-stream-integration`: FAIL del banco, no del
  comparador numérico. Completó casos 1–12; el estímulo de cancelación usó un
  contador residual antes de comenzar el nuevo MAC y el cargador quedó
  esperando una activación después de cancelar. La corrección exige un MAC
  actual en curso. Se conserva el timeout y el código Vivado 0 engañoso.
- `20261003T014314Z-V7-gemv-stream-integration-r2`: fallo Tcl antes de simular;
  el valor `-testplusarg` se interpretó como opción. Se corrigió con `-dict`.
- `20261003T014535Z-V7-gemv-stream-integration-r3`: **66/66 PASS**, 129,492 s
  reales. Cada semilla: 64 ráfagas, 2441 MAC y 75 filas publicadas. El revisor
  independiente rechazó nueve alteraciones de logs reales.
- `20261003T015039Z-V7-gemv-stream-reset`: **72/72 PASS**, 129,857 s.
  Cada semilla: 68 ráfagas, 2443 MAC; 76–83 pausas de resultados y 120–122
  de completion. Once alteraciones de logs rechazadas. Solo tres avisos
  XSIM 43-3431 del entorno del compilador, conservados.
- `20261003T015649Z-V7-gemv-pipelined-validation`: **72/72 PASS**, 120,919 s;
  misma aritmética y errores tras registrar la validación de comandos.
  Totales de 60.341/60.377/60.373 ciclos por semilla; mismos 68 bursts y
  2443 MAC. La latencia adicional por tile no cambia los resultados.

La versión previa al pipeline midió estos intervalos desde inicio del job
hasta aceptar su completion, incluyendo carga de activaciones, transporte y
pausas del banco. Reloj **simulado** de 5 ns:

| Forma | Ciclos (tres semillas) | Tiempo simulado |
| --- | ---: | ---: |
| 3 × 2048 | 4652–4662 | 23,260–23,310 µs |
| 2 × 8192 | 12512–12523 | 62,560–62,615 µs |
| 32 × 1024 | 24346–24391 | 121,730–121,955 µs |

### Síntesis y ruteo aislado

`20261003T015049Z-V7-gemv-stream-synth`, 207,876 s: **10.443 LUT, 7.644 FF,
32 RAMB36 + 3 RAMB18, 27 DSP**, sin cajas negras. WNS **−1,322 ns**, TNS
−213,259 ns y 200 endpoints fallan setup; WHS +0,046 ns.
La ruta `rows_reg[3] → computed_blocks_reg[0]/CE` atravesaba cálculo de
tamaño, multiplicación y comprobaciones de rango. Se corrigió la estructura
en el RTL, sin añadir excepciones ni cambiar el reloj solicitado.

Conserva 100 avisos de puertos sin carga de operadores generados; DRC muestra
95 avisos de pipeline DSP y 16 recomendaciones AVAL. Metodología incluye
35 SYNTH-6, ocho SYNTH-9, 179 TIMING-16 y 1000 TIMING-18 reportados.
Faltan retardos de 1238 entradas/282 salidas y `HD.CLK_SRC`; cero endpoints
internos sin restricción. Estos conteos corresponden al primer intento.

La versión segmentada, `20261003T015929Z-V7-gemv-pipelined-synth`, terminó en
195,762 s: **10.396 LUT, 7.704 FF, 32 RAMB36 + 3 RAMB18 y 26 DSP**.
WNS +1,785 ns, WHS +0,046 ns, cero fallos internos; mantiene las mismas
entradas/salidas sin retardos. Desaparecen los TIMING-16; quedan los avisos
de RAM/multiplicadores, retardos ausentes y pipeline DSP.

`20261003T021557Z-V7-gemv-pipelined-route-r2` completó opt/place/route en
242,335 s: **10.293 LUT, 7.655 FF**, mismas memorias y 26 DSP. El informe
final da **WNS +0,564 ns, WHS +0,010 ns**, TNS/THS cero; el +0,541 ns del
log del router era intermedio. La ruta crítica es BRAM de pesos → reservorio Q8.
No se agregaron excepciones temporales.

El contexto OOC supone `HD.CLK_SRC=BUFGCE_X0Y60`, sin buffer del padre
instanciado ni retardos/localizaciones de puertos. `report_route_status`
cuenta 18.779 nets ruteadas y cero errores, **pero una net conserva gaps
explícitos**. La auditoría identifica `clk` como `PARTIAL`, puerto jerárquico
sin driver; por tanto no es un reloj completo ni cierre temporal de tarjeta.
Conserva 100 avisos Route 35-198 de puertos sin PARTPIN_LOCS (límite de
mensajes), cuatro DRC 23-814, 54 avisos DSP, un RTSTAT-10, 16 AVAL y retardos
faltantes. `20261003T022306Z-V7-gemv-route-context-audit` (82,188 s) encuentra
203 segmentos jerárquicos del mismo reloj, todos con idéntica ruta parcial;
no son 203 relojes independientes. No modifica restricciones ni checkpoint.

El primer intento `20261003T021016Z-V7-gemv-pipelined-route` terminó antes de
implementar por comprobar NAME contra el top RTL; al abrir un DCP, NAME era
`checkpoint_input`. La revisión verifica parte, reloj y puertos y exporta
las propiedades reales. Ambos intentos se conservan.

La semántica del contexto externo está documentada por AMD en
[UG905](https://docs.amd.com/api/khub/documents/6AuxHARoy_bn8JC57bYlIQ/content).
El sitio supuesto se debe reemplazar por el del sistema integrado y repetir
implementación; no se aprobaron pinout, CDC externos ni restricciones de placa.

## Proyecciones completas con pesos de Brandon

`20261003T023437Z-V7-gemv-brandon-distinct-tiles`: **162/162 operaciones PASS**,
356,925 s reales; 54 operaciones por semilla 1/7/42. Matrices completas de
la capa única 0: `w1`/`w3` de 720×256 y `w2` de 256×720, con columnas down
padded a 736. Los grupos finales de gate/up tienen 16 filas; K se recorre en
tiles de cuatro/ocho bloques, incluido el tail de siete bloques de down.
Cada proyección recibe una entrada sintética independiente con RMS uno;
no se encadena SwiGLU ni se ejecuta una FFN completa en este banco.

La revisión encontró que el primer banco, `20261003T021547Z-V7-gemv-brandon-projections`
(359,584 s), reutilizaba una dirección
entre tiles K menores de 32 bloques, cambiando el contenido del modelo AXI.
Su PASS numérico no acredita pesos inmutables por dirección. La corrección
asigna 64 KiB distintos por tile y pasó la nueva tanda; se conserva
el primer archivo sin modificar. Los tiles sintéticos anteriores de 32
bloques ya usaban direcciones distintas. La repetición conserva las mismas
salidas, ciclos y conteos; los 54 jobs tienen rangos alineados, disjuntos
entre tiles y contenidos en su región registrada.

Modelo `xaskasdf/brandon-tiny-10m-instruct`, revisión
`ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`; SHA-256 del modelo
`2ff727cc8921d21e5e7b153babd288efc5524b166e9417e1b439ac8c11fe44cf`.
El generador verifica modelo/configuración y registra hashes de cada tensor,
archivo de prueba, entradas y referencias. Manifest:
`7758590dec8fac758b7f2920703d06ccf73b3a5d571b7c3af0626ffe1624cf97`.

Las **5088 salidas** coinciden numéricamente con la referencia Q8 escalar:
error absoluto máximo **0**. Frente al producto FP32 independiente, el error
L2 relativo es **0,792496% gate, 0,718751% up y 0,716756% down**.
El revisor rechaza doce corrupciones de logs, incluidas salidas omitidas,
duplicadas, alteradas y NaN.

Por semilla: 17.408 MAC de bloque, 360 ráfagas AXI, 1696 filas y
418.427/418.497/418.510 ciclos. A 5 ns simulados, son 2,092135–2,092550 ms
para las tres proyecciones con carga de activaciones y pausas del banco.
591.872 bytes de pesos por tanda implican aproximadamente **0,283 GB/s útiles
simulados**, no caudal DDR físico ni tokens/s. El MAC escalar serial limita
el aprovechamiento del lector; medir ocupación por etapa antes de replicarlo.

```bash
python tools/prepare_ntrpu_gemv_fixture.py --model-dir RUTA_BRANDON \
  --output-dir NUEVA_CARPETA_FIXTURE
vivado -mode batch -source tools/ntrpu_board_gemv_stream_sim.tcl \
  -tclargs RUTA_HLS_MAC_COMPLETADO NUEVA_SALIDA RUTA_FIXTURE
python tools/check_ntrpu_gemv_fixture.py RUTA_LOG --fixture RUTA_FIXTURE --seed 1
```

La regresión `20261003T021623Z-V7-gemv-fixture-bank-regression` volvió a pasar
72/72 y once alteraciones de logs. Ambas tandas conservan tres avisos
XSIM 43-3431 y tres Wavedata 42-489: el array de fixture excede el límite de
visualización de ondas, sin impedir su simulación ni lectura.

## Iteración de paralelismo

La instrumentación y el candidato de dos MAC ya se ejecutaron: véanse
[arquitectura dual, medidas y evidencias](Q8_GEMV_PARALLEL.md). Brandon baja
de unos 418 mil a 218 mil ciclos, manteniendo sus 5088 salidas. Este archivo
conserva el contrato y los resultados del escalar; la propuesta que sigue
explica la decisión de diseño original.

Antes de replicar, instrumentar ciclos de MAC ocupado, Q8 bloqueado, espera
de AXI, carga de activaciones y publicación. El cociente global del banco
Brandon ronda 24 ciclos por bloque, frente al intervalo aislado de 22 del MAC.
Es evidencia para investigar el cómputo como límite; no sustituye esa medición
por etapa ni permite extrapolar tokens/s de Llama.

Probar inicialmente **dos MAC** con la misma referencia y los bancos de
errores/reset. El layout actual recorre K dentro de una fila; dos bloques
consecutivos de esa fila dependen del mismo acumulador. Replicar el MAC sin
cambiar la planificación no elimina esa dependencia. La alternativa a
evaluar intercala filas dentro de cada bloque K, preservando K creciente por
fila y declarando explícitamente una versión nueva del layout de tiles.

Cada resultado debe volver al acumulador de su fila; no se pueden sumar
subtotales ni publicar mientras quede un MAC o una validación pendiente.
Dimensionar selección de activaciones y puertos de acumuladores, propagar
cancelación/errores a ambos motores y drenar ambos antes de completion.
Comparar ciclos y recursos contra esta revisión archivada; integrar DDR y
restringir el reloj del padre antes de atribuir rendimiento a una tarjeta.
