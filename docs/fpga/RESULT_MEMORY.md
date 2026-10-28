# Resultados GEMV en memoria AXI

Estado al 2026-10-03 UTC: `ntrpu_result_writer.sv` recibe un vector privado
y lo escribe por AXI; `ntrpu_q8_gemv_store.sv` lo conecta al
[GEMV con activaciones desde memoria](ACTIVATION_MEMORY.md). El conjunto
lee activaciones y pesos, calcula y escribe resultados sin intervención
del banco en sus datos internos. Conserva los dos MAC HLS y su orden de
acumulación. Las pruebas usan memoria simulada, sin DDR/PCIe conectados.

## Descriptor, formato y publicación

El job registra dirección, base, longitud y época del destino junto con
los descriptores de entrada. Admite 1–32 resultados IEEE754 binary64
little-endian, ocho bytes por fila. La dirección debe estar alineada a
64 bytes; el rango debe cubrir `ceil(rows/8) * 64` bytes. El último beat
activa únicamente los bytes válidos mediante `WSTRB`: el padding y las
guardas del destino conservan su contenido.

Antes de leer entradas se comprueban rango, época, sumas de 65 bits y
ausencia de solapamiento del destino redondeado con cualquiera de las
dos regiones de entrada registradas. Ambas entradas deben permanecer
inmutables. Se mantienen los 64 bits de dirección; el banco escribe a
640 GiB + 4032 y cruza páginas de 4 KiB.

El escritor almacena primero hasta 32 filas. Comprueba orden, cantidad,
secuencia, época, `last` y valores finitos. Solo un commit correcto con
identidad coincidente habilita AW/W. También admite el último resultado
y el commit en el mismo ciclo. Un fallo de carga o cómputo no escribe
el destino. El wrapper espera las respuestas B de todas las ráfagas
emitidas antes de presentar su completion.

**El destino permanece privado hasta `done_status == 0`.** Un fallo de
escritura puede haber modificado un prefijo o todo el vector; el runtime
debe descartarlo. `done_bytes` cuenta bytes habilitados en handshakes W,
no bytes persistidos ni resultados publicables. No existe transacción
atómica de memoria ni coherencia automática entre tarjetas.

## Errores y recuperación

| Estado del escritor | Significado |
| ---: | --- |
| 0 | Vector escrito y respuestas AXI correctas |
| 1 / 2 | Descriptor inválido / época incorrecta |
| 3 | BRESP o BID incorrectos |
| 5 | Resultado, conteo o identidad inválidos |
| 7 / 8 | Cancelación / watchdog |

El estado fallido del cómputo se propaga con su significado original.
La cancelación antes del commit drena el stream hasta su completion;
después del commit no se acepta. AW y W progresan independientemente,
con una sola ráfaga pendiente. El watchdog predeterminado es de 65.536
ciclos sin progreso: retiene el fallo, mantiene señales pendientes y
drena la transacción. Un peer que nunca responde exige reset o aislamiento
coordinado; el timeout no revoca una escritura emitida.

Todas las interfaces comparten reloj. El reset debe alcanzar también al
peer AXI y descartar el destino. El banco del escritor verifica reset
durante W y esperando B; el banco integrado verifica reset durante carga
y cómputo, y su ampliación añade ambas fases de escritura. CDC, DMA host,
interrupciones y recuperación del sistema con DDR/PCIe aún requieren
integración.

## Validación y síntesis

| Tanda | Resultado |
| --- | --- |
| `20261003T050240Z-V7-result-writer-unit` | 126/126 casos, 42 por semilla 1/7/42; 49,268 s |
| `20261003T051344Z-V7-gemv-store-unit` | 105/105 casos integrados; 138,225 s |
| `20261003T052315Z-V7-gemv-store-regression` | 105/105 tras corregir declaraciones del banco; 146,923 s |
| `20261003T052315Z-V7-result-writer-netlist-r3` | 126/126 sobre netlist sintetizada; 91,859 s |
| `20261003T051836Z-V7-gemv-store-fixture` | 174 jobs / 5376 resultados calculados y escritos exactos; 544,532 s |
| `20261003T054211Z-V7-gemv-store-write-reset` | 111/111 casos; cuatro fases de reset por semilla; 158,114 s |

Los bancos comprueban tails, guardas, páginas, AW antes/después de W,
pausas, primer/segundo BRESP y BID incorrectos, watchdog en AW/W/B,
cancelación, descriptores alterados tras aceptación y recuperación.
Por semilla del banco integrado se calculan 128 valores privados y
se escriben 83; solo 26 pertenecen a jobs completados correctamente.
Los prefijos escritos en casos fallidos no cuentan como publicados.
Los revisores rechazan 10, 11 y 12 mutaciones de logs de los bancos
integrado, netlist e imágenes, respectivamente.

La ampliación conserva los 35 casos previos y añade dos resets de escritura:
después de un beat (64 bytes) y después de tres beats (136 bytes) esperando
B. Captura los conteos antes del reset, prohíbe completion para esos jobs,
comprueba interfaces inactivas tras reset coordinado y ejecuta un job de
recuperación. Por semilla calcula 162 valores privados y escribe 108;
mantiene 26 publicados. El revisor rechaza doce alteraciones, incluidas
evidencias de reset eliminadas y una respuesta B inventada. No cambió RTL
sintetizable; también se comprueba estabilidad de `done_bytes` bajo pausa.

La netlist usa el watchdog real de 65.536 ciclos, con pausas de 65.584.
No lleva SDF: es simulación funcional posterior a síntesis. Se conservan
dos intentos fallidos: `20261003T051540Z-V7-result-writer-netlist` por
una opción Tcl, y `20261003T051836Z-V7-result-writer-netlist-r2` porque
el estímulo esperaba 32 ciclos sobre el watchdog de producción. La
corrección cambia el banco; reutiliza el mismo checkpoint sintetizado.

| Síntesis OOC, VU5P, reloj 5 ns | LUT (LUTRAM) | FF | RAMB36 + RAMB18 | DSP | WNS / WHS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Escritor | 1215 (320) | 446 | 0 + 0 | 0 | +2,521 / +0,060 ns |
| Cargador + GEMV dual + escritor | 18382 (320) | 13744 | 37 + 2 | 52 | +0,979 / +0,046 ns |

Checkpoints: `20261003T050528Z-V7-result-writer-synth` y
`20261003T051836Z-V7-gemv-store-synth`. El conjunto agrega 1389 LUT y
1306 FF respecto al wrapper de lectura, sin aumentar DSP/BRAM. No hay
cajas negras ni endpoints internos sin reloj/restricción. Hay 1335
entradas y 804 salidas sin retardos externos; estas estimaciones no
cierran timing de placa.

Se retienen avisos de puertos aritméticos sin carga, recomendaciones
DSP/RAM y contexto OOC. La simulación mapeada conserva el aviso de
redefinición del módulo estándar `glbl`. El banco integrado inicial y el
fixture tienen dos avisos de declaración de `case_id/mode`; la regresión
posterior los elimina y conserva los conteos y fallos inyectados.

## Implementación fuera de contexto

`20261003T053852Z-V7-gemv-store-route-r2` coloca y rutea el checkpoint
del conjunto en 381,702 s. El informe final registra **18.239 LUT**,
**13.644 FF**, 320 LUTRAM incluidas en esas LUT, 37 RAMB36 + 2 RAMB18
y 52 DSP. Cumple las restricciones internas de 5 ns con **WNS +0,395 ns,
WHS +0,010 ns** y margen de pulso +1,958 ns, sin endpoints fallidos.

Vivado cuenta 30.741 nets enrutables completas y cero errores de ruteo,
pero también un net con huecos explícitos. El diseño conserva el contexto
provisional `HD.CLK_SRC=BUFGCE_X0Y60`, sin buffer de reloj padre, ubicaciones
de puertos ni retardos de las 1335 entradas/804 salidas. No se añadieron
excepciones, cambios de severidad ni waivers. La auditoría
`20261003T054620Z-V7-gemv-store-route-audit` terminó en 213,366 s:
los 446 segmentos jerárquicos con huecos tienen el mismo hash de ruta,
estado `PARTIAL` y corresponden al reloj padre. Los puertos carecen de
ubicación de partición. **No es cierre de timing de placa**.

DRC conserva DPIP-2 (42), DPOP-3 (20), DPOP-4 (46), RTSTAT-10 (1),
AVAL-155 (22) y AVAL-156 (10). Metodología conserva SYNTH-6 (39),
SYNTH-9 (16) y TIMING-18 (1000). El reporte CDC omite puertos sin
retardos: no acredita sus futuros cruces. Los 100 avisos Route 35-198
y cuatro DRC 23-814 permanecen en la evidencia.

El intento `20261003T053410Z-V7-gemv-store-route` se detuvo antes de
colocar: el lanzador consultaba `NAME` (nombre de sesión del checkpoint)
en lugar de `TOP` (módulo HDL). La corrección usa la propiedad observada
y el mismo checkpoint; el intento fallido se conserva.

## Pesos reales y medidas

Se reutilizan las imágenes inmutables de activaciones y pesos, 465
archivos más manifiesto, SHA-256
`ae519d84dd530d6e457bc44ae25f956e63cd2e62671c4fec808bc008ee2e4f81`.
La auditoría CPU decodifica todos los registros físicos y recalcula las
referencias Q8 con acumulación binary64 ordenada desde los pesos densos.
También se comparan los bits físicos de cada fila escrita.

Cada semilla ejecuta 54 proyecciones de Brandon y cuatro matrices
sintéticas: 32×2048, 32×8192, 1×8192 y 31×2048. Brandon conserva revisión
`ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`; las entradas de las
proyecciones son sintéticas e independientes, sin encadenar una FFN.

| Medida por semilla 1 / 7 / 42 | Resultado |
| --- | --- |
| Ciclos de los 58 jobs | 377729 / 377009 / 377262 |
| Ciclos de las 54 proyecciones Brandon | 219853 / 219453 / 219552 |
| Ciclos de 32×8192 sintético | 100806 / 100630 / 100692 |
| Lectura: bytes de activaciones / pesos | 76288 / 1016192 |
| Escritura: bytes habilitados / bytes de beats | 14336 / 14400 |
| Ráfagas de activaciones / pesos / resultados | 122 / 496 / 115 |

Los 1.106.880 bytes de beats leídos/escritos dan 2,930–2,936 bytes por
ciclo de job bajo estas pausas. Es tráfico simulado, no rendimiento DDR,
PCIe ni tokens/s. El cambio de patrón de pausas impide atribuir la resta
de tiempos frente al banco anterior exclusivamente al escritor.

## Reproducción y siguiente integración

Usar Vivado 2026.1, el MAC HLS aprobado y directorios nuevos:

```bash
vivado -mode batch -source tools/ntrpu_board_result_writer.tcl \
  -tclargs sim NUEVA_SALIDA_ESCRITOR
vivado -mode batch -source tools/ntrpu_board_gemv_store.tcl \
  -tclargs "$NTRPU_MAC_RUN" sim NUEVA_SALIDA "$NTRPU_MEMORY_FIXTURE"
vivado -mode batch -source tools/ntrpu_board_writer_netlist.tcl \
  -tclargs SALIDA_SINTESIS_ESCRITOR NUEVA_SALIDA_NETLIST
```

Omitir el fixture para el banco de fallos; cambiar `sim` por `synth`
para síntesis OOC. Cada ejecución conserva logs, revisión, fuentes y
checkpoint cuando corresponde, con respaldo verificado en
`artifacts/ntrpu-board-vivado/<run_id>/`. Los hashes de los RTL coinciden
entre síntesis y simulaciones; las fuentes del GEMV anterior permanecen
intactas.

Siguientes piezas: integrar reloj y fronteras del conjunto, registro
de buffers y publicación en el runtime, conexión DMA host y secuenciación
FFN con conversión FP32, SwiGLU, cuantización y down. La calificación
DDR/placa sigue las dependencias de [V2–V8](board/VIVADO_STATUS.md).

El nuevo [secuenciador de matrices](MATRIX_MEMORY.md) reutiliza este motor
sin modificarlo: genera grupos/tiles y mantiene privado el destino completo
hasta la completion raíz. Pasan 30 casos integrados y 5088 valores escritos
con matrices Brandon completas; la composición FFN continúa pendiente.
