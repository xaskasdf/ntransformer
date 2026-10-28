# Escritura y lectura de comprobación con HBM nativo

## Alcance

Continuación aislada de [la regresión del netlist](PEER_RX_HBM_NETLIST.md).
El nuevo banco instancia el IP HBM de AMD sobre VU47P, con dos stacks y
16 GB configurados. Activa lectura y escritura en AXI 00 y 16; conserva
el enlace anterior, el transporte aprobado y la cola del hilo principal.

Es una simulación del controlador con el **responder interno de XSIM**,
no una prueba en silicio. AMD documenta ese modelo y los dos archivos
`xpm_internal_config_file_sim_*.mem` necesarios en
[PG276: Simulation](https://docs.amd.com/r/en-US/pg276-axi-hbm/Simulation).
El modelo instalado contiene almacenamiento asociativo y operaciones de
lectura/escritura; se comprueba el dato recuperado, no sólo la respuesta AXI.

## Direcciones y protocolo

Se mantienen AXI3 de 256 bits, reloj de 450 MHz y direcciones de 34 bits.
En la disposición 8H, el bit 33 selecciona stack; el segundo comienza en
`0x200000000`. Los bits 32:29 seleccionan el destino dentro del stack y
los accesos se alinean a 32 bytes. Véase
[PG276: Port Descriptions](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions).

Los mismos offsets se escriben con patrones distintos en cada stack.
El banco espera la inicialización de ambos, añade 32 ciclos APB y después
habilita tráfico. No fuerza señales internas ni reinicia AXI después de
comenzar. La paridad de usuario permanece desactivada; no se califica.

## Casos aprobados

| Caso | Offset dentro de cada stack | Beats por puerto | Comprobación |
| --- | --- | ---: | --- |
| 1 | `0x1000` | 1 | Dirección antes de datos, patrón distinto por stack |
| 2 | `0x2000` | 16 | Ráfaga máxima AXI3 e IDs no nulos |
| 3 | `0x3000` | 8 | WVALID ofrecido antes de AWVALID, con pausas del productor |
| 4 | `0x2000` | 16 | Sobrescritura parcial alternando strobes; preservar los otros bytes |
| 5 | `0x1fffffe0` | 1 | Último bloque de 32 bytes del pseudo-canal local |

Cada caso ejecuta ambos puertos en paralelo. BREADY retrasa la aceptación
de cada respuesta y RREADY introduce pausas periódicas. Los monitores
exigen estabilidad de B y R bajo bloqueo, IDs/respuestas correctos y WLAST/RLAST
en el beat correspondiente. Que se ofrezca W antes de AW no obliga al
controlador a aceptarlo antes: se registra el orden real de handshakes.

Un replay Python independiente reconstruye la memoria por dirección y byte,
aplica WSTRB y compara cada lectura con lo escrito. También comprueba
causalidad, totales y ausencia de transacciones anteriores a inicialización.
Sus controles negativos eliminan eventos y alteran direcciones, datos,
strobes, IDs y RLAST.

## Resultado revisado

La tanda `20261007T013911497808Z-traffic` pasa con estado
`NATIVE_HBM_WRITE_READBACK_PASS`, en **714,869 s**. La compilación final
`20261007T012409067803Z-compile` termina en 848,260 s. Reutilizar la caché
no evitó el coste dominante de elaboración del modelo.

| Evidencia | Resultado |
| --- | ---: |
| Casos completados en ambos puertos | 5/5 |
| Ráfagas de escritura / lectura | 10 / 10 |
| Beats escritos / leídos | 84 / 84 |
| Bytes habilitados por WSTRB / leídos | 2.176 / 2.688 |
| Direcciones de byte distintas verificadas | 1.664 |
| Muestras de respuesta bloqueada B / R | 70 / 58 |
| Controles negativos de traza rechazados | 26 |

Cada ráfaga mantuvo B bloqueado durante siete flancos de muestreo.
Dos ráfagas aceptaron W antes de AW; las ocho restantes aceptaron AW
antes o simultáneamente. El replay verifica los datos de cada lectura y
de cada pausa R, además de IDs, causalidad, máscaras y contadores.
Los controles negativos alteran las trazas; no son inyecciones de fallo
en el controlador nativo ni acreditan supervisión térmica.

El tráfico se habilita a 48,657356 µs y termina a 50,010554 µs simulados.
Ambos indicadores públicos e internos de inicialización permanecen en uno.
`CATTRIP` permanece en X durante los 609 ciclos observados con tráfico:
la protección térmica sigue **sin validar**. La paridad de usuario está
desactivada. Es un pase del camino de datos del modelo, no de hardware,
integración CDC/transporte, timing ni rendimiento.

## Observación térmica y primer rechazo

La tanda `20261007T003625940402Z-traffic` terminó en 672,226 s y se
**rechazó**: `NATIVE_READY_AND_THERMAL` a 48,658467 µs, antes de aceptar
datos. Ambos `apb_complete` y los dos `init_seq_complete` estaban en uno.
Xsim devolvió código cero incluso con `$fatal`; el revisor detecta el
diagnóstico y no lo confunde con un pase.

La revisión del banco separa disponibilidad y temperatura. Exige ambos
`apb_complete` conocidos en uno y rechaza un `CATTRIP` afirmado. Registra
los valores térmicos sin forzarlos; X/Z contabiliza ciclos sin información
térmica válida y deja esa parte **sin calificar**. Este tratamiento sólo
acota la simulación de datos: no define una política aceptable para el
supervisor físico. Los estímulos y el controlador permanecen iguales.
`BENCH_DELTA.patch` conserva los cambios de monitor.

La tanda `20261007T010801997098Z-traffic` completó los cinco casos, pero
se rechazó a 50,010554 µs por `NATIVE_COUNTS_AND_STALLS`. Una auditoría
separada reconstruyó 2.176 bytes habilitados por WSTRB y comparó los
2.688 bytes leídos, sin diferencias; no convierte ese rechazo en pase.
Se confirmó `CATTRIP=X` en ambos stacks durante inicialización y tráfico,
con temperatura reportada `00`. La temperatura cero no acredita seguridad
térmica y el origen exacto del estado desconocido sigue abierto.

Un banco mínimo (`20261007T012341200124Z-hold-probe`) reprodujo el defecto
del estímulo: siete flancos negativos pueden contener sólo seis flancos
positivos con BVALID afirmado y BREADY bajo. Cinco variantes de fase
confirman que esperar siete flancos positivos antes de habilitar BREADY
garantiza siete muestras. `STALL_DELTA.patch` conserva la corrección.
El banco revisado registra cada muestra bloqueada B/R y los contadores
antes de la aserción final; el revisor comprueba cada pausa y sus datos.

## Ejecución y evidencia

La compilación y el arranque/tráfico son tandas separadas de dos CPU,
prioridad baja y límite de 900 s cada una. La segunda usa una copia del
snapshot cuya identidad se comprueba antes y después; no modifica la
validación previa ni depende de señales forzadas.

```bash
python3 -B tools/run_ntrpu_hbm_native_traffic.py /compilacion/output --stage compile
python3 -B tools/check_ntrpu_hbm_native_traffic.py /compilacion/output --review /compilacion/REVIEW.json
python3 -B tools/run_ntrpu_hbm_native_traffic.py /trafico/output --stage traffic \
  --prior /compilacion/output --review /compilacion/REVIEW.json
python3 -B tools/check_ntrpu_hbm_native_traffic.py /trafico/output --review /trafico/REVIEW.json
```

Para revisar el banco con la misma IP puede añadirse `--prior
/compilacion/output --review /compilacion/REVIEW.json` a una nueva tanda
`--stage compile`. Se copia y verifica el proyecto anterior, se regeneran
únicamente los archivos propios y se ejecutan `xvlog`/`xelab --incr`.
El donante se comprueba por hash después; no se edita en su ubicación.

Evidencia local, revisiones, hashes y diagnósticos conservados:
`C:\Users\xasko\artifacts\ntrpu-hbm-native-traffic\20261007T001942443978Z`.
Respaldo incremental SHA-256/gzip por tanda en
`D:\AI\artifacts\ntrpu-content-store`; sin ZIP de transferencia retenidos.
El ejecutable XSIM se conserva como caché regenerable en la VM.

## Integraci?n posterior

La conexi?n del transporte as?ncrono al controlador nativo est? documentada
con evidencia en [Transporte as?ncrono conectado al HBM nativo](PEER_RX_HBM_NATIVE_TRANSPORT.md).
Esa composici?n pasa cuatro casos y 24 descriptores, con lectura de
comprobaci?n de 18.752 bytes, dos dominios de reloj y un ?nico arranque.
Conserva los dos `apb_complete` reales y los destinos AXI 00/16.

El nuevo pase valida escritura, completaciones, fences y drenaje desde el
transporte. La lectura de comprobaci?n todav?a procede de un maestro del
banco; no valida un retorno de datos por CDC o PCIe. La ocupaci?n observada
es de dos bancos y no agota las reservas. La parte t?rmica sigue sin validar.

Faltan presi?n sobre las reservas, contexto f?sico completo, m?rgenes
temporales de integraci?n, PCIe/DMA, hardware y pruebas de rendimiento.
Estos bancos no acreditan 512 GB, coherencia entre NTRPUs ni saturaci?n
de sus enlaces.
