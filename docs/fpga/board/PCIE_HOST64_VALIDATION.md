# DMA con direcciones de host de 64 bits

Extensión del [banco DMA aislado](PCIE_DMA_VALIDATION.md), Vivado/XSIM 2026.1,
XDMA 4.2, VU5P, Gen3 x16 y AXI de 512 bits. Estado: **DMA completo PASS**
en la tercera tanda, tras corregir dos fallos del banco público. Usa PIPE y pausas
W/R. No califica DDR, IOMMU, P2P, coherencia ni rendimiento físico.

## Fixture y criterios

Dos descriptores encadenados por sentido, 768 bytes cada uno, 1536 bytes totales.
Destino/origen AXI: intervalos consecutivos `[0,768)` y `[768,1536)`. La BRAM
del ejemplo conserva su capacidad; no se presenta como memoria de 512 GiB.

| Región | Dirección inicial del host | Bytes |
| --- | --- | ---: |
| Descriptor H2C 0 / 1 | `0x0000000100000100` / `0x0000000200000100` | 32 cada uno |
| Descriptor C2H 0 / 1 | `0x0000000300000300` / `0x0000000400000300` | 32 cada uno |
| Payload H2C 0 / 1 | `0x0000000100000fc0` / `0x0000000200000fc0` | 768 cada uno |
| Payload C2H 0 / 1 | `0x0000000300000fc0` / `0x0000000400000fc0` | 768 cada uno |

Las regiones comparten bits bajos y tienen patrones distintos: `byte(i ^
(i >> 8) ^ 0xa5)`, con `i` global de 0 a 1535. El modelo decodifica los 64 bits
antes de convertir a índice local; no basta con enmascarar la dirección. Este
fixture ejercita palabras altas 1–4, no todas las combinaciones de 64 bits.

Cada payload empieza 64 bytes antes de un límite de 4 KiB. Con MPS/MRRS de
256 bytes se exige fragmentación `64,256,256,192`, sin peticiones que crucen
la página. Se esperan ocho lecturas de payload, ocho escrituras y cuatro
lecturas de descriptor. El verificador permite prefetch de los descriptores;
exige cada dirección una vez y orden correcto del payload.

El primer descriptor no detiene la lista; el segundo tiene Stop/Completed.
Los punteros siguientes incluyen la palabra alta. Se programan `0x4084` y
`0x5084`, además de las mitades bajas, antes de activar el canal. Se exigen
conteos de dos descriptores y estados finales `6` por sentido. Formato y
registros contrastados con AMD: [descriptores](https://docs.amd.com/r/en-US/pg195-pcie-dma/Descriptors),
[dirección alta H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-SGDMA-Descriptor-High-Address-0x84)
y [dirección alta C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-SGDMA-Descriptor-High-Address-0x84?contentId=XVULdZPnb4upnaIJ5XhPhA).

## Modelo y adaptación

`ntrpu_host_memory.sv` mantiene ocho regiones acotadas, validez por byte y
contabilidad independiente de lecturas únicas y escrituras. Rechaza direcciones
desconocidas/no mapeadas, lecturas no inicializadas, escrituras en regiones
de origen y escrituras duplicadas. Al final exige 1664 bytes leídos únicos
(128 de descriptores y 1536 de payload) y 1536 bytes de destino comparados.

`patch_ntrpu_pcie_host64.py` actúa después del adaptador `multi`, siempre sobre
una copia nueva. Amplía el registro de dirección CQ y el argumento del
completador de 11 a 64 bits; sustituye 296 lecturas de bytes por accesos al
modelo. Los siete bits del campo PCIe Lower Address se conservan como tales.
Los lanes no válidos del último beat se rellenan con cero, sin leer fuera del
intervalo solicitado. La dirección, longitud y permisos de la petición se
validan antes de ese relleno. No cambia el RTL protegido de AMD.

Los datos C2H aceptados por CQ se guardan en el destino de host y se comparan
de nuevo al finalizar. El monitor AXI conserva comprobación de datos, respuestas,
direcciones y LAST por ráfaga; las compuertas comprueban estabilidad con pausas.

## Evidencia unitaria

Fuentes comunes: `source-20261002T150813Z`, SHA-256
`5917a634d0d4d8c50e1c66d09bd01715221974f13a92f4202cf0f886dcc44916`.

| Tanda | Resultado | Duración real |
| --- | --- | ---: |
| `20261002T150840Z-V4-host-memory-unit` | 1664 bytes leídos/1536 escritos; siete rechazos negativos | 64,588 s |
| `20261002T150850Z-V4-cq-host64-unit` | 44 handshakes, 16 pausas, fin 604 ns; nueve negativos | 77,214 s |
| `20261002T150919Z-V4-cq-baseline-regression` | Fixture anterior de 768 bytes, fin 264 ns; siete negativos | 63,446 s |

El banco de memoria prueba funciones sin latencia y termina a tiempo cero;
no acredita timing de bus. Sus negativos incluyen truncamiento de bits altos,
cruce de página, permisos, duplicación, byte no inicializado, rango y dirección
desconocida. CQ añade detección de alias entre regiones con bits bajos iguales.
Solo queda el aviso de entorno XSIM 43-3431 en elaboración; no se suprime.

Respaldos locales verificados, respectivamente:

- 49.932 bytes, 56 archivos: `12eb1e9f8c24f07f3f71133316a92d0959c4383e3f6a7d397196716005bbfc7c`.
- 65.722 bytes, 61 archivos: `748d085f509a5d8cdc7af34bcad807acd86bdfc1594ee6e4cd536c697c579725`.
- 60.907 bytes, 57 archivos: `1b17bb5fb95de254b20fd35511b6bd41ab9523adeefcf0fce56473cd2343269b`.

El revisor Python pasó ocho pruebas, incluyendo pérdida de marcas, alteración
del conteo de descriptores, dirección alta aliasada y partición incorrecta.

## Fallo del logger y reproducción

`20261002T151608Z-V4-host64-dma-pipe` terminó en 511,878 s reales y
108,884 µs simulados. H2C comparó 1536 bytes y completó dos descriptores;
el logger paró al interpretar la primera respuesta de descriptor C2H.
`Received an invalid frame` y `NTRPU_PCIE_MULTI_INCOMPLETE` hacen que la
revisión sea **FAIL**, aunque Vivado devolvió 0. Respaldo verificado:
110.636.571 bytes, 842 archivos, SHA-256
`a9bd5082484d5b1907e202b1bb9f37b0dd6b842885711cca43b203d056f35543`.
Revisión manual de avisos: mismos grupos del PIPE anterior, sin supresión
(`VRFC 10-3705` ×84, `10-9543` ×43, `10-5021` ×6; `10-3027`,
`Project 1-5713`, `XSIM 43-3431` y `43-4328`, uno cada uno).
Los dos diagnósticos funcionales citados mantienen el resultado FAIL.

El ejemplo público empaqueta las cabeceras RQ en la mitad baja de un registro
de 512 bits, pero las CC empiezan en el byte 0. El decodificador leía siempre
el byte 32: confundía la palabra alta de destino C2H con el tipo de paquete.
Además, la tabla de remanentes repetía entradas 0–7 y la reconstrucción CQ
usaba formato de lectura para escrituras de dirección de 64 bits.

`patch_ntrpu_pcie_logger.py` normaliza las solicitudes de control de un beat
y hasta 32 bytes, conserva CC multibeat, corrige el conteo de bytes y el formato
CQ. Rechaza explícitamente RQ fuera de ese alcance. Mantiene intactos el parser
y su rechazo de paquetes inválidos; no altera señales AXIS ni el DUT.

`20261002T153855Z-V4-pcie-logger-unit` extrae las tareas reales del ejemplo:
reproduce el tipo incorrecto 3 con el decodificador original y pasa **42
comprobaciones** corregidas. Incluye los 16 remanentes en ambas direcciones,
orden de bytes, concatenación CC y formatos RQ/CQ. Cinco negativos se rechazan.
Duración 51,881 s; API sin latencia, fin a tiempo cero. Solo aviso XSIM 43-3431.
Fuentes `source-20261002T153834Z`, SHA-256
`53078030aa669144d8f3d3b1756c593d03e8497bca053470ad46f87c3cabebe5`.
Respaldo verificado: 122.706 bytes, 59 archivos, SHA-256
`1cdb5397cee39905d9ce33875d71d62266eadcf221140e31ec4462385f640e0b`.

La repetición `20261002T154049Z-V4-host64-dma-pipe-r2` también terminó **FAIL**:
493,774 s reales, 100,616 µs simulados. Al procesar una respuesta de 64 bytes,
las dos ramas del `fork` del completador llamaron a `TSK_TX_SYNCHRONIZE`.
Sus argumentos estáticos se sobrescribieron durante la espera; el logger
procesó el cierre dos veces. El banco corto anterior cubría concatenación
pero no esta concurrencia del sincronizador. Se amplió para reproducirla y
comprobar una única invocación de cierre con la tarea declarada `automatic`.
El parser completo y el rechazo de tramas inválidas permanecen intactos.

Fuentes de r2: `source-20261002T154015Z`, SHA-256
`44292daffc147aa18e19d285c3e452d76a746efed985253836f7c319fafb3bae`.
Respaldo verificado: 109.718.400 bytes, 842 archivos, SHA-256
`20d944f21231ae638ac5ea2c24377cb72a472d6feb0df3a7cbe9ab1aa6fa2363`.

`20261002T155149Z-V4-pcie-logger-concurrent-unit` **PASS** en 51,710 s:
reproduce dos cierres con argumentos estáticos y exige uno con 76 bytes al
usar argumentos automáticos. Termina a 6 ns. Conserva las 42 comprobaciones
anteriores y cinco negativos; solo aviso XSIM 43-3431. El observador de conteo
es propio del unitario; el DMA completo mantiene el parser AMD sin cambios.
Respaldo verificado: 132.318 bytes, 61 archivos, SHA-256
`a1ddcf417eb31d8350d08b6a28b0680646b3e089d28b342552b99dce2d52125d`.

## Resultado completo

`20261002T155357Z-V4-host64-dma-pipe-r3` **PASS**, salida 0, 534,477 s reales;
fin a **116,1091 µs simulados**. Fuentes `source-20261002T155311Z`, SHA-256
`820582312a781a37221cfa874e6a4aeed9ff52ddb93d1e2f26c86d153934c45a`.

Compara 1536 bytes por sentido: dos descriptores por canal, cuatro en total,
con estados finales `6`. Observa ocho lecturas de payload y ocho escrituras,
fragmentadas `64/256/256/192` en cada región, y cuatro lecturas de descriptor.
La memoria de host confirma 1664 bytes únicos leídos y 1536 escritos/comparados.
CQ registra 44 handshakes y 24 pausas. AXI registra 24 beats W y 24 R, ocho
AW/B y dos AR; máscaras, respuestas, LAST y datos coinciden.

Las compuertas introducen **30 ciclos de espera del productor W** y **20 del
productor R**. RREADY del XDMA no introduce espera adicional (`r_stalls=0`).
Enlace a 54,0481 µs; W primero/último a 100,908/101,484 µs; R a
109,216/109,704 µs; datos C2H observados por CQ a 109,464–109,988 µs.

Incluyendo un periodo AXI de 4 ns después del último handshake, las ventanas
W/R abarcan 580/492 ns: tasas locales del fixture de 2,648/3,122 GB/s decimales.
Incluyen huecos entre sus dos descriptores; excluyen enumeración, preparación
y notificación final. **No representan caudal sostenido, DDR ni placa física.**
Preparación hasta tiempo 0: 228 s; CPU del kernel XSIM: 270,530 s.

Revisión independiente `--require-host64 --require-stalls`: PASS, cero errores.
Log SHA-256 `3bc885f48fb59b3b33b904cee1d77e71d453f352dcf8f552baabc9ce7a63f291`.
Revisión manual: sin errores ejecutados; mismos siete grupos de avisos PIPE
enumerados arriba. No se suprimen. Esto completa el campo de revisión manual
pendiente del respaldo automático, sin modificar el archivo sellado.
Respaldo local verificado: 111.419.968 bytes, 842 archivos, SHA-256
`29b3f823d6c9750479e5eeefd626e77ac6ed6ca38c26af7d2e856d4e887ae28b`.

## Reproducción

```bash
vivado -mode batch -source tools/ntrpu_board_host_memory_unit.tcl -tclargs NUEVO_UNIT
vivado -mode batch -source tools/ntrpu_board_pcie_cq_unit.tcl -tclargs NUEVO_CQ host64
vivado -mode batch -source tools/ntrpu_board_pcie_logger_unit.tcl \
  -tclargs EJEMPLO_PIPE_LIMPIO NUEVO_LOGGER
vivado -mode batch -source tools/ntrpu_board_pcie_sim.tcl \
  -tclargs EJEMPLO_PIPE_LIMPIO NUEVO_DMA host64-stall
python3 tools/check_ntrpu_pcie_multi.py LOG_DMA --require-host64 --require-stalls \
  --json REVISION_NUEVA.json
```

El PASS unitario no reemplaza el resultado completo. La ampliación siguiente
cubre retardos AW/AR/B; quedan más bits altos/rangos, interrupciones,
recuperación de reset y la integración con DDR según V2/V3.

## Pausas en los cinco canales AXI

`20261002T181538Z-V4-all-channel-gate-unit` pasó tres casos positivos
(periódico y retardos de 4/9 ciclos por oferta, 64 beats cada uno) y rechazó
tres regresiones de estabilidad de VALID/payload. Duración: 79,703 s.

`20261002T181750Z-V4-host64-all-channel-dma` pasó el DMA completo en
535,208 s reales; fin a 116,2291 µs simulados. Conserva los 1536 bytes por
sentido, cuatro descriptores, regiones altas y fragmentos 64/256/256/192.
Las compuertas retrasan cada oferta AW/AR/B en 7/11/13 ciclos; W/R mantienen
pausas periódicas. El monitor correlaciona BID con AW y WLAST, y RID con AR.

| Canal | Handshakes | Ciclos detenido el productor |
| --- | ---: | ---: |
| AW | 8 | 56 |
| W | 24 | 45 |
| B | 8 | 104 |
| AR | 2 | 22 |
| R | 24 | 21 |

El consumidor XDMA no añade pausas B/R. Se verifican ocho BID y 24 beats
RID; faltan pruebas RTL negativas específicas de esas comprobaciones.
CQ conserva 44 handshakes/24 pausas y el host compara 1536 bytes escritos.
Ventanas W/R, incluido el último ciclo: 584/544 ns, equivalentes a
2,630/2,824 GB/s locales del fixture; no son caudal sostenido ni DDR.

El revisor con `--require-host64 --require-stalls --require-all-stalls` pasa.
Los 138 avisos tienen exactamente las mismas formas y multiplicidades que
r3 al normalizar rutas: no aparecen nuevos diagnósticos. Se conservan todos.
Fuentes: `source-20261002T181504Z`, SHA-256
`e4bd71558348daaab4a67f9ff361689293426caaad379130b9651cc94b7b4939`.
Reproducir con el modo `host64-all-stall` del mismo Tcl y una carpeta nueva.
