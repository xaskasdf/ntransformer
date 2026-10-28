# FFN con host y motor en relojes independientes

La [revisión con ABI2 y retiro IRQ](FFN_IRQ_INTEGRATION.md) ya está integrada
y validada con el motor real. Este documento conserva las métricas y la
corrección RRESP de la referencia ABI1.

Actualizado el 2026-10-03. **Dos perfiles funcionales aprobados; recodificación
RRESP verificada en síntesis, ruteo exploratorio y auditoría independiente**.
Desaparecen TIMING-37 y Route 35-470; los límites OOC y CDC-15 permanecen.
Se conectaron los [conversores AMD](HOST_CLOCK_INTEGRATION.md)
al [fabric host/FFN](AXI_MEMORY_FABRIC.md), conservando el motor RTL/HLS.
El iniciador y la memoria son modelos del banco; todavía no participan
XDMA, DDR ni un driver real.

## Revisión RRESP: funcionamiento y ruteo revisados

`ntrpu_axil_cdc_recode.sv` envuelve la IP AMD sin modificar su HDL ni XDC.
Aplica `{b1,b0} → {b1,b0 XOR b1}` antes del transporte y la misma operación
al recibir. Es reversible y conserva todos los códigos en los extremos:

| RRESP externo | Dentro de la IP | RRESP restaurado |
| --- | --- | --- |
| OKAY `00` | `00` | `00` |
| SLVERR `10` | `11` | `10` |
| DECERR `11` | `10` | `11` |

`01` también se conserva matemáticamente, pero no se genera como respuesta
AXI-Lite válida. El canal B y todos los handshakes permanecen directos.
La IP transporta RRESP junto con RDATA como payload del mismo handshake.
Los errores internos codificados no se exponen al host ni cambian la ABI.
Para el CSR, cuyas lecturas producen `00/11`, un bit codificado se vuelve
constante. La implementación resultante elimina el registro fuente
compartido y ambos avisos, sin añadir waivers.

`20261003T164300Z-V7-host-recode-sim`: **54 casos/24 resets PASS** en
72,928 s. Incluye 27 casos/12 resets AXI-Lite con el adaptador nuevo y
27/12 AXI4 de referencia, con tres perfiles de reloj y pausas. Se comparan
OKAY/SLVERR/DECERR; las 22.486 líneas de trazas externas `NTRPU_*` comunes,
incluidos tiempos, coinciden con el banco anterior. Las 36 mutaciones de
evidencia son rechazadas.

`20261003T164300Z-V7-host-recode-corrupt`: fallo esperado en 41,721 s al
alterar un bit de RRESP después del decodificador. XSIM detecta la primera
respuesta del caso 3, tras reset, `lite=1/sc=20/mc=16`.

Ambas tandas tienen respaldo local/NFS y todos los miembros verificados:
32.645.902 bytes,
`ab81c4f81308331bc292b00bed4b940724df346775b8059a0633e344fc0e871d`;
31.607.481 bytes,
`2a88d6e95dcbf2ff9e7fd6006cca96b40e0d6748469537d68a0141744807932e`.
Fuente `source-20261003T164156Z`, SHA-256
`b526bdab279ca7c0da4d310be15dd46110a789af865c6e7a358ff9f09605f248`.
La síntesis `20261003T164514Z-V7-ffn-recode-synth` termina en 492,116 s:
**44.194 LUT/44.819 FF/93 DSP**, 39 RAMB36 y tres RAMB18. Añade una LUT
frente a la referencia; WNS/WHS estimados +0,735/+0,013 ns. **TIMING-37
desaparece**, sin cambiar las restricciones del fabricante. Se mantienen
24 CDC-3, diez CDC-9 y un CDC-15 protegido pendiente; no equivale a cierre
de todos los cruces.

Su archivo está respaldado: 112.028.222 bytes,
`3f4e8b02c80dcc5ae1bdc322e2b2c38076943d677821ced1fafe0ce548df279d`;
checkpoint `72701168192c8bf4ab1f817f450120342f0c2f42b61811933e0a4da831002bcf`.
Las tandas `20261003T164514Z-V7-ffn-recode-{brandon,slow}` terminan en
797,301/814,207 s: **tres FFN Brandon y 768 FP32 exactos por perfil**.
Cada perfil carga 1.777.152 bytes por AXI, escribe 42.048 bytes del cálculo
y recupera 3.072 bytes finales por el host. Todas las trazas `NTRPU_*`,
incluidos ciclos e IRQ, coinciden con las respectivas referencias sin
recodificación. Las 17 mutaciones de evidencia por perfil son rechazadas.
Se verifican por hash motor RTL, HLS, fixture y conversores AMD; el wrapper
solo cambia la instancia de control por el adaptador.

Respaldos completos local/NFS: Brandon, 181.311.568 bytes,
`749891f9e59f85a4a59be4f2641239af98a670dee84d4c1a339e3879d3f54aba`;
perfil lento, 177.675.007 bytes,
`45b1441e0c58c197970f71ccb2bf8f6775b74bccf47e77ddf0c15fb6655c3b7c`.
El ruteo `20261003T165611Z-V7-ffn-recode-route` termina en 753,579 s;
su auditoría `20261003T170857Z-V7-ffn-recode-audit`, en 99,958 s.

| Medida de la revisión posrutado | Resultado |
| --- | ---: |
| LUT / FF / DSP | 44.038 / 44.607 / 93 |
| RAMB36 / RAMB18 / URAM | 39 / 3 / 0 |
| WNS / WHS / pulso | +0,068 / +0,010 / +1,468 ns |
| Redes enrutables completas / errores | 79.143 / 0 |
| Grupos bus skew / margen mínimo | 15 / +3,281 ns |

Los 33 registros fuente variables alimentan un único destino cada uno.
El registro de captura `[32]` permanece y recibe `GND/G`; `all_fanout`
no devuelve endpoints temporales para esa constante. Se conserva el aviso
de consulta Vivado 12-830, acotado a ese pin. Ya no aparecen TIMING-37 ni
Route 35-470, ni excepciones temporales ignoradas. Clocks, máximos retardos
y bus skew coinciden con síntesis. Solo desaparecen las tres excepciones
locales de reset-busy del FIFO ya identificadas en la referencia.

La auditoría conserva **dos huecos de reloj padre**, con 1.186/147 alias
PARTIAL y ninguna ruta parcial ajena a esos relojes. Faltan retardos de
1.397 entradas/1.333 salidas y ubicación de puertos. Persisten 21 CDC-3,
diez CDC-9, un CDC-15 y los avisos DSP/BRAM previos. No hay latches ni
cajas negras. El margen setup es menor que el de la referencia inicial;
esta corrección elimina una restricción ignorada, no acredita más frecuencia.
No es cierre de timing de placa ni validación DDR/PCIe.

El revisor acepta la evidencia original y rechaza doce alteraciones:
checkpoint, error de herramienta, skew ignorado, latch, excepción global,
identidad de reloj, PARTIAL ajeno, fuente compartida, constante equivocada,
captura ausente, restricción skew ausente y error de ruteo. Los dos perfiles
FFN están ligados por hashes a la misma síntesis. Checkpoint ruteado:
`8aca11d64859aefb4850beb46490373ec0f2cc746918eda87057888d0d0d0ace`.

Las siete tandas de esta revisión tienen respaldo local/NFS verificado.
Ruteo: 90.855.333 bytes,
`2322cbe719ae4b7640a891ec8e1387c51ffe24c81164091af6fa1b7c6354a988`;
auditoría: 38.846.313 bytes,
`96019a32018b43bdf02f18a670d1df25cc98ba5619170881df7b0ef88f76fe85`.
Ambos usan el snapshot `source-20261003T164156Z` identificado arriba.
En el ensayo de mutaciones, `reviewer_sha256` identifica el texto Python
ejecutado con saltos de línea normalizados; `FILES.json` conserva además
los hashes de bytes originales. El handoff incluye ambas identidades.

```bash
vivado -mode batch -source tools/ntrpu_board_host_recode.tcl \
  -tclargs sim build/RECODE_NUEVO
# sim_corrupt debe fallar en la comparación de RRESP, nunca emitir suite PASS.
```

## Implementación y contrato

`fpga/rtl/ntrpu_ffn_async_fabric.sv` contiene:

- `memory_cdc`: AXI4 asíncrono de 512 bits, dirección de 64 bits e ID de
  cuatro bits. QoS/region se fijan a cero, como en la interfaz disponible.
- `control_cdc`: AXI4-Lite asíncrono de 32 bits de datos/dirección.
- `fabric`: árbitro de dos entradas y FFN existentes, sin cambios.
- `irq_cdc`: nivel de finalización retenido en CSR, registro de origen y
  tres etapas de destino `XPM_CDC_SINGLE`. Se confirma por MMIO después
  de leer los resultados.

`host_clk/host_rst_n` corresponden al iniciador; `clk/rst_n`, al motor y
al puerto de memoria. Ambos conversores son `axi_clock_converter:2.1`,
revisión 36, `ACLK_ASYNC=1`, tres etapas configuradas. Los modelos generados,
parámetros e XDC internos coinciden con los conversores aislados calificados.

El wrapper exige reset coordinado de ambos dominios y sus pares. El banco
mantiene cada reset 64 flancos y espera antes del tráfico. **Solo se prueba
el reset inicial de esta integración**: los 24 resets con transacciones en
vuelo de los conversores aislados no califican automáticamente al conjunto.
Faltan supervisor de reloj/enlace, recuperación de época y ensayos de aborto.

## Resultados funcionales

| Tanda | Periodos host/motor | Duración de herramienta | Resultado |
| --- | --- | ---: | --- |
| `20261003T155432Z-V7-ffn-async-brandon` | 4/5 ns | 820,797 s | Tres FFN exactas |
| `20261003T155637Z-V7-ffn-async-slow` | 6,2/5 ns | 815,549 s | Tres FFN exactas |

Fase host de 0,7 ns en ambos perfiles. Cada uno carga **1.777.152 bytes**
hacia RAM inicialmente vacía, compara los **42.048 bytes** escritos por el
motor y recupera **3.072 bytes** mediante AXI host. Las **768 salidas FP32
por perfil** coinciden exactamente con la referencia Q8 independiente.
Son tres FFN de Brandon, capa 0, D=256/H=720, pesos fijados e inputs
sintéticos; no son prompts ni inferencia de un modelo completo.

El banco incluye lecturas host concurrentes con el cálculo, pausas,
programación del descriptor, polling y confirmación de IRQ. Los 37 archivos
del fixture y todos los RTL/HLS previos coinciden con la referencia de un
reloj. `check_ntrpu_ffn_async.py` reconstruye bytes, resultados, secuencia
MMIO, tiempo absoluto y flancos de ambos relojes. Rechaza **17 alteraciones
de evidencia por perfil**, incluidas métricas de reloj e IRQ incorrectas.

| Intervalo por muestra, en tiempo simulado | Host 4 ns | Host 6,2 ns |
| --- | ---: | ---: |
| Carga host | 382,0552–382,4752 µs | 483,5278–483,8658 µs |
| Programación, cálculo y polling | 1551,7048–1552,0788 µs | 1554,0872–1554,6694 µs |
| Lectura de resultados | 0,7312–0,7532 µs | 0,9416–0,9514 µs |

Los campos históricos `*_cycles` cuentan el reloj del motor, de 5 ns.
`NTRPU_ASYNC_METRICS` añade duración en ps y flancos host; no intercambiarlos.
AMD advierte que sus FIFO conductuales asíncronos **omiten retardos de
sincronización**. Estos intervalos no miden latencia física, caudal PCIe,
MTBF ni tokens/s. Permanecen registrados los avisos FIFO, XSIM y Wavedata.

## Referencia inicial: síntesis y restricciones

`20261003T155637Z-V7-ffn-async-synth`, 487,065 s, parte provisional
`xcvu5p-flva2104-2-i`, Vivado 2026.1:

| Recurso o margen | Resultado |
| --- | ---: |
| LUT, incluidas 906 LUTRAM y 244 SRL | 44.193 |
| FF / DSP | 44.819 / 93 |
| RAMB36 / RAMB18 | 39 / 3 |
| WNS / WHS estimados | +0,735 / +0,013 ns |
| Bus skew | 15 restricciones cumplidas; mínimo +3,851 ns |

Incremento respecto al fabric de un reloj: 1.047 LUT y 3.455 FF; DSP y BRAM
se conservan. No hay latches, cajas negras ni endpoints internos sin reloj
o restricción. Son estimaciones OOC, antes de colocación/ruteo.

CDC informa 24 CDC-3 y diez CDC-9 informativos, más **un CDC-15 pendiente**
en el FIFO AXI protegido. La IRQ aparece como CDC-3 de tres etapas.
Permanece **TIMING-37** en la captura de respuesta AXI-Lite R, bit 32;
su fanout debe auditarse. Se conservan las restricciones y waivers AMD,
sin excepciones globales ni waivers propios; ninguna excepción ignorada.

Faltan ambos relojes físicos y retardos de 1.403 entradas/1.333 salidas.
TIMING-18 está limitado a 1000 entradas en el listado. Persisten los mismos
avisos DSP/RAM, puertos aritméticos y alias de arrays del motor anterior.
El DRC OOC tiene cobertura de conectividad incompleta.

## Frontera pendiente con XDMA

El wrapper XDMA generado en `20261002T091826Z-V4-pcie-r2` confirma
AXI4 512/64/4 y AXI-Lite 32/32, con un par `usr_irq_req/ack` de un bit.
Su atributo de interfaz `FREQ_HZ=100000000` es metadato; el banco PCIe
midió 250 MHz. No usar ese atributo como medición del reloj.

**La IRQ de ABI1 no puede conectarse como un simple cable a XDMA.** AMD
exige conservar la solicitud hasta recibir `usr_irq_ack` y hasta que el
host atienda/limpie la interrupción. El primero confirma envío al bloque
PCIe, no ejecución del ISR.
[PG195: MSI/MSI-X](https://docs.amd.com/r/en-US/pg195-pcie-dma/MSI-and-MSI-X-Interrupts).

El adaptador siguiente deberá conservar ambas confirmaciones en cualquier
orden, impedir que una confirmación vieja retire una finalización nueva y
definir el rearme antes de admitir otro trabajo. Probar ACK de transporte
temprano/tardío, polling que limpia primero, finalizaciones consecutivas y
reset coordinado. El [transporte IRQ aislado](IRQ_TRANSPORT.md) ya tiene
45 casos y 24 resets aprobados, síntesis y auditoría. La
[ABI 2.0 opcional](FFN_IRQ_CONTROL.md) ya lo conecta al CSR con 54 casos
adicionales. Su [integración con esta FFN](FFN_IRQ_INTEGRATION.md) ya pasa
los bancos de fallos/reset y ambos perfiles Brandon; falta el modelo XDMA.
La regresión del CSR/FFN ABI1 conserva las trazas anteriores.
El [supervisor](RESET_SUPERVISOR.md) ya tiene una implementación aislada
probada con relojes detenidos; todavía falta conectarlo a los pares de esta
integración y validar su recuperación completa.

## Reproducción y respaldo de la referencia inicial

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_RUN" "$SWIGLU_RUN" sim_async_fp32 \
  build/ASYNC_RUN_NUEVO "$BRANDON_FIXTURE"
python3 tools/check_ntrpu_ffn_async.py build/ASYNC_RUN_NUEVO/seed-1.log \
  --fixture "$BRANDON_FIXTURE" --host-period-ps 4000
# sim_async_slow_fp32 usa 6200 ps; synth_async_fp32 no requiere fixture.
```

Las tres tandas están verificadas en NFS y
`artifacts/ntrpu-board-vivado/<run_id>/`, incluidos todos los miembros y
fuentes. Snapshot `source-20261003T155407Z`, SHA-256
`87201d317ba698667b279fc62630f69911a26e8e35ff376a7b2c46623ccc1f3d`.
El revisor independiente se añadió después y acompaña cada archivo funcional.

| Archivo | Bytes | SHA-256 |
| --- | ---: | --- |
| Host 4 ns | 181337381 | `97a67b5d7a2b8f33397446eccfeb39a88027cc7e4f0da5433e3e2bdaab117062` |
| Host 6,2 ns | 177699362 | `a372fb58b69f517ccde7ad8260ae4fca19c36f76c003a3a2eccf20c6b70bdf60` |
| Síntesis | 111961743 | `0ad078e747a9b368873308e8fbba38930ebdd37ea698c41c4656688bddbbd00c` |

Checkpoint sintetizado:
`0ce637aad61dea0097ad79329aae2b12706ea804e39325f6bd7ec9496d1f021c`.
V2/V3, DIMM exacto y reset DDR permanecen abiertos; este avance V7 no los cierra.

## Referencia inicial: ruteo exploratorio y auditoría

`20261003T161520Z-V7-ffn-async-route` termina en 786,281 s;
`20261003T162839Z-V7-ffn-async-route-audit`, en 98,738 s. La auditoría
abre una copia del checkpoint sin modificarlo. Se asumen ubicaciones
externas `BUFGCE_X0Y60` para el motor y `BUFGCE_X0Y61` para el host;
no se instancian ni califican los generadores de reloj de una placa.

| Medida posrutado | Resultado |
| --- | ---: |
| LUT / FF / DSP | 43.857 / 44.596 / 93 |
| RAMB36 / RAMB18 | 39 / 3 |
| WNS / WHS | +0,177 / +0,010 ns |
| Redes enrutables completas / errores de ruteo | 79.143 / 0 |
| Grupos bus skew analizados / margen mínimo reportado | 15 / +2,985 ns |

**Estos márgenes no cierran la implementación.** Hay dos redes de reloj
con huecos explícitos: 1.185 alias jerárquicos coinciden exactamente con la
ruta de `clk` y 146 con `host_clk`; ninguna otra red PARTIAL aparece.
Faltan retardos de 1.397 entradas/1.333 salidas y ubicaciones de puertos.
Los seis bits `AxPROT` de control dejan de aparecer en el listado de entradas
sin retardo después de optimización; no se les añadieron constraints.

CDC pasa de 24 a 21 CDC-3 dentro del FIFO protegido. Desaparecen tres
excepciones locales `cdc_wr_rst_busy_ic_*`; los cruces visibles, incluida
la IRQ de tres etapas, coinciden con síntesis. Permanecen diez CDC-9 y un
CDC-15. No aparecen clocks extra, latches ni endpoints internos sin
restricción. El DRC conserva los avisos DSP, y RTSTAT-10 agrupa 1.372 redes
sin cargas enrutables, en este contexto OOC con fronteras pendientes.

### Fallo de bus skew en la referencia inicial

La auditoría identifica el origen de TIMING-37: un único
`src_hsdata_ff_reg[33]/Q` alimenta los destinos `[32]/D` y `[33]/D` del
mismo handshake de respuesta R. El CSR produce OKAY=`00` o DECERR=`11`;
los dos bits iguales se fusionaron en el origen. Los 32 bits de datos
tienen un destino cada uno. Esto acota el fanout, pero no elimina el aviso.

El router emite **Route 35-470 e ignora ese grupo de bus skew durante la
optimización**. El análisis posterior todavía mide +11,413 ns para ese
grupo y la excepción de máximo retardo de 12 ns cubre sus 34 destinos.
`report_exceptions -ignored` vacío no demuestra que el router haya usado
todas las restricciones de bus skew. Se registra el estado
`EXPLORATORY_OOC_REPORTED_TIMING_MET_ROUTER_SKEW_GAP`.

Todos los clocks, máximos retardos y restricciones de bus skew originales
se conservan; no se agregaron waivers ni excepciones. Este hallazgo motivó
la recodificación reversible descrita al principio: conservar los códigos
externos, probar errores y pausas, y repetir FFN y ruteo. El resultado de
esta referencia se conserva con su advertencia original; la revisión nueva
tiene evidencia y archivos propios.

Fuente de ruteo `source-20261003T161500Z`, SHA-256
`ac8e11a062e3c58303d7d2a480b31b50012642fbdecf2332e1e37dcca62584f7`;
fuente de auditoría `source-20261003T162221Z`,
`c5d0a071c24da913416168b8b73476d055df86e50a4039245a7c8f9f1d6c6e72`.
Checkpoint ruteado:
`9e0b7c2bd07417da257ab6a99473b0e890b5be2c915ea89f83f2cc4cae92d990`.

Ambas tandas tienen respaldo local y NFS verificado, incluidos todos los
miembros. Archivo de ruteo: 90.913.560 bytes, SHA-256
`b770f78d89af35136ec3595b1ddb7b65f9085dae1c1df4101ff90057d1bcebf8`;
auditoría: 38.994.901 bytes,
`46ec2ca3b8c373f0cdd7b4a8cde524bc9ab528939249cd11309d2baee4432399`.

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_async_route.tcl \
  -tclargs "$ASYNC_SYNTH_DCP" BUFGCE_X0Y60 BUFGCE_X0Y61 build/ASYNC_ROUTE_NUEVO
vivado -mode batch -source tools/ntrpu_board_ffn_async_route_audit.tcl \
  -tclargs build/ASYNC_ROUTE_NUEVO/route_attempt.dcp build/ASYNC_AUDIT_NUEVO
```
