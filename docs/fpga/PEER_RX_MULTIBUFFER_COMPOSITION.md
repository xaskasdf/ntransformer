# Composición física del receptor multibuffer

Paso posterior: [primer controlador MIG y arranque en frío](PEER_RX_MULTIBUFFER_MIG.md).
Esa integración tiene evidencia funcional propia; los tiempos físicos de
este documento corresponden únicamente a la composición aquí descrita.

## Resultado y alcance

Experimento lateral del 2026-10-04: el receptor R4, sus colas elásticas,
dos convertidores de reloj AMD y dos filtros de cuatro créditos pasan
colocación, ruteo, auditoría CDC y regresión funcional del netlist exportado.
Se conserva íntegro el RTL de la [comparación de créditos](PEER_RX_MULTIBUFFER_CREDITS.md).
Esta variante sigue separada del diseño de placa y de la cola principal.

El pool mantiene ocho bancos de 4 KiB: dos por destino y cuatro de reserva,
32 KiB lógicos en total. Hay 16 descriptores y hasta cuatro escrituras
pendientes por destino. La entrada compartida de 512 bits a 200 MHz tiene
un techo bruto agregado de 12,8 GB/s. Añadir buffers no aumenta ese techo.
Las ventanas de dirección de 128 GiB por destino sólo comprueban límites;
no acreditan memoria instalada ni un sistema de 512 GiB.

## Implementación y tiempos

Vivado 2026.1, `xcvu5p-flva2104-2-i`, colocación `Default` y ruteo
`RuntimeOptimized`. Tres BUFGCE independientes distribuyen los relojes.
El generador del banco registra cada interfaz en su propio dominio:
811 bits en el origen y 648 en cada destino. Estos registros son una
frontera de medición, **no un adaptador AXI válido para integración**.
La regresión conecta directamente el núcleo exportado.

| Dominio | Periodo | Frecuencia | Setup después de ruteo | Hold |
| --- | ---: | ---: | ---: | ---: |
| Receptor | 5,000 ns | 200 MHz | +0,497 ns | +0,013 ns |
| Destino 0 | 4,000 ns | 250 MHz | +0,642 ns | +0,010 ns |
| Destino 1 | 4,300 ns | 232,558 MHz | +0,750 ns | +0,012 ns |

Se incluyen incertidumbres explícitas de 0,100 ns para setup y 0,050 ns
para hold. Las frecuencias de destino representan los máximos de los
perfiles previos, no relojes de una placa seleccionada. Margen mínimo de
ancho de pulso: +1,468 ns. Cero fallos de setup, hold o pulso; cero endpoints
internos sin restringir, latches, black boxes o errores DRC en los chequeos
ejecutados. Las 23.019 nets enrutables y los tres relojes están ruteados.

Recursos del núcleo, excluyendo el banco: **7.054 LUT, 17.801 registros,
15 BRAM, 238 CARRY8 y cero DSP**. De las LUT, 752 implementan memoria.
Respecto del receptor elástico R4 previamente ruteado, el conjunto añade
2.072 LUT y 5.826 registros; conserva 15 BRAM. Es una diferencia observada
entre implementaciones, no una suma universal de costes de IP.

## CDC, restricciones y reset

Cada destino usa `axi_clock_converter:2.1`, AXI4, datos de 512 bits,
direcciones de 64 bits, ID de 4 bits, modo asíncrono y tres etapas de
sincronización. Los canales de lectura están inactivos. Se verifican las
cuatro conexiones de reloj de las dos IP, con exactamente un reloj por
pin, conforme a [AMD PG059](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Asynchronous-Clock-Domain-Crossing-Constraints).

Las restricciones implementadas contienen 20 `set_max_delay -datapath_only`,
20 `set_bus_skew` y 44 false paths, todos dentro del contexto de las IP CDC.
No se añaden excepciones globales entre relojes ni hay excepciones ignoradas.
Los 20 controles de bus skew cumplen: máximo desfase observado 0,785 ns,
margen mínimo +3,215 ns. Se usa `report_bus_skew -delay_type min_max`,
además del timing ordinario, según [UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_bus_skew).

Los informes CDC normal y `-no_waiver` muestran 20 CDC-3 y 20 CDC-9,
todos informativos y asociados a los convertidores. Sus elementos cifrados
aparecen como `<hidden>`: esto no constituye una auditoría independiente de
la implementación interna AMD ni una medición de MTBF. El informe general
de waivers dice que no hay exenciones visibles, pero el XDC sí conserva
28 definiciones internas del proveedor: CDC-6 (20), CDC-10/11/15 y LUTAR-1
(dos de cada una). No se interpreta ese informe como ausencia de waivers.

El contrato sigue exigiendo reset coordinado, sincronizado con cada reloj,
con solapamiento entre extremos; no se admite reset parcial durante tráfico.
El banco mantiene cada reset durante 64 ciclos locales y espera otros 64
ciclos de origen antes de operar. Esto supera la recomendación de 16 ciclos
del reloj más lento de [PG059](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).
Un reset ocupado abandona trabajo y no revierte bytes ya escritos.

## Regresión del circuito exportado

`write_verilog -cell core -mode funcsim` exporta el núcleo ruteado con sus
primitivas y modelos protegidos AMD. Se ejecutan los tres perfiles previos,
sin SDF. La adaptación del testbench sólo cambia el nombre del módulo,
retira la parametrización del núcleo fijo y tres inyecciones jerárquicas
propias del RTL, y exige el perfil positivo de cuatro créditos.
Los generadores de tráfico y scoreboards externos permanecen iguales.

Los tres perfiles pasan 14 escenarios cada uno: **210 comandos, 207
completions, tres abandonos deliberados por reset y 443.520 bytes**.
Incluyen doce DECERR por límites y seis SLVERR inyectadas; los bytes cuentan
también el trabajo escrito antes de un abandono. Se saturan los cuatro
créditos en ambos destinos, se retienen AW/W/B/completions y se verifica
aislamiento, pausa de reloj, stop, drenaje, rechazo intercalado y recuperación
con reset coordinado. El verificador Python detecta 18 corrupciones de
trazas: datos, alias de dirección, B ausente, estado, duplicado y completion
obsoleta.

Los caudales funcionales coinciden con la referencia de cuatro créditos:
11,609566 / 8,943842 / 10,536334 GB/s. Son resultados del banco sin SDF y
con destinos sintéticos; no estiman rendimiento de PCIe, DDR4 o HBM.

## Límites y siguiente integración

El banco OOC deja 675 entradas y 1.432 salidas sin delays externos; tampoco
asigna pines, ubicaciones de partición ni fuentes físicas `HD.CLK_SRC`.
Los avisos se conservan. La metodología señala además 16 BRAM sin registro
de salida integrado, aunque cumplen los tiempos solicitados. Los avisos de
puertos internos sin uso y eliminación de lógica se revisan frente a la
configuración de sólo escritura. No hubo avisos críticos en las tandas
completadas.

La siguiente prueba debe conectar esta variante a un controlador concreto,
con su reloj, calibración, supervisor de reset y arbitraje; comprobar el
contrato de cuatro pendientes, el orden de B y el drenaje ante cierre de
admisión. Después corresponde integrarla con el endpoint PCIe y validar
P2P/BAR/DMA, visibilidad de escrituras y recuperación entre NTRPU. El banco
actual no acredita esos componentes, coherencia entre tarjetas, firmware
final, cierre de placa ni bitstream listo para hardware.

## Reproducción y respaldo

Cada tanda usa un snapshot `inputs/` y una carpeta de salida nueva:

```bash
python3 -B tools/run_ntrpu_composition.py /tanda-fisica/output --timeout 1200
python3 -B tools/check_ntrpu_composition.py physical /tanda-fisica
# Copiar el netlist a inputs/evidence/routed_core_funcsim.v en otra tanda.
python3 -B tools/run_ntrpu_composition_netlist.py /tanda-netlist/output --timeout 1200
python3 -B tools/check_ntrpu_composition.py netlist /tanda-netlist --physical /tanda-fisica
# Copiar el checkpoint a inputs/evidence/routed.dcp en otra tanda.
python3 -B tools/run_ntrpu_composition_audit.py /tanda-auditoria/output --timeout 300
python3 -B tools/check_ntrpu_composition.py audit /tanda-auditoria --physical /tanda-fisica
```

Ejecutar los runners desde las herramientas del snapshot, con Vivado cargado.
Usan dos CPU, `nice=10`, dos hilos Vivado, timeout y respaldo automático.
El resultado físico requiere también la auditoría CDC/bus skew; su primer
`REVIEW.json` conserva explícitamente ese requisito pendiente.

Evidencia local: `D:\AI\artifacts\ntrpu-multibuffer-composition\`.

| Tanda UTC | Resultado | Duración |
| --- | --- | ---: |
| `20261004T203313737117Z-composition-physical` | P&R y tiempos correctos | 576,834 s |
| `20261004T204338961909Z-composition-netlist` | Tres perfiles correctos | 169,331 s |
| `20261004T204705224336Z-composition-audit` | Fallo del script al capturar ayuda Tcl | Ver manifiesto |
| `20261004T204828943227Z-composition-audit` | Auditoría corregida completa | 48,571 s |

El intento fallido se preserva; no modificó el checkpoint. El verificador
compara las filas CDC por dominio y contenido, conservando multiplicidades:
Vivado puede reordenar filas equivalentes al reabrir el DCP.
`FINAL_REVIEW.json`, `SOURCE_PRESERVATION.json` y `HANDOFF.md` registran
resultados, hashes y recuperación. `20261004-composition-handoff.zip`
incluye fuentes, scripts, checkpoints y las cuatro tandas; su archivo
`.VERIFIED.json` registra CRC y SHA-256. Se preservan `AGENTS.md`, el RTL
original, los testbenches anteriores y el trabajo del hilo principal.
