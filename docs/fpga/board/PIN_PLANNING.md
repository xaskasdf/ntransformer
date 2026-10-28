# Exploración del pinout DDR4

Estado 2026-10-02: candidato genérico, **sin aprobación de PHY, timing ni placa**. Parte provisional `xcvu5p-flva2104-2-i`, Vivado 2026.1, DDR4 2.2 rev30; dos slots, cuatro ranks x4, ECC y 128 GiB útiles a 1600 MT/s.

## Síntesis y distribución generada

`20261002T112130Z-V3-generic-pin-synth` sintetiza el controlador y el ejemplo completo, con tráfico/calibración: 30.556 LUT, 34.675 registros, 26,5 BRAM, 3 DSP y 154 IOB. No incluye cómputo NTRPU ni representa utilización posrutado. El checkpoint `output/synthesized.dcp` conserva las asignaciones originales antes del intento de recolocación.

| Banco original | Puertos | Alternativa trasladada |
| --- | ---: | --- |
| 44 | 46 | 64 |
| 45 | 48 | 65 |
| 46 | 48 | 66 |
| 47 | 12 | 67 |

Los 154 puertos incluyen 72 DQ, 36 DQS y señales de dirección/control, reloj, reset y estado. El ejemplo consume cuatro bancos; la reserva de tres en `io_plan.json` sigue sin validar. `place_ports -iobank` rechaza memoria con 12-5364: ese rechazo no demuestra que tres bancos sean imposibles. Revisar con el planificador específico y las [reglas DDR4 de AMD](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-Pin-Rules), incluidas las restricciones de dirección/control y nibbles x4.

## Alternativa en 64–67 y PERST

`20261002T113237Z-V3-bank64-remap` usa `tools/ntrpu_board_ddr_bank_remap.tcl` para preservar coordenadas de byte/nibble y capacidad GC al trasladar las bolas. `mapping.tsv` contiene 154 puertos y bolas únicos. **AM17/PERSTN0 queda libre; AP17/PERSTN1 lo ocupa `c0_ddr4_dq[25]`**. No se ha integrado PCIe ni aceptado eléctricamente PERST; sigue pendiente la traducción del reset del host al banco de 1,2 V.

El DRC inicial emite 48 `DCIRST-1` y cuatro `REQP-1877`. La tanda `20261002T121632Z-V3-dci-integrated-synth` incorpora el [secuenciador DCI](DCI_STARTUP.md), probado con la primitiva instalada y reloj independiente de UI: desaparecen los 48 `DCIRST-1`, permanecen los cuatro `REQP-1877`. Incluye pulso de 256 ns, espera de LOCKED y liberación sincronizada de MIG. La síntesis integrada usa 30.646 LUT y 34.722 registros; mantiene BRAM/DSP y AM17 libre. Falta cerrar implementación y revisar restricciones.

`20261002T122611Z-V3-ddr-place` conserva el rechazo de colocación 30-675/30-716: GCIO en banco 64 y MMCM de MIG en 66. La revisión aplica BACKBONE al net global de referencia MIG para utilizar la columna de reloj dedicada. La disponibilidad documental de bolas no sustituye esa validación física.

`20261002T123715Z-V3-ddr-place-backbone` sí completa colocación: 28.805 LUT, 34.962 registros CLB, 26,5 BRAM y 3 DSP. AM17 permanece libre. El reporte previo al ruteo presenta WNS +0,553 ns y WHS −0,277 ns con 499 endpoints de hold incumplidos. DRC: ruteo incompleto, tres avisos PDCN-1569 del debug hub y cuatro buffers sin carga; sin DCIRST-1. No constituye cierre de timing.

## Auditoría de restricciones del checkpoint

`20261002T124830Z-V3-constraint-audit` confirma que las consultas RIU originales y una consulta jerárquica independiente seleccionan los mismos 150 pines de dirección y 400 de datos. `report_exceptions -coverage` muestra cobertura del 100% para ambas excepciones de hold y ninguna aparece ignorada. El diagnóstico de síntesis sigue sin explicación, pero no implica que falten estas excepciones en el checkpoint colocado.

Cuatro restricciones de bus skew del debug hub tienen holgura positiva antes de ruteo (mínima 19,596 ns). CDC clasifica como seguros 1551 endpoints con relojes definidos, incluidos tres cubiertos por excepciones existentes; omite puertos sin retardo y no cierra la revisión completa. **Corrección del `result.md` de esa tanda:** el total escrito allí como 1549 es un error aritmético; el reporte original preservado suma 1551.

Faltan retardos/contratos para `sys_rst`, `c0_data_compare_error`, `c0_ddr4_reset_n` y `c0_init_calib_complete`. La metodología conserva siete LUTAR-1, incluido el combinador de reset nuevo. No se inventan retardos de PCB ni se amplían excepciones para hacer pasar el reporte.

## Criterios antes de elegir bancos

`20261002T125230Z-V3-ddr-route` completa las **70.578 nets enrutables**, sin errores. Tras ruteo: WNS +0,334 ns, WHS +0,011 ns y ancho de pulso +0,123 ns; cero endpoints incumplidos bajo las restricciones actuales. Las cuatro restricciones de bus skew tienen holgura positiva (mínima 19,558 ns). Recursos iguales a colocación. El DRC conserva tres PDCN-1569, cuatro REQP-1877 y un RTSTAT-10 por nets sin cargas enrutables; no presenta DCIRST-1. No se genera bitstream.

Explicar los mensajes `extra characters after close-brace` de síntesis, conservando las excepciones RIU cuya cobertura ya se comprobó. Completar los cuatro contratos de E/S, revisar reset y calificar DIMM antes de aceptar timing de placa. Actualizar el manifiesto solo con una decisión explícita respaldada. El ruteo genérico no cierra V3–V6 ni autoriza fabricación.

La [prueba conjunta de netlists DDR/PCIe](INTERFACE_COEXISTENCE.md) conserva
221 bolas únicas y confirma BIVC-1=0/DCIRST-1=0 con PERST AM17 LVCMOS12.
El circuito de adaptación a 1,2 V sigue pendiente. La colocación conjunta de
un canal genérico terminó, con WNS −0,811 ns y WHS −0,581 ns antes de ruteo.
El ruteo conjunto completó 177.212 nets sin errores y mejoró esas holguras a
−0,288 ns / +0,010 ns; setup sigue incumplido.
No sustituye la revisión del DIMM ni el diseño de cuatro canales.
