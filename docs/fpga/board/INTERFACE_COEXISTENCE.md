# Coexistencia de un canal DDR4 y PCIe

Estado 2026-10-02: **PASS de conectividad y DRC de tensión/DCI sobre netlists
reales**; colocación y ruteo completos, con setup/hold positivos tras optimizar
antes de rutear. La revisión de metodología sigue abierta. Es una exploración V4 con memoria
genérica, sin conexión DMA→DDR. V2/V3 siguen bloqueados por la calificación del
DIMM y el fallo de reinicio; no inicia V5/V6 de cuatro canales.

## Entradas y reproducción

Parte provisional `xcvu5p-flva2104-2-i`, Vivado 2026.1 SW 6511674/IP 6504888.
Se usan el canal DDR4 ruteado en bancos 64–67 de
`20261002T125230Z-V3-ddr-route` y el endpoint serial Gen3 x16 de
`20261002T160051Z-V4-pcie-lane-synth-r2`. Los checkpoints originales permanecen
intactos. Los comandos exactos y las rutas están en cada `invocation.json`.

1. `tools/ntrpu_board_interface_inventory.tcl DDR_DCP PCIE_DCP NEW_DIRECTORY`
   exporta stubs nativos, propiedades de puertos, primitivas, XDC y reportes.
2. `tools/ntrpu_board_interface_coexist.tcl INVENTORY DDR_DCP PCIE_DCP NEW_DIRECTORY`
   genera un wrapper con `prepare_ntrpu_interface_top.py` e importa ambos DCP
   bajo `u_ddr` y `u_pcie`. Reaplica las propiedades válidas al top.
3. `tools/check_ntrpu_interface_coexist.py INVENTORY OUTPUT CONSOLE --json REVIEW`
   coteja puertos, propiedades, relojes y resultados de los DRC específicos.
4. `tools/ntrpu_board_interface_place.tcl COMBINED_1V2_DCP NEW_DIRECTORY`
   elimina el ruteo heredado solo en una copia nueva, optimiza, coloca y guarda
   reportes/checkpoint incluso si la colocación es rechazada. No genera bitstream.

El inventario `20261002T162036Z-V4-interface-checkpoint-inventory` termina con
salida 0 en 97,111 s. Sus 154 puertos DDR y 67 PCIe ocupan **221 bolas únicas**.
Los avisos por consultar propiedades no aplicables a GT se conservan; sus
campos vacíos no se convierten en constraints. `report_io` muestra SLOW por
defecto en TX GT, aunque esa propiedad no esté disponible mediante Tcl.

## Resultado estructural conjunto

`20261002T162725Z-V4-ddr-pcie-coexist`: salida 0, 204,624 s. Fuente
`source-20261002T162653Z.zip`, SHA-256
`916a7ca5b063cff8a1a20551f253487308951c7ae64337715d7b72aab20799c4`.

La netlist final tiene cero blackboxes, una DCIRESET, una STARTUPE3, una
PCIE40E4 y 16 GTYE4_CHANNEL. Conserva BACKBONE en la referencia DDR. Los relojes
primarios siguen siendo 250 MHz DDR y 100 MHz PCIe; no significa que todos
los dominios internos trabajen a esas frecuencias.

| Configuración PERST en AM17, banco 65 | BIVC-1 | DCIRST-1 |
| --- | ---: | ---: |
| LVCMOS18 original, DDR a 1,2 V | 1 | No ejecutado en esa variante |
| LVCMOS12, DDR a 1,2 V | 0 | 0 |

Se guardan `combined_original_voltage.dcp` y `combined_1v2.dcp`. El único
cambio de E/S es IOSTANDARD de PERST; se conserva PULLUP. El circuito de placa
debe adaptar el reset del host a 1,2 V y satisfacer su secuencia eléctrica.
Esta prueba no califica un traductor ni demuestra enumeración en hardware.

El revisor confirma las 221 bolas, direcciones y propiedades contra el inventario
y los reportes originales. Rechaza siete alteraciones deliberadas: bola,
dirección, tensión, puerto ausente, reloj, conteo DCI y marca de terminación.
Los ejemplos siguen independientes: DDR conserva su generador de tráfico y
XDMA utiliza BRAM. No se ha probado aquí transporte entre ambos bloques.

Recursos conjuntos medidos: **94.668 LUT, 93.615 registros, 231,5 BRAM y 3 DSP**.
No se obtienen sumando reportes aislados: la representación/packing de los
checkpoints importados puede diferir.

## Avisos y límites

El DRC completo tiene 46 hallazgos: RTSTAT-13 ×1 por ruteo incompleto,
PDCN-1569 ×3 del debug DDR, REQP-1858 ×22 de BRAM y REQP-1877 ×20 de buffers
sin carga. Solo los chequeos específicos de tensión y DCI están aprobados.

Las tres advertencias críticas Project 1-486 aparecen al crear el esqueleto
antes de importar los DCP; la comprobación posterior exige cero blackboxes.
Se conservan advertencias de referencias Tcl invalidadas, DRIVE/SLEW internos
y waivers heredados; los objetos se vuelven a consultar y los puertos se
cotejan independientemente. No se agregan waivers ni se degradan severidades.

## Colocación y ruteo

`20261002T164022Z-V4-ddr-pcie-place` completa colocación en **864,280 s**,
salida 0. Usa una copia del DCP a 1,2 V, cuatro hilos y presupuesto de 1800 s.
Fuente `source-20261002T163956Z.zip`, SHA-256
`0dc5453806b138c5d9e904559d0139237c9339400458b160e51f3b0ae0716a00`.
Las propiedades de los 221 puertos coinciden exactamente con el reporte previo.
Recursos tras optimizar/colocar: **84.265 LUT, 90.076 registros, 231,5 BRAM,
3 DSP, 16 GT y un PCIe**.

| Métrica antes de rutear | Resultado |
| --- | ---: |
| WNS / TNS | −0,811 / −34,532 ns |
| Endpoints que incumplen setup | 74 |
| WHS / THS | −0,581 / −587,012 ns |
| Endpoints que incumplen hold | 5268 |
| WPWS / fallos de pulso | 0,000 ns / 0 |

La peor ruta de setup pertenece a XDMA a 250 MHz: `tlp_stg4_vld_reg_rep__2`
→ `tag_did_conti_val_reg_0_255_5_5`, con nueve niveles lógicos y 4,780 ns de
camino estimado. No es una ruta entre DDR y PCIe, todavía independientes.
El ruteo `20261002T165523Z-V4-ddr-pcie-route` termina con salida 0 en
**654,465 s**, mediante `tools/ntrpu_board_interface_route.tcl`. Las **177.212
nets enrutables** están completas, sin errores; conserva las propiedades de
los 221 puertos. Fuente `source-20261002T164956Z.zip`, SHA-256
`2e463d82b3a6ae692177ee9c260ad8943018a7ca5061ba63e184014d7ad45cb3`.

| Métrica después de rutear | Resultado |
| --- | ---: |
| WNS / TNS | −0,288 / −44,569 ns |
| Endpoints que incumplen setup | 633 |
| WHS / fallos de hold | +0,010 ns / 0 |
| WPWS / fallos de pulso | 0,000 ns / 0 |

Resuelve hold y reduce la peor violación de setup; aumenta el número de
endpoints con pequeñas violaciones de setup. **Todavía no cumple timing**.
El peor camino sigue dentro de XDMA a 250 MHz, ahora con once niveles lógicos.
Recursos iguales a colocación. Las cuatro holguras de bus skew son positivas,
con mínimo 19,511 ns.

DRC tras colocar: RTSTAT-13 ×1, PDCN-1569 ×3, REQP-1858 ×22 y REQP-1877 ×4.
Metodología conserva 14 LUTAR-1, un TIMING-9, 48 TIMING-16 y un TIMING-18.
CDC clasifica 5247 endpoints como seguros dentro de su alcance, que excluye
puertos sin retardos y usa excepciones heredadas; no resuelve por sí solo
TIMING-9. `check_timing` no identifica endpoints internos sin constraints ni
registros sin reloj. Faltan retardos en dos entradas —una con false path—
y tres salidas.

Se retienen siete Netlist 29-47 por DONT_TOUCH del PHY DDR, dos Constraints
18-550, un Place 30-1021 y avisos de bus skew/múltiples relojes. No se altera
el PHY protegido ni se ocultan esos diagnósticos.

Después de rutear desaparece RTSTAT-13 y aparece RTSTAT-10 por nets sin cargas
enrutables; los otros 29 avisos DRC se mantienen. Metodología conserva 14
LUTAR-1, un TIMING-9, 52 TIMING-16 y un TIMING-18. Los 550 endpoints RIU
mantienen cobertura del 100%. Hay 20 excepciones heredadas ignoradas: 17 por
caminos inexistentes, una por endpoint inválido y dos reemplazadas por otras.
El respaldo conserva el detalle; la revisión global de constraints sigue abierta.

`20261002T170751Z-V4-ddr-pcie-postroute-physopt` completa una pasada
`phys_opt_design -directive Explore` en otra copia: salida 0, **425,872 s**.
No cambia frecuencias, excepciones ni retardos. AMD documenta esta optimización
sobre diseños colocados o ruteados en
[UG835 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/phys_opt_design).
Fuente `source-20261002T170350Z.zip`, SHA-256
`b15b51de395c49128a41ec002a93d9f6c2c983629ccf2236e77c63215e423e25`.

| Métrica después de optimizar | Resultado |
| --- | ---: |
| WNS / TNS | −0,215 / −39,289 ns |
| Endpoints que incumplen setup | 612 |
| WHS / fallos de hold | +0,001 ns / 0 |
| WPWS / fallos de pulso | 0,000 ns / 0 |

Las 177.213 nets enrutables están completas, sin errores; conserva las 221
E/S y añade una LUT (84.266 en total). Mejora setup, pero reduce el margen de
hold y **no cierra timing**. El peor camino sigue en XDMA a 250 MHz, con
nueve niveles y 4,378 ns de retardo. DRC, metodología, CDC y contratos externos
conservan los pendientes anteriores. Los
[pendientes de DDR](MEMORY_QUALIFICATION.md) y
[contratos de E/S](PIN_PLANNING.md) siguen vigentes.

## Optimización antes del ruteo

`20261002T172152Z-V4-ddr-pcie-preroute-physopt` termina con salida 0 en
**846,522 s**. Parte de
`20261002T164022Z-V4-ddr-pcie-place/output/placement_attempt.dcp`, aplica
`phys_opt_design -directive Explore` y después `route_design`. El modo
`preroute-explore` de `tools/ntrpu_board_interface_route.tcl` conserva checkpoint
y reporte intermedios. Cuatro hilos, presupuesto de 1800 s; fuentes del snapshot
`source-20261002T171957Z.zip`, SHA-256
`9a551b4a4aa3d4ec687db62381dcdf3169b96893d6bc29e3bdc758017505ac33`.

| Métrica final | Resultado |
| --- | ---: |
| WNS / TNS | +0,006 / 0 ns |
| Endpoints que incumplen setup | 0 de 243.595 |
| WHS / THS | +0,004 / 0 ns |
| Endpoints que incumplen hold | 0 de 242.821 |
| WPWS / fallos de pulso | 0 ns / 0 de 100.854 |
| Nets enrutables completas / errores | 177.381 / 0 |

Conserva las 221 E/S, frecuencias y constraints; BIVC/DCI=0. Recursos:
84.311 LUT, 90.127 registros, 231,5 BRAM y 3 DSP. Las cuatro holguras de
bus skew son positivas, mínimo 19,277 ns. **Cumple el resumen temporal actual
con márgenes muy pequeños**. El aviso Route 35-328 corresponde a una estimación
intermedia; el reporte final queda separado. Persisten 30 avisos DRC y los
contratos externos pendientes. No inicia los cuatro canales.

## Control PCIe y auditoría temporal

El endpoint aislado `20261002T172439Z-V4-pcie-isolated-implementation` completa
optimización, colocación y ruteo en **802,651 s**, salida 0. Solo adapta PERST
a LVCMOS12; conserva las otras propiedades de sus 67 puertos. Tiene 106.680
nets enrutables completas, cero errores, WNS +0,031 ns y WHS +0,010 ns,
sin fallos de setup/hold/pulso. Recursos: 55.383 LUT, 55.137 registros y
205 BRAM; DRC REQP-1858 ×22. Metodología: LUTAR-1 ×7, TIMING-9 ×1 y
XDCC-5 ×1 por el cambio explícito de IOSTANDARD. No presenta TIMING-16.

El conjunto mantiene **55 TIMING-16** pese al resumen positivo. La auditoría
de solo lectura `20261002T174842Z-V4-joint-timing-audit` reabre el checkpoint
y reproduce los avisos antes y después de reportar timing: salida 0, 206,293 s.
TIMING-16#1 informa −1,036 ns; la consulta directa de la misma pareja
CORECLK→`clk_msi_req_reg[0]/D` da **+0,964 ns**, requisito 4 ns y excepción
heredada `MultiCycle Path Setup -start 2`. La diferencia coincide con un ciclo
de origen de 2 ns. Esta observación motiva revisar las 55 parejas y el XDC
de origen.

`20261002T175745Z-V4-multicycle-path-audit` amplía la comprobación a las
**55 parejas**, salida 0 en 151,962 s: todas tienen slack activo entre
**+0,293 y +0,964 ns**, requisito de 4 ns y setup multicycle `-start 2`.
Todas difieren exactamente 2 ns del aviso. Al endurecer **una sola pareja**
a `-start 1` en memoria se reproduce −1,036 ns con requisito 2 ns. No se
guarda un checkpoint alterado; el XDC de ese diagnóstico no es una propuesta
de implementación. El XDC original AMD conserva setup 2/hold 1 en esas
salidas PCIe y se archiva como procedencia, SHA-256
`a52495c958e8dce31b8ce89f0b2c5b0166962b3c46624046b42a33639403cd83`.

Esto acota la discrepancia al ciclo adicional de la excepción heredada;
no demuestra un defecto interno concreto de Vivado ni permite omitir la
revisión funcional de constraints. Los avisos quedan conservados, sin waiver.

CDC detallado conserva 5247 endpoints clasificados seguros, pero excluye puertos
sin retardos y no resuelve por sí solo TIMING-9. No se añaden waivers ni
excepciones para aprobar el diseño. Falta calificar memoria, reset y placa.

## Evidencia y respaldo

Inventario respaldado localmente: 21 archivos, comprimido de 153.324 bytes,
SHA-256 `a413bfdbcb9f52afb60deed94476051b6a893bc1eaf2e4a31c09562df81d1fd7`.
Coexistencia respaldada: 35 archivos, 215.204.426 bytes, SHA-256
`bf5b3a7891ae80ac9199b98a21ecb749cf89fe0f2747f1a313f8417aa3595ea3`;
verificación local de comprimido, miembros y fuentes completada a las 16:48 UTC.
Colocación respaldada: 32 archivos, 134.544.170 bytes, SHA-256
`becbfe9aab09f937a172a844c5083c663a6a11aeff736a729b509f3217e641d1`;
verificación local completada a las 17:01 UTC.
Ruteo respaldado: 29 archivos, 124.208.472 bytes, SHA-256
`274f25f99c0138c91225a86de5ccb87e2c33ccba57d301fb49dc43acb86ba31f`;
verificación local completada a las 17:11 UTC.
Optimización posterior respaldada: 30 archivos, 142.335.088 bytes, SHA-256
`c421fc6e6e9ca7d27b6ef3509fc906aa75e49d9d1d284e7d8e74f60847f6b339`;
verificación local completada a las 17:19 UTC.
Optimización previa respaldada: 33 archivos, 177.988.655 bytes, SHA-256
`7edcedc9fc645620e2c386955819eb21ffc12dbf0daa42ad971773c95c11b880`,
verificada a las 17:50 UTC. PCIe aislado: 43 archivos, 188.335.633 bytes,
SHA-256 `5d6b8b4b6890a87290fbdd8609282a89fc08c41cdbf1798163140cd8c50a0499`,
verificado a las 17:52 UTC. Auditoría temporal: 90 archivos, 229.777 bytes,
SHA-256 `d4cd158b612b75844c8cc82a05f491eb1fda9b94ecfe375cb4ec872cf0b61b99`,
verificada a las 17:57 UTC.
Auditoría de las 55 parejas: 69 archivos, 101.129 bytes, SHA-256
`8f8825376a031c0fced29e034b174a9544c518752489f4c5f38bd1a35f7846c7`,
verificada a las 18:04 UTC.
Las fuentes, resultados, checkpoints y reportes de cada ejecución se conservan
en `artifacts/ntrpu-board-vivado/<run_id>/`, mediante junction a D:.
`VERIFIED.json` acredita el hash del comprimido, todos los miembros y las fuentes;
el [estado de la cola](VIVADO_STATUS.md) distingue pruebas en curso de respaldos
completos.
