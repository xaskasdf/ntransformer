# Síntesis y netlist del inventario PCIe

## Resultado del 2026-10-07

R5 reduce la utilización a **46.359 LUT** manteniendo 9.524 registros.
Pasó la regresión RTL y el netlist de 52 escenarios binarios, tras
investigar dos avisos intermedios de bucles. El DCP reabierto confirma
cero excepciones temporales válidas o ignoradas.
**Todavía no cumple timing a 250 MHz** y no autoriza integración física
ni drenaje global. La tabla conserva las medidas anteriores.

| Medición | R2 | R3 | R4 |
| --- | ---: | ---: | ---: |
| LUT de lógica, ajustadas por combinación | 63.050 | 61.883 | 58.225 |
| Registros | 9.524 | 9.524 | 9.524 |
| Latches / BRAM / URAM / DSP | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| CARRY8 | 2.199 | 1.984 | 1.986 |
| Período de caracterización | 4 ns | 4 ns | 4 ns |
| WNS / WHS con I/O completos | −17,501 / −0,075 ns | −12,597 / −0,075 ns | −11,964 / −0,075 ns |
| Endpoints que fallan setup / hold | 15.758 / 1.366 | 15.755 / 1.368 | 15.749 / 1.369 |
| Endpoints analizados por setup y hold | 27.644 | 27.642 | 27.387 |
| Camino crítico de setup | 21,483 ns, 88 niveles | 16,579 ns, 68 niveles | 15,974 ns, 65 niveles |

El camino crítico termina en `fault_sources_reg[13]` y parte de
`allocated_tags_reg[67]` en R2 o `[11]` en R3. En R3 incluye 4,082 ns
de lógica y 12,497 ns de interconexión estimada sin colocación/ruteo.
En R4 parte de `rc_user[94]`: 3,867 ns de lógica y 12,107 ns de interconexión.
No representa una medida física de la placa.

## Tandas y evidencia

| ID abreviado | Alcance | Duración | Archivo D:/NFS SHA-256 |
| --- | --- | ---: | --- |
| `20261007T132006Z`, síntesis R2 | Mapeo completo; defecto en XDC de entrada identificado | 633,921 s | `35990227269242c9a9fa7dda355991c0285035c5576af5fb03194d807704f881` |
| `20261007T133643Z`, timing R1 | Restricciones completas sobre copia del DCP | 175,484 s | `4830626ccbb248e479de144819ca22ce7ee8c5123e9b75fe45cbcff0bdba8629` |
| `20261007T134048Z`, netlist R1 | 21.933 ciclos, 42 escenarios, revisión funcional aprobada | 146,158 s | `4ed7930d9eae895750f44ef189196375b96cfca64d152d15c92fd1fd0421dfd6` |
| `20261007T134901Z`, banco R7 | Diagnóstico de preparación: nombre de testbench inexistente; no ejecutó RTL | 14,593 s | `55dd14ae41f11f58ea830df338135e4629fdb5e1f00091533b61b3762a4cb63a` |
| `20261007T135640Z`, banco R8 | RTL R3: 21.942 ciclos, 43 escenarios, revisión aprobada | 35,374 s | `a74254568ef0c2f395d401f64fd847cb707211572bbd265f5bb92298e0f325db` |
| `20261007T135946Z`, síntesis R3 | Mapeo completo; cobertura XDC completa; falla timing | 538,759 s | `176105db421d59570a0bfdeb2631343b47eafe87f5ad9f77711c5dce5752d382` |
| `20261007T141051Z`, banco R9 | RTL R3: 22.160 ciclos, 46 escenarios, revisión aprobada | 36,084 s | `2ca7b68b4dc40cfcf6296d52f342b838021166eaaaf68081fd1c62af78613a31` |
| `20261007T141227Z`, netlist R2 | RTL R3 mapeado: 21.933 ciclos/42 escenarios, revisión aprobada | 133,569 s | `e708897135e4cbef866a2c243d742aa4b5dd2d1ee989c3c5d45fc723ad16db67` |
| `20261007T141631Z`, banco R10 | RTL R4: 22.160 ciclos/46 escenarios, revisión aprobada | 37,587 s | `b61d8117857f1f2bca9e3ba22d71b5d57b2c971bcb9452744b39acfad3568920` |
| `20261007T142016Z`, síntesis R4 | Mapeo completo; cobertura XDC completa; falla timing | 738,390 s | `b814e6d61962a558158e49627cc0807266ba507433567158fd1ba7765c0c48f3` |
| `20261007T143719Z`, netlist R3 | RTL R4 mapeado: 22.151 ciclos/45 escenarios, revisión aprobada | 128,931 s | `0e113cbc32c6a8784c1b337b0fbffa00a3c54941157ab3614e769c938618b4e3` |
| `20261007T144059Z`, banco R11 | RTL R4: 22.219 ciclos/53 escenarios, revisión aprobada | 37,479 s | `6e3706028131b9b89522ee1407e8f8d853abab31999d6c93f40666a94959ec89` |
| `20261007T144524Z`, banco R12 | RTL R5: mismo banco, revisión aprobada | 37,338 s | `3fcaba0d6c4fe9a61ab5f72055d8f1f5341734f87304d642bad8c7d62c8c76f5` |
| `20261007T145153Z`, síntesis R5 | Mapeo completo; avisos intermedios investigados; falla timing | 1.289,854 s | `b35c78828e71d4e94c49d150c0e40becc75e8b84bed069dfe36640126fe01253` |

Las revisiones independientes ligan informes, logs, fuentes y artefactos
al paquete de cada ejecución. El primer netlist R2 probado tiene SHA-256
`4cc14c7af84e883a1cccfd20eee9e565607bcde22d13418b508919c21fffa5cd`.
Su revisión tiene SHA-256
`e5e2b57336c5c4fd023a413be55c28c28c8927ef751eb8349e6c44e4fcb0ecee`.

## Corrección de restricciones

El lector XDC rechazó `remove_from_collection`; la síntesis original dejó
1.351 entradas sin retardo. Se conservó ese diagnóstico y se abrió una
copia idéntica del DCP para leer un XDC con
`get_ports -filter {DIRECTION == IN && NAME != clk}`. Esta consulta pertenece
a los [comandos admitidos en XDC por AMD](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Valid-Commands-in-an-XDC-File).

La revisión corregida conserva el inventario de primitivas y el reloj de
4 ns, con retardos mínimos/máximos de 0 ns sobre 1.351 entradas y 412
salidas. Los doce grupos de `check_timing` dan cero, incluidas entradas y
salidas sin retardo, endpoints internos sin restricciones y bucles.
Esto verifica cobertura, **no cumplimiento temporal**: setup y hold fallan.

Se mantienen los avisos de origen de reloj OOC sin `HD.CLK_SRC`, DRC de
conectividad incompleto, 1.000 avisos de setup y 1.763 de min/max I/O
iguales. Los retardos de placa y el reloj del diseño padre siguen pendientes.

## Cobertura funcional y siguiente cambio

El banco de netlist reproduce todos los estados binarios de la referencia:
256 tags, 257 lecturas, 1.052.672 bytes pendientes, cuatro actualizaciones
RC, reutilización simultánea con RQ2 y 21 escenarios de fallo. Reproduce
3.778 ciclos de una captura previa; no ejecuta un endpoint nativo nuevo.
La revisión rechazó nueve corrupciones de evidencia.

Se excluye explícitamente el único caso de detección X/Z y se concretan
619 bits que la referencia declaró irrelevantes. El banco espera 200 ns
para el arranque UNISIM y muestrea 1 ns después del flanco. Los modelos
instalados declaran GSR de 100 ns y retardo de salida FDRE de 100 ps;
no se fuerzan ni modifican. Es una simulación funcional sin SDF.

R3 (`fpga/rtl/ntrpu_pcie_request_inventory_r3.sv`) agrega los incrementos/decrementos de cada beat mediante sumas
agrupadas y actualiza cada contador una vez. Conserva el orden RC→RQ,
la liberación de tag en SOP, la obligación de payload hasta EOP y todo
el perfil. El banco R8 pasó los 43 escenarios, incluidos 22 negativos,
3.778 ciclos capturados y nueve corrupciones de evidencia rechazadas.
Revisión SHA-256: `e19574e7dec7b559dc95c0fe36b0237e2b8b8be218b93d0e146a282ca854ff27`.
R7 conserva el error de nombre de banco; R8 corrigió solo esa referencia
a `tb_ntrpu_pcie_request_inventory_r2.sv`, con el mismo RTL y vectores.

La síntesis R3 `20261007T135946Z` terminó, con fuente
`99ab651552174087357dd9f1e787f55e14a70cc9b1d8150a04c09afaa230d798`,
dos hilos, reloj de 4 ns y consulta XDC corregida antes del mapeo.
La revisión independiente `349ca4ea4e2cad06c070b107786650f0106b1947ae8fe445c79d10a80cc10134`
verifica recursos, primitivas y cobertura completa: los doce grupos de
`check_timing` dan cero. Conserva los avisos OOC y el fallo temporal;
no acredita restricciones físicas de placa.

El netlist R3 `48f0d8e9a0d9bb9297ba19de8acc5d39bb9f09daec039830da7f1a3dc156f726`
pasó el banco binario de 42 escenarios. Revisión independiente:
`8a29da88dddb09b07b0d8d4e172d84697bdce84f0bbfdac595ae0311b81fc03d`.
Este banco precede a los tres escenarios nuevos del generador R5; su
extensión está preparada para el próximo netlist, con 22.151 ciclos y
45 escenarios binarios. Solo excluye detección X/Z, como antes.

## Actualización de tags R4

`fpga/rtl/ntrpu_pcie_request_inventory_r4.sv` registra hasta cuatro
liberaciones RC y dos asignaciones RQ. Calcula el estado de cada tag
respetando RC→RQ y actualiza cada bit del mapa una vez por ciclo.
Una segunda completion del mismo tag ya liberado sigue siendo huérfana;
dos solicitudes del mismo tag siguen produciendo fallo. La obligación
de payload permanece separada hasta EOP.

El generador R5 añade tres escenarios a la referencia: tabla llena con
cuatro liberaciones y dos reasignaciones cruzadas, dos RQ duplicadas tras
liberar el tag y dos completions terminales del mismo tag en un beat.
R3 pasó primero este banco de 22.160 ciclos/46 escenarios, con 24 negativos
y nueve corrupciones de evidencia rechazadas. R4 pasó después el mismo
banco, conservando todos los máximos y fuentes de fallo. Revisión:
`ad2abda205ba7bfda182b3935d528a65e4925ca94b88171d2288c121a10dd5ec`.
La síntesis R4 y su netlist ampliado ya tienen revisión independiente:
`fd29dae47d9ef3d029c022388cb6d66d44f025ed9666840e323c9226bfbcc966`
y `ad898750ab19d53353449dedb9c1d97cbcfc848fce46931101884c8ea4be9f47`.
El banco compila el netlist `97be2016ef5aa084a27efef302b28c2a4770b94500c7922e905895ec1c3a7f58`,
sin DUT RTL: 45 escenarios binarios, 23 negativos y nueve corrupciones
rechazadas. El fallo temporal permanece abierto.

## Temporales acotados R5

El generador R6 incorpora siete casos de longitudes RQ/RC inválidas,
incluidos span negativo y counts de 11 bits fuera del perfil permitido.
R4 y R5 pasaron los 53 escenarios, 31 negativos, todos los máximos y
nueve controles de corrupción por revisión. Además, los **22.219 estados
emitidos** coinciden entre ambas revisiones, incluidos estados posteriores
a fallo. Comparación SHA-256:
`f80bb6f47b324f6b42f0f7ae6426e04671c5e554522ca309efc8421ae03fb0fd`.
Es una regresión finita, no una prueba de equivalencia para toda entrada.

La síntesis R5 `20261007T145153Z` usa fuente
`7b3dd7345b38b9468760e81412e135bc40c3265a3582cfd537ba6f91a5bd148a`,
dos hilos y las mismas restricciones OOC de entrada. Terminó y está
respaldada. Utiliza 46.359 LUT, 9.524 FF y 258 CARRY8, sin latches,
BRAM, URAM ni DSP. El informe reporta WNS −11,266 ns y WHS −0,075 ns:
15.747 endpoints fallan setup y 1.370 fallan hold, de 27.387.
El camino `address_next_reg[51][0]` → `fault_sources_reg[13]` tiene
60 niveles y 15,248 ns estimados. No son medidas de placa.

Durante optimización emitió dos avisos `Synth 8-326` que infieren
`set_false_path` para romper bucles en `rq_update_valid`. El resultado
final reporta cero bucles; la auditoría independiente del EDIF guardado
comprueba un grafo acíclico de **66.238 primitivas, 70.853 nets y 388.094
referencias a pines**, con inventario idéntico al de Vivado. Trata cada
primitiva combinacional de forma conservadora y corta únicamente en FF.
Rechaza un ciclo inyectado y tres alteraciones de parsing/interfaces.
Los XDC guardados dentro del DCP no contienen excepciones de camino.
Por tanto, los avisos intermedios no prueban un bucle en el netlist final.

Auditoría SHA-256:
`c3912bb2255c1afc7cfc79ff45832de8937ba016bae56870bee50b686cc7308a`.
Revisión de síntesis:
`6b88679ae89755d89eb42ed6cc111c0f3eb0ad8b2020311d8422a4e307801041`.
La reserva histórica de esa revisión quedó resuelta en la tanda
`20261007T153808Z`: el DCP idéntico reabierto reporta cero excepciones
válidas e ignoradas, primitivas idénticas y los mismos resultados temporales.
Revisión SHA-256:
`904029ed921a4451eecb88fee287f0f730b3c911940b16d29f1bc132771532a7`.
La comparación con R4 es admisible; R5 sigue fallando setup y hold.

El netlist R5 pasó **22.210 ciclos/52 escenarios binarios**, 30 negativos,
3.778 ciclos capturados y nueve corrupciones rechazadas. Todos los estados
emitidos coinciden con el RTL R5, incluidos los posteriores a fallo;
se excluye únicamente el escenario de payload X/Z, fuera del dominio
binario del netlist. Comparación SHA-256:
`8249ec3010f8579c764cb35cc313f6c5d907fd80b399515e0570de2a845538e7`.

R6 expresa las asignaciones RQ mediante prefijos combinacionales
explícitos. Es una candidata para estudiar el mapeo y los avisos; no
se presenta como reparación de un bucle final confirmado. R13/R14
pasaron R4/R6 con 68 escenarios, incluidos los 15 pares de punteros
SOP inválidos del generador R7. Coinciden los **22.339 estados emitidos**;
comparación SHA-256:
`5d663faa4c2423695cd841e81bf317dd2f084ec046a6ff12d014a9b680d37b3d`.
La síntesis R6 `20261007T154755Z` terminó en 1.389,468 s y fue revisada:
**46.359 LUT/9.524 FF/258 CARRY8**, WNS **−11,266 ns**, WHS **−0,075 ns**.
No mejoró R5. Su EDIF final es acíclico y los tres informes de excepciones
dan cero. Revisión:
`47da15f94235cbeb9b79688b850649e724f735889570286b70b010a4c5ac06db`.
Su netlist pasó 22.330 ciclos/67 escenarios binarios, 45 negativos y
nueve controles. Todos los estados coinciden con RTL R6; comparación:
`7c60a509ab3e80c972f754d62f2c9cd03d6f6e9b5230b9bf1ea35f1d636f933a`.
Las revisiones previas permanecen intactas.

| Tanda revisada | Duración | Archivo D:/NFS SHA-256 |
| --- | ---: | --- |
| `20261007T153056Z`, R4 RTL/68 | 37,324 s | `ab5b90cc996243305d693a2e7d398d892fc69590e549349763b36f93cf0f779d` |
| `20261007T153325Z`, R5 netlist/52 | 95,415 s | `85fc18c3a34edb718dab176c4fc8387567d4964959f2c9a257f395e21367860a` |
| `20261007T153625Z`, R6 RTL/68 | 36,543 s | `599d1a14f5eba25f87b78f4cace7c97ede904f152972311e914a2059f4432fa6` |
| `20261007T153808Z`, DCP R5 reabierto | 138,028 s | `b96ca0f7c2896069db31bbac1d74f0da8e533487def807745ce21881fe61c75c` |
| `20261007T154755Z`, síntesis R6 | 1.389,468 s | `bf01dbd2cfb6fd3b92f392a3c3611d892d8012947a7b7ad13fa6ec2a5b9594ce` |
| `20261007T161525Z`, netlist R6/67 | 95,456 s | `01d80bcde68ef707c4922e03fffe17c47482e695c4591f173a71b7b8f2a84174` |
| `20261007T161823Z`, RTL R7/68 | 37,592 s | `1e05851fb6ada18a7ca6415ca942fb67d926d0244ed1ba9d14868d123dfec56c` |

R7 mueve la guarda de rango después de los registros y conserva los bits
altos descartados. Pasó los 68 escenarios y coincide con todos los estados
emitidos de R6. Su síntesis `20261007T162508Z` terminó: **46.617 LUT/9.531 FF**,
WNS **−11,267 ns**, WHS **−0,075 ns**. No mejora el peor camino de R6.
El EDIF de 66.847 primitivas/71.310 nets es acíclico y las excepciones son cero.
Revisión: `548198667ead5ee384005a4a1fb3f9a98d003f4dc244c9fe16c63acb9b71b55d`.
Su netlist también pasó los 67 escenarios binarios y coincide con RTL;
revisión `81b0e7952bd50d1ddb8319e68301f9b747519e61bda3d0bc814ba0226a4b424f`.
La guarda de longitud/bytes RC es ahora el peor camino, con 62 niveles.
[Contrato, medidas y siguiente arquitectura](PCIE_INVENTORY_TIMING_PLAN.md).

| Tanda R7 | Duración | Archivo D:/NFS SHA-256 |
| --- | ---: | --- |
| `20261007T162508Z`, síntesis | 570,727 s | `fc9a8d14314c2ba04f96d90fae716ba6b3cdcf64273f917b1b6579705aa2b02c` |
| `20261007T163945Z`, netlist/67 | 95,720 s | `29a1b71886d184e80dfd4a029b9fe6a5b45263119b38501618753faeed57bea0` |

[Rangos, criterio temporal e integración XDMA](PCIE_INVENTORY_TIMING_PLAN.md).

Después siguen observación nativa, conexión
de producción y los siete contratos de drenaje; la tarjeta de 512 GiB,
la inferencia real y el cierre físico conservan sus propios criterios.

[Arquitectura y bancos RTL](PCIE_REQUEST_INVENTORY_RTL.md) ·
[Contrato completo de la cola](VIVADO_QUEUE.md).
