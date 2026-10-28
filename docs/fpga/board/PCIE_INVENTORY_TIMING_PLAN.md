# Cierre temporal e integración del inventario PCIe

## Diagnóstico medido

R4 conserva 256 tags, RQ2/RC4 y hasta 257 lecturas pendientes. Su síntesis
usa 58.225 LUT y 9.524 registros, pero falla a 4 ns: WNS −11,964 ns y
WHS −0,075 ns. El camino crítico va de `rc_user[94]` a
`fault_sources_reg[13]`: 65 niveles y 15,974 ns estimados, incluidos
12,107 ns de interconexión sin colocar. No basta para integración física.
La referencia R4 pasó el netlist de 45 escenarios binarios y el banco
RTL ampliado de 53 escenarios.

## Revisión de anchos R5

`ntrpu_pcie_request_inventory_r5.sv` conserva la interfaz y los registros
de estado de R4. Reduce temporales combinacionales de tipo `integer`:

| Temporal | Rango que debe representar | Tipo R5 |
| --- | --- | --- |
| Cantidades SOP/EOP | −1 a 4 | 4 bits con signo |
| Punteros y último EOP | −1 a 15 | 5 bits con signo |
| Primer/último byte habilitado | 0 a 3 | 2 bits |
| Span RQ | −6 a 8.188, incluso descriptores inválidos | 15 bits con signo |
| Bytes disponibles RC | −3 a 8.188 | 15 bits con signo |
| Dirección inicial temporal | 0 a 4.098 | 13 bits |
| Máscaras de bytes | 0 a 15 | 4 bits |

Los operandos de comparación de punteros siguen teniendo signo: el
sentinela −1 debe ordenar antes que lane 0. Se conservan las conversiones
mixtas del cálculo de longitud y los guardas de desbordamiento.

El generador R6 añade counts RQ 0/1025/2047 y RC 0/1025/2047, incluidas
combinaciones de byte enables que producen span negativo. El banco completo
tiene 22.219 ciclos/53 escenarios, con 31 negativos. Las tandas R11/R12
pasaron revisión independiente y la comparación de **todos** los estados
emitidos R4/R5, incluidos los posteriores a un fallo. La síntesis R5
`20261007T145153Z` terminó: 46.359 LUT/9.524 FF/258 CARRY8, con WNS
reportado de −11,266 ns. Los dos avisos intermedios `Synth 8-326` se
investigaron sobre el resultado final: su EDIF es acíclico y sus XDC
guardados no contienen excepciones de camino. El netlist pasó los
52 escenarios binarios y sus 22.210 estados emitidos coinciden con RTL,
incluidos los estados posteriores a fallo. Esta regresión finita no
demuestra equivalencia para toda entrada.
[Revisión y hashes](PCIE_REQUEST_INVENTORY_SYNTHESIS.md).

La tanda `20261007T153808Z` reabrió una copia idéntica del DCP y emitió
`report_exceptions`, su resumen y excepciones ignoradas: **cero excepciones
válidas o ignoradas**. Primitivas y resultados temporales coinciden con
la síntesis R5. Queda resuelta la reserva sobre la comparación temporal;
setup y hold siguen fallando. No se cambiaron restricciones ni retardos.
[UG835: informe de excepciones](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_exceptions).

## Candidata R6 y punteros inválidos

R6 mantiene anchos y registros de R5, pero cada posición de cabecera RQ
lee solamente el prefijo anterior de tags válidos, identidades y cuenta.
Es una cadena combinacional explícita de cuatro posiciones, con el mismo
perfil RQ2 y sin añadir ciclos. No se presume que mejore el mapeo.

El generador R7 añade los 15 pares SOP distintos de `(0,2)` que son
inválidos sin un paquete parcial anterior: duplicados, orden inverso y
posiciones no admitidas. Son **22.339 ciclos/68 escenarios**, con 46
negativos. R13 (R4) y R14 (R6) pasaron revisión independiente, incluidos
nueve controles de corrupción por tanda. Los 22.339 estados emitidos
coinciden, también después de fallo. La síntesis R6 `20261007T154755Z`
terminó en 1.389,468 s: **46.359 LUT/9.524 FF/258 CARRY8**, WNS
**−11,266 ns** y WHS **−0,075 ns**. Son las mismas métricas que R5;
el prefijo explícito no mejoró el resultado medido.

R6 también emitió dos avisos intermedios `Synth 8-326`, ahora sobre
`rq_prefix_valid`. Su propio EDIF final es acíclico: 66.238 primitivas
y 70.854 nets. Los informes finales confirman cero excepciones válidas
o ignoradas; no hay una excepción que justifique omitir caminos.
Auditoría `0cfb66abe752ef4f7ca1fe0b51fc760d6bdad81be85c10654e444a5134648745`;
revisión `47da15f94235cbeb9b79688b850649e724f735889570286b70b010a4c5ac06db`.
El banco binario preparado para R6 contiene 22.330 ciclos/67 escenarios,
45 negativos, y excluye únicamente el caso de payload X/Z. Su ejecución
pasó en `20261007T161525Z`, incluidos nueve controles de corrupción.
Sus 22.330 estados emitidos coinciden con el RTL R6 para esos escenarios.

## Candidata R7: retiming de la guarda de rango

El camino crítico medido termina en el bit 13 de `fault_sources`.
R7 conserva los contadores de salida y captura sus seis bits altos
descartados: dos para tags, dos para lecturas y dos para bytes. Así
reconstruye exactamente los candidatos con signo de 11/11/23 bits
después del registro, incluidos valores inválidos y negativos.

El fallo visible combina el fallo persistente anterior y la comparación
del candidato recién registrado. El registro persistente captura esa
combinación en el siguiente flanco. Esta transformación pretende mantener
el fallo visible en el mismo ciclo, sin reducir capacidad ni ritmo.
Una marca de validez impide comprobar un candidato antes de observar;
el reset frío limpia ambos componentes. `===1'b1` conserva la semántica
del `if` anterior cuando la comparación contiene X.

El banco R15 `20261007T161823Z` pasó los 68 escenarios: R6/R7 coinciden
en los 22.339 estados emitidos, incluidos los posteriores a fallo y el
bit 13 en los casos de span negativo. Comparación SHA-256:
`a9fdcb5acb12b31fe4429e131f093becc76a0c9a95188c2d78c67f83bd43ff23`.
La síntesis R7 `20261007T162508Z` terminó en **570,727 s**: **46.617 LUT,
9.531 FF y 258 CARRY8**. No emitió los avisos intermedios de bucles;
su EDIF final es acíclico y las excepciones válidas/ignoradas son cero.
WNS **−11,267 ns** y WHS **−0,075 ns** siguen fallando. TNS de setup
mejora de −134.364,406 a −97.355,531 ns, pero el peor camino no mejora.
No se admite como cierre temporal.

El camino crítico ahora va de `address_next_reg[51][1]` a
`fault_latched_reg[10]`: **62 niveles/15,249 ns**, con 3,239 ns de lógica
y 12,010 ns estimados de interconexión. El cuello de botella sigue en
la validación RC de longitud/bytes, dependiente de los metadatos del tag.
El netlist R7 pasó 22.330 ciclos/67 escenarios binarios y coincide con
todos los estados de RTL en ese dominio. Comparación SHA-256:
`f422cf601e7f7032c400e648fa87fcbf882c22d527b5d3fb22ea007ffca49260`.

La siguiente revisión debe separar decodificación, consulta de metadatos, validación y
contabilidad en etapas sin reducir el ritmo de entrada. Cualquier latencia
adicional exige inventariar los eventos en vuelo, forwarding para tags
reutilizados y revocación de certificados; nunca publicar vacío mientras
queden eventos sin retirar. Esa arquitectura aún no está implementada.
La prueba deberá correlacionar cada beat aceptado con su retirada, conservar
RC→RQ dentro del mismo beat y la obligación de payload tras liberar el tag,
y medir todos los caminos restantes. No basta con mover otro bit de fallo.
El [contrato de arquitectura](PCIE_INVENTORY_PIPELINE_CONTRACT.md) concreta
etapas, conflictos de metadatos, revocación de vacío y orden de pruebas;
todavía no es una implementación.

El primer bloque de esa investigación, un banco registrado 256×30 con
cuatro consultas y seis escrituras, ya pasó RTL, netlist y shadow contra
los eventos de R7. Su setup OOC es +1,924 ns; hold sigue en −0,042 ns.
No demuestra aún el encadenamiento de etapas: los valores escritos llegan
calculados desde fuera. [Resultados y siguiente dependencia](PCIE_TAG_METADATA.md).

## Frontera de producción XDMA

La inspección `pcie-inventory-seam-dcp-inventory-20261007T142750Z.json`
no encontró DCP en la copia del ejemplo usada para la captura nativa.
La síntesis posterior de esa IP con el perfil exacto permitió inspeccionar
la frontera sin modificar el ejemplo original.

Se congeló `tools/ntrpu_board_xdma_observation_boundary_synth.tcl` y su
runner en `xdma-observation-boundary-synth-frozen.json`, SHA-256
`300335681834589fcb947e0ca882c6944ef6fe994d275ecad651713b3f1d4248`.
Los 756 archivos del ejemplo coinciden con el archivo nativo verificado.
La sonda `20261007T151433Z` terminó y está respaldada: copió el ejemplo, comprobó que XCI y fuentes de
síntesis resuelven dentro de la copia, conserva el perfil, sintetiza solo
XDMA y exporta DCP, jerarquía, pines y nets. El runner comprueba el original
antes y después, y respalda también cualquier diagnóstico. Falta revisar
el resultado de conectar el monitor. La revisión ligó los 122 archivos
del orden de compilación y conservó 153 fuentes/configuraciones de la copia.
Los buses de datos y TUSER RQ/RC tienen conexión; READY[1:3] y el número
de bus no tienen net en la consulta del nivel superior. Falta realizar
la integración completa del monitor. La revisión de segmentos y primitiva
posterior encontró ceros en esas salidas superiores, pero conservó los
ocho pines `CFGBUSNUMBER` en la `PCIE40E4`. El ensayo `164306Z` añadió
ocho salidas pasivas conservando celdas y transporte; `165658Z` confirmó
su persistencia al reabrir el DCP. Faltan muestreo funcional y procedencia
del Requester ID completo.
[Resultado, límites y respaldo](XDMA_OBSERVATION_BOUNDARY.md).
[UG835: creación de una ejecución OOC de IP](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/create_ip_run).
[UG835: fuentes de síntesis de una IP](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_files).

El ensayo en otra copia confirmó que `connect_net -hierarchical` permite
exportar esos ocho nets. No demuestra una carga de muestreo ni equivalencia
funcional. Exigir conectividad preservada, ausencia de nuevos drivers de
transporte, equivalencia, timing y CDC al integrar el observador completo.
[UG835: conexión de nets](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/connect_net).

Fuente/host, siete contratos coherentes, reset nativo de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y entrega física/ECAD permanecen en
[VIVADO_QUEUE.md](VIVADO_QUEUE.md).
