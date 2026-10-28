# Selección de fragmentos RC registrada

Este documento conserva el experimento R4. La revisión posterior
[R5](PCIE_RC_EVENT_PAYLOAD.md) mejora recursos y timing y pasa los mismos
bancos en RTL y netlist. La combinación posterior [R6](PCIE_RC_EVENT_COMBINED.md)
es ahora la referencia temporal medida.

Revisión R4, ejecutada el 2026-10-07 y revisada el 2026-10-08 UTC, en
`fpga/rtl/ntrpu_pcie_rc_event_stage_r4.sv`. Continúa el experimento de
[validación paralela R3](PCIE_RC_EVENT_FRAGMENTS.md). La interfaz, las
dos etapas y la aceptación de un beat RC4 por ciclo se conservan.

## Cambio

Cada DWORD registra durante la captura la última cabecera anterior,
la presencia de un EOP previo, si el fragmento nuevo sigue activo y
la distancia desde SOP. Estos campos dependen exclusivamente del beat
capturado; se alinean con los descriptores y las propuestas aritméticas.

El contexto retenido no se captura anticipadamente. Sin nueva cabecera
ni EOP previo, la validación usa `context_active` y `context_seen + lane`
del ciclo actual, que puede contener el resultado del beat anterior.
Se conservan el wrap de once bits, el comportamiento posterior a fallo,
`discontinue`, reset frío y el arreglo de arranque `always_comb` de R3.
Los registros nuevos no añaden una etapa de salida.

La hipótesis es acortar la selección que precede a los controles de
byte enables y errores. La mejora temporal debe medirse sobre el circuito
generado; insertar estos registros no la demuestra por sí solo.

## Evidencia funcional

El banco de cuatro estados pasó **22.339 ciclos/68 escenarios**. Todas
las filas de eventos, estados públicos, propuestas y metadatos coinciden
con R3 y con el modelo independiente. Las dos mutaciones ejecutadas en
XSIM se detectaron en los ciclos 132 (omisión RC) y 133 (metadatos ausentes);
el revisor rechazó diez corrupciones del registro.

La tanda binaria conserva los mismos tres bancos: 22.330 ciclos del shadow,
370 casos límite y 281.904 filas de fragmentación, con 80.544 casos.
Las dimensiones, limitaciones y referencias están descritas en el
[documento de R3](PCIE_RC_EVENT_FRAGMENTS.md). Las **304.604 filas binarias**
pasaron la revisión independiente, incluida la comparación de la matriz
contra R1 y quince controles de corrupción.

El netlist propio pasó las mismas **304.604 filas binarias**, todas
idénticas a las del RTL y al modelo independiente. Se verificaron el
artefacto compilado y la ausencia del DUT RTL; R7 y R1 renombrado se usan
solo como referencias del banco. Los tres cuerpos duplicados de `glbl`
son idénticos. Quince corrupciones adicionales fueron rechazadas en esta
revisión. No se aplicó SDF ni se demostró equivalencia general X/Z.

La simulación del netlist duró 553,479 s en la VM; no es una latencia del
circuito. Su transferencia de logs utilizó compresión SSH/SCP, conservando
los bytes originales comprobados por hash. No se midió aisladamente el
efecto de la compresión sobre el tiempo total de respaldo.

## Síntesis: no mejora respecto de R3

La síntesis R4 está respaldada y revisada para `xcvu5p-flva2104-2-i`,
período de 4 ns y los mismos retardos I/O explícitos de cero. No hay
excepciones temporales válidas ni ignoradas.

| Medida OOC | R3 | R4 |
| --- | ---: | ---: |
| LUT / FF | 7.333 / 1.223 | 7.417 / 1.305 |
| CARRY8 | 186 | 185 |
| WNS / TNS (ns) | −1,263 / −8,049 | −1,463 / −8,496 |
| Setup: fallidos / total | 7 / 3.100 | 7 / 3.181 |
| WHS / THS (ns) | −0,075 / −32,329 | −0,075 / −32,276 |
| Hold: fallidos / total | 550 / 3.100 | 544 / 3.181 |
| Peor retardo / niveles | 5,245 ns / 20 | 5,445 ns / 21 |

R4 empeora el margen setup **0,200 ns** y añade **84 LUT y 82 FF**.
**R3 sigue siendo la mejor variante medida para el siguiente intento**;
R4 se conserva como experimento funcional. Ambas fallan setup y hold.
El peor camino de R4 va de `context_lower_address_reg[0]/C` a
`packet_checked_reg[2]/D`: adelantar la selección no elimina el coste del
cálculo de byte enables a partir del contexto. Los 5,445 ns incluyen
1,210 ns de lógica y 4,235 ns de interconexión estimada, sin colocación.

No hay latches, BRAM, URAM ni DSP. Los doce controles de cobertura temporal
dan cero incidencias; quedan TIMING-16 (5) y XDCH-2 (1.334). El EDIF final
tiene 10.310 primitivas y 13.797 declaraciones de net sin bucles; ocho
controles del auditor fueron rechazados. Se registran 172 declaraciones
vacías, incluida una anotación duplicada sin conexiones.

El primer revisor se detuvo ante tres avisos de constantes cuyo nombre
de puerto no coincidía con su patrón. Una nueva revisión conserva ese
diagnóstico y admite exactamente los bits `[3:1]` de
`payload_lane[3].selected_header_stage`: para ese DWORD, `4*h<=3` permite
solo la cabecera cero; inicialización, reset, captura y retención mantienen
los otros bits en cero. Los 36 avisos internos restantes corresponden al
patrón ya revisado. No se cambió el RTL ni el netlist para resolverlo.
La conversión de comparaciones X/Z y la eliminación de `data_stage_reg`
siguen limitando la reproducción sintetizada al dominio binario.

## Tandas y conservación

| ID | Alcance |
| --- | --- |
| `20261007T212703Z-V4-ffn-supervised-pcie-rc-event-prefix-rtl` | Cuatro estados, dos mutaciones y diez controles; revisión y respaldo completados |
| `20261007T213045Z-V4-ffn-supervised-pcie-rc-event-prefix-binary-rtl` | Tres bancos binarios, quince controles, revisión y respaldo completados |
| `20261007T213905Z-V4-ffn-supervised-pcie-rc-event-prefix-synth` | Síntesis y EDIF revisados; regresión temporal frente a R3 conservada |
| `20261007T214651Z-V4-ffn-supervised-pcie-rc-event-prefix-binary-netlist` | Tres bancos reproducidos en el netlist; revisión independiente y respaldo completados |

Fuentes y resultados se conservan por tanda en
`artifacts/ntrpu-board-vivado/<ID>/`, respaldados en D: y NFS. No se
modifican las revisiones anteriores para incorporar esta variante.

Las cuatro tandas tienen respaldo verificado y revisión independiente.
El netlist usado tiene SHA-256
`dd024cef1477993a56b89c69b215404d5e0045594814977b4239d23b2da56c3f`.

## Siguiente intento temporal

Partir de R3, que mantiene menor retardo y coste, y separar los predicados
de payload de la selección de metadatos. Evaluar límites de byte enables
con anchos explícitos y sin encadenar selección, suma y comparación en
el mismo camino. Conservar también longitudes inválidas, wrap de once bits,
contexto posterior a fallo, RC4 y latencia de dos etapas. Cualquier cambio
debe pasar las mismas pruebas y su propia síntesis/reproducción de netlist.

## Integración que sigue abierta

R7 sigue proporcionando pertenencia y metadatos esperados en el banco
shadow. R4 no es el monitor completo: faltan asignación RQ, escrituras,
consultas y forwarding, más retirada y contabilidad correlacionadas.
La liberación del tag en la cabecera terminal y la terminación del payload
en EOP deben mantener obligaciones separadas, incluso al reutilizar el tag.

Los eventos y `packet_checked` no autorizan vacío ni ACK. Se mantienen
los requisitos del [contrato de integración](PCIE_INVENTORY_PIPELINE_CONTRACT.md)
y de [VIVADO_QUEUE.md](VIVADO_QUEUE.md), incluidos identidad independiente,
CDC/reset nativo, DIMM exacto, cuatro DDR/512 GiB, inferencia y cierre físico.
