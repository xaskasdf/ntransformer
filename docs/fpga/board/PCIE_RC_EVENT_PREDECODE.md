# Predecodificación registrada de eventos RC

Continuación medida: [validación paralela R3 y diagnóstico R2](PCIE_RC_EVENT_FRAGMENTS.md).
Este documento conserva los resultados históricos de R1.

Revisión R1 de `fpga/rtl/ntrpu_pcie_rc_event_stage_r1.sv`, 2026-10-07.
Conserva el nombre de módulo y la interfaz de la
[etapa RC original](PCIE_RC_EVENT_STAGE.md); cada proyecto selecciona
explícitamente una sola revisión del DUT.

## Cambio y contrato

Los mapas SOP/EOP, los slots de EOP y los errores estáticos de framing
se calculan durante la captura, junto a la propuesta aritmética. El ciclo
siguiente consume esos registros y comprueba las condiciones que dependen
del contexto retenido: EOP huérfano, solapamiento y continuidad del paquete.
No se cambia la latencia de dos etapas ni la aceptación de un beat por
ciclo con hasta cuatro cabeceras/eventos.

Se conservan la interfaz de metadatos esperados, el requester independiente,
el contexto hasta EOP, los candidatos inválidos y las obligaciones sin
resolver. Los eventos y `packet_checked` siguen sin autorizar vacío o ACK.
Faltan asignación RQ, consultas/escrituras y forwarding del banco integrado,
contabilidad global, épocas y revocación coherente.

## Evidencia funcional

El shadow RTL pasó **22.339 ciclos/68 escenarios**. Todas las filas de
eventos, estados y metadatos emitidas son idénticas a la versión anterior,
incluido tráfico inválido y estados posteriores a fallo. El verificador
reconstruyó cada salida independientemente y rechazó diez corrupciones.
Las mutaciones ejecutadas en XSIM siguen fallando: omisión RC en el ciclo
132 y metadatos ausentes en el 133.

Para la comparación con hardware sintetizado se preparó un banco común
con **22.330 ciclos/67 escenarios binarios**, más **370 ciclos/15 fases**
dirigidas. Ambos pasaron sobre el RTL R1 y revisión independiente:

- El R7 sin modificar proporciona pertenencia y metadatos; todas sus
  salidas públicas coinciden con la referencia binaria revisada.
- El shadow conserva 17.461 beats retirados, 16 actualizaciones parciales,
  1.075 completions, dos eventos sin resolver y 520 orígenes entre beats.
- Se conserva la cobertura de 256 tags, 257 lecturas y 1.052.672 bytes
  pendientes del banco R7. Esto no integra todavía esa contabilidad en R1.
- El banco dirigido conserva cancelaciones por reset frío, wrap, pausas,
  stalls y 256 ciclos consecutivos de cuatro fragmentos del mismo tag.
- Diez corrupciones del registro, repartidas entre ambos bancos, se rechazan.

El generador binario excluye únicamente `fault_rc_unknown_payload` y
concreta a cero 619 bits X/Z de campos que no cambian los estados esperados
en estos vectores. Se regeneraron todos los estados antes de comparar.
Esta prueba finita no establece equivalencia general X/Z. Los 68 escenarios
de cuatro estados siguen preservados por separado.

El banco binario espera 200 ns al arrancar y muestrea 1 ns después del
flanco. La copia `ntrpu_pcie_rc_proposal_reference.sv` difiere de la unidad
aritmética original solo en el nombre del módulo; evita una colisión con
el módulo interno del netlist. El R7 y esa aritmética son referencias RTL
del banco, no sustitutos del DUT sintetizado.

## Síntesis y reproducción del netlist

La síntesis R1 terminó y fue revisada. Se mantiene la misma parte
`xcvu5p-flva2104-2-i`, reloj de 4 ns, I/O explícito de cero y ninguna
excepción temporal válida o ignorada. La interfaz, latencia y RC4 son iguales:

| Medida OOC | Original | R1 predecodificada |
| --- | ---: | ---: |
| LUT / FF | 7.845 / 1.161 | 9.639 / 1.266 |
| CARRY8 | 160 | 160 |
| WNS / TNS (ns) | −3,354 / −135,617 | −2,624 / −39,825 |
| Setup: fallidos / total | 353 / 2.660 | 46 / 3.150 |
| WHS / THS (ns) | −0,075 / −32,260 | −0,075 / −32,734 |
| Hold: fallidos / total | 515 / 2.660 | 596 / 3.150 |
| Peor retardo / niveles | 7,336 ns / 26 | 6,606 ns / 24 |

El margen setup mejora **0,730 ns**, a coste de **1.794 LUT y 105 FF**
adicionales. Sigue sin cumplir 4 ns ni hold; no hay cierre físico. El peor
camino ahora va de `arithmetic/header_seen_reg[0]_rep__3/C` a
`fault_latched_reg[10]/D`, en la detección de longitud/byte enables.
No hay latches, BRAM, URAM ni DSP; los doce controles de cobertura temporal
no reportan incidencias. Permanecen TIMING-16 (12) y XDCH-2 (1.334).

El EDIF final contiene 12.754 primitivas y 15.954 declaraciones de net,
sin bucles combinacionales; el auditor jerárquico rechazó ocho controles.
Se registran 125 declaraciones vacías, incluida una segunda anotación sin
conexiones de `expected_valid_3_`. No se admiten nets conectadas duplicadas.
Se conservan 59 avisos de puertos constantes en celdas internas `__GB1` y
`__GB4`, los avisos de conversión de igualdad y la eliminación de
`data_stage_reg`, cuya detección de X no se implementa en lógica binaria.

El netlist R1 pasó **los 22.330 ciclos del shadow binario y los 370 del
banco dirigido**. Todas las filas de eventos, metadatos y referencia pública
son idénticas a las del RTL; el modelo independiente las reconstruye y
rechaza diez corrupciones. Se verificó el hash del archivo sintetizado
compilado y que no se compiló el DUT RTL. El R7 y la aritmética renombrada
se compilan únicamente como referencias del banco.

Los dos bancos conservaron los avisos de definición duplicada de `glbl`:
sus cuerpos archivados son idénticos al incluido en el netlist. No se aplicó
SDF. Esta prueba funcional de la etapa no resuelve setup/hold, no integra
el banco de metadatos ni califica el endpoint nativo o el diseño físico.

## Tandas y diagnóstico

Fuentes y ejecuciones se respaldan en D: y NFS bajo
`artifacts/ntrpu-board-vivado/<ID>/`. Las revisiones y hashes forman parte
del checkpoint del hilo principal.

| ID, prefijo `20261007T` | Trabajo |
| --- | --- |
| `195515Z-V4-ffn-supervised-pcie-rc-event-predecode-rtl` | Fallo Tcl: ruta incorrecta al banco; no hubo compilación |
| `195737Z-V4-ffn-supervised-pcie-rc-event-predecode-rtl-r1` | Ruta corregida; mismo RTL, shadow completo y mutaciones revisados |
| `200444Z-V4-ffn-supervised-pcie-rc-event-predecode-binary-rtl` | Shadow binario y casos límite revisados |
| `200846Z-V4-ffn-supervised-pcie-rc-event-predecode-synth` | Síntesis y EDIF revisados; setup/hold fallidos |
| `201435Z-V4-ffn-supervised-pcie-rc-event-predecode-binary-netlist` | 22.330 + 370 ciclos revisados e idénticos al RTL |

Un intento local previo recibió `Errno 28` al crear el registro de tanda.
Se comprobó que no existía carpeta ni invocación remota; se conservan el
registro local vacío, configuración y diagnóstico. No cuenta como una
simulación ejecutada. Se comprimieron seis archivos locales de evidencia
ya certificados, verificando sus hashes antes y después, sin borrar archivos.

## Siguiente dependencia

La siguiente revisión debe paralelizar los controles por fragmento y
obtener las posiciones mediante prefijos de SOP/EOP, reduciendo el recorrido
secuencial de 16 DWORD que todavía domina la validación del payload.
El primer intento conserva latencia de dos etapas. Si requiere más etapas,
definir antes su latencia, alineación de metadatos, forwarding y obligaciones
en vuelo; no retrasar silenciosamente la revocación de vacío/ACK.
Preservar RC4/RQ2 y la capacidad; no insertar burbujas en el observador
para obtener un resultado temporal más favorable.

Los criterios globales de [VIVADO_QUEUE.md](VIVADO_QUEUE.md) siguen abiertos:
identidad BDF/host, siete contratos coherentes, CDC, reset nativo de 50 ms,
DIMM exacto, cuatro DDR/512 GiB, inferencia con pesos reales y cierre físico.
