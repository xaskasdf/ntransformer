# Etapa registrada de eventos RC

Actualizado el 2026-10-07 UTC. Implementación:
`fpga/rtl/ntrpu_pcie_rc_event_stage.sv`. Esta etapa combina las
[propuestas aritméticas](PCIE_RC_PROPOSAL.md) con contexto de paquete,
validación de cabeceras y comprobación de longitud/byte enables hasta EOP.
Acepta un beat por ciclo y hasta cuatro cabeceras/eventos por beat.

La [revisión R1 con predecodificación](PCIE_RC_EVENT_PREDECODE.md) conserva
estas salidas y añade la comparación completa del shadow binario con su
propio netlist. Los resultados siguientes corresponden a la versión original.

## Contrato y alcance

Un beat aceptado en el flanco N produce sus eventos registrados en N+1.
`header_owned` contiene las decisiones de pertenencia por prefijo del
asignador externo. `expected_valid` y `expected_metadata` deben responder
al beat capturado el ciclo anterior. Cada palabra de 30 bits contiene
bytes pendientes, dirección, TC y atributos. El requester esperado es una
entrada independiente; nunca se aprende de una respuesta RC.

El contexto conserva tag, propietario, aritmética, errores y origen hasta
EOP, aunque cambie la propiedad de las nuevas cabeceras. `observing=0`
pausa la captura y conserva el contexto. Solo el reset frío lo descarta.
Secuencia y origen son diagnósticos de 32 bits, con wrap; no identifican
la propiedad ni sustituyen una época de asignación.

`update_valid`, `complete` y `unresolved` son **eventos candidatos**.
`packet_checked` certifica únicamente las comprobaciones locales del
paquete: una respuesta inválida puede emitir un candidato con ese bit a
cero. Ni este bit ni `pipeline_pending` autorizan vaciado, revocación o ACK.
Todavía faltan asignación RQ, banco integrado, forwarding y contabilidad
global de obligaciones. El DUT no expone una señal de inventario vacío.

## Simulaciones revisadas

El shadow completo ejecutó **22.339 ciclos/68 escenarios** junto al R7
sin modificar. Sus salidas públicas, incluidas las posteriores a fallo,
coinciden con la referencia R15. Un modelo Python independiente reconstruye
todas las salidas de la nueva etapa, conservando las diferencias de
semántica inválida documentadas en la unidad de propuestas.

Se comprobaron 17.462 beats retirados, 16 actualizaciones parciales,
1.076 completions, dos eventos sin resolver y 520 eventos cuyo origen
precede al beat de EOP. Hubo 1.080 eventos con comprobación local correcta.
Dos mutaciones ejecutadas en XSIM —omitir RC y suprimir metadatos— fallaron
como se esperaba. Diez corrupciones de los registros fueron rechazadas.
La pertenencia y los metadatos de este banco proceden del R7: no demuestra
la programación integrada del nuevo banco de metadatos.

Otro banco dirigido ejecutó **370 ciclos/15 fases**, en RTL y en el netlist
funcional. Todas las filas coinciden entre ambos y con el modelo:

- Reset frío cancelando dos beats capturados y un contexto retenido.
- Finalización de un paquete que atraviesa el wrap de secuencia.
- Pausa de observación y stalls antes y durante el payload.
- 256 beats consecutivos con cuatro fragmentos del mismo tag por beat:
  1.023 actualizaciones, una completion y 1.024 comprobaciones correctas.
- Metadatos ausentes, identidad/atributos erróneos, byte enables inválidos,
  respuesta no normal, contexto solapado, discontinue y EOP huérfano.

Cada revisión del banco dirigido rechazó diez corrupciones. La prueba del
netlist compiló el archivo sintetizado sin DUT RTL, con 200 ns de arranque
y muestreo a 1 ns, sin SDF. Se compararon los dos cuerpos `glbl` archivados:
son idénticos. **Falta reproducir el shadow R7 completo en este netlist**;
370 ciclos dirigidos no sustituyen esa cobertura.

## Síntesis y limitación de frecuencia

Vivado 2026.1, `xcvu5p-flva2104-2-i`, reloj de 4 ns, I/O de cero para
caracterización OOC, sin excepciones temporales válidas ni ignoradas:

| Medida | Resultado |
| --- | ---: |
| LUT / FF / CARRY8 | 7.845 / 1.161 / 160 |
| Latches / BRAM / URAM / DSP | 0 / 0 / 0 / 0 |
| WNS / TNS | −3,354 / −135,617 ns |
| Setup: endpoints fallidos / total | 353 / 2.660 |
| WHS / THS | −0,075 / −32,260 ns |
| Hold: endpoints fallidos / total | 515 / 2.660 |
| Peor camino | `user_stage_reg[78]/C` → `packet_checked_reg[2]/D` |
| Retardo / niveles | 7,336 ns / 26 |

La etapa **no cumple 4 ns**. Son estimaciones OOC, sin cierre físico del
diseño padre; no se pueden comparar directamente con el monitor completo
como si tuvieran el mismo alcance. Los doce controles de cobertura temporal
no reportaron incidencias. Permanecen TIMING-16 (14) y XDCH-2 (1.334).

El EDIF final contiene 10.501 primitivas y 13.608 declaraciones de net,
con jerarquía expandida mediante alias explícitos de puertos. No hay ciclos
combinacionales; ocho controles incluyen un bucle entre padre e hijo,
recursión, puertos inválidos y nets duplicadas con conexiones. Se registran
136 declaraciones vacías: una repite el identificador `expected_valid_3_`
con otro nombre visible y sin conexiones. Solo esta anotación vacía se
reconcilia; una segunda declaración conectada se rechaza.

Se conservan los avisos de conversión `===`/`!==` y **Synth 8-6014**:
Vivado eliminó `data_stage_reg`, usado únicamente para detectar X en
payload. La comprobación del netlist se limita a entradas binarias; no
establece equivalencia X/Z ni convierte la detección de X en hardware.

## Tandas y respaldo

Los archivos de ejecución y fuentes están verificados en
`artifacts/ntrpu-board-vivado/<ID>/` (almacenamiento D: y copia NFS).
Las revisiones independientes y su cadena de hashes se conservan en el
checkpoint y en `build/cloud-setup/20261002-board/`.

| ID, prefijo `20261007T` | Resultado |
| --- | --- |
| `185751Z-V4-ffn-supervised-pcie-rc-event-stage-rtl` | Diagnóstico: normal correcto; Tcl no reconocía el fallo esperado de falso vacío |
| `190628Z-V4-ffn-supervised-pcie-rc-event-stage-rtl-r1` | Shadow completo y dos mutaciones revisados; RTL idéntico |
| `191242Z-V4-ffn-supervised-pcie-rc-event-stage-synth` | Síntesis y auditoría revisadas; setup/hold fallidos |
| `192312Z-V4-ffn-supervised-pcie-rc-event-boundaries` | 370 ciclos RTL revisados |
| `194009Z-V4-ffn-supervised-pcie-rc-event-boundaries-netlist` | 370 ciclos de netlist revisados e idénticos al RTL |

## Siguiente revisión

Predecodificar SOP/EOP y sus mapas en la captura, separando errores
estáticos de framing y comprobaciones que dependen del contexto retenido.
La primera revisión debe mantener la interfaz, la latencia y RC4, y
compararse con todas las filas del shadow y del banco dirigido antes de
volver a sintetizar. Si aún domina la cadena de 16 DWORD, segmentar su
cálculo con forwarding y obligaciones en vuelo explícitas; no introducir
burbujas en el observador ni reducir capacidad para cumplir timing.

Después: integrar consultas/escrituras del [banco de metadatos](PCIE_TAG_METADATA.md),
asignación y retiro correlacionados, manteniendo 256 tags, RQ2/RC4,
257 lecturas y 1.052.672 bytes. Continúan pendientes identidad BDF/host,
unión coherente de los siete contratos, CDC, reset nativo de 50 ms,
DIMM exacto/cuatro DDR/512 GiB, inferencia con pesos reales e implementación
física según [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
