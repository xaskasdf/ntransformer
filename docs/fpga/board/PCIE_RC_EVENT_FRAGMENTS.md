# Validación paralela de fragmentos RC

Resultados históricos de R3. La revisión posterior
[R5](PCIE_RC_EVENT_PAYLOAD.md) conserva las salidas, reduce recursos y
mejora el timing; pasa también los bancos completos en su propio netlist.

Revisión RTL R3, 2026-10-07. Archivo:
`fpga/rtl/ntrpu_pcie_rc_event_stage_r3.sv`. Conserva la interfaz de la
[etapa predecodificada R1](PCIE_RC_EVENT_PREDECODE.md), dos etapas y un
beat por ciclo con hasta cuatro cabeceras/eventos. Cada proyecto elige
explícitamente una sola revisión del módulo `ntrpu_pcie_rc_event_stage`.

## Cambio y conservación del contrato

Los dieciséis DWORD calculan en paralelo el fragmento al que pertenecen,
su posición, los byte enables esperados y los errores de payload. Cada
lane busca el último SOP y los EOP anteriores; usa el contexto retenido
cuando no hay una nueva cabecera. El contador ya no atraviesa dieciséis
validaciones consecutivas. El recorrido ordenado de cabeceras/EOP conserva
los eventos, contexto, errores locales y pérdida de obligaciones.

Se mantienen requester y metadatos esperados independientes, la propiedad
del tag, contexto hasta EOP, wrap, observación pausada y reset frío.
`packet_checked` sigue siendo una validación local: no autoriza vacío ni
ACK. El banco R7 proporciona metadatos/pertenencia en los bancos shadow;
su integración con asignación, escrituras, forwarding y contabilidad
global aún está pendiente. No se redujeron RC4/RQ2 ni capacidad.

## Pruebas y fallo descubierto

La primera implementación paralela R2 pasó los 22.339 ciclos/68 escenarios
del shadow de cuatro estados y las mutaciones reales. También pasó el
shadow binario y los 370 ciclos dirigidos. Sin embargo, la nueva matriz
de fragmentación detectó un fallo en la fila 2: el primer beat aceptado,
sin cabecera, devolvía errores X en lugar de `0x0404`.

Los bloques `always @*` por lane podían quedar sin evaluar al comienzo
de la simulación con entradas estables. R3 cambia únicamente esos bloques
a `always_comb`, que también los evalúa al inicio. Se conservaron R2,
la ejecución fallida y los vectores; el paso de su banco anterior no
autoriza sintetizar R2 como revisión aceptada.

R3 pasó los 22.339 ciclos/68 escenarios, idénticos a R1. Las mutaciones
ejecutadas en XSIM siguen fallando en los ciclos 132 (omisión RC) y 133
(metadatos ausentes); el verificador rechazó diez corrupciones del registro.

La tanda binaria R3 pasó tres bancos, revisados independientemente:

| Banco | Cobertura |
| --- | --- |
| Shadow R7 | 22.330 ciclos, 67 escenarios; 256 tags, 257 lecturas y 1.052.672 bytes pendientes |
| Casos límite | 370 ciclos, 15 fases; reset, wrap, pausa, stalls y 256 beats consecutivos RC4 |
| Matriz de fragmentos | 281.904 filas, 80.544 casos y 120.816 beats aceptados/retirados |

La matriz cruza los 16 subconjuntos físicos de SOP, los 2.517 conjuntos
de cero a cuatro posiciones EOP y dos estados iniciales: sin contexto,
o con un paquete de 32 DWORD cuyo contexto ya vio 16 DWORD. Incluye
disposiciones inválidas. Distribuye ocho tamaños y distintas direcciones
y byte enables; no cruza exhaustivamente esos campos ni todas las
codificaciones malformadas de punteros.

Cada fila compara todas las salidas registradas con el modelo secuencial
independiente y con una copia R1 cuyo único cambio funcionalmente neutro
es renombrar el módulo y su instancia aritmética. Quince corrupciones,
cinco por banco, se rechazan. La matriz es binaria; el shadow original
conserva por separado el caso X/Z excluido de la reproducción sintetizada.

## Síntesis y reproducción del netlist

La síntesis R3 de la tanda `20261007T205728Z` está respaldada y revisada.
Mantiene `xcvu5p-flva2104-2-i`, reloj de 4 ns, I/O explícito de cero y
ninguna excepción temporal válida o ignorada.

| Medida OOC | R1 secuencial | R3 paralela |
| --- | ---: | ---: |
| LUT / FF | 9.639 / 1.266 | 7.333 / 1.223 |
| CARRY8 | 160 | 186 |
| WNS / TNS (ns) | −2,624 / −39,825 | −1,263 / −8,049 |
| Setup: fallidos / total | 46 / 3.150 | 7 / 3.100 |
| WHS / THS (ns) | −0,075 / −32,734 | −0,075 / −32,329 |
| Hold: fallidos / total | 596 / 3.150 | 550 / 3.100 |
| Peor retardo / niveles | 6,606 ns / 24 | 5,245 ns / 20 |

El margen setup mejora **1,361 ns**, con **2.306 LUT y 43 FF menos**;
usa 26 CARRY8 adicionales. Aún fallan setup y hold. El peor camino va de
`arithmetic/header_seen_reg[1]_rep__1/C` a `packet_checked_reg[0]/D`.
Otros caminos fallidos parten de `end_map_stage_reg[0]`: la selección
por SOP/EOP todavía precede a los controles de payload.
El detalle atraviesa `context_delivered`, el cálculo de
`payload_lane[4].enable_expected` y la acumulación de `context_faults[10]`.
Los 5,245 ns incluyen 1,222 ns de lógica y 4,023 ns de interconexión
estimada sobre celdas sin colocar; no son una medida de ruteo físico.

No hay latches, BRAM, URAM ni DSP. Los doce controles de cobertura temporal
dan cero incidencias; quedan TIMING-16 (5) y XDCH-2 (1.334). El EDIF final
tiene 10.101 primitivas y 13.443 declaraciones de net sin bucles; ocho
controles del auditor fueron rechazados. Se registran 119 declaraciones
vacías, incluida una anotación duplicada sin conexiones; no se admiten
declaraciones duplicadas conectadas.

Se conservan 57 avisos de puertos constantes internos `__GB0/2/4`, las
conversiones de igualdad y la eliminación de `data_stage_reg`, usado para
detectar X en simulación. La estimación OOC no incluye colocación/ruteo
del conjunto y no establece equivalencia X/Z.

El netlist funcional exacto pasó los tres bancos: **304.604 filas binarias**
en total, todas idénticas a las del RTL y reconstruidas por el modelo
independiente. La matriz también coincidió fila por fila con la referencia
R1 renombrada; quince corrupciones del registro fueron rechazadas. Se
verificaron el hash del netlist compilado y la ausencia de un DUT RTL.
R7 y R1 se compilaron exclusivamente como referencias del banco.

Los tres avisos de `glbl` duplicado corresponden a cuerpos archivados
idénticos. No se aplicó SDF: esta reproducción confirma los resultados
funcionales de los vectores binarios, no los tiempos de propagación.
La simulación RTL binaria duró 172,220 s, la síntesis 300,456 s y la
reproducción del netlist 608,372 s; son tiempos de ejecución en la VM,
no latencias del circuito. Setup, hold e integración siguen abiertos.

## Tandas y respaldo

Prefijo de los IDs: `20261007T`; rutas completas y hashes en las revisiones
de `build/cloud-setup/20261002-board/` y certificados bajo
`artifacts/ntrpu-board-vivado/<ID>/`, con respaldo físico en D: y en NFS.

| ID sin el prefijo de fecha | Resultado |
| --- | --- |
| `202854Z-V4-ffn-supervised-pcie-rc-event-fragment-rtl` | R2 pasó el shadow original; después invalidado para admisión por la matriz ampliada |
| `203714Z-V4-ffn-supervised-pcie-rc-event-fragment-binary-rtl` | R2: fallo de arranque X en fila 2 de la matriz, conservado y revisado |
| `204317Z-V4-ffn-supervised-pcie-rc-event-fragment-rtl-r1` | R3: shadow de cuatro estados y mutaciones revisados |
| `204904Z-V4-ffn-supervised-pcie-rc-event-fragment-binary-rtl-r1` | R3: los tres bancos binarios revisados; quince controles rechazados |
| `205728Z-V4-ffn-supervised-pcie-rc-event-fragment-synth-r1` | R3: síntesis y EDIF revisados; mejora de recursos/setup, timing todavía fallido |
| `210435Z-V4-ffn-supervised-pcie-rc-event-fragment-binary-netlist-r1` | R3: los tres bancos reproducidos en su netlist; todas las filas idénticas al RTL |

Las seis tandas tienen revisión independiente y respaldo verificado.
Netlist usado: SHA-256
`5b3ed8d754c28ad4555559f0778b3ac77b7ef33e19fc58ff8cfe683f9193bbd2`.

Cuatro archivos locales de evidencia ya certificados se comprimieron
con NTFS después de verificar sus bytes contra los archivos de respaldo.
Los hashes lógicos no cambiaron y no se borró ningún archivo.

## Dependencias abiertas

La continuación R4 ya fue [simulada, sintetizada y reproducida en su
netlist](PCIE_RC_EVENT_PREFIX.md). Conserva las salidas, pero empeora
el peor camino a 5,445 ns y añade recursos. R3 sigue siendo la mejor
variante de esa comparación. R5 la supera posteriormente; los resultados
anteriores de este documento son de R3.

El experimento R4 adelantó a captura la selección de fragmento y los
prefijos de posición que solo dependen de SOP/EOP. El contexto retenido
debe seguir calculándose con el estado correcto del ciclo de validación.
Por lane, registrar qué cabecera anterior se selecciona, si un EOP ya
cerró ese fragmento y la distancia desde SOP. Sin nuevo SOP ni EOP previo,
seguir usando `context_seen + lane` y `context_active` del ciclo actual;
no capturar anticipadamente ese estado. Aplicar `discontinue` al contexto
heredado antes de sustituirlo por una nueva cabecera. Verificar también
seen inactivo, wrap de once bits y errores de EOP después de un cierre.
Medir si esa separación reduce el camino completo. Si continúa dominando
el cálculo de byte enables, evaluar predicados de payload antes de elegir
los metadatos del fragmento, conservando los mismos anchos y resultados.
Cualquier revisión debe conservar los tres bancos y las mutaciones,
y reproducir su propio netlist. Si se añaden etapas, definir primero
alineación de metadatos y obligaciones en vuelo; no retrasar silenciosamente
la revocación de vacío/ACK ni introducir burbujas para conseguir timing.

Permanecen abiertos los criterios de [VIVADO_QUEUE.md](VIVADO_QUEUE.md):
pipeline completo, identidad BDF/host, siete contratos coherentes, CDC,
reset nativo de 50 ms, DIMM exacto, cuatro DDR/512 GiB, inferencia con pesos
reales y cierre físico/ECAD.
