# R4 con preparación de candidato y fronteras de evaluación

Estado: **protocolo RTL y equivalencia funcional de la netlist validados;
timing interno cerrado a 200 MHz en el montaje ruteado. Integración física
PCIe/HBM pendiente**.

Experimento aislado sobre [cabeceras registradas](PEER_RX_REGISTERED_DISPATCH.md).
La referencia y la cola principal permanecen separadas de esta evaluación.

## Cambio de admisión

Antes, la disponibilidad del receptor elegía un destino y multiplexaba el
descriptor que R4 revalidaba y asignaba en ese mismo ciclo. La variante
`ntrpu_candidate_dispatch` añade una etapa: guarda el descriptor y el destino
candidatos; al ciclo siguiente vuelve a consultar la oferta real antes de
emitir `reserve_take` y el comando receptor.

Preparar un candidato no retira la entrada de su FIFO ni reserva banco,
descriptor o resultado. La única asignación sigue siendo el handshake de
comando de R4. Si desaparece la oferta, se vuelve a elegir entre las
cabeceras actualmente elegibles. Un fallo cancela los candidatos sin reserva;
stop cierra publicaciones nuevas y permite drenar los trabajos encolados.
Las reservas ya comprometidas conservan sus reglas de drenaje.

Se mantienen la validación de alineación/longitud/página en R4, la selección
round-robin, ocho entradas por destino y el pool de ocho bancos de 4 KiB
(dos dedicados por destino y cuatro compartidos). La etapa añade un ciclo
de preparación a cada admisión; no aumenta capacidad ni ancho de bus.

`tools/prepare_ntrpu_candidate_dispatch.py` genera las variantes a partir de
fuentes preservadas, con diferencias y hashes de procedencia. No se sustituye
ningún módulo de referencia.

## Pruebas funcionales

La composición directa, con productor y destino AXI de comportamiento, pasa
los ocho escenarios previos: 87 publicaciones, 85 transferencias y
138.944 bytes; 84 completions OKAY, una SLVERR inyectada y dos trabajos en
cuarentena. El revisor reconstruye 4.769 ciclos, las cabeceras y los 85
candidatos/handshakes, además del payload, bancos, generaciones y completions.
Se mantiene cobertura de reutilización de banco en el mismo flanco.
La comparación independiente conserva los metadatos, bytes y estados
exactos de la tanda anterior. Los ciclos acumulados de los ocho casos
pasan de 4.719 a 4.769; sus pausas dependen del ciclo y no constituyen un
benchmark de ancho de banda.

Dos escenarios adicionales comprueban el intervalo nuevo:

- Fallo antes del compromiso: trabajo 500 en cuarentena, cero asignaciones
  y cero bancos ocupados; desaparece el candidato.
- Stop antes del compromiso: trabajo 501 completa sus 64 beats/4.096 bytes
  y cierra la admisión de publicaciones nuevas.

El banco de contrato `tb_ntrpu_candidate_credit.sv` retira la oferta del
destino seleccionado y comprueba que avance el otro. También retira ambas
ofertas durante ocho ciclos y comprueba reanudación sin consumir las colas.
Este banco usa ofertas controladas, no modela el consumo físico de bancos;
esa parte corresponde a la composición integrada anterior.

Los revisores rechazan 14 alteraciones de trazas integradas y tres del
contrato de crédito: corrupción, pérdidas, identidad/generación incorrecta,
validación falsa, crédito obsoleto y candidato alterado.

## Contexto de medición física

Se generan dos montajes iguales, uno para la referencia registrada y otro
para el candidato. Cada puerto de datos/control del núcleo termina en
registros de muestreo: 2.864 registros de frontera y un BUFGCE explícito.
El periodo es 5 ns, con incertidumbre setup/hold de 0,1/0,05 ns.

**Estos montajes son instrumentos de timing, no adaptadores de transporte.**
Registrar independientemente ready y valid no implementa una frontera
elástica utilizable en producción. Las pruebas de protocolo conectan el
núcleo directamente; no atribuyen funcionamiento de transporte al montaje.
Los puertos externos del muestreador carecen de presupuesto de placa y se
registran como no cualificados. Los informes usados para comparar incluyen
solo caminos entre registros; no se añaden false paths ni multicycle paths.

Esto separa el núcleo del hold artificial observado desde puertos externos
idealizados en la tanda anterior. No transforma un informe de síntesis o
placement en cierre de timing ruteado. La capacidad y conectividad reales
de PCIe/P2P, HBM, MIG, CDC y recuperación de época siguen fuera de esta tanda.

## Comparación de síntesis

Tanda `20261005T075632519511Z-candidate`, Vivado 2026.1,
`xcvu5p-flva2104-2-i`. Ambas columnas incluyen el mismo montaje de frontera;
no se comparan directamente con los márgenes del contexto OOC anterior.

| Métrica | Cabeceras registradas | Candidato registrado |
| --- | ---: | ---: |
| LUT totales | 5.989 | 6.176 |
| Registros totales, incluidas fronteras | 15.876 | 16.116 |
| BRAM equivalentes de 36 Kib | 15 | 15 |
| CARRY8 | 249 | 249 |
| Setup entre registros, estimado | −0,499 ns | +0,869 ns |
| Hold entre registros, estimado | −0,126 ns | −0,126 ns |
| Niveles lógicos del peor setup | 21 | 14 |

La diferencia es de 187 LUT y 240 registros, sin BRAM adicional. Ambas
síntesis tienen cero latches y cero cajas negras. El camino crítico aún
nace en una generación del asignador y termina en el enable de un descriptor,
pero su retardo estimado baja de 5,158 a 3,790 ns. El peor hold ahora parte
de un registro de frontera hacia la RAM de la FIFO; ya no nace en un puerto
idealizado con latencia de lanzamiento cero. Sigue siendo un incumplimiento
estimado que debe revisarse físicamente, no una excepción que ignorar.

## Resultado de placement

Ambas variantes usan el BUFGCE explícito situado en `BUFGCE_X0Y60`, periodo
de 5 ns y las mismas incertidumbres. Se conservaron sus checkpoints.

| Camino entre registros | Referencia registrada | Candidato |
| --- | ---: | ---: |
| Peor setup estimado | −0,076 ns | +0,522 ns |
| Peor hold estimado | −0,165 ns | −0,194 ns |

El peor setup del candidato parte del registro de reset de la frontera y
termina en el enable de estado de un descriptor, con 12 niveles lógicos.
El peor hold parte de `boundary_payload_data_reg[463]` hacia una entrada de
la BRAM del pool; es un camino entre registros/memoria del montaje. No se
han eliminado esos caminos del análisis ni se ha relajado el periodo.
Al terminar placement, el hold seguía fallando. La sección siguiente registra
el resultado posterior del ruteo.

La primera tanda de placement, `20261005T080802786696Z-place`, guardó la
referencia completa y se interrumpió durante el candidato a los 577,277 s
para separar su presupuesto de ejecución. Su estado global permanece
`FAILED_OR_TIMEOUT`, con salida −15 y `timed_out=false`; no se presenta como
una tanda doble completada. El candidato se ejecutó solo desde síntesis en
`20261005T081818086830Z-candidate-place`, terminando en 205,394 s.

La referencia empleó dos hilos y el reintento del candidato cuatro; ambos
con prioridad reducida. La comparación física es diagnóstica y no controla
esa diferencia de ejecución. La comparación de síntesis sí utilizó dos
hilos para ambas variantes. Estas tandas de placement no solicitaron ruteo.

## Ruteo y cierre interno a 200 MHz

La tanda `20261005T183553994957Z-route` reutilizó el checkpoint colocado,
sin repetir síntesis ni simulación RTL y sin modificar lógica o restricciones.
Terminó en 728,328 s con cuatro CPU y prioridad reducida, dentro del límite
de 900 s. Vivado 2026.1 conserva el mismo dispositivo, BUFGCE, periodo de
5 ns e incertidumbres setup/hold de 0,1/0,05 ns.

| Comprobación final | Resultado |
| --- | ---: |
| Peor setup entre registros | +0,182 ns |
| Peor hold entre registros | +0,007 ns |
| Margen mínimo de ancho de pulso | +1,958 ns |
| Extremos con fallo de setup/hold/pulso | 0 / 0 / 0 |
| Redes ruteables / completamente ruteadas | 17.653 / 17.653 |
| Redes con errores de ruteo | 0 |
| Extremos internos sin restricción de timing | 0 |
| LUT / registros / BRAM equivalentes de 36 Kib | 6.134 / 16.116 / 15 |

El peor setup nace en una generación del banco 6 y termina en
`fill_generation_reg[44]`, con 14 niveles lógicos. El peor hold va de la
RAM de la cola 1 a su registro de cabecera. El margen de hold es de solo
7 ps: este cierre pertenece al checkpoint y contexto medidos; la integración
debe volver a implementarse y verificarse. No se extrapola una Fmax mayor.

El ruteador detectó congestión local en `return_generation[71]` y `[72]`.
Una primera solución sin solapamientos tenía setup −0,540 ns y hold
+0,007 ns; una iteración posterior produjo los márgenes finales positivos.
Se conservan los diagnósticos intermedios, sin confundirlos con el cierre.

No hay errores DRC ni advertencias críticas. Sí permanece `RTSTAT-10`:
1.795 salidas del muestreador no tienen cargas físicas externas. La auditoría
`20261005T185041481946Z-boundary`, de 47,312 s y dos CPU, expandió la lista
completa: cada red proviene de un registro de frontera, termina en un puerto
externo único y tiene cero cargas internas. El checkpoint y las restricciones
se mantuvieron intactos; no se añadió ninguna exención.

Se mantienen 1.069 entradas y 1.795 salidas sin presupuestos externos.
Los avisos por ausencia de `HD.PARTPIN_LOCS` y `TIMING-18` pertenecen a esa
frontera no cualificada. Metodología también informa 16 avisos `SYNTH-6`
por BRAM sin registro de salida fusionado y el aviso `CLKC-3` por fijar el
BUFGCE mediante LOC. El análisis OOC no ejecuta todos los DRC de una placa.
**El resultado acredita timing interno del montaje, no una tarjeta lista
para desplegar ni compatibilidad física PCIe/HBM.**

Se exportaron el checkpoint ruteado y netlists funcionales del montaje y
del núcleo. La comprobación posterior usa la netlist completa, como se
describe a continuación; no repite síntesis, placement ni ruteo.

## Equivalencia funcional de la netlist ruteada

La tanda `20261005T220923988722Z-netlist` compara la netlist completa contra
el RTL original. Terminó en 162,212 s con dos CPU y prioridad reducida,
dentro del límite de 900 s. La extracción `write_verilog -cell core` cambió los nombres
y la distribución de puertos durante la optimización: **no conserva la
interfaz RTL para conectarla directamente al banco**. Se mantiene intacto
`candidate_fixture_funcsim.v`, con SHA-256
`28f8508054e7bfd746a29efdb91dca5f7111b10150a017d130cf0913a1fa3b09`.

El banco aplica la misma historia de entradas al núcleo RTL y al montaje
ruteado. Compara las salidas con dos flancos de diferencia: uno por el
muestreo de entrada y otro por el de salida. Comprueba los 28 puertos de
salida; datos y metadatos se comparan cuando valid está activo, incluso
durante contrapresión. Esta reproducción de estímulos no convierte los
registros de frontera en un adaptador ready/valid utilizable en producción.

La referencia usa nombres `ref_*`, con transformación reversible verificada,
para evitar colisiones con módulos de la netlist. Las fuentes generadas
originales coinciden por hash con la tanda RTL anterior. El arranque espera
150 ns para respetar los 100 ns de GSR del modelo estándar. No se usan
`force`, SDF ni opciones que desactiven comprobaciones de timing.

| Comprobación | Resultado |
| --- | ---: |
| Escenarios completos | 10 |
| Ciclos comparados, principales + adicionales | 4.772 + 197 |
| Publicaciones / transferencias completas / cuarentenas | 89 / 86 / 3 |
| Payload ruteado verificado | 143.040 bytes |
| Divergencias funcionales detectadas | 0 |
| Alteraciones de traza rechazadas por revisores | 18 |

Se conservan los casos de saturación, reutilización de banco en el mismo
flanco, errores, barreras, stop y fallo antes de reservar. El revisor compara
independientemente los fetches, beats y completions observados en los puertos
ruteados. Los monitores internos de reservas, generaciones y candidatos
pertenecen a la referencia RTL; no se presentan como observabilidad interna
reconstruida de la netlist. La diferencia de ciclos frente a la tanda anterior
proviene de la fase de arranque y pausas dependientes del ciclo; no es una
medición de rendimiento.

La auditoría conserva los avisos de declaraciones posteriores del banco,
`LIBRARY_PATH` y redefinición de `glbl`; esta última sustituye una definición
estándar por otra cuya equivalencia textual normalizada se verificó. No hubo
errores ni advertencias críticas. La prueba es funcional, sin retardos SDF;
el cierre físico sigue sustentado por STA sobre el checkpoint anterior.

## Reproducción y respaldo

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_candidate_dispatch.py /tanda/output --timeout 900
python3 -B tools/check_ntrpu_candidate_dispatch.py /tanda
python3 -B tools/run_ntrpu_candidate_credit.py /credito/output --timeout 120
python3 -B tools/check_ntrpu_candidate_credit.py /credito
# Placement de cada variante por separado, sin repetir simulación/síntesis:
python3 -B tools/run_ntrpu_candidate_place.py /placement/output \
  --suite /tanda/output/suite --variant candidate --threads 4 --timeout 600
# Usar otra salida y --variant registered para la referencia.
python3 -B tools/run_ntrpu_candidate_route.py /ruteo/output \
  --placed /placement/output --threads 4 --timeout 900
python3 -B tools/check_ntrpu_candidate_route.py /ruteo \
  --placed /placement/output
python3 -B tools/run_ntrpu_candidate_boundary_audit.py /frontera/output \
  --routed /ruteo/output --threads 2 --timeout 180
python3 -B tools/run_ntrpu_candidate_netlist.py /netlist/output \
  --routed /ruteo/output --timeout 900
python3 -B tools/check_ntrpu_candidate_netlist.py /netlist
```

Cada salida debe ser nueva. Los runners limitan CPU, prioridad y duración,
comprueban fuentes antes/después y producen ZIP con SHA-256, CRC y manifiesto
por archivo. El montaje físico no solicita ruteo automáticamente.
Respaldo local: `D:\AI\artifacts\ntrpu-candidate-dispatch\20261005T075312418258Z`.
Las cuatro tandas, revisores, procedencia de checkpoints y fuentes se
consolidan en `20261005-candidate-dispatch-handoff.zip`. Se conserva también
el intento interrumpido y la evolución de los scripts para seleccionar una
sola variante, sin cambios en la lógica sintetizada.

El ruteo y la auditoría de frontera se respaldan por separado en
`D:\AI\artifacts\ntrpu-candidate-route\20261005T183421914699Z`, con
`20261005-candidate-route-handoff.zip`, manifiestos, revisores e informes.
La revisión física comprueba procedencia del checkpoint, fuentes, cobertura
interna, setup/hold/pulso, ruteo completo y ausencia de excepciones de timing.
La revisión de frontera compara las 1.795 conexiones contra todos los puertos
de salida declarados, sin limitarse al extracto abreviado del informe DRC.

La simulación ruteada se respalda en
`C:\Users\xasko\artifacts\ntrpu-candidate-netlist\20261005T220701402665Z`,
con `20261005-candidate-netlist-handoff.zip`, fuentes, trazas, revisores,
procedencia y manifiestos SHA-256/CRC. Se utilizó C: porque D: agotó espacio
durante la extracción; el ZIP íntegro llegó a D:, pero su extracción quedó
incompleta y no debe confundirse con el respaldo verificado en C:.

## Criterios para continuar

La [frontera elástica de transporte](PEER_RX_CANDIDATE_TRANSPORT.md) ya añade
colas con handshakes y validación funcional de stop, fallos y barreras.
Su evaluación física posterior acredita timing interno y equivalencia de netlist;
la integración PCIe/HBM sigue pendiente.
Medir el coste
de la etapa de admisión con bloques de 64 B, 512 B y 4 KiB, tráfico equilibrado
y destino bloqueado, antes de atribuirle una mejora de ancho de banda.
