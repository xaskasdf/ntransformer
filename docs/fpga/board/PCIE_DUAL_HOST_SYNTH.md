# Síntesis del transporte NTRD, reservas y pool

## Resultado y alcance

`20261005T072212Z-V7-ffn-supervised-managed-host-synth` terminó en
368,981 s, pero queda **sin aprobación de implementación**:
`SYNTHESIS_DIAGNOSTIC_UNRESOLVED_RESET_TIMING`. La tanda usa el mismo RTL
de la [simulación PCIe nativa aprobada](PCIE_DUAL_NATIVE_PLAN.md), sin
cambios funcionales. Parte: `xcvu5p-flva2104-2-i`, Vivado 2026.1.

La frontera OOC incluye control NTRD, recolector, puente de reservas,
pool, supervisor frío y cuatro instancias CDC AMD. Excluye endpoint PCIe,
controladores y PHY DDR, cómputo y placa. Sus recursos no son el total
necesario para la NTRPU de cuatro canales.

| Recurso sintetizado | Uso |
| --- | ---: |
| LUT | 9.939 |
| Registros | 23.279 |
| LUT usadas como RAM distribuida | 1.112 |
| Bloques RAM de 36 Kb equivalentes | 15 |
| URAM / DSP / latches | 0 / 0 / 0 |

## Hallazgos que impiden aprobar

Las estimaciones dan WNS +0,178 ns y WHS +0,013 ns, sin endpoints internos
sin reloj o restricción. Sin embargo, metodología informa seis `TIMING-6`
y seis `TIMING-7`: se temporizan relaciones entre gestión y fuente/UI
sin un reloj primario común. El margen positivo no resuelve esas relaciones.

CDC informa 56 cruces `CDC-3`, 48 `CDC-9` y un `CDC-26` en
`external_reset_async` → `management_reset/arststages_ff_reg[0]/PRE`.
La etiqueta del aviso menciona LUTRAM; falta examinar esa red y los
registros reales antes de atribuirle una causa o aplicar excepciones.

Quedan 861 entradas y 3.847 salidas sin retardos externos, además de
cinco orígenes de reloj padre sin ubicación. Se conservan 16 avisos
`SYNTH-6` de salidas RAM sin registro integrado, tres avisos de jerarquía
de waivers del proveedor y los diagnósticos de puertos opcionales y
canales de lectura eliminados. DRC OOC informa cero infracciones, pero
no cubre todas las comprobaciones de una implementación completa.

## Procedencia y siguiente revisión

Snapshot `source-20261005T072200Z.zip`, SHA-256
`56889bb08708815226189c57ad59ba210581ebda11e405532878e620ade10d66`.
Se verificaron 28 entradas de síntesis y tres configuraciones IP.
El asignador original `ntrpu_buffer_pool_allocator.sv` está incluido en
las fuentes, pero no se compila por no estar referenciado; esta variante
usa el asignador decodificado. Se conserva el primer intento del revisor
que detectó esa diferencia de inventario.

Respaldo local/NFS verificado: 121 archivos, 54.693.612 bytes; SHA-256
`864024360682470497ca526d0c457e753861473d400c522d1ef4601a1c037baf`.
No se suprimieron avisos ni se añadieron excepciones de temporización.

El siguiente paso es una auditoría de solo lectura del DCP original con
`ntrpu_board_pcie_managed_reset_audit.tcl`: identificar las cadenas de
liberación, INIT, ASYNC_REG, conexiones D/Q y fuentes CLR/PRE; enumerar
las rutas de los seis pares de relojes señalados. Derivar después las
restricciones justificadas para sus endpoints concretos, en otra revisión.
No ejecutar ruteo como tanda aprobada antes de revisar estos hallazgos.

## Auditoría estructural aprobada

`20261005T073530Z-V7-ffn-supervised-managed-reset-audit` abrió una copia
idéntica del DCP y terminó en 63,707 s. Se verificaron 15 cadenas de tres
FDCE con INIT=0 y ASYNC_REG, reloj y continuidad D/Q correctos. Las 90
rutas max/min de los seis pares señalados terminan exclusivamente en
sus 45 CLR. El reset externo llega directamente a los PRE de tres FDPE
INIT=0/ASYNC_REG, sin un driver interno en esa red.

La revisión independiente rechazó 11 corrupciones. El XDC exportado
conserva todos sus bytes salvo la ruta del comando en un comentario;
se guarda también el primer revisor que exigía igualdad de ese comentario.
Los avisos originales siguen presentes: la auditoría identifica las
estructuras, no cambia restricciones ni aprueba timing físico.

Respaldo local/NFS: 33 archivos, 6.238.214 bytes; SHA-256
`52ae68cd38cf45eabacfc5c1a268b32f5d4c0a12fcf963e7c563a137ad51709c`.
La nueva tanda `ntrpu_board_pcie_managed_reset_constraints.tcl` parte del
DCP original, vuelve a comprobar esas cadenas y aplica excepciones solo
a los 45 CLR identificados. Exporta netlists antes/después para verificar
que no cambie la lógica y repite los informes. No se excluyen pares
completos de relojes ni rutas D/Q.

El intento `20261005T074225Z-V7-ffn-supervised-managed-reset-constraints`
se detuvo después de seleccionar los 45 CLR: la consulta de rutas aún
devolvió objetos para gestión → fuente. No alcanzó a exportar el DCP
modificado ni la netlist posterior. Se conserva como
`FAIL_EXPECTATION_AFTER_AUDITED_CLR_CONSTRAINTS`, sin atribuir todavía
el resultado a una ruta omitida o a la semántica de la consulta.
Respaldo: 34 archivos, 11.755.443 bytes; SHA-256
`cba0f8f31e4ed6e7defb2ae67626e4bfb63ed21298078afa67763c00eb81010f`.

El diagnóstico separado `ntrpu_board_pcie_managed_reset_constraints_probe.tcl`
mantiene exactamente esos 45 endpoints y exporta todos los objetos de
ruta restantes con sus propiedades, además de los informes y ambas
netlists. No añade excepciones para hacer desaparecer el resultado de
la consulta. Su evidencia debe revisarse antes de aprobar la corrección.

## Resultado del diagnóstico de restricciones

`20261005T074710Z-V7-ffn-supervised-managed-reset-constraints-probe`
terminó en 87,347 s. Los 122 objetos que devuelve la consulta están
marcados **False Path**, sin slack ni tiempo requerido. La expectativa
de recibir una lista vacía era insuficiente para clasificar el resultado.

El único cambio ejecutado es la excepción de los 45 CLR auditados.
Desaparecen los seis `TIMING-6` y seis `TIMING-7`; los endpoints internos
temporizados disminuyen exactamente en 45. WNS estimado +1,700 ns,
WHS +0,013 ns, recursos idénticos. Es una corrección del análisis de
restricciones; no representa aceleración física del circuito. Permanecen
los avisos RAM, el contexto de los relojes padre y los retardos externos.

La parte visible de las netlists antes/después coincide. Sin embargo,
Vivado vuelve a exportar un bloque cifrado con bytes diferentes; excluirlo
de la comparación no demuestra equivalencia completa. El resultado queda
`RESET_CONSTRAINT_PROBE_REVIEWED_LOGIC_AUDIT_PENDING`, sin habilitar ruteo.
La comparación siguiente abre ambos DCP y exporta de forma determinista
propiedades de primitivas, conexiones, puertos y EDIF mediante Vivado.

El XDC exportado conserva una consulta de `release_ff_reg` con comodín.
En este DCP se verifican sus 45 coincidencias exactas. Para reconstruir
otra revisión deben mantenerse los controles de cadenas/nombres/cantidad
o emitirse una lista fija de endpoints; no reutilizar la consulta sin ellos.

Respaldo del diagnóstico: 183 archivos, 23.660.229 bytes; SHA-256
`156879d5dc6dd8f30cfd112e1f868dfa624633dfd9c5aec4ad3e4dbaf061ca80`.

La primera exportación exhaustiva, `075734Z`, consultaba todas las
propiedades de cada primitiva. Se detuvo de forma deliberada a los
281,696 s: produjo 608.698.368 bytes parciales y avisos por propiedades
HD que no aplican al contexto de síntesis. Solo se enviaron señales de
terminación a los dos procesos de esta auditoría de lectura, después de
verificar su directorio y `launch.tcl`; los trabajos concurrentes se
conservaron. No acredita equivalencia ni una comparación terminada.
Respaldo: 17 archivos, 44.276.090 bytes; SHA-256
`6b1a9663850cd594251c970db72a11ae2283717384a855a9fbd49c06e6ed944f`.

La revisión separada `ntrpu_board_pcie_managed_logic_edif.tcl` limita el
trabajo a exportar EDIF y la identidad de las primitivas desde ambos DCP,
con un presupuesto de 600 s. Conserva los archivos completos para
comparar parámetros y conexiones sin interpretar como equivalentes
los bytes cifrados de las netlists de simulación.

## Comparación EDIF parcial: corrección de alcance

`20261005T080626Z-V7-ffn-supervised-managed-logic-edif` terminó en
60,407 s. La revisión archivada registró `PASS_MANAGED_RESET_LOGIC_PRESERVED_EDIF`,
pero **se retira la afirmación de preservación completa** tras la auditoría
de auxiliares del 2026-10-05 a las 12:07 UTC. Ambos DCP contienen
38.557 primitivas, 131.645 redes, 4.767 puertos y cinco relojes; sus listas
de identidad de primitivas coinciden byte por byte.

Se compararon los dos EDIF principales de 31.513.502 bytes cada uno.
Las 109 diferencias son una fecha y la permutación de 18 propiedades
sin cambios de valor en seis instancias RAM32M16. El comprobador
`tools/check_ntrpu_managed_edif.py` exige igualdad de todos los demás
bytes, incluidas conexiones, nombres y parámetros INIT. Rechazó 14
alteraciones de inicialización, geometría, conexiones y primitivas.
Sin embargo, Vivado también escribió **189 auxiliares protegidos `.edn`**.
Las dos exportaciones usaron una carpeta común y solo se conservó el
conjunto final. No existe comparación separada de esos auxiliares; por
ello, la igualdad del archivo principal no demuestra la lógica completa.

Los XDC coinciden con el diagnóstico ya revisado, salvo el comentario
con la ruta de exportación. Se conserva el aviso Vivado 12-23575 al abrir
el checkpoint restringido. La observación es parcial y no cierra la
dependencia del diagnóstico anterior. Tampoco acredita equivalencia
secuencial formal ni timing físico. Los replays funcionales posteriores
mantienen sus observaciones independientes, limitadas a los casos probados.

La corrección conserva intactos los archivos y revisiones originales.
Inventario y procedencia: `guarded-owner-prior-edif-caveat-20261005T120700Z.json`,
SHA-256 `c8c1182d6b112d6ea598e0343942c1a577c1763b08935b5cb6978aebb38c3aa0`,
guardado localmente y en la sesión NFS. No reutilizar el PASS histórico
como aprobación de preservación lógica completa.

Respaldo local/NFS verificado a las 08:16:59 UTC: 215 archivos,
20.067.933 bytes; SHA-256
`a28a7b4423fb05bc94d55b7798498d3ac9bcd893fbff8dd2e2091636211c6153`.

La revisión siguiente abre una copia del DCP restringido y examina
únicamente las 16 RAM del pool señaladas por SYNTH-6: registros de salida,
anchos, conexiones y rutas max/min. No altera RTL, restricciones ni
severidades. Los 596 avisos de puertos sin uso se concentran en 38 módulos
FIFO/CDC; su inventario debe contrastarse con los perfiles AXI4/AXI4-Lite y los
canales de lectura desconectados explícitamente. El ruteo exploratorio
sigue pendiente de esa clasificación.

## Caracterización de las RAM del pool

`20261005T081846Z-V7-ffn-supervised-managed-ram-audit` terminó en
62,337 s y pasó la revisión de lectura. Son 14 RAMB36E2 y dos RAMB18E2
(15 bloques equivalentes de 36 Kb), con DOA_REG/DOB_REG=0. Se verificaron
5.708 pines y los 32 pines de reloj pertenecen a `source_clk`. Las 20
rutas max y 20 min exportadas van de RAM a registros FIFO: margen mínimo
estimado +3,706 ns en setup y +0,261 ns en hold. Las 35 entradas del
informe de bus skew tienen margen estimado mínimo +3,851 ns.

El aviso `remaining_reg` corresponde a 72 FDRE: dos puertos × cuatro
descriptores × nueve bits. El RTL actualiza el contador al aceptar AW
y al consumir W; se conserva la implementación en registros.

Esta evidencia caracteriza SYNTH-6; no elimina el aviso ni acredita
timing tras ruteo. Activar registros adicionales en BRAM cambia la
latencia y exigiría otra revisión RTL con su regresión funcional.

Corrección de atribución: el revisor sellado de `080626Z` mencionaba el
checkpoint original como origen de Vivado 12-23575. La posición del
mensaje en el log identifica la apertura del **restringido**; reaparece
en esta tanda. El informe de metodología recién generado contiene solo
16 SYNTH-6 y 1.000 TIMING-18, sin hallazgos críticos. Se conserva el
mensaje y no se supone su causa. La comparación estructural no cambia;
esta corrección figura en el nuevo `review.json` sin modificar el respaldo
anterior.

Respaldo local/NFS verificado a las 08:23:03 UTC: 22 archivos,
6.262.348 bytes; SHA-256
`8456f8ce3d721c02d269b0438d83f7d289219f5340b7019990e68560de892bb7`.

El índice local `ffn-supervised-managed-synth-warning-index.json`, incluido
en el checkpoint de recuperación, reúne los 596 avisos por puerto/módulo.
Siete módulos tienen una ubicación de fuente AMD explícita en el log;
31 módulos FIFO no exponen allí esa ubicación. Ninguno lleva prefijo
`ntrpu_`, pero eso no sustituye revisar la configuración generada y la
jerarquía. Esa revisión continúa pendiente; no se han añadido waivers.

## Implementación exploratoria admitida

La revisión de los wrappers generados confirma AXI4 de 512/64/4 bits en
ingreso y memoria, AXI4-Lite de 32/32 bits en control y tres etapas CDC.
El wrapper público de AMD selecciona FIFO asíncrono AXI y desactiva
contadores, umbrales programables, señales USER, ECC/inyección, estados
opcionales y varias entradas de reset. Los canales AR de los tres caminos
AXI4 están explícitamente inactivos en nuestro RTL; las lecturas AXI-Lite
siguen conectadas y tienen evidencia funcional nativa.

Las fuentes FIFO de síntesis están protegidas. Sus 31 módulos internos
se clasifican como diagnósticos de implementación del proveedor para
esta exploración, sin afirmar una auditoría independiente de su lógica.
La revisión `ffn-supervised-managed-route-preflight.json` conserva la
justificación, hashes y límites. No se suprimen avisos, no se cambian
severidades y no se añaden waivers para los tres objetivos LUTAR-1.

`20261005T083408Z-V7-ffn-supervised-managed-host-route` inicia una
implementación OOC aislada con cuatro hilos y límite de 1.800 s.
Usa el DCP cuya lógica se verificó y comprueba nuevamente los 45 CLR
por nombre, cantidad y primitiva. Solo añade orígenes de reloj **supuestos**:
BUFGCE_X0Y60 a X0Y64 para fuente, host, UI0, UI1 y gestión.

Esto permite medir colocación y ruteo interno; no asigna clocks de PCB,
retardos externos, pines ni pblock padre. El resultado todavía debe pasar
revisión de DRC, timing, CDC, excepciones y rutas parciales. No cierra
V6/V8 ni sustituye las dependencias de DIMM, cuatro canales o cómputo.

## Ruteo y auditoría de límites completados

La tanda `20261005T083408Z-V7-ffn-supervised-managed-host-route` termina
en 417,995 s: 9.684 LUT, 23.231 FF y 15 bloques RAM, sin DSP/URAM.
Con los cinco orígenes supuestos obtiene WNS +0,380 ns, WHS +0,010 ns
y margen mínimo de bus skew de 3,312 ns. Las 28.887 redes ruteables
están ruteadas; no hay errores de ruteo. Permanecen cinco discontinuidades,
un aviso RTSTAT-10 sobre 1.434 redes, 16 SYNTH-6 y 1.000 TIMING-18.
Las 861 entradas y 3.847 salidas carecen de retardos externos por ser OOC.
Estas cifras no acreditan timing de placa ni incluyen MIG PHY o cómputo.

La auditoría de solo lectura `20261005T084534Z-V7-ffn-supervised-managed-route-audit`
clasifica las 660 representaciones jerárquicas de rutas parciales en
exactamente cinco relojes padre. Ninguna red RTSTAT-10 tiene cargas
internas: 1.287 llegan a puertos del límite y 147 son redes vendor sin
carga. De 774 registros ASYNC_REG quedan 723, todos con tipo, INIT,
reloj y conductor D idénticos, incluidos los 78 propios. Se eliminaron
15 FDPE sin cargas internas y nueve cadenas de cuatro FDRE cuyo último
Q no tenía cargas internas. Esto explica CDC-3 56→47; no constituye
una equivalencia formal de todas las optimizaciones. CDC-9 permanece
en 48 y el aviso CDC-26 del reset externo sigue presente.

Respaldos local/NFS verificados:

| Tanda | Archivos / bytes | SHA-256 del archivo |
| --- | --- | --- |
| Ruteo `083408Z` | 49 / 31.118.791 | `d80b65d32efb7a2b7355ccd9d408b7151c42964da49ff8a886c65a0e5d1bd03e` |
| Auditoría `084534Z` | 26 / 18.793.193 | `179b26969c20251b60a5bebe4b31bfd3a1e7f0b1643de8a42811bca511eda81f` |

## Replay funcional de la netlist

La netlist ruteada tiene SHA-256
`6fd52cc5c83bbd022e6ea8d4efed58349e5a36b0fea77ac790733967ce939e73`.
El replay usa el banco frío R2 y dos MIG, sin SDF ni endpoint PCIe nativo.
El adaptador conserva las transacciones y comparaciones: retira el parámetro
ya sintetizado, relocaliza dos observadores optimizados y prolonga el reset
inicial a 200 ns para comprobar la liberación de GSR.

`20261005T085427Z-V7-ffn-supervised-managed-routed-replay` termina en
68,185 s con código 1: `set_property` interpretó el valor que empieza por
`-L` como una opción. Los 13 casos independientes de arranque pasaron,
pero **la simulación principal no llegó a lanzarse**. La revisión R2 usa
`-name`, `-value` y `-objects`, conforme al ejemplo de
[AMD UG900](https://docs.amd.com/r/2021.1-English/ug900-vivado-logic-simulation/Universal-Verification-Methodology-Support).
El primer intento y sus fuentes quedan congelados; no se cambia HDL,
modelo, reloj ni criterio de aceptación para corregir este fallo.

Respaldo del fallo verificado a las 09:01:06 UTC: 379 archivos,
44.847.186 bytes, SHA-256
`8c6d8e94e384df25bf15e602519a2d1d04c0c71632d164034e47ea961b4e96cd`.
La revisión `20261005T090132Z-V7-ffn-supervised-managed-routed-replay-r2`
superó la elaboración y entró en simulación con límite de 2.400 s.
Sus fuentes son `source-20261005T090121Z`, SHA-256
`da6558670819e3a1f99122c6575505f8b825c6bcd0eda128f5808df84dd5aca7`.
El aviso de doble definición de `glbl` se contrastó: ambos cuerpos de
módulo son idénticos.

### Resultado funcional revisado de R2

R2 termina con código 0 en 703,340 s, a los 90,865 µs simulados.
Estado de revisión: `PASS_MANAGED_ROUTED_NETLIST_TWO_MIG_FUNCTIONAL`.
El verificador frío original permanece idéntico; comprueba 18 trabajos,
35 tramas, un rechazo, 16 escrituras DDR (1.024 bytes) y 28 lecturas
comparadas (1.792 bytes), incluidos los cuatro ranks de cada controlador.
En ambos extremos CDC coinciden 35 AW/W/B de datos y 276 AW/W/B más
10 AR/R de control. Las 54 barreras ARM/publicación/ACK incluyen las
respuestas pendientes, y el cierre conserva ocho ciclos fuente sin actividad.

La auditoría identifica 124 archivos compilados en el proyecto principal:
la netlist exacta, el banco adaptado y los modelos, sin RTL de producción
del transporte. Los 35 archivos del ejemplo DDR mantienen su procedencia;
solo cambian las tres macros de geometría/tasa ya documentadas. Pasan
los 13 casos separados de arranque y se rechazan 12 corrupciones de evidencia.

| Medida de este banco funcional, sin SDF | Rango |
| --- | --- |
| Oferta AW de datos → aceptación en destino CDC | 42,919–46,819 ns |
| Oferta W de datos → aceptación en destino CDC | 42,519–47,319 ns |
| Oferta B en fuente → aceptación host | 78,181–85,381 ns |
| Línea → B DDR | 88,498–609,229 ns |
| AR → R del readback DDR | 115–655 ns |
| Línea → ACK de propiedad | 815–1.340 ns |

Son latencias observadas con host de 7,3 ns y fuente/UI de 5 ns;
no miden retardos posruteo ni caudal sostenido. Los 15 avisos de herramienta
coinciden con la referencia, descontando los FIFO comportamentales/IP
standalone y añadiendo el `glbl` idéntico. Los 2.016 avisos del modelo
DDR se asignan íntegramente a 14 lecturas periódicas sin inicializar;
se conservan y verifican sus comandos, pines y retornos. Se drenan las
56 solicitudes periódicas del canal 0 y las 51 del canal 1.

Respaldo local/NFS verificado a las 09:17:21 UTC: 991 archivos,
98.722.233 bytes; SHA-256
`d8191e3760fd7acffc2ed3464e88659c6dd9ba9396f257245fd9f47024c0304b`.
Este PASS cierra el replay frío
funcional de este bloque OOC; no el endpoint PCIe nativo con esta netlist,
la recuperación PERST, el DIMM comercial, los cuatro canales ni V6/V8.
