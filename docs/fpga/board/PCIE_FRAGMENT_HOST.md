# Fragmentos, doble CDC y memoria DDR

## Camino implementado

`ntrpu_pcie_fragment_host.sv` conecta el [control NTRF](PCIE_FRAGMENT_CONTROL.md)
y `ntrpu_pcie_fragment_async.sv` mediante dos IP AMD
`axi_clock_converter:2.1`. Datos: AXI4, direcciones de 64 bits, datos de
512 bits e ID de cuatro bits. Control: AXI-Lite, direcciones y datos de
32 bits, con el recodificador de respuestas ya calificado. Ambos cruces
son asíncronos y tienen tres etapas de sincronización configuradas.

La captura de fragmentos, autorización de destino, receptor y filtro de
capacidad operan en el reloj UI de MIG. Los registros de diagnóstico
expuestos como pines permanecen en ese dominio; el host los consulta por
AXI-Lite. `transport_idle` describe el ensamblador, no las colas CDC ni el
tráfico pendiente aguas arriba. La barrera externa sigue siendo obligatoria.
Reset frío y coordinado en ambos dominios.

## Banco de integración

Se usa el MIG generado para la geometría exploratoria de dos RDIMM x4 con
ECC, 37 bits de dirección y 512 bits de datos. Es el modelo genérico ya
conservado, no la calificación de un DIMM comercial concreto. READY, B y R
de memoria provienen del MIG. El productor host, el aislamiento externo y
el productor de lecturas de retorno son BFMs.

El banco recorre seis ubicaciones: cero, `0x100000040`, `0x800000000`,
`0x1000000000`, `0x1800000000` y `0x1fffffffc0`. Tras cada reserva vuelve
a leer todas las ubicaciones escritas y compara los bytes completos.
Incluye tres destinos fuera de capacidad, duplicados, conflicto, magic
incorrecto, ventana incorrecta, identidades con bits altos, tramas antiguas
tras reutilizar una ventana, STOP durante la escritura y STOP con AW
admitido y W pendiente durante 100 ciclos UI.

Se observan los cinco canales de control y los tres canales de escritura
en ambos lados de los CDC. El revisor comprueba contenido, identidad,
orden, conservación de transferencias, bordes de reloj y retiro del trabajo.
Las capturas de rechazos se consultan por el control real, incluido un
banco congelado mientras llegan rechazos nuevos.

## Tandas y criterios

| Tanda UTC del 2026-10-05 | Perfil | Estado |
| --- | --- | --- |
| `003422Z` y `003449Z` | Lanzamientos iniciales | Script DWORD seleccionado por error; no califican NTRF |
| `003714Z` | Host 4 ns, fase 0,7 ns; UI 5 ns | R2 aprobado y respaldado |
| `003749Z` | Host 7 ns, fase 1,1 ns; UI 5 ns | R2 aprobado y respaldado |

Los primeros lanzamientos conservan sus fuentes, trazas y resultados.
R2 usa `tools/ntrpu_board_pcie_fragment_host.tcl`, comprobado en el
lanzador, la invocación efectiva y los archivos realmente compilados.
Un código de salida cero o un nombre de tanda no demuestra el perfil probado.

Cada perfil nuevo reconstruye 15 reservas/resultados, 35 tramas
(28 aceptadas y siete rechazadas, tres por identidad), 10 escrituras de
memoria (640 bytes), 75 lecturas de retorno (4.800 bytes), 695 escrituras
y 147 lecturas de control; se rechazan 90 escrituras de control y tres
lecturas de direcciones inválidas. Cada revisión detecta 343 alteraciones
de evidencia. Se cotejan 134 archivos compilados: 11 del proyecto y 123
AMD idénticos a la referencia DWORD; también coinciden los 35 archivos
importados del modelo.

## Métricas verificadas

| Medida | Host 4 ns | Host 7 ns |
| --- | --- | --- |
| Tiempo de herramienta | 452,243 s | 506,726 s |
| Final de simulación | 129,648498 µs | 160,673498 µs |
| Transferencias periódicas solicitadas/aceptadas/devueltas | 76/76/76 | 104/104/104 |
| Lecturas en pines: usuario + periódicas | 75 + 76 | 75 + 104 |
| Activaciones observadas por rank 0–3 | 27, 12, 11, 19 | 32, 14, 13, 24 |
| AW de datos: host → UI | 39,298–43,298 ns | 42,398–46,398 ns |
| W de datos: host → UI | 40,298–43,298 ns | 42,398–46,398 ns |
| B de datos: UI → host | 49,702–52,702 ns | 83,602–87,602 ns |

La calibración finaliza a 28,051098 µs en ambos bancos. Se conservan 21
avisos de herramienta y cinco avisos FIFO por perfil. Los 288 y 432 avisos
de lectura sin inicializar se explican, respectivamente, por dos y tres
ráfagas periódicas: se cotejan órdenes, pines, dirección, beats del modelo
y retorno correspondiente. No hay avisos sin esa correspondencia. El
drenaje final observa todos los retornos antes de `$finish`.

Respaldo de cada tanda en `artifacts/ntrpu-board-vivado/<ID>/` y NFS:

- Rápido, `20261005T003714Z-V7-ffn-supervised-fragment-host-fast-r2`:
  608 archivos, 81.049.100 bytes, verificado a las 00:48:15 UTC;
  SHA-256 `a6119f49fc79aa03045310b9312043aa03f19aa8a465b77f34763dd597c30323`.
- Lento, `20261005T003749Z-V7-ffn-supervised-fragment-host-slow-r2`:
  608 archivos, 81.104.056 bytes, verificado a las 00:49:43 UTC;
  SHA-256 `b029b489d89ae9c0b2d905707785d00ddb55cb59319a2ccb55ad753eadf67743`.
- Los originales `003422Z` y `003449Z` también están respaldados, con
  estado `NOT_QUALIFIED_WRONG_TCL`; sus SHA-256 son
  `e751a7ec0b9f3ed125f8c7f274a2a8d4db0607cde980b4df539a335cdec1dd16`
  y `ca678f82b01f8848905f97e1456cde53c5fd6033416f53750ea2f5fb32d1f06b`.

## Síntesis de la composición

`20261005T005058Z-V7-ffn-supervised-fragment-host-synth` revisada y
respaldada: 3.251 LUT (682 de RAM distribuida), 6.277 registros, sin
latches, BRAM, URAM ni DSP. Relojes de 4 y 5 ns; WNS interno sin colocar
1,972 ns, WHS 0,013 ns y WPWS 1,468 ns. Los 844 inputs y 1.289 outputs
carecen todavía de presupuesto temporal: esto no cierra timing de placa.

CDC informa 23 cruces de un bit y diez resets sincronizados, todos Info,
con las restricciones y waivers AMD activos. Se revisaron las 15
restricciones de bus skew; margen mínimo estimado 3,851 ns. No se añadieron
excepciones globales. Los 870 avisos de puertos opcionales, sin truncamiento,
coinciden con la composición anterior en 42 módulos especializados AMD;
los siete inputs de síntesis AMD también conservan sus hashes.

Respaldo: 93 archivos, 33.631.620 bytes, SHA-256
`25a1b5e982017a071182829ad239ad95e03e33f4de1ecb45f5129e621875af7d`,
verificado a las 00:58:25 UTC. Netlist funcional SHA-256
`bb99fc3629df9a2e338d340d9b9dfd0c48e049f720427b079fb280100cf619ae`.
Su replay contra MIG frío pasó en `20261005T005916Z`: 7.283 eventos y
tiempos idénticos al RTL de 4 ns; 343 alteraciones de evidencia detectadas.
Se compilaron 123 inputs: banco, netlist y 121 fuentes AMD sin cambios
respecto del replay anterior; ningún RTL de producción fue recompilado.
Solo se adaptan rutas de observación pasiva a los pines conservados por
síntesis; estímulos, comprobaciones y modelo de memoria se mantienen.
Duración: 429,399 s. Respaldo: 676 archivos, 61.243.259 bytes, SHA-256
`69f7cb8ab8ec3b60bcf76a9d1e371c211db83fe0b31cd979bec4bd2a6305eea1`,
verificado a las 01:10:30 UTC. El primer recolector usó por error el mapa
de fuentes anterior; se preservó y se repitió únicamente la recolección
correcta del mismo resultado terminal. El registro de corrección forma
parte del respaldo; no se volvió a ejecutar Vivado.

## Ruteo exploratorio

`20261005T010456Z` implementó el mismo checkpoint con ubicaciones parentales
provisionales `BUFGCE_X0Y60/61`. Quedaron 3.214 LUT y 6.251 registros;
8.949 redes enrutables completas, cero errores de ruteo y dos discontinuidades
explícitas pendientes de clasificación. WNS interno 0,647 ns, WHS 0,010 ns,
WPWS 1,468 ns; margen mínimo de bus skew 2,692 ns en 15 restricciones.

CDC informa 20 cruces y diez resets, frente a 23 y diez en síntesis.
La auditoría `20261005T011154Z` compara los dos checkpoints sin editarlos:
los 268 alias PARTIAL coinciden exactamente con uno de los dos relojes
parentales (133 UI y 135 host). No aparecen redes de datos parcialmente
ruteadas sin explicación.

De 252 registros ASYNC_REG quedan 235. Se eliminaron tres cadenas AMD de
cuatro etapas cuyo último Q no tenía cargas y cinco etapas `rst_d4`
también sin cargas. Las 235 celdas retenidas conservan tipo, propiedad
ASYNC_REG y redes D/Q; diez fanouts pierden únicamente los pines D/PRE de
esas cinco etapas eliminadas. Los tres cruces de profundidad cuatro que
faltan en CDC coinciden con esa diferencia. El replay funcional de la
netlist ruteada sigue siendo una comprobación separada.

Quedan 844 entradas y 1.289 salidas sin presupuesto y 653 redes sin
cargas enrutables; se conservaron los 844 avisos de puertos sin ubicación
de frontera, sin truncarlos ni cambiar su severidad. Esto no califica
interfaces físicas, placa ni puertos de entrada/salida.

Duración: 206,163 s. Respaldo: 39 archivos, 10.785.035 bytes, SHA-256
`8b9acc85e362bca5432cbe03be366b9bab3387b7dbbb9990b74f961a3c514406`,
verificado a las 01:11:17 UTC. El archivo auxiliar `launch.py` de ese
respaldo corresponde al lanzador anterior; la invocación efectiva y el Tcl
son correctos. El lanzador real se conserva mediante un suplemento en la
auditoría siguiente y el checkpoint local, sin modificar el archivo sellado.

Auditoría: 53,982 s; 35 archivos y 6.411.267 bytes, respaldo verificado a
las 01:15:32 UTC; SHA-256
`5f7387fef02cc3344282e503a6a19ff603338ed441c9e0e4e604336947f20e84`.
Incluye el suplemento del lanzador real.

## Observación de la netlist ruteada

`20261005T011556Z` terminó en 436,829 s y queda
`NOT_QUALIFIED_ROUTED_RRESP_OBSERVER`. Sus 147 trazas internas de RRESP
leen `X`: el ruteo dejó desconectado el puerto opcional del bit codificado
que antes observaba el banco. Los demás campos, eventos y tiempos coinciden
con RTL; esa coincidencia parcial no aprueba la prueba completa.
Respaldo de 662 archivos y 59.744.280 bytes verificado a las 01:26:42 UTC,
SHA-256 `8c81fa877062a7f8402c7e74b844bb38a92f12de61352c2abe062d84c50f8d02`.

R2, `20261005T012347Z`, aprueba el replay funcional y mantiene la misma
netlist, estímulos y modelo.
Solo cambia un bit de la expresión pasiva de observación: consulta el Q
del registro retenido `dest_hsdata_ff_reg[32]`. Su INIT, D y R son cero en
la netlist exacta; el revisor comprueba tanto esa estructura como la
jerarquía. No se fuerza ninguna señal ni se corrigen las trazas anteriores.
Coinciden los 7.283 eventos y tiempos completos con RTL; se detectan 343
alteraciones de evidencia. Las 15 reservas, 35 tramas, 10 escrituras y 75
lecturas DDR, 695 escrituras y 147 lecturas de control conservan los
resultados anteriores. Los 121 inputs AMD compilados son idénticos a la
referencia; ningún RTL de producción se recompila. Se conservan 17 avisos
de herramienta y los 288 avisos DDR, todos asociados causalmente al
tráfico periódico. Duración: 428,383 s; final a 129,648498 µs simulados.
Ambos bancos son funcionales, sin SDF; no simulan retardos físicos.

R2 está respaldado localmente y en NFS: 672 archivos, 59.814.358 bytes,
verificado a las 01:34:48 UTC; SHA-256
`dd62c525bca70b809a74235d8eae433146cefd9b5b903e5e5effeeb3772ab6ab`.
La netlist ruteada compartida por ambos intentos tiene SHA-256
`3ecf5ab566123a54518f0a1bb45e4414680c9a05b7054440ec1da7038bd05a36`.

## Límites y siguiente puerta

Las latencias medidas son de simulación. Los modelos FIFO de AMD advierten
que no modelan retardos de sincronización; no prueban metastabilidad ni
latencia física. En AXI-Lite el handshake del receptor precede al
reconocimiento del origen: su diferencia es un retardo de reconocimiento,
no el tiempo de entrega del dato. La integración XDMA nativa tiene su
banco posterior separado. Faltan restricciones físicas completas, reset con tráfico,
DIMM exacto y validación física.

La [integración nativa posterior](PCIE_FRAGMENT_NATIVE.md), `014817Z`,
aprueba BAR0 NTRF, BAR2 con tramas completas, cuatro reservas y
comprobaciones independientes de RQ/RC, ambos CDC y MIG. Esa cobertura
corresponde a su propia tanda y no se atribuye a las pruebas anteriores.
