# Drenaje AXI4: capacidad y ambos extremos del CDC

## Capacidad probada

El monitor original permanece intacto. La tanda `20261005T212032Z` usa
`WRITE_DEPTH=32`, ocho lecturas por ID y los 16 IDs. Pasa once escenarios:
32 escrituras pendientes, 128 lecturas agregadas, vuelta de índices,
reemplazo de entradas llenas en el mismo flanco y W completa/parcial antes
de AW. Incluye ráfagas de 1/2/4/16/64/256 transferencias; las de 256 usan
16 bytes por transferencia para respetar 4 KiB.

Los cuatro casos sanos completan 384 escrituras y 480 lecturas. Se observan
7.288 transferencias W y 1.344 R. Un caso mantiene simultáneamente las
32 escrituras y 128 lecturas durante 96 altas/retiros consecutivos. Los
siete casos inválidos comprueban overflow, WLAST ausente/largo, B prematura
y RLAST erróneo con inventario lleno. El fallo permanece con un tag nuevo.

La adenda revisa las mismas trazas y rechaza 22 corrupciones de evidencia.
Conserva el diagnóstico inicial: el selector de una corrupción R buscaba en
un caso que solo escribía. No hubo repetición de Vivado para esa corrección.

Síntesis `212342Z`: **2.482 LUT / 2.824 FF**, incluyendo 24 LUT de memoria
distribuida; incremento de 44 LUT y 53 FF frente al perfil de ocho escrituras.
Sin BRAM/DSP/latches/cajas negras. A 7 ns: WNS +3,673 ns, WHS +0,020 ns.
Faltan retardos de 1.357 entradas/101 salidas y colocación física.

## Composición funcional con IP real

`ntrpu_axi4_pair_drain.sv` compone dos monitores de capacidad 32/8, dos
mailboxes de generación/tag y doce cruces XPM de fuentes de fallo
registradas. Primero solicita la prueba host y después la del dispositivo.
La fuente host debe permanecer cerrada de forma independiente; este bloque
es pasivo y no controla admisión, READY ni respuestas inválidas.

La tanda R2 `214547Z` usa Clock Converter AMD AXI4 real, perfil 512/64/4,
relojes host de 7 ns, gestión de 10 ns y dispositivo de **5 y 37 ns**.
REGION y QOS quedan fijados en cero. En cada perfil:

- Pasan 24 escrituras y 24 lecturas, con **1.045 transferencias W y 1.045 R**.
  Se comparan cada payload de 512 bits, WSTRB, ID y atributos de dirección
  a ambos lados. Incluye direcciones sobre 4 GiB, W anterior a AW,
  respuestas B reordenadas entre IDs y R intercaladas.
- Retener B/R y detener el dispositivo mantiene la prueba pendiente. Al
  reanudar y drenar, se obtiene el primer certificado. Un tag nuevo con
  host detenido queda pendiente; cambiar generación exige otra prueba.
- Después del tercer certificado se ofrece una R huérfana real en D,
  aceptada por el CDC. Se detiene D 1 ps después del registro de fallo:
  gestión revoca en **21,9/22,9 ns**, sin otro flanco D. Al reanudar, la
  respuesta llega a H y ambos monitores conservan su fallo. Una identidad
  nueva no abre mailboxes.

La respuesta inválida **sí atraviesa** esta composición pasiva. No hay
contención AXI4 ni un endpoint XDMA protegido en este banco. Algunos títulos
heredados de los informes de orquestación mencionan contención; el alcance
autoritativo de esta tanda es detección, transporte y revocación externa.

La revisión R2 reconstruye las transacciones, comprueba todas las
observaciones de reloj y rechaza 26 corrupciones. Se verifican las fuentes
compiladas, parámetros IP y 86 archivos de la instalación AMD. La primera
tanda `213844Z` se conserva como diagnóstico: REGION sin conectar y un
import incorrecto del revisor. La aprobación corresponde a R2.

## Límite del modelo y controles negativos

AMD emite cinco avisos por perfil: sus FIFO de relojes independientes
representan las transacciones, pero **no modelan retardos de sincronización**.
No convertir el caudal o latencia de datos simulados en timing físico.
Los demás avisos revisados son entorno XSIM, parámetro glbl y límites de
visualización de las matrices del banco; sus datos sí aparecen en las trazas.

La tanda `214938Z` compila dos copias incorrectas privadas contra el banco
R2 intacto, con dispositivo de 5 ns:

| Mutación | Aserción que la detecta |
| --- | --- |
| Se pierde la fuente R de fallo D hacia gestión | Reporte obligatorio con D detenido, 26.545.001 ps |
| El mailbox mantiene ACK de un tag anterior | Invalidación inmediata al cambiar tag, 25.455.502 ps |

## Síntesis de la composición pasiva

`215303Z` mide **5.320 LUT / 6.648 FF**, incluyendo 48 LUT de memoria
distribuida; sin BRAM, URAM, DSP, latches ni cajas negras. El Clock Converter
de datos AMD queda fuera de este top OOC: estos recursos corresponden a los
dos inventarios, mailboxes y transporte de alarmas.

Se revisan **16 CDC-3 informativos**, todos de tres etapas: cuatro controles
de mailbox y doce fuentes de fallo. Cada fuente parte directamente del
registro del monitor, sin exigir otro flanco de su reloj. Las excepciones
son 16 false paths a primeras etapas y dos límites XPM de datos agrupados
de 5/7 ns. No se añaden excepciones generales entre relojes.

Con H/D/gestión a 7/5/10 ns: WNS +1,773 ns, WHS +0,020 ns y WPWS
+1,968 ns. Faltan retardos de 2.711 entradas/87 salidas y colocación/ruteo.
Los nueve avisos se explican por señales XPM sin uso y estimación de reloj
OOC. Hay 6.644 registros INIT=0 y cuatro INIT=1; estos últimos corresponden
a los estados iniciales one-hot de los mailboxes. Todos tienen
inicialización explícita. No se ha probado equivalencia funcional del netlist.

## Evidencia y dependencias pendientes

Todos los IDs de esta página tienen prefijo `20261005T`; las carpetas
completas se encuentran en `artifacts/ntrpu-board-vivado/<ID>/` y NFS.

| Tanda | SHA-256 del archivo |
| --- | --- |
| `212032Z-V4-ffn-supervised-axi4-capacity` | `251f7ea84ef07b15b59e00a927d26cec304a2f6a415827c640198abdbcf128c4` |
| `212342Z-V4-ffn-supervised-axi4-capacity-synth` | `6aa031b29bedb32e28d8380386ddf6c67beaf2e3b53ef2c3bcb291534854fb36` |
| `213844Z-V4-ffn-supervised-axi4-pair`, diagnóstico | `938192be4e22291e5f5fb2ee54a14f8c138e7997b1f5449874750d757853f47c` |
| `214547Z-V4-ffn-supervised-axi4-pair-r2` | `99d0c66e4b5d0899b18c344c7f22032291b5e8a95304fed2dfdfa5eb910b7c6d` |
| `214938Z-V4-ffn-supervised-axi4-pair-mutants` | `1e00f6d5297bd2885c1b434f64b9a0eba087f26756872f07f785cd798032ca3b` |
| `215303Z-V4-ffn-supervised-axi4-pair-synth` | `dbc574b3f784be90e3b61167da636e62272d42942fe6a3c03ed7f4730ebd318c` |

Adendas de capacidad y síntesis:
`3a997774ce0ba634f3ab0ab5397f9323427794e40860b9fa1ec14d2882d31948` y
`7950893b927a20d15a0f10823d1f28c9f0d70ddb392caae7078f21d7bf80e89d`.
Adenda de síntesis/CDC del par:
`7bcce472490bf460677bebf27dd809af3238da494f5a28616a325a309e972e72`.

El [plan completo](AXI4_DRAIN_PLAN.md) sigue exigiendo
frontera activa con grupo finito y contención de respuestas,
inventario de las colas protegidas XDMA y coordinación global. Las pruebas
externas no autorizan reset del endpoint ni reutilización NTRD.
