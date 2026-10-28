# Pérdida de enlace y reset del transporte NTRD

## Estado y dependencia

Extensión V4/V7 en desarrollo. El primer adaptador de frontera ya tiene
RTL y un banco con CDC AMD. El [guard automático](HOST_LOSS_GUARD.md) pasa
diez escenarios aislados, pero su timing e integración NTRD/PCIe siguen pendientes.
La tanda `20261005T063837Z-V7-ffn-supervised-managed-native` aprobó su
banco frío y revisión; el respaldo local/NFS está verificado. Véanse
[evidencia y límites](PCIE_DUAL_NATIVE_PLAN.md). Conservar sus fuentes
y crear una revisión separada.

`ntrpu_pcie_fragment_dual_host_cold.sv` no implementa este contrato.
Actualmente recibe resets de transporte externos, no observa pérdida
del enlace y conserva la IRQ en el dominio fuente. El supervisor del pool
vigila fuente y memorias, pero no el reloj del endpoint PCIe.

## Separar detección, invalidación y reset

El reloj de gestión debe seguir funcionando cuando desaparezca
`EP.user_clk`. Observar reset del endpoint, estado del enlace y una prueba
de ida/vuelta del reloj host; un nivel de enlace sincronizado por sí solo
no detecta un reloj detenido. Registrar la causa y el momento de detección.
El estado inicial sin enlace permite arranque frío; perderlo después de
abrir la sesión debe dejar una invalidación persistente hasta nueva
configuración de FPGA. Recuperar el enlace o escribir ARM/ACK no la borra.

Cerrar la admisión de peticiones y propagar la invalidación al propietario
retenido antes de permitir reutilizar memoria. No conectar PERST al reset
de vida del recolector, control o puente: se perderían época, cookie y
diagnóstico de una reserva expuesta. Una escritura DDR ya aceptada puede
terminar después de detectar el fallo; eso no autoriza publicar éxito ni
reciclar su reserva.

El reset de los dos extremos de cada CDC necesita un protocolo propio.
No se puede borrar un FIFO y considerar devueltas sus respuestas. Tampoco
se debe retirar VALID unilateralmente mientras el receptor siga operativo.
La revisión de integración deberá decidir y probar cómo aislar el núcleo
retenido, consumir localmente respuestas abandonadas y distinguir ese
descarte de una aceptación real del host. El adaptador aislado ya tiene
evidencia; conectarlo al guard y propietario sigue pendiente. Un detector
aislado no cierra la recuperación del transporte.

## Obligaciones que debe conservar la evidencia

Contar por separado AW, W, B, AR y R en ambos extremos, con identidad y
datos completos. En AXI-Lite, el destino puede aceptar antes de que vuelva
la confirmación al productor; incluir también las peticiones presentadas
que causaron efectos sin aceptación todavía observable en el host.

| Punto de inyección | Comprobación necesaria |
| --- | --- |
| Antes de abrir la primera sesión | Arranque retenido cerrado, sin pérdida ficticia |
| AW sin W, o W sin AW | No unir mitades de sesiones distintas ni perder su inventario |
| Petición aceptada en host, dentro del CDC | Identificar lo entregado y lo abandonado |
| Efecto de control en destino, confirmación pendiente | Conservar efecto y diagnóstico aunque se pierda B/R |
| Línea o comando DDR expuestos | No liberar cookie, publicar éxito tardío ni reasignar buffer |
| Resultado pendiente de ACK | Mantener resultado e invalidación; rechazar retiro/rearmado |
| B/R esperando al host | Contabilizar respuesta perdida; no inventar finalización PCIe |
| Reloj host detenido sin reset | Detección acotada por gestión, cierre aun sin flancos host |
| Enlace/restauración de reloj posterior | Sesión anterior sigue inválida |

Probar primero el adaptador con los CDC AMD y memoria controlable; después
repetir los casos representativos con root/endpoint nativos y dos MIG.
Variar fase, pulsos cortos de reset y orden AW/W. Verificar la detección
por ciclos de gestión y observar las señales reales, además de los logs.
Conservar cada intento fallido y exigir revisión independiente y respaldo
por tanda. Una prueba con reset del BFM no acredita PERST nativo.

## Adaptador retenido de frontera

`ntrpu_host_boundary_cut.sv` coloca ocho colas de dos entradas entre los
CDC de host y el núcleo retenido: AW/W/B de datos y AW/W/B/AR/R de control.
`ntrpu_retained_channel.sv` mantiene datos y contadores bajo el reset de
vida de configuración; el reset de la interfaz CDC no borra ese inventario.
Al invalidar, deja de aceptar peticiones nuevas. Las ya aceptadas conservan
VALID y payload hacia el núcleo hasta su aceptación, aunque el CDC se resetee.
Una mitad AW o W no completada queda pendiente; no se fabrica la mitad ausente.

Las respuestas conservan su presentación mientras el CDC siga operativo.
Solo después de invalidación y reset de la interfaz se descartan localmente
las pendientes, y se consumen del mismo modo las tardías. La invalidación
permanece al restaurar el enlace/reset: este adaptador no implementa rearme.
Los contadores saturan e indican overflow; un reset operativo anterior a la
invalidación también registra fallo. El banco diagnóstico específico de siete
casos y las pruebas con RTL alterado se documentan más abajo.

El integrador debe aplicar la invalidación a la propiedad NTRD, esperar
su reconocimiento y entonces resetear ambos extremos CDC y las interfaces
adyacentes, con solapamiento. AMD no admite reset parcial de estas IP y
recomienda al menos 16 ciclos del reloj más lento. Véase
[PG059: resets](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).
El banco usa 64 ciclos host, pero controla pérdida y reset explícitamente;
todavía no valida un generador de reset ni PERST físico.

### Qué significan los contadores

`accepted` cuenta capturas locales; `delivered`, handshakes de salida de
esa cola; `abandoned`, respuestas locales pendientes retiradas después del
reset. **Ninguno sustituye el contador de recepción del endpoint host.**
En AXI-Lite, una petición puede causar efectos antes de volver su ACK al
host, y el host puede recibir una respuesta antes de volver el ACK a su
productor. Por tanto, una respuesta descartada localmente puede haber sido
recibida ya por el host. La integración necesita reconciliar ambos extremos
con identidad, conservando el estado desconocido hasta tener evidencia;
no puede inferir éxito, pérdida ni reintento seguro de un único contador.

### Primer intento conservado

`20261005T093552Z-V7-ffn-supervised-host-boundary-cut` falla en el caso 10:
esperaba un ACK de productor al ver BVALID en host. La traza independiente
observa un B de control aceptado por el adaptador, cero handshakes de salida,
un descarte local posterior y cero recepciones host. Los primeros diez casos
terminan, pero el banco completo permanece **FAIL**.

Respaldo local/NFS verificado a las 09:44:19 UTC: 127 archivos,
32.351.206 bytes; SHA-256
`7344e4a3353a00d5a167c8898dae3a8dbd8274c5478ea3c5a8a7a129f78dc2e6`.
R2 conserva ambos RTL y el cableado; corrige el criterio AXI-Lite y el
padding diagnóstico de B, y añade dos ventanas de ACK para un total de
15 escenarios.

### R2: banco del adaptador aprobado y respaldado

`20261005T094448Z-V7-ffn-supervised-host-boundary-cut-r2` termina con
código 0 en 43,769 s y 36,925 µs simulados. La revisión independiente
comprueba conservación y orden por canal, estabilidad bajo backpressure,
invalidación persistente y ausencia de readmisión tras reset; rechaza
12 alteraciones de evidencia y el banco fallido anterior. Hay 74 capturas,
61 handshakes de salida y 13 descartes locales, sin entradas pendientes al
final de cada caso. Estos conteos agregan canales; no son transacciones DDR.

| Caso observado | Handshake host | Efecto/handshake en fuente |
| --- | --- | --- |
| 13: AW/W de control, pérdida antes del ACK host | 0 AW y 0 W | 1 AW y 1 W entregados al peer retenido |
| 14: B recibido antes del ACK al productor | 1 B recibido | 0 ACK de salida; 1 respuesta retirada localmente tras reset |

Esto confirma por qué `abandoned` no significa «no recibido por el host».
Los casos 6–8 verifican retirar dos respuestas almacenadas y capturar una
tercera en el mismo flanco de reset. Los casos 1–5 conservan dos peticiones
ya aceptadas hasta que el peer vuelve a estar listo. El addendum de trazas
comprueba solapamiento real de los resets en los 14 casos con pérdida:
al menos 64 muestras host dentro del intervalo compartido, mínimo 461,119 ns.

Respaldo local/NFS verificado a las 09:50:05 UTC: 127 archivos,
32.549.637 bytes; SHA-256
`23d80c3e7159505f6381d1e2e9a9e7cd430b5938d476d0dac42a975187c68381`.
Se conservan ocho avisos: dos regeneraciones de constraints IP, uno de
entorno `LIBRARY_PATH` y cinco sobre el modelo comportamental FIFO sin
retardos de sincronización. Este PASS **solo** acredita el mecanismo de
frontera con pérdida/reset controlados y un peer de prueba. Todavía faltan
detector, secuenciador, inventario host y la
conexión a invalidación/ACK NTRD antes del ensayo PCIe nativo y dos MIG.

### Diagnósticos y coste del adaptador

`20261005T100539Z-V7-ffn-supervised-retained-faults-synth` pasa siete
escenarios RTL con contadores de dos bits, sin forzar registros internos:
reset inicial permitido, reset operativo prematuro en ambas direcciones,
overflow por tráfico real, pérdida breve, reset repetido y respuesta retenida
durante pérdida con la interfaz todavía activa. El verificador independiente
contabiliza 20 capturas, 11 entregas y nueve retiros; rechaza diez alteraciones
de evidencia. Tras overflow, los contadores saturados dejan de expresar los
totales exactos. El banco también comprueba drenaje de peticiones previas y
bloqueo de nuevas peticiones después del fallo.

La tanda conjunta queda **FAIL**: el Tcl de síntesis apuntaba al puerto
inexistente `source_clk` en vez de `clk`; el guard de reloj lo rechazó.
Se conserva el intento completo, incluida la simulación válida.
`20261005T101001Z-V7-ffn-supervised-retained-synth-r2` corrige únicamente
la síntesis y termina en 98,411 s con el RTL original y contadores de 64 bits:
**1.132 LUT, 2.863 registros, 152 CARRY8, cero BRAM/URAM y cero latches**.
La estimación interna da WNS +2,876 ns y WHS +0,077 ns a 5 ns; falta
`HD.CLK_SRC`, colocación/ruteo y retardos de 820 entradas y 2.069 salidas.
Los 1.000 avisos TIMING-18 reflejan el límite del reporte; DRC advierte
cobertura incompleta fuera de contexto. No hay excepciones de timing.
Estos datos miden recursos del adaptador, no timing físico ni del conjunto.

### Pruebas con fallos deliberados en el RTL

`20261005T101054Z-V7-ffn-supervised-host-boundary-mutants` compila dos
copias alteradas con el banco de 15 casos y los CDC AMD originales; verifica
los hashes de los archivos realmente compilados. Ambas quedan rechazadas:

- Borrar estado durante reset de interfaz hace desaparecer dos peticiones
  retenidas, sin handshake de consumo; el testbench aborta el caso 1.
- Descartar respuestas al detectar pérdida, antes del reset, retira VALID
  bajo backpressure mientras la interfaz sigue activa (caso 6). Los conteos
  finales del testbench aún producen PASS en los 15 casos; la revisión
  independiente por ciclo lo rechaza. **El marcador PASS por sí solo no
  aprueba la tanda.**

El PASS de esta prueba acredita la detección de esos defectos, nunca las
copias alteradas. El RTL aprobado permanece intacto. Se conservan 16 avisos
de generación/entorno/modelo, equivalentes a dos ejecuciones del banco base.
No se validan todavía el detector automático ni la propiedad NTRD.

Los tres respaldos incluyen fuentes, logs, revisión y hashes local/NFS:

| Tanda | Archivos / bytes comprimidos | SHA-256 del archivo |
| --- | --- | --- |
| `100539Z` | 65 / 160.738 | `42a6e8ff70b6ced2934218ae0e31c52e4c4a34a76e1d033b6fc7e32c6811fd17` |
| `101001Z` | 33 / 933.567 | `28fdfa62a6b44f9f8a8ea3046978017a7e32c2105bfeb58243aa1c705fa19ddf` |
| `101054Z` | 242 / 64.258.215 | `711dc8f546098d4cee2d1d0ad4dce0ae577b246eefd95834d850af430c289f56` |

## Límites de esta etapa

Esta extensión busca invalidación segura y contabilidad tras pérdida del
host. La recuperación caliente DDR sigue abierta y tiene evidencia fallida;
no se deriva de una nueva enumeración PCIe. MSI/MSI-X, lectura DDR por BAR,
cuatro canales, P2P e inferencia mantienen sus propias dependencias en
[la cola de diseño](VIVADO_QUEUE.md).
