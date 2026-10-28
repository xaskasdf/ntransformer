# Control de carga por BAR: contrato experimental

`ntrpu_pcie_ingress_control.sv` controla una reserva de 64 bytes del camino
`ntrpu_pcie_ingress_guarded`. Implementa AXI-Lite de 32 bits en el reloj UI.
El acceso privilegiado, el cruce del BAR hacia UI y el supervisor de aislamiento
son responsabilidades de integración; este bloque no los sustituye.

## Registros y secuencia

Las direcciones son offsets de byte alineados. Los pares de 32 bits contienen
primero la mitad baja. No se descartan bits altos de dirección de AXI-Lite.

| Offset | Contenido |
| --- | --- |
| `00/04/08` | Identificador NTRI, ABI `00010000`, requisito de barrera externa |
| `0c` | Estado: IRQ, ARM pendiente, ejecución, resultado, STOP, fallo, ARM ready, reserva, error, transporte idle, guard idle, AXI busy; bits 0–11 |
| `10` | Comando: 1 ARM, 2 STOP, 4 ACK; cero no hace nada |
| `14/18/1c` | Habilitar IRQ, reservas aceptadas, resultados retirados |
| `20–3c` | Descriptor: época, etiqueta, ventana BAR, destino DDR; cuatro valores de 64 bits |
| `40–4c` | Época y etiqueta de la confirmación ACK |
| `50–5c` | Época actual y bitmap de bytes recibidos |
| `60–74` | Época, etiqueta y estado del resultado; código de error de ingreso |
| `78–80` | Solo lectura: source_stopped, ingress_isolated y quiesce_epoch |
| `84–8c` | Época del descriptor aceptado por el control; fallo de protocolo |

1. Escribir descriptor con ventana y destino alineados a 64 bytes y época
   estrictamente mayor que la actual. No se permite volver a cero por overflow.
2. ARM congela el descriptor; `OKAY` significa solicitud encolada. Esperar su
   aceptación efectiva antes de enviar datos. Los registros del descriptor
   quedan bloqueados hasta ACK. Una solicitud pendiente no puede cancelarse.
3. STOP solo se admite tras aceptar ARM. Permanece activo hasta el siguiente
   ARM; no cancela canales AXI ya expuestos ni reinicia el transporte.
4. Esperar resultado válido. Leer identidad/estado, escribir esa identidad en
   `40–4c` y emitir ACK. Una identidad incorrecta no libera la reserva. El
   resultado permanece estable hasta ACK; leerlo después carece de significado.

Las escrituras parciales respetan WSTRB. Direcciones desconocidas o desalineadas
devuelven DECERR; comandos inválidos y escrituras de registros protegidos,
SLVERR. Los canales AW/W son independientes y las respuestas soportan pausas.
Un resultado inesperado activa un fallo persistente; requiere recuperación
coordinada, no un bit escribible para borrar el problema.

## Límite de seguridad y validación

Ni `transport_idle`, ni un temporizador, ni una escritura en BAR acreditan que
no quedan paquetes PCIe antiguos. La barrera debe abarcar productores, colas
PCIe/CDC y respuestas pendientes. Las entradas de diagnóstico aquí son de
solo lectura; el camino de datos conserva sus comprobaciones de época y drenaje.
Las escrituras DWORD actuales no llevan época: reutilizar una ventana sin esa
barrera puede atribuir datos antiguos a una nueva reserva.

La primera tanda prueba el control aislado con un interlocutor de simulación
que retiene readiness y resultados. No califica PCIe, DDR, CDC, reinicio en
caliente ni aislamiento real. El Tcl reproduce RTL, fallos deliberados,
síntesis y simulación funcional de netlist. Los resultados revisados y sus
respaldos se registran en `VIVADO_STATUS.md` al terminar la tanda.

## Resultados conservados

`20261004T210107Z-V7-ffn-supervised-ingress-control` aprobó seis replays
(semillas 1, 7 y 42, en RTL y netlist sintetizada). Por replay: 10 escenarios,
116 escrituras, 25 lecturas y tres reservas/confirmaciones; los eventos y
ciclos coinciden entre RTL y netlist. Cinco variantes RTL defectuosas y 22
corrupciones de evidencia fueron rechazadas. Duración de la tanda: 200,433 s.
Síntesis: 821 LUT y 816 FF. Los márgenes internos sin colocar no califican
las fronteras: 443 entradas y 302 salidas carecen de retardos externos.
Archivo local/NFS verificado: `4324d3e44808b0ff81ecaae63efe9d7e39fd95f5fee8aa5993721e0dd046bcae`.

La composición `ntrpu_pcie_ingress_managed.sv` mantiene los seis módulos
existentes del camino de datos. El generador del banco conserva sus 21 casos
y añade programación/lectura de registros. Los comandos STOP se **inician**
en cinco fases; su aplicación ocurre después de la latencia AXI-Lite.
No se atribuye a este banco la inyección de STOP en el mismo ciclo de la fase.

La tanda `20261004T210950Z` conserva un fallo del script: la retirada
automática deliberadamente incorrecta dispara `MONITOR_RESULT_STABLE`,
pero el script esperaba `LEASE_BEFORE_RETIRE`. Los tres replays positivos
RTL pasaron; no se ejecutó síntesis/netlist en esa tanda. Respaldo verificado:
`5521cf00872394b6395636ba7c5bed6a2216a82845725a35a646b6a3a4c4d49f`.
`ntrpu_board_pcie_ingress_managed_r2.tcl` corrige únicamente esa expectativa,
sin modificar DUT ni estímulos, y utiliza una carpeta nueva.

`20261004T211253Z-V7-ffn-supervised-ingress-managed-r2` aprueba la composición:
los tres replays RTL coinciden evento a evento y ciclo a ciclo con los tres de
netlist. Cada uno realiza 21 reservas, 309 escrituras AXI de entrada, 14
escrituras de memoria (896 bytes), 954 escrituras de control y 105 lecturas de
control. Rechaza 128 comandos incorrectos; las lecturas indicadas son de
registros, no readback DDR. Comprueba respuesta B ausente durante 100 ciclos,
ACK con etiqueta incorrecta, barrera obsoleta, destinos inválidos y recuperación
sin reset. Tres defectos RTL deliberados y 33 corrupciones de evidencia son
detectados. Duración de la tanda: 222,533 s.

Síntesis del conjunto: 2.317 LUT, 3.907 FF, 298 LUTRAM y cero BRAM/URAM/DSP.
Con reloj de 5 ns, setup interno sin colocar 1,757 ns y hold 0,066 ns.
No hay latches, cajas negras, endpoints internos sin restricción ni bucles
detectados. Faltan retardos para 843 entradas y 1.215 salidas. Se conservan
ocho avisos de síntesis/OOC y cinco de entorno de elaboración; no acreditan
timing físico. DCP: `dc27218c5e6aaffa8dff0e30dfd242c6ec38fcfbe5b2275c9bc238fe88e8f2e9`.
Respaldo local/NFS: `f5b135a22c0360e959eb5e60f61b0a92b163366e3efe45035dd55789cbde3647`.

## Próxima integración

Conectar el AXI-Lite y el camino de datos al modelo PCIe nativo de XDMA;
el montaje con ambos cruces reales y MIG observado ya pasa el banco descrito
abajo. Demostrar después el protocolo de parada/drenaje de productores y
colas: un readback por otra ruta de BAR no se acepta como barrera sin evidencia
de ordenamiento. Hasta resolverlo, la reutilización de ventanas sigue
condicionada a la barrera externa. Evaluar un protocolo que lleve época por
transferencia si no puede garantizarse el drenaje del formato DWORD actual.

## Integración host y control CDC contra MIG

`ntrpu_pcie_ingress_host.sv` une el control UI con el camino asíncrono ya
validado. Reutiliza dos IP AMD `axi_clock_converter:2.1`: AXI4 de 512/64/4
bits para datos/dirección/ID, y AXI4-Lite de 32/32 bits para registros; ambos
con tres etapas de sincronización. El segundo reutiliza
`ntrpu_axil_cdc_recode.sv`, cuya codificación reversible preserva RRESP externo.
Los diagnósticos e IRQ expuestos por este top pertenecen al dominio UI.

El banco `tb_ntrpu_pcie_ingress_host.sv` conserva el MIG genérico y sus 35
archivos de modelo. Programa reservas, STOP y ACK desde el host; observa cada
handshake a ambos lados de ambos cruces y lee seis ubicaciones de DDR,
incluyendo más de 4 GiB y la última línea de 128 GiB. Las tandas aisladas
`20261004T212852Z` y `20261004T213121Z` usan host de 4 ns/fase 0,7 ns y
7 ns/fase 1,1 ns, respectivamente, con UI de 5 ns. Ambas pasan la revisión y
tienen copia local/NFS verificada. Por tanda: 13 trabajos, 194 transacciones
AXI4, nueve escrituras de memoria (576 bytes), 63 lecturas (4.032 bytes),
588 escrituras y 70 lecturas de control. Se rechazan 78 comandos, tres lecturas
con DECERR y 179 corrupciones deliberadas de evidencia por tanda.

El perfil rápido completa 19.777 ciclos UI; el lento, 26.378. Se explican por
causalidad los 432 y 720 avisos respectivos del modelo de DDR y se drenan
80 y 106 lecturas periódicas. Duraciones de ejecución: 413,058 y 481,008 s.
Los 35 archivos de memoria coinciden con el banco frío previamente calificado;
esto sigue sin homologar un DIMM comercial ni el reinicio en caliente.

El AXI-Lite de AMD usa `lite_async`/`xpm_cdc_handshake`: el receptor completa
su handshake antes de que READY confirme la transferencia al emisor. La
revisión inicial suponía el orden del FIFO AXI4; se corrigió tras contrastarlo
con el RTL instalado, sin cambiar DUT, estímulos ni logs. Los tiempos medidos
son **recepción → confirmación en origen**, no latencia de llegada de payload:
AW/W/AR 16,702–19,702 ns en el perfil rápido y 28,602–34,602 ns en el lento.
Las respuestas B/R demoran 20,298–24,298 ns y 20,398–24,398 ns, respectivamente.
La revisión conserva controles estrictos de payload, orden y drenaje en los
cinco canales a ambos lados; la simulación no prueba metastabilidad física.

Respaldos SHA-256: rápido
`68f2761f67ddcee35a6ccfbcd5fd2433944cc103766a8e3bd1f43d6def724543`;
lento `4e5734ba69f8b6e8ee4117c4ee13d84ec55ed27a9114fdc18db3ebe4ce15603e`.
Los productores, la barrera de aislamiento y las lecturas de comprobación
siguen siendo interlocutores de simulación. El resultado no demuestra aún
control nativo PCIe ni drenaje de paquetes anteriores al rearme.

## Síntesis del control con ambos cruces

`20261004T214501Z` sintetiza los diez RTL de producción sin cambios respecto
al banco rápido: **2.946 LUT, 6.117 FF y 682 LUTRAM**, sin latches, cajas negras,
BRAM, URAM ni DSP. A 4/5 ns, setup interno sin colocar 1,747 ns y hold
0,013 ns; 15 grupos de skew con margen mínimo 3,851 ns. CDC informa 23
sincronizadores y diez resets informativos con los waivers AMD activos.
No se añaden excepciones ni waivers propios. DRC OOC no encuentra violaciones,
pero faltan retardos para 844 entradas y 1.215 salidas; TIMING-18 permanece
abierto. Esto no cierra timing de puertos ni de placa.

El respaldo de síntesis es
`b61e84d00cad8ed9760787ce9b60fcfc198e3c078b364c490509eee9bd03e5ed`.
La tanda separada `20261004T215325Z` amplía solo el límite del mensaje
`Synth 8-7129` a 10.000 y revela 870 avisos, frente a los primeros 100
impresos. Corresponden a 42 especializaciones de 20 familias AMD: campos
opcionales AXI, interfaces FIFO no seleccionadas, indicadores/umbrales y
relojes de registros de entrada no utilizados. No hay módulos propios en
esa lista. Se conservan también los otros 21 avisos de síntesis/OOC.

Los 17 inputs de síntesis, recursos y márgenes coinciden. La netlist visible
coincide tras normalizar comentarios y rutas; el bloque cifrado AMD tiene
otro hash y no se declara equivalencia binaria completa. El replay funcional
usa la netlist **original** de `214501Z`. Las dos tandas duran 161,993 y
160,522 s; la repetición diagnóstica no sustituye su identidad.
Respaldo diagnóstico verificado local/NFS:
`91ee704f2c730695d66b545507958a6b085d1950c21f4396142b53e98b3fb0b2`.

## Replay sintetizado del control y datos

`20261004T215351Z-V7-ffn-supervised-ingress-host-netlist-r2` pasa con
123 inputs compilados y **ningún RTL de producción como sustituto** de la
netlist. Los 7.401 eventos coinciden con el perfil RTL rápido, incluidos
payloads, respuestas y timestamps. Conserva los 13 trabajos, 588/70 accesos
de control, 78 comandos rechazados, nueve escrituras y 63 lecturas DDR;
179 corrupciones de evidencia son detectadas. Los 432 avisos del modelo se
correlacionan y las 80 solicitudes periódicas quedan drenadas. Duración:
415,786 s. Se revisan 17 avisos de compilación/entorno; los cuerpos `glbl`
duplicados coinciden tras normalizar comentarios y espacios.

Solo cambian observadores pasivos del banco: toman los pines conservados del
convertidor porque síntesis desplazó lógica y renombró nets. Para mostrar
RRESP de UI se invierte la codificación reversible ya existente. No cambian
estímulos, monitores, memoria ni netlist. Hash de la netlist ensayada:
`e0651f6f49e056205036add6038487e1d5f47a7020d972d7602f4a656e04d9b9`.
Es simulación funcional de primitivas sin SDF; no acredita metastabilidad
ni retardos físicos. La preparación anterior fue rechazada por el filtro
de fuentes antes de crear una tanda Vivado; su diagnóstico se conserva.
Respaldo local/NFS verificado a las 22:03:58 UTC: 674 archivos,
59.660.829 bytes, SHA-256
`521dea3b02dabef9b57ddcbac1585b671db17431863c7b508998cf23099948c6`.

## Alcance de las solicitudes del host

La máscara `WSTRB=5` y los accesos desalineados de diagnóstico ejercitan la
robustez del AXI-Lite interno. No demuestran soporte de tales solicitudes
desde PCIe: [PG195, limitaciones](https://docs.amd.com/r/en-US/pg195-pcie-dma/Others)
incluye restricciones sobre máscaras no contiguas. Se deben separar de los
patrones admitidos en una prueba nativa.

La consulta de [PG195, Write Flush Timeout](https://docs.amd.com/r/en-US/pg195-pcie-dma/Config-Write-Flush-Timeout-0x60)
confirma que ese registro corresponde a canales C2H AXI4-Stream y cierre de
descriptor. No constituye evidencia del drenaje de escrituras entrantes por
BAR; no se usará como prueba de aislamiento del formato DWORD.

## Control PCIe nativo y MIG frío

`20261004T223702Z-V7-ffn-supervised-native-host-r2` **aprueba** la conexión
del BAR0 al controlador AXI-Lite y del BAR2 al ingreso DWORD, con ambos
convertidores CDC AMD y el MIG genérico frío. Los diez RTL de producción
coinciden con los de la netlist `215351Z`; se conservan 134 archivos AMD
sin cambios y se verifican 206 entradas compiladas y 180 señales VCD.

| Tráfico comprobado | Cantidad |
| --- | --- |
| Escrituras de control / rechazos | 78 / 16 |
| Lecturas de control del fixture | 52, con 52 completions y tags asignados |
| Lectura adicional de enumeración | Una consulta real de NTRI en BAR0 |
| Escrituras DWORD nativas | 83, con máscaras contiguas `f/3/c/1/e` |
| Reservas / resultados retirados | 4 / 4 |
| Escrituras al MIG / bytes publicados | 3 / 192 |
| Lecturas DDR comparadas | 8 / 512 bytes |

Se ejercitan ARM pendiente sin prueba de aislamiento, descriptor bloqueado,
ACK anticipado y con etiqueta incorrecta, conflicto de bytes, STOP y
recuperación. Los destinos incluyen más de 4 GiB y la última línea de 128 GiB.
Una respuesta B del AXI-Lite rechazado se observa en el endpoint; sigue sin
ser una completion PCIe para el MemWr publicado.

El root-port AMD asigna BAR0 a cero y BAR1 a `0x10000`. Su tarea original
`TSK_XDMA_FIND_BAR` lee BAR0 antes del fixture. La tanda `221559Z` hizo todo
el tráfico pero falló al esperar 52 AR/R totales en vez de 53: **se conserva
como fallo**, aunque Vivado terminó con código cero. Su respaldo verificado
es `1b53854d58933338e00e44a4d98bf8c9ec8138fd45929da1bab3a8d6affbbd69`.
R2 cambia únicamente ese total y adelanta la declaración del contador;
no modifica estímulos, DUT, IP ni memoria. Los 1.630 eventos de pines y
11.959 muestras de estado UI coinciden exactamente entre ambas tandas,
incluidos tiempos, payloads y pausas; esto no convierte el primer fallo en PASS.

La revisión correlaciona los 6.624 avisos del modelo con sus comandos y
retornos y drena 79 solicitudes periódicas. Detecta 200 corrupciones de logs
y VCD. Los 156 avisos de herramienta coinciden con el banco PCIe/MIG previo;
se retienen cinco avisos de FIFO comportamental y un aviso de caso único
en tiempo cero. Final: 113,118498 µs simulados; duración de R2: 902,600 s.

El cruce AXI-Lite mide confirmaciones de recepción: 16,002–19,002 ns para
AW/W/AR y 21,998–24,998 ns para B/R en este perfil, no latencia física ni
metastabilidad. El aislamiento del productor y la comprobación de DDR
siguen siendo interlocutores del banco. No hay MSI integrado ni prueba
de drenaje real antes de reutilizar el BAR. El estudio separado de
[fragmentos con identidad](PCIE_INGRESS_FRAGMENTS.md) aborda esa limitación
sin modificar ni dispensar la barrera actual.

Respaldo R2 verificado localmente y en NFS a las 22:57:17 UTC: 1.401
archivos, 203.150.022 bytes, SHA-256
`0249c94facf247fc957f544bc892baffb6e1e4f34559e3db0830f9d13dbf4ebd`.
Reproducción: `tools/ntrpu_board_pcie_ingress_native_host_r2.tcl`, con el
ejemplo XDMA `154440Z`, el ejemplo MIG `20261002T102039Z` y salida nueva.
Los revisores y sus hashes viajan con la evidencia; el primer intento
y sus fuentes permanecen intactos.
