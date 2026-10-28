# Fragmentos con identidad en el ingreso PCIe

Propuesta del 2026-10-04. Complementa el [ingreso DWORD](PCIE_INGRESS_PLAN.md)
y su [control](PCIE_INGRESS_CONTROL.md); **no cambia esos bloques ni elimina
la barrera externa exigida actualmente**. El decodificador aislado está en
`fpga/rtl/ntrpu_pcie_fragment_decode.sv`. Su validación Vivado se registra
por separado; esto todavía no es un endpoint ni un protocolo aprobado en placa.

## Problema que debe resolver

Un DWORD antiguo y otro nuevo con la misma dirección, máscara y datos son
idénticos para el decodificador actual. El control puede incrementar su época,
pero ese número no viaja en la escritura. Un contador local vacío no demuestra
que no quede un paquete anterior en el productor, PCIe o un cruce de reloj.

Cada fragmento del formato propuesto lleva su propia identidad. No sirve
enviar una cabecera por un BAR y después DWORDs sin identidad por otro:
seguiría siendo posible asociar un dato retrasado a la cabecera nueva.

## Perfil experimental NTRPUF01

Una unidad de ingreso es una pareja AW/W ya capturada de **64 bytes completos**:
dirección alineada, `AWLEN=0`, `AWSIZE=6`, `AWBURST=INCR`, sin exclusividad,
`WLAST=1` y `WSTRB=~0`. El caller debe drenar también las transferencias que
rechace. No se presupone que un `memcpy` o varias escrituras MMIO produzcan
esa unidad; el mapeo desde TLP al bypass AMD necesita su propia prueba.

| Bytes | Campo | Validación |
| --- | --- | --- |
| 0–7 | ASCII `NTRPUF01` | Valor de 64 bits `0x313046555052544e`, little endian |
| 8–15 | Época de 64 bits | Igual a la reserva autorizada |
| 16–23 | Etiqueta de 64 bits | Igual a la reserva autorizada |
| 24–31 | Desplazamiento de fragmento | Únicamente 0 o 32 |
| 32–63 | 32 bytes de datos | Se colocan en la mitad indicada de la línea |

La dirección AW completa debe coincidir con la ventana alineada de la
reserva. Época y etiqueta identifican trabajo; la autorización de destino
sigue siendo responsabilidad del controlador de reservas. No son credenciales.
Dos fragmentos transportan una línea útil de 64 bytes con 64 bytes adicionales
de cabeceras NTRPU; esa proporción no incluye overhead PCIe ni mide caudal.

`faults[6:0]` identifica, en orden de bit ascendente: reserva, formato,
ventana, magic, época, etiqueta y desplazamiento. Cualquier fallo entrega
`byte_valid=0`. El bloque no captura AW/W, no genera B, no acumula fragmentos,
no asigna épocas ni publica resultados. La comparación conserva los 64 bits.

## Reglas para la futura integración

1. Capturar y comprobar la unidad completa antes de entregar bytes al
   ensamblador. Rechazar cabeceras parciales, longitudes y strobes ajenos al
   perfil; nunca usar campos guardados de un paquete anterior.
2. Drenar un fragmento de época/etiqueta antigua sin escribir DDR ni modificar
   bitmap, descriptor o estado de la reserva vigente. Registrar el rechazo
   aparte: reutilizar sin cambios el fallo global del ingreso DWORD podría
   invalidar un trabajo nuevo por tráfico antiguo.
3. Asociar cada buffer y cada respuesta pendiente a su identidad capturada.
   Mantener inmutable lo expuesto por VALID hasta READY y esperar los B de
   memoria antes de publicar éxito. No prometer reversión de un prefijo escrito.
4. No reciclar épocas por wrap ni después de un reset local mientras pueda
   quedar tráfico anterior. Un nuevo arranque necesita una sesión distinta
   o aislamiento externo probado; el formato no resuelve por sí solo resets.
5. Conservar el perfil DWORD y su ABI existentes durante la exploración.
   La negociación de este formato y sus registros es trabajo pendiente.

## Pruebas y puertas pendientes

La tanda aislada `20261004T224226Z-V7-ffn-supervised-fragment-decode` **aprueba
RTL y la netlist sintetizada**. Cada positivo completa 1.176 casos, con 516
aceptaciones y 660 rechazos: posiciones de payload, todos los bits de los cuatro
campos, dirección completa, strobes, longitud, tamaño, tipo y una reserva nueva
sobre la misma ventana. Las seis mutaciones RTL fallan en el caso esperado;
la revisión compara las 1.176 decisiones y máscaras con una tabla independiente.
Ambos positivos terminan en 11,760 µs simulados. La tanda consume 169,150 s.

Síntesis VU5P: 678 LUT y cero registros, sin latches ni cajas negras. El retardo
combinacional máximo reportado sin colocar es 1,206 ns bajo presupuesto de
5 ns; no es cierre temporal de placa. DRC OOC informa cero comprobaciones
fallidas, con su limitación de conectividad. Las ocho elaboraciones conservan
el aviso de `LIBRARY_PATH`; no hay avisos de compilación ni de simulación en
los positivos. Respaldo local/NFS verificado a las 22:46:18 UTC: 273 archivos,
1.887.406 bytes, SHA-256
`0da03262982919bc60b891c8c8ef21e2b1760c2bf9b8a769b46eae4a6e6e5c89`.

## Tanda nativa y corrección del generador

`20261004T230609Z-V7-ffn-supervised-fragment-link` termina como **fallo** a
70,216 µs: `NTRPU_LOG_RQ_MULTIBEAT_UNSUPPORTED`, en la primera escritura,
antes de una decisión del decodificador. Vivado devuelve cero, pero el fatal
impide aprobar la prueba. Duración: 412,523 s. El respaldo local/NFS conserva
839 archivos y 106.052.373 bytes; SHA-256
`7a70c37b09fcdabad159e2ea689b7bde8f235c9abcc9e6364953b93ffcb8ff75`.

Hay dos problemas distintos en el banco reutilizado: su registrador RQ
admite únicamente paquetes de un beat y hasta 32 bytes; además, la tarea
pública `TSK_TX_MEMORY_WRITE_64` conserva empaquetado de 256 bits y un
`TKEEP` no final de ocho DWORDs en un puerto de 512 bits. Ampliar solamente
el registro de paquetes no corrige el estímulo.

La revisión `20261004T231900Z-V7-ffn-supervised-fragment-link-r2` **aprueba**
el comprobador independiente y sus 128 corrupciones de evidencia. La tanda
usa un productor acotado nuevo del banco: descriptor de 16 bytes más 48 de
payload en el primer beat (`TKEEP=ffff`), seguido de los 16 bytes restantes
(`TKEEP=000f`, `TLAST=1`). Mantiene VALID y los datos bajo contrapresión.
El registro atómico de 80 bytes ocurre tras el segundo handshake. La IP AMD,
el decodificador de producción y los estímulos de identidad permanecen
intactos; el generador de tráfico sí cambia.

Esta disposición sigue las reglas de
[escritura RQ de PG213](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Memory-Write-Operation?contentId=bReuOBHlK75iDWWV3g8f5w)
y sus campos de
[TKEEP, SOP/EOP y byte enables de 512 bits](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Request-Interface?contentId=PCHE6KE3RpHa2Ftt_qJTcg).
La prueba deberá comparar el paquete registrado con los handshakes RQ,
AW/W/B, decisiones del decodificador y lecturas de retorno. Un PASS del
productor por sí solo no acredita el mapeo ni demuestra que MMIO de CPU
genere estas transferencias.

### Evidencia y alcance de R2

Se reconstruyen 28 peticiones PCIe: 16 escrituras de dos beats y 12 lecturas
de un DWORD. Las 44 transferencias RQ se correlacionan con 16 AW/W/B,
12 AR/R, 12 tags asignados por el core y 12 completions. Cada escritura
produce `AWLEN=0`, `AWSIZE=6`, INCR y `WLAST=1` en el bypass real. Se revisan
todos los bytes habilitados, incluido el último bloque de la ventana BAR de
1 MiB; la traducción conserva las direcciones de 64 bits por encima de 4 GiB.

El decodificador acepta nueve fragmentos y rechaza siete: identidad antigua,
magic incorrecto, etiqueta distinta en un bit alto, offset inválido, último
byte deshabilitado, época antigua y ventana incorrecta. El banco cambia cuatro
reservas y ensambla cuatro líneas, incluida una repetición idéntica de la
mitad alta. El estado se compara en cada ciclo: los rechazos no modifican el
buffer de la reserva nueva. Las tres líneas finales se leen en offsets
0, 28, 32 y 60. **Reservas, ensamblado y memoria son BFMs**, no el ingreso de
producción ni DDR; el cambio de reserva no es una operación ARM por BAR0.

La captura contiene 61 señales; se cotejan 75 inputs compilados y 134 archivos
AMD intactos. Los 139 avisos de herramienta y el aviso de `unique case` en
tiempo cero conservan exactamente el conjunto normalizado del primer intento.
Duración de herramienta: 453,242 s; final simulado: 81,436 µs. Hay 16 ciclos
de stall AW y 23 AR; **cero stalls W/RQ/RC/B/R**. No afirmar cobertura de
contrapresión de esos canales. Cada pareja RQ está separada por 4 ns; desde
el segundo beat hasta la decisión pasan 196 ns y desde allí hasta B, 28 ns.
Son tiempos de esta simulación PIPE y su BFM, no latencia ni caudal de placa.

Respaldo local/NFS verificado a las 23:28:44 UTC: 844 archivos,
107.367.835 bytes, SHA-256
`c0af15aee4e9d57fd960f5a87ee90af85698ab586aea5840725a4be797b945e7`.
El certificado `VERIFIED.json` conserva la comprobación del archivo completo
y de cada miembro. La copia local reside en `artifacts/ntrpu-board-vivado/`
mediante el enlace al volumen D:.

## Siguiente integración

El [capturador y ensamblador de producción](PCIE_FRAGMENT_INGRESS.md) ya aprueba
20 escenarios por RTL/netlist y conserva la barrera externa. Su conexión
nativa a XDMA también aprueba; esos resultados deben componerse con
las puertas de control, CDC y memoria, sin inferir aprobación transitiva.

El banco del capturador ejerce contrapresión y paquetes de identidad antigua
después del rearme. Repetir con control nativo, ambos CDC, errores, STOP y MIG
frío, incluyendo tramas retenidas en el transporte antes de ARM. Solo entonces revisar
qué parte de la barrera externa puede sustituirse. Siguen pendientes el DIMM
exacto, reset con tráfico, MSI, múltiples productores, P2P y validación física.

El contrato del ensamblador distingue duplicados idénticos y conflictivos,
tráfico posterior a publicación/retiro y rechazos locales. Sus pruebas
aisladas cubren B pendiente y drenaje de ráfagas, incluida una de 261 beats.
Conservar esas reglas y la identidad capturada durante la futura composición;
la evidencia aislada no sustituye la prueba de cada cadena integrada.
