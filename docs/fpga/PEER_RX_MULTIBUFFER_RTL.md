# Prototipo RTL de RAM, colas y escrituras pendientes

Tanda local del 2026-10-04: **16 simulaciones aprobadas, 1174 transferencias,
581.248 bytes comparados y 72.404 ciclos de banco de pruebas**. El prototipo
integra el [asignador validado](PEER_RX_BUFFER_ALLOCATOR.md), RAM de payload,
colas por destino, emisores AXI y seguimiento de respuestas. No está
conectado al endpoint ni sustituye al receptor con épocas y rearme.

## Organización implementada

El [módulo](../../fpga/rtl/ntrpu_peer_multibuffer.sv) tiene una entrada de
comandos y datos de 512 bits, y un puerto AXI de escritura por destino.
Por defecto usa dos destinos, dos bancos propios por destino, cuatro de
reserva, 64 beats por banco, 16 descriptores por destino y cuatro escrituras
pendientes por puerto. Eso supone 32 KiB de payload; reserva ocho supone
48 KiB. Metadatos y registros de pipeline son adicionales.

- Cada banco físico tiene una escritura y una lectura síncrona. La propiedad
  exclusiva evita que dos emisores lean el mismo banco. Bancos distintos
  pueden alimentar varias salidas simultáneamente.
- Cada salida cuenta con dos entradas de datos y un crédito para la lectura
  en vuelo, incluido dentro de esas dos plazas. Así mantiene WDATA/WLAST
  estables cuando WREADY está bajo.
- Los descriptores recorren `FREE → FILL → READY → WRITE → WAIT_B → DONE`.
  Los punteros de admisión, envío, respuesta y resultado son independientes;
  admiten profundidades que no sean potencias de dos.
- El banco se devuelve **en el handshake de WLAST**. La cola de respuestas
  conserva la escritura hasta B; el descriptor se libera al consumir CPL.
  Se reservan todos esos metadatos desde la aceptación del comando.
- Se reserva capacidad de escritura pendiente antes de exponer AWVALID.
  Una salida termina sus datos antes de comenzar la ráfaga siguiente, pero
  puede hacerlo mientras aún espera respuestas B anteriores.

Se utiliza ID cero por puerto, ráfagas INCR, beats completos de 64 bytes y
AxCACHE cero. El destino debe conservar el orden de respuestas del mismo
ID. AW se acepta antes de exponer WVALID. Las respuestas y resultados de
destinos distintos pueden progresar independientemente.

## Entrada y comportamiento ante pausas/fallos

El comando incluye destino, dirección, longitud AXI y tag de 64 bits.
El adaptador de entrada debe entregar exactamente `LEN+1` beats completos
del span prevalidado. **Esta interfaz no tiene TLAST ni comprueba framing
PCIe**; tampoco valida ventanas BAR, permisos o épocas. Una trama incompleta
mantiene ocupado el banco. El prototipo comprueba destino, alineación,
capacidad del banco y frontera de 4 KiB antes de aceptar el descriptor.
Un descriptor inválido bloquea admisión y activa fault; no genera CPL.

`stop` cierra nuevas admisiones, pero permite terminar datos de entrada y
escrituras ya aceptadas. Es un **drenaje cooperativo**, diferente del abort
con descarte previo a AXI del receptor original. La fuente debe completar
las tramas aceptadas y el destino responder. No se presupone progreso si
alguno deja de hacerlo.

Un B con error activa fault, conserva el código en su resultado y descarga
esa transacción. Un BID ajeno se consume y activa fault, pero **no descarga
la escritura esperada**: todavía debe llegar su B de ID cero. En ambos
casos se cierra admisión y continúan los trabajos aceptados. CPL informa
el BRESP local; no certifica por sí solo SUCCESS/FENCE del protocolo NTRPU
ni visibilidad final de pesos en la memoria del sistema.

El reset es síncrono y requiere coordinación externa, aislamiento y drenaje.
No hay recuperación ni cambio de época dentro de este módulo. `drained`
cubre bancos, entrada y descriptores internos, incluyendo CPL pendiente;
no certifica que el endpoint o enlaces externos carezcan de tráfico tardío.

La entrada sigue siendo una única secuencia de comandos: un comando a un
destino lleno puede bloquear al productor. Las colas separadas permiten
progresar a trabajos ya admitidos, pero no seleccionan otro comando aguas
arriba. El controlador deberá ofrecer destinos elegibles para aprovechar
ese paralelismo. Cada destino tiene su propio canal CPL; un agregador
compartido posterior deberá preservar sus créditos y retención.

## Validación y cobertura

El [banco de pruebas](../../fpga/tb/tb_ntrpu_peer_multibuffer.sv) ejecuta dos
semillas por perfil. Se prueban 1/2/3 destinos, bancos de 1/4/64 beats,
reserva 0/3/4/8, colas de 3/5/16 descriptores y límites de 1/2/4 escrituras
pendientes. Las longitudes de transferencia se mezclan dentro de cada caso.

La memoria de destino simulada demora B 160–182 ciclos después de WLAST.
AWREADY, WREADY y el consumidor de resultados sufren pausas; pruebas
dirigidas detienen un destino, llenan créditos de resultado, preparan
salidas simultáneas y presentan comandos durante stop/fault. Se prueban
error B, BID ajeno, direcciones desalineadas, longitud excesiva, cruce de
4 KiB y destino fuera de rango cuando se puede representar.

El [revisor de transacciones](../../tools/check_ntrpu_peer_multibuffer.py)
compara cada beat de entrada con un patrón independiente, cada beat W con
la entrada, AW con el descriptor, WLAST con la longitud y CPL con el B
correcto. Reconstruye préstamos y generaciones; verifica orden, créditos,
ausencia de pérdidas/duplicados y drenaje antes del reset. El banco también
comprueba estabilidad bajo backpressure, atributos AXI, propiedad de RAM
y capacidad de los registros de lectura.

Resultados agregados de la tanda final:

- 827 reutilizaciones de banco mientras su B anterior seguía pendiente.
- 1176 ciclos con handshakes W simultáneos en varios destinos.
- Hasta cuatro escrituras AXI aceptadas y pendientes **por puerto**.
- 16 respuestas B con error y 16 respuestas de ID ajeno ejercitadas.
- Cinco mutaciones detectadas: corrupción de datos, liberación antes de
  WREADY, exceso de pendientes, CPL anticipado y descarga por BID ajeno.

La primera tanda se detuvo por falta de cobertura de salidas simultáneas
en el perfil de un beat. Se añadió una prueba dirigida y se mantuvo el
criterio. La segunda pasó; la tercera amplió intentos de admisión durante
stop/fault y descriptores inválidos. El RTL del datapath no cambió entre
esas tandas. Los diagnósticos previos están conservados, sin presentarlos
como validación final. Icarus solo emitió avisos de sensibilidad de arrays
en la tanda final; los logs completos se conservan.

## Reproducción y respaldo

Desde Linux/WSL, con Icarus en PATH:

```bash
python3 -B tools/run_ntrpu_peer_multibuffer.py artifacts/ntrpu-peer-multibuffer/nueva-tanda
```

La carpeta debe ser nueva. El [ejecutor](../../tools/run_ntrpu_peer_multibuffer.py)
acepta `--iverilog`, `--vvp` y `--ivl-base` para la herramienta local extraída.
La tanda final tardó 28,72 s según el ejecutor. Cada proceso tiene un límite
de 45 s y cada simulación un watchdog. Se ejecutó localmente, sin usar la VM.

- [Revisión final](../../artifacts/ntrpu-peer-multibuffer/20261004-datapath-r3/REVIEW.json):
  fuentes exactas identificadas por hash, perfiles, resultados y mutaciones.
- Las carpetas de evidencia incluyen comandos, salidas, diagnósticos,
  ejecutables y entradas copiadas antes de cada ejecución.
- [Respaldo local verificado](../../artifacts/ntrpu-peer-multibuffer/20261004-datapath-handoff.zip),
  con las tandas y documentación, manifiesto SHA-256 y comprobación del ZIP.

## Límites y trabajo pendiente

La [síntesis aislada posterior](PEER_RX_MULTIBUFFER_SYNTHESIS.md) ya confirma
BRAM para los perfiles de dos destinos con reserva cuatro y ocho: 60/90
tiles equivalentes para 32/48 KiB de payload. El slack de setup estimado
a 5 ns es +0,365/−0,653 ns: reserva ocho incumple el objetivo. La organización
con un array independiente por banco aprovecha solo el 11,85 % de los bits
BRAM físicos brutos. Falta comparar agrupaciones más profundas y puertos
de reserva, además de colocar/rutear y restringir las interfaces externas.
Ese documento conserva informes, avisos y la validación de netlist.
La [variante compacta aislada](PEER_RX_MULTIBUFFER_COMPACT.md) compara otra
organización de RAM y del asignador conservando este módulo original.

Estos ensayos son funcionales, sin SDF ni CDC. Sus ciclos y bytes no son
una medición de ancho de banda de una FPGA ni reproducen todavía el modelo
ideal de 12,8 GB/s. Hay costes reales de admisión, inicio de ráfaga y lectura.

Antes de integrar faltan el framing y créditos PCIe, asociación con ventanas
y épocas, stop/abort/rearme completo, semántica de visibilidad y agregación
de resultados. Todo el camino AXI debe admitir la concurrencia requerida;
el árbitro existente de una sola escritura pendiente puede limitarla.
El receptor original, su banco, la cola Vivado y los trabajos del principal
permanecieron sin cambios.
