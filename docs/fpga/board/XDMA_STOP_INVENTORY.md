# Parada e inventario público de XDMA

Estudio V4 iniciado el 2026-10-06 UTC. Depende del ejemplo nativo
`20261004T154440Z-V4-ffn-supervised-pcie-bypass-probe` y conserva los
certificados [AXI4](AXI4_FORWARDED_FAULTS.md). Este documento no autoriza
reset global ni reutilización de pesos, buffers o descriptores.

## Configuración observada

El XCI preservado es XDMA 4.2, revisión 4, sobre
`xcvu5p-flva2104-2-i`, Vivado 2026.1. Tiene un canal H2C y uno C2H,
AXI Memory Mapped, maestro BAR bypass de 64 bits y maestro AXI-Lite.
`xdma_dsc_bypass=false`, `xdma_sts_ports=false`, `xdma_axilite_slave=false`
y `soft_reset_en=false`. El Completion Timeout permanece en **50 ms**.

Por tanto, este banco consulta registros mediante el Root Port PCIe
nativo. No presupone puertos de estado que el XCI no ha generado.
El BAR de registros DMA es BAR1; BAR2/3 es la ventana de bypass. No
confundir bypass de descriptores, bypass de transacciones PCIe y DMA.

## Contrato documental

Al borrar `Run`, el motor SGDMA deja terminar su descriptor actual.
La escritura W1C de `0x000c=1` permite borrar únicamente ese bit y
conservar la configuración de notificación. Debe comprobarse la lectura
posterior del control; enviar una escritura posted no acredita su aplicación.
[Control H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Control-0x04),
[alias W1C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Control-0x0C?contentId=0MC0Pmb_C9~L~EnhMMYgxQ).

AMD define `Busy` como descriptor almacenado o solicitudes DMA pendientes.
`0x0048` cuenta descriptores completados y se reinicia al volver a activar
`Run`; no es un contador acumulado permanente de propiedad.
[Estado para depuración](https://docs.amd.com/r/en-US/pg195-pcie-dma/Initial-Debug-of-the-DMA/Bridge-Subsystem-for-PCIe),
[contador de completados](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Completed-Descriptor-Count-0x48?contentId=gTyxdHzmeBK1K7kdu1BZYQ).

Las peticiones dirigidas al BAR bypass alimentan otro maestro AXI4.
La separación exige verificar el inventario de ese camino además de los
motores SGDMA. Es una condición de diseño, pendiente de calificación
global, no una garantía de coherencia entre tarjetas.
[Maestro bypass](https://docs.amd.com/r/en-US/pg195-pcie-dma/Host-to-Card-Bypass-Master?contentId=8Yf0lLVRT3wlmJ2Ygv_5xA).

## Banco y criterios

Fuentes vigentes: `fpga/tb/ntrpu_xdma_stop_inventory_r2.sv`,
`tools/patch_ntrpu_xdma_stop_inventory_r3.py` y
`tools/ntrpu_board_xdma_stop_inventory_r6.tcl`. Se copia el ejemplo a
una carpeta nueva. Cambian cuatro archivos públicos del banco:
instancia de sonda, conexión de respuestas B, selección de prueba y
rutina de completions del host. La lógica protegida y los XCI conservan
sus hashes. La revisión inicial de tres archivos permanece preservada.

1. Transferir un descriptor terminal de 64 bytes desde la memoria del
   Root Port hasta AXI `0x1000`. Comparar los 64 bytes independientemente.
2. Retener la respuesta B en la interfaz pública de la aplicación.
   Borrar `Run`, leerlo como cero y observar `Busy=1`, contador cero y
   respuesta sin retirar durante dos rondas de lectura.
3. Liberar B y exigir una única respuesta, `Busy=0`, contador uno y
   ausencia de errores de canal.
4. Manteniendo ambos motores parados, enviar una escritura BAR de
   cuatro bytes y retener su B. Comprobar que el inventario BAR continúa
   pendiente después de observar SGDMA en reposo. Lanzar una lectura de
   control y liberar B desde una rama independiente tras 1.024 ciclos del
   endpoint. Comprobar el retorno de esa lectura, volver a consultar el
   estado y leer los mismos bytes BAR mediante PCIe.
5. Reconstruir handshakes, contenido, orden, resets y consultas desde VCD;
   contrastarlos con las marcas del banco. Revisar fuentes compiladas,
   avisos, hashes y respaldo antes de aprobar el alcance observado.

Límite: 180 µs simulados y 1.800 segundos de ejecución; dos hilos. Este
ensayo no aplica reset ni reduce el Completion Timeout. El descriptor
terminal único no prueba que `Run=0` cancele una cadena prefetched.

## Registro de ejecución

- `20261006T030808Z-V4-ffn-supervised-xdma-stop-inventory`: fallo Tcl
  previo a simulación, opción con guion interpretada por `set_property`.
  **No evidencia funcional.** Respaldado: 474 archivos, 42.155.911 bytes;
  SHA-256 `a3878c545df23e41ad96da9a2428a7a7bb2c9c8711ce676c23df3e64f3b0f34e`.
- `20261006T031010Z-V4-ffn-supervised-xdma-stop-inventory-r2`: revisión
  Tcl que usa `set_property -dict`; falló antes de elaborar porque
  `-mt 2` duplicaba el `--mt 2` generado por Vivado. Respaldado: 604
  archivos, 41.386.960 bytes; SHA-256
  `0b8b146dc848a7590e61cefe6c85796b97906de0eca62e836e4c30faf8cc520d`.
- `20261006T031300Z-V4-ffn-supervised-xdma-stop-inventory-r3`: usa el
  límite generado por `general.maxThreads=2`. Terminó a **98,640 µs**
  simulados tras 583,385 segundos de ejecución, cuando el registrador TX
  del Root Port rechazó una trama. Vivado devolvió **0**, pero no hubo
  marca de éxito ni ningún handshake AW/W/B en DMA o BAR. **No aprobado.**
  Respaldado: 842 archivos, 109.069.696 bytes; SHA-256
  `518140815331091203232e5bb53f77f13ef06a93777e8c2b161d48e5f79a865b`.
- `20261006T032956Z-V4-ffn-supervised-xdma-stop-inventory-r4`: emisor
  público de completions y trazas CQ/CC completas. A **98,164 µs**, la
  rutina rechazó `attr=2` porque esperaba cero. CQ acredita dirección
  `0x100`, ocho DWORD, TC cero y atributo dos; no se emitió ningún CC ni
  hubo handshakes AXI. Diagnóstico del estímulo, sin aprobación funcional.
  Respaldado: 841 archivos, 108.596.523 bytes; SHA-256
  `851ed53d534c76bfdbba957cd2924b0982c8fa04f28b76db8a8fa0fbd49b338d`.
- `20261006T034802Z-V4-ffn-supervised-xdma-stop-inventory-r5`: conserva
  el atributo observado y lo devuelve en CC. Termina por watchdog a
  **180 µs**, en fase BAR. Hubo una AW/W/B DMA y una AW/W BAR; B del BAR
  quedó retenida. La consulta de control posterior fue aceptada por RQ
  pero no recibió RC durante el resto del ensayo. Se conserva como
  diagnóstico, sin aprobación del banco completo. SHA-256 del respaldo:
  `cfa420318155c472f13527df269ebea8376b042675f5b004b03291d0c9524c29`.
- `20261006T040407Z-V4-ffn-supervised-xdma-stop-inventory-r6`: conserva
  el atributo observado y libera B desde una rama independiente. **Observación aprobada** tras
  reconstrucción independiente: una transferencia H2C de 64 bytes, una
  escritura/lectura BAR de cuatro bytes y 24 consultas de registros.
  Termina a 151.170000 µs simulados. Respaldo SHA-256
  `7d531d9330c10dc8b16792b970fc5f911de65c0778c6dbca34385afa16166d34`.

## Diagnóstico del modelo de host

En las fuentes públicas concretas del ejemplo, `TSK_DECIPHER_FRAME`
busca la cabecera TX en el byte 32, mientras `TSK_TX_COMPLETION_DATA`
construye una cabecera de 12 bytes desde el byte 0. El registrador acaba
leyendo el byte 20 del payload. En nuestro patrón ese byte es `0xa1`,
coherente con el tipo de trama rechazado. La revisión conserva hashes
de las fuentes, la salida de la simulación y los 139 avisos heredados
contrastados en 48 ubicaciones. El ensayo R3 no trazó CC/CQ, por lo que
no demuestra si la transacción real de completion era correcta.

Además, la rutina generada toma el primer fragmento de `DATA_STORE` y
los siguientes de `DATA_STORE_2`. La nueva rutina
`fpga/tb/ntrpu_xdma_host_completion_r2.vh` define explícitamente el estímulo
de este banco: descriptor de 32 bytes y datos de 64 bytes desde
`DATA_STORE`. Conduce únicamente la interfaz pública CC, mantiene VALID
hasta READY y calcula KEEP/SOP/EOP para los fragmentos de 52 y 12 bytes.
No modifica estados internos del endpoint ni fabrica registros de estado.

El atributo `2` observado activa Relaxed Ordering: CQ lo presenta en el
bit 125 y CC lo devuelve en el bit 93. TC y atributos se copian de la
petición según los formatos de
[CQ](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Completer-Request-Descriptor-Formats) y
[CC](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Completer-Completion-Descriptor-Format)
de PG213. La revisión preserva ese comportamiento; no lo desactiva para
pasar la prueba.

El revisor R7 debe reconstruir las dos peticiones CQ y los tres beats CC,
comprobar tag/requester, longitud, contenido del descriptor y cada byte de
datos, además de conservar TC y atributos; después verifica su recepción
por AXI y las consultas PCIe de estado. La correlación RQ/RC de estas
consultas usa una sola lectura pendiente y su orden temporal: el núcleo
asigna los tags y sus puertos de asignación no se trazaron. La comparación
exacta CQ/CC sí usa el tag recibido del núcleo. Esta sustitución es un
componente de estímulo acotado, no una
reparación ni una homologación general del modelo AMD.

## Consulta bloqueada detrás del BAR

R5 retuvo la B del BAR desde 134,718 µs hasta el watchdog. La petición de
lectura del registro de control fue aceptada a 134,724 µs, pero solo hubo
19 completions para 20 lecturas. La traza conserva BVALID de aplicación y
BREADY cero durante 11.320 flancos posteriores; reset y enlace siguen
activos. Esto demuestra la espera en este ensayo, sin identificar su cola
interna ni generalizar el ordenamiento a todos los atributos o dispositivos.

El banco tenía una dependencia circular: liberaba B después de terminar
esa consulta. R6 mantiene la consulta pendiente y libera B tras 1.024
ciclos desde otra rama. El revisor exige que la aceptación real de B
preceda a la completion de la consulta; no basta que el temporizador corra.
La enumeración de este banco asignó el acceso host `0x100002000`; la
revisión inicial esperaba incorrectamente `0x900002000` de otro ejemplo.
Se corrige el comprobador a la dirección trazada, conservando ambas revisiones.

## Resultado revisado y límites

La respuesta B de H2C permaneció pendiente durante
**25.140000 µs** simulados y la de BAR durante
**4.096000 µs**. Se compararon los 64 bytes DMA
y los cuatro bytes BAR. El revisor reconstruyó 25 peticiones
de lectura PCIe, 25 completions y 6 escrituras posted,
además de las dos peticiones CQ y tres beats CC del emisor acotado.

Al borrar Run, H2C conserva Busy y contador cero mientras falta B. Al
consumir B, Busy baja y el contador llega a uno. Después una escritura BAR
mantiene su propia respuesta pendiente con ambos motores SGDMA parados.
Esto exige inventarios separados en el diseño de parada.

La consulta pendiente fue aceptada a **134,724 µs**; B del BAR terminó
a **138,816 µs** y la completion de la consulta llegó a **139,072 µs**.
Se verifica así la eliminación de la espera circular del banco. Los tiempos
de retención se imponen como estímulo; no son resultados de rendimiento.

Se rechazaron **52 alteraciones de evidencia**.
Los 139 avisos heredados se contrastaron con sus fuentes compiladas.
Archivo de revisión independiente: `37e442cdeb1df90285caff7756f5c18ceaccee85da5d6b38eabad8af09fe706e`;
verificado localmente y en NFS. Son observaciones funcionales simuladas,
sin medida de rendimiento o latencia física. No califican cadenas, C2H,
reset del endpoint, vaciado global de XDMA ni placa completa.

## Trabajo posterior

Seguir la [matriz de cierre](XDMA_DRAIN_ACCEPTANCE.md): C2H, cadenas con descriptores precargados y obligaciones de
completion PCIe; definir el cierre del productor host y su fence BAR.
Después integrar ese certificado con las secciones AXI-Lite/AXI4 y el
[coordinador de reset](ENDPOINT_RESET_COORDINATION.md). Una observación de
reposo o un ACK de sección por separado no cierran las colas protegidas
de XDMA. Siguen pendientes DDR exacta, integración y validación física.
