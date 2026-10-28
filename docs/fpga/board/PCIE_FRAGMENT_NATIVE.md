# Integración nativa de fragmentos

Estado: **`PASS_NATIVE_PCIE_FRAGMENT_CONTROL_DATA_MIG_PERIODIC`**, tanda
`20261005T014817Z`, finalizada el 2026-10-05 a las 02:06:19 UTC.
Revisión independiente aprobada; respaldo local y NFS verificado a las
02:12:55 UTC.
Reutilizar el [host NTRF con doble CDC y MIG](PCIE_FRAGMENT_HOST.md),
sin cambiar su RTL para adaptar el banco.

## Ejecución aislada

`tools/ntrpu_board_pcie_fragment_native_host.tcl` usa el banco
`fpga/tb/ntrpu_pcie_fragment_native_host.sv` y el productor
`ntrpu_pcie_fragment_native_host_tasks.vh`. El generador conserva el banco
anterior y crea archivos nuevos; el parche modifica seis archivos de una
copia nueva del ejemplo público. Se desconectan 17 puertos AXI-Lite de la
aplicación de ejemplo, manteniendo las conexiones de la IP XDMA real.

La tanda parte de la netlist ruteada revisada `20261005T012347Z` y del
productor PCIe revisado `20261004T234908Z`. Fuente fijada:
`source-20261005T014806Z.zip`, SHA-256
`c2d32a944b55ba4aa80ccf46f7517c0a9110358fdae8e0986ab84fbd989d3f83`.
El RTL de producción conserva sus hashes; se observan 184 señales.
Se verificaron 82 escrituras y 72 lecturas de control del fixture, más una
lectura BAR0 de descubrimiento anterior. La revisión conserva las
178 transferencias RQ del fixture: 24 beats de datos y 154 de control.

## Resultado y mediciones

| Comprobación | Resultado observado |
| --- | --- |
| Tramas de 64 bytes | 12; ocho aceptadas y cuatro rechazadas, dos por identidad |
| Control | 82 escrituras, 16 rechazadas según contrato; 73 lecturas con descubrimiento |
| Datos y memoria | Tres escrituras DDR, 192 bytes; ocho lecturas comparadas, 512 bytes |
| Cancelación | Resultado 1, sin escritura de la línea parcial; contenido anterior intacto |
| Recuperación | Job 3 finaliza sin reset en la última línea del rango exploratorio de 128 GiB |
| Evidencia | 184 señales; 206 inputs compilados; 134 archivos de IP PCIe conservan sus hashes |
| Pruebas negativas | 309 alteraciones de evidencia rechazadas |

La simulación termina a 124,523498 µs; Vivado tarda 1.077,333 s de pared.
Desde el primer beat RQ hasta B del bypass se miden 372–408 ns. Esa B
confirma recepción del fragmento, no escritura DDR ni retiro del trabajo.
Desde la aceptación de una línea hasta B del MIG se miden 195, 35 y 155 ns
para jobs 0, 1 y 3; DONE llega 20 ns después. Son latencias funcionales de
este banco pequeño. Los modelos FIFO no representan retardos de
sincronización: no son caudal sostenido ni mediciones de placa.

Se conservaron 156 avisos de herramienta y seis de ejecución, coincidentes
con la referencia nativa tras normalizar rutas y ordinales temporales.
Los 6.912 avisos DDR corresponden a 48 ráfagas periódicas con cadena causal
completa. Se verifican 90 solicitudes/aceptaciones/retornos periódicos,
98 READ físicos, tres WRITE físicos y vaciado final. Ningún aviso se
suprimió ni se forzó estado del modelo.

Una prueba negativa inicial descubrió que el verificador admitía un tiempo
de intención anterior al fixture. Se añadió el límite inferior; se
conservaron el diagnóstico y el verificador anterior en el respaldo.
El RTL, los estímulos, las trazas y la única ejecución Vivado no cambiaron.

Archivo sellado: 1.404 archivos, 204.531.064 bytes; SHA-256
`a6c632519219cb86756ed108d8c01385d98a7030863f5927c566f788a7ccad0f`.
La copia local vive en `artifacts/ntrpu-board-vivado/20261005T014817Z-V7-ffn-supervised-fragment-native-host/`,
mediante junction al disco D. `VERIFIED.json` certifica legibilidad y hashes
de todos sus miembros; `review.json` registra la aprobación funcional.

## Dependencias y conexiones

Partir del ejemplo XDMA conservado en `20261004T154440Z`. El control BAR0
debe conducir el AXI-Lite real de XDMA hacia `ntrpu_pcie_fragment_host`.
BAR2 conduce su interfaz AXI bypass hacia el CDC de datos del mismo bloque.
Desconectar exclusivamente los puertos correspondientes de la aplicación
de ejemplo; conservar las IP AMD y sus hashes.

El productor de tramas debe reutilizar `ntrpu_pcie_frame512_driver.vh`, ya
probado en `20261004T234908Z`: transmite los 64 bytes de una trama mediante
dos beats RQ. Conservar el observador de paquetes completos; el productor
DWORD anterior no sirve para esta longitud. Mantener las lecturas MMIO
por solicitudes y completions PCIe reales, incluyendo la lectura de
descubrimiento BAR0 que hace el ejemplo antes del fixture.

MIG, modelo genérico, macros de geometría, lecturas de retorno y observación
de tráfico periódico deben permanecer iguales a las tandas calificadas.
El aislamiento y la prueba de vaciado aguas arriba siguen siendo BFMs:
no se debe presentar esa barrera como un fence PCIe físico.

## Fixture ejecutado

Cuatro reservas consecutivas con época `0x100000001 + job` y tag
`0x8000000100002000 + job`. Reutilizar la ventana intermedia después de
retirar la reserva anterior, sin reset entre trabajos.

| Job | Destino DDR | Secuencia de tramas | Resultado esperado |
| --- | --- | --- | --- |
| 0 | `0` | Magic incorrecto; mitades 0 y 32 válidas | Una línea; un rechazo local |
| 1 | `0x100000040` | Tag incorrecto; mitad 0; duplicado idéntico; mitad 32 | Una línea; un rechazo por identidad |
| 2 | `0x100000040` | Trama de job 1; mitad 0 actual; STOP por BAR0 | Sin escritura; reserva cancelada y dato anterior intacto |
| 3 | `0x1fffffffc0` | Ventana incorrecta; mitades 0 y 32 válidas | Una línea; recuperación sin reset |

Se cumplieron los totales del fixture: 12 tramas, ocho aceptaciones, cuatro
rechazos (dos por identidad), tres escrituras DDR de 64 bytes y cuatro
resultados. Se vuelven a leer todas las ubicaciones escritas tras cada
retiro: ocho lecturas de 64 bytes en total.

## Criterios de revisión

- Leer identificación NTRF, capacidades y formato por BAR0; capturar y
  comprobar contadores, identidad, motivo y validez de cada snapshot.
- Comprobar bloqueo del descriptor, ACK adelantado, ACK con tag alto
  incorrecto y rechazo de un ARM con época ya retirada.
- Reconstruir solicitudes RQ, tags asignados, completions RC, AXI-Lite y
  bypass desde pines VCD. No omitir tráfico de enumeración ni beats de datos.
- Exigir igualdad de contenido y conservación de handshakes en ambos CDC,
  resultados posteriores al vaciado y readback completo desde MIG.
- Revisar cada aviso DDR mediante órdenes, pines y retornos periódicos;
  verificar drenaje final. Conservar intentos fallidos y respaldo por tanda.

## Identificación antes de integrar cómputo

La revisión de fuentes encontró una colisión experimental: tanto
`ntrpu_ffn_control.sv` como `ntrpu_pcie_fragment_control.sv` devuelven
`0x4652544e`. FFN ABI 1.0 también comparte la versión `0x00010000`,
pero interpreta los descriptores y comandos de otra manera. El ID aislado
no autoriza a utilizar el mapa de fragmentos.

| Bloque actual | Versión | Capacidades `0x08` |
| --- | --- | --- |
| FFN, retiro local | `0x00010000` | `0x06` o `0x07`, según formato de salida |
| FFN, retiro externo | `0x00020000` | `0x0e` o `0x0f` |
| Fragmentos F01 | `0x00010000` | `0x1f` |

Para este endpoint, el driver deberá exigir la tupla completa ID, versión
y capacidades; después comprobar magic y tamaños en `0x9c..0xa8`, antes
de escribir comandos. Rechazar perfiles desconocidos. Esta es una regla
de compatibilidad propuesta; todavía no existe aquí un driver de hardware
que la implemente. El banco sí lee y compara esos campos por PCIe.

La futura composición FFN/transporte necesita bloques de registros
distintos y una identificación inequívoca, con una revisión de ABI propia.
No reinterpretar un bloque como otro ni modificar las tandas anteriores.

Esta prueba cubre integración de transporte y control sobre un canal frío.
Quedan separados el DIMM comercial exacto, reset con tráfico, cuatro
canales, temporización de placa, hardware y cómputo con pesos reales.
La [conexión propuesta con multibuffer](PCIE_FRAGMENT_POOL_INTEGRATION.md)
define las obligaciones del siguiente adaptador y el orden de sus pruebas;
su especificación no acredita una integración ejecutada.
