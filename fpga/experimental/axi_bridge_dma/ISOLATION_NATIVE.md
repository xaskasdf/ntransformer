# Prueba nativa de frontera host y guardian

Candidato privado posterior a [GUARD.md](GUARD.md). El banco
`ntrpu_axi_bridge_isolation.sv` conecta el cliente y el guardian reales con
el mismo XDMA/Bridge 4.2 rev. 4 generado en Vivado 2026.1 para VU47P.
El reloj de gestion del banco es independiente, de 100 MHz; el AXI nativo
se mide sobre su puerto publico. No cambia el top ni la cola principal.

## Cobertura y criterios

Cada campaña envia cuatro escrituras BAR y sus lecturas de comprobacion,
mas cuatro escrituras Type 0 de configuracion y sus lecturas: 16 solicitudes,
12 completions. Los offsets BAR son `0x208`, `0x20C`, `0x10208`, `0x1020C`;
los de configuracion, `0x208`, `0x20C`, `0xA08`, `0xA0C`. Es un conjunto
seleccionado de direcciones y aliases, no una exploracion exhaustiva de PCIe.

| Campaña | Estado exigido durante solicitudes y completions |
| --- | --- |
| 0 | Traduccion completa, sin contexto de transferencia reservado |
| 1 | Mitad superior escrita; B retenida por backpressure antes de escribir la inferior |
| 2 | Contexto reservado, esperando al lector de origen |

La campaña 1 mantiene BVALID nativo y BREADY bajo. No fuerza registros internos
ni inventa respuestas. El lector de inspeccion privado usa exclusivamente AR/R
y solo cuando el mapper no tiene transacciones pendientes. Sigue habiendo un
unico escritor de la tabla. Tanto ese mux como el backpressure son instrumentacion
del banco; no constituyen un arbitro de produccion.

El Root Port debe recibir completions atribuidas y terminar las lecturas BAR
con el dato escrito. El puerto entrante `m_axib_*` debe mostrar esas operaciones
contra la BRAM separada. Para configuracion se distingue Successful Completion
de Unsupported Request; un timeout o Completer Abort no acredita aislamiento.
El revisor comprueba tag, estado, error y Request Completed del descriptor,
segun [PG213](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Completion-Descriptor-Format).
El Root Port utiliza asignacion interna: el tag valido se captura de sus puertos
publicos `pcie_rq_tag0/pcie_rq_tag_vld0`, con una solicitud no posted pendiente
por vez. El campo propuesto en RQ no sustituye esa asignacion.
[PG213: gestion de tags](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Tag-Management-for-Non-Posted-Transactions).

Despues de cada campaña se leen `0x208/0x20C` por el puerto privado. Ambas
palabras deben conservar el valor autorizado. Dos transferencias posteriores
de 64 y 128 bytes deben llegar a los destinos PCIe completos esperados. Esta
comprobacion ejercita la separacion documentada por
[PG194](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Bridge-Register-Memory-Map).

Finalmente PERSTn se aplica con un tercer contexto pendiente del lector. Se
exigen fallo retenido del guardian, admision cerrada y cold reset liberado,
con identidad y ownership conservados. Esto no prueba recuperacion de TLPs
en vuelo ni una interrupcion independiente del reloj nativo.

## Resultado del 2026-10-09

La tanda `20261009-native-r1` obtuvo
`NATIVE_SAMPLED_HOST_ISOLATION_AND_GUARD_PASS` mediante revision local del log
y VCD descargados. Los 48 RQ, 36 tags asignados y 36 RC coinciden con los
handshakes nativos. Las 12 escrituras BAR y sus 12 lecturas llegaron a la BRAM;
las 24 operaciones de configuracion devolvieron Successful Completion y sus
12 lecturas devolvieron cero. Una completion de escritura sin payload se
registra con `words=0, data=null`, sin interpretar los bits de datos inactivos.

Las seis lecturas privadas conservaron las traducciones
`0x1234567800001000` y `0x9ABCDEF000002000`. El VCD confirma 28 handshakes RQ/RC
con la escritura superior pendiente y otros 28 con el contexto reservado.
La inspeccion de la campana 1 ocurre despues de completar el mapping; no
demuestra por si sola ausencia de cambios transitorios en la palabra inferior.
Dos paquetes salientes entregaron 64 y 128 bytes a sus destinos completos.
El reloj AXI midio 250 MHz. PERSTn retuvo el tercer contexto y produjo
`fault=1, code=1, cold=1, owned=1, stop=1, link=0`.

El revisor rechazo siete controles con evidencia alterada: payload host,
tag de completion, lectura privada, destino saliente, estado de mapping,
asignacion de tag omitida y payload entrante modificado solo en el VCD.
Compilacion, elaboracion y simulacion duraron 2,49, 361,14 y 251,53 segundos;
la tanda completo 615,18 segundos con dos CPU y nice 10. Los 623 archivos de
la semilla y el board original permanecieron intactos.

El archivo remoto descargado contiene 30 archivos verificados, ocupa 700342
bytes y tiene SHA-256
`6c545209da72d14601c08987d711ef9f8014e4dda6dcd9e151dc193b9ab0b3e3`.
`20261009-OUTCOME.json` recoge el resultado y `20261009-backup-receipt.json`
registra la restauracion del paquete local y la repeticion del revisor.

## Ejecucion y evidencia

La tentativa r0 del 2026-10-09 se detuvo tras el primer acceso BAR: el dato
`0xBAD00000` llego correctamente, pero el banco comparaba la completion con
el tag propuesto en RQ. El Root Port habia asignado el tag `0x00`. La revision
r1 atribuye cada completion mediante la salida publica de asignacion de tags.
El fallo del banco, sus fuentes y el VCD quedan conservados en
`20261009-native-r0/remote-evidence.tar.gz` dentro del directorio de artefactos;
se verificaron sus 30 archivos y el SHA-256
`305ceb05e48649736c6ae7591f3838ab8162884e5facb98d409cb5f69882c234`.

`run_isolation_native.py` se ejecuta en la VM con una salida nueva bajo
`/home/ubuntu/vivado/ntrpu-side/`. Copia el snapshot compilado anterior y
recompila solo el nuevo banco, guardian y binding de `board`. Limita afinidad
a dos CPU y usa nice 10. Los limites son 90 segundos de compilacion y 600
por elaboracion/simulacion; solo puede terminar el grupo de procesos propio.
Conserva logs, fuentes, comandos, hashes y comparacion de la semilla antes/despues.

```text
python3 -B run_isolation_native.py NUEVA_SALIDA --seed SNAPSHOT_WORK --example EJEMPLO_R3 --settings VIVADO_SETTINGS
python -B fpga/experimental/axi_bridge_dma/review_isolation_native.py SIMULATE_LOG NATIVE_VCD NUEVO_REVIEW_JSON
```

El codigo de salida del simulador no equivale a PASS. El revisor exige las
tres campañas completas, verifica los handshakes del log contra el VCD,
reconstruye los accesos y contrasta la evidencia de datos de la prueba anterior.
El marcador heredado `host_isolation=unproven` sigue indicando que el banco
base no certifica aislamiento universal; el nuevo veredicto solo cubre las
operaciones y estados descritos aqui.

Los artefactos se guardan en `D:/AI/artifacts/ntrpu-axi-bridge-isolation/`.
El paquete remoto excluye el nuevo cache compilado para reducir transferencia;
incluye fuentes y comandos para reconstruirlo usando la semilla anterior ya
respaldada. Para recuperacion se requieren Vivado y los respaldos nativos r4
(snapshot original) y r3 (ejemplo), ademas de esta nueva tanda.

Pendientes independientes: aislamiento exhaustivo, implementacion del control
privado de produccion, CDC/STA del guardian, reloj de gestion fisico, lector
DDR/HBM nativo, ACK entre tarjetas y recuperacion coordinada durante DMA.
El [ensayo de reloj AXI](CLOCK_LOSS.md) posterior paso ambos niveles detenidos
en Vivado/XSIM el 2026-10-11, con evidencia y respaldo propios. Valida la
retencion ante una interrupcion inyectada en la salida AXI; no amplia la
cobertura de aislamiento host ni acredita un fallo fisico del reloj PCIe.
