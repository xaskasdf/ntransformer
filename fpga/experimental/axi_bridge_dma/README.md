# Prototipo de escritor DMA para AXI Bridge

Estado: **`AXI_BRIDGE_WRITER_LOCAL_PASS`** en `20261009-local-r1`: 16 escenarios
en cuatro ejecuciones (64 ejecuciones de casos), con replay despues del
retiro, tres mutantes RTL rechazados y tres corrupciones de evidencia
rechazadas por ejecucion. El alcance sigue siendo exclusivamente local.
Carpeta aislada, sin conectar a la cola Vivado ni a los tops existentes.

## Contrato del prototipo

Un contexto y un reloj; escritura AXI4 de 512 bits, ID cero, hasta 4096 bytes.
Longitud y direcciones deben ser multiplos de 64; origen y destino no cruzan
4 KiB. Una ventana de traduccion de 4 KiB tiene bases AXI de 48 bits y PCIe
de 64 bits. El origen se valida a 39 bits antes de entregarlo al lector.
No incluye lector DDR/HBM, segmentador de solicitudes grandes ni multicanal.

`cfg_valid/cfg_ready` captura la configuracion. `map_valid/map_ready` entrega
la imagen a un gestor local; `map_rsp_ok` debe certificar programacion y
lectura de comprobacion del puente. Solo entonces admite comandos. El banco
modela ese gestor: este modulo NO programa registros AMD. Debe ser la unica
ruta de escritura a esos registros y quedar fuera del mapa del host.

El comando captura tag (sesion y secuencia), generacion, origen, offset y
longitud. La secuencia crece durante toda la sesion. La ventana permanece
reservada hasta consumir el resultado. Una nueva configuracion exige una
generacion mayor y la misma sesion; una sesion nueva requiere arranque
coordinado, sin reutilizar mappings antiguos.

El buffer de 4 KiB recibe un frame atribuido y su finalizacion de lectura
antes de ofrecer AW/W. AW y W avanzan independientemente. Las ofertas ya
presentadas conservan VALID y contenido bajo backpressure, incluso si falla
el enlace. B con OKAY acredita la respuesta local, no recepcion remota.
El resultado requiere ademas ACK con tag, generacion y longitud exactos.
El receptor y la visibilidad que sustentan ese ACK aun son externos.

Un error bloquea admision/configuracion y retiene el contexto. Volver a
levantar el enlace no lo libera. Un resultado exitoso ya ofrecido mantiene
su contenido; un fallo posterior impide reutilizar el contexto. No existe
recuperacion operativa, timeout, CDC ni garantia frente a reloj detenido.
`cold_rst_n` solo se usa para configuracion coordinada; no conectar al reset
PCIe para borrar obligaciones. El buffer no se borra, pero sus contenidos no
pueden emitirse hasta recibir y validar todos los beats de un comando nuevo.

## Prueba local

```powershell
python -B fpga/experimental/axi_bridge_dma/run_local.py NUEVA_SALIDA --toolchain TOOLCHAIN --seeds 4
```

Icarus disponible en
`D:/AI/artifacts/ntrpu-hbm-dma-rtl/20261009T005545469986Z/restore-toolchain`.
El runner congela fuentes, registra hashes del toolchain y escribe logs, VCD,
revision y manifiesto en una carpeta nueva. Cada proceso tiene limite de 30 s.
Prueba 16 escenarios por semilla: AW/W en ambos ordenes, payload de 64/128/4096
bytes, direcciones altas, captura frente a cambios, generacion obsoleta,
limites, replay despues del retiro, errores B/BID, ACK temprano/incorrecto,
frame/CPL defectuosos y fallo con resultado retenido. Revisa bytes AXI y
rechaza tres corrupciones de logs. Compila tres mutantes RTL y exige un fallo
semantico concreto: direccion mutable, perdida de propiedad y falta de ACK.
Los cambios de espera son deterministas; no representan cobertura exhaustiva.

Tandas locales: `D:/AI/artifacts/ntrpu-axi-bridge-writer/`. El PASS del banco
no califica un FIFO, un mapping o la recepcion PCIe reales. La ruta de ACK
del banco es un estimulo atribuido despues de los handshakes AW/W.

## Integracion pendiente

Generar un endpoint en modo AXI Bridge compatible con el dispositivo elegido;
implementar el gestor de mapping con respuesta atribuida y exclusion real;
conectar lector, ACK remoto y supervisor independiente. Vincular constantes
de atributos AXI exigidas por la IP (incluidos PROT/CACHE/LOCK) y comprobar
puertos, resets, errores, tiempos y recepcion nativa. No trasladar un PASS
local a PCIe nativo, seguridad del host, capacidad DDR o placa fisica.

PG194 documenta el [uso de un DMA local con el puente](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Example-Use-Mode)
y el [acceso local a registros VSEC](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Bridge-Register-Memory-Map).
La compatibilidad exacta de configuracion debe verificarse en Vivado.
