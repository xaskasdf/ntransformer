# Proveedor de inventario XDMA: contrato de implementación

Estado al 2026-10-06. El maestro AXI-Lite local tiene integración nativa
revisada y el [registro de descriptores](XDMA_DESCRIPTOR_LEDGER.md) pasó su
contrato unitario, comparación RTL/netlist y banco nativo con dos descriptores.
En ese banco la contabilidad termina 103,999 ns antes de recibir todos los
bytes en el host. El [secuenciador W1C/CSR](XDMA_STOP_REGISTERS.md) pasó
89 casos con el maestro real y un modelo de registros, además de síntesis
fuera de contexto. Su comparación RTL/netlist pasa los mismos 89 casos,
y la integración nativa revisada completa dos W1C, tres rondas CSR y
la recepción de 64 bytes con revocación retenida. La cadena H2C nativa de
dos elementos también está revisada: 128 bytes, rondas 0/1/1/2 y dos Done
posteriores a B reales; el snapshot tarda 1,448 µs desde el inicio aceptado.
El proveedor que autoriza drenaje completo sigue pendiente: faltan cadenas
generales, inventarios PCIe/BAR, contrato real del host y transferencia
coherente al join. La [composición DMA/BAR nativa](XDMA_DMA_BAR_INVENTORY.md)
ya está revisada con dos observadores de 32 escrituras: BAR conserva una
obligación durante 36 ns después del snapshot de descriptores; solo B real
la retira. Sin cierre del productor, los ACK permanecen en cero incluso
con inventarios vacíos. La composición bidireccional también está revisada:
64 bytes H2C/C2H, ledger completo 103,999 ns antes de la recepción host y
BAR pendiente 36 ns después del snapshot CSR. Pasaron 65 controles de
corrupción de evidencia. Siguen los
[inventarios PCIe y su conexión de producción](XDMA_PCIE_INVENTORY_PLAN.md)
y el join; Busy=0 o un snapshot no sustituyen ese cierre.
Dependencias y evidencia: [maestro local](XDMA_LOCAL_PROVIDER_DESIGN.md) y
[criterios de drenaje](XDMA_DRAIN_ACCEPTANCE.md).

## Identidad y fuentes de evidencia

El proveedor recibe un cierre coherente `{generation, tag}` del propietario
de la sesión. Debe mantener esa identidad desde la solicitud hasta el reporte
retenido. Una respuesta de otro intento, un cambio de identidad pendiente o
reutilización de identidad revoca la autorización. Se conserva el orden
lexicográfico `{generation, tag}` del coordinador: una sesión liberada puede
dar paso a un tag mayor de la misma generación. El maestro transporta
128 bits opacos; la validación de frescura pertenece al proveedor superior.

El cierre del host debe acreditar el bloqueo de productores y doorbells,
la barrera de sus escrituras, el inventario de descriptores publicados y los
destinos DMA que todavía deben recibirse. Un pulso del banco o un registro
Busy=0 no reemplaza ese contrato. El control de canales debe ser exclusivo:
ningún escritor concurrente puede reactivar Run o borrar diagnósticos.

## Secuencia local e inventario

1. Capturar identidad, cantidades esperadas y asociación de contadores con
   la sesión de cada canal. Verificar cierre del host y propiedad exclusiva.
2. Solicitar W1C Run en H2C `0x000c` y C2H `0x100c`, dato `1`. No esperar
   vaciado de H2C antes de solicitar parada de C2H. Cada escritura conserva
   su obligación hasta B real; el watchdog externo cubre un bus bloqueado.
3. Leer control, estado y contador de ambos canales mediante el maestro.
   Retener respuestas AXI no OKAY, timeout, identidad incorrecta y errores
   observados. Confirmar Run=0 y Busy=0; revisar todos los errores aplicables.
4. Confrontar contadores con el ledger de descriptores. Un descriptor publicado
   sin completion conserva su obligación: ni Run=0 ni un contador estable
   prueban cancelación. Un reinicio o wrap ambiguo del contador invalida la ronda.
5. Reunir también lecturas PCIe pendientes de datos/descriptores, escrituras
   AXI H2C, respuestas AXI C2H, BAR y recepción del host. El cierre se mantiene
   mientras se reúnen los contratos; cualquier nueva admisión o fallo lo revoca.
6. Transferir el resultado completo a `mgmt_clk` con el mailbox coherente,
   generación/tag y revocación. Integrarlo con el join de siete contratos;
   nunca conectar una lectura de registros directamente a `complete_ack`.

### Lecturas sin borrar diagnósticos

El secuenciador debe leer estado en `0x0040/0x1040`, nunca en
`0x0044/0x1044`: estas últimas direcciones borran bits de estado al leer.
Las dos escrituras permitidas durante parada son W1C Run, dato `1`,
strobes `0xf`; no escribir el registro de estado para borrar errores.
[PG195: H2C RC](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Status-0x44?contentId=oij5s8pLh5vfpYOkHrkPow),
[C2H RC](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Status-0x44?contentId=8efGbjhwNubY_d9KMQEAkw).

Máscaras derivadas de PG195 v4.2, revisadas el 2026-10-06 para AXI-MM:

| Registro | Bits de error definidos | Máscara derivada |
| --- | --- | --- |
| H2C estado | 23:19, 15:14, 13:9, 5:3 | `0x00f8fe38` |
| C2H estado | 23:19, 10:9, 5:3 | `0x00f80638` |

H2C separa errores de lectura PCIe, escritura AXI y descriptor; C2H
informa lectura AXI y descriptor. Un bit reservado observado en 1 debe
producir un diagnóstico separado y bloquear aprobación. Los bits 6, 2 y 1
son eventos de parada/completación; no equivalen a errores. Algunos eventos
dependen de habilitaciones de control: su ausencia no demuestra que nunca
ocurrieron. Conservar control y estado crudos junto con cada respuesta.
[PG195: H2C estado](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Status-0x40),
[C2H estado](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Status-0x40).

El estado de ocho bits del puerto XDMA es otra codificación: allí el bit 6
es Run y el 3 es Descriptor Done. No aplicar las máscaras CSR a ese puerto.

### Casos exigidos al próximo secuenciador

Además del tráfico positivo, exigir B/R retrasadas o con error, identidad
ajena, timeout con respuesta tardía, fallo mientras VALID espera READY,
contador inferior/superior al inventario, Run reactivado y certificado
revocado después de aprobar. Ningún fallo retira una petición ya ofrecida.
Tras completar ambas escrituras, comparar control/estado/contador de ambos
canales en rondas completas y conservar los seis valores con su identidad.

El maestro actual permite una transacción pendiente: la segunda parada
espera la respuesta real de la primera escritura, pero no espera que el
primer motor vacíe su DMA. Si el bus de gestión no responde, conservar la
obligación y señalar el bloqueo al supervisor independiente. Un watchdog
no debe inventar B/R ni convertir ese bloqueo en permiso de reset.

AMD especifica que el contador se reinicia al subir Run; por eso su valor
requiere asociación con la sesión, no solo igualdad numérica.
[PG195: contador H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Completed-Descriptor-Count-0x48).
La finalización C2H acredita emisión por el bloque PCIe; el contrato del host
debe acreditar además la recepción de las escrituras.
[PG195: C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel).

## Tandas y criterios de aceptación

| Tanda | Evidencia exigida antes de aprobar |
| --- | --- |
| Secuenciador e inventario local | VCD de ambas paradas, datos/errores reales, identidad retenida; cierre ausente, obsoleto o revocado nunca autoriza |
| Ledger de descriptores | Completion tardía, cadena prefetched, contador reiniciado/wrap y descriptor no completado conservan obligaciones |
| Integración XDMA nativa | Mismos contratos sobre tráfico BAR/H2C/C2H, esperas reales y contenido host verificado; sin modificar estado protegido del core |
| CDC y join | Payload coherente, reloj detenido, respuesta obsoleta, fallo simultáneo y revocación después de ACK; aprobación solo con siete contratos válidos |
| Reset completo | Timeout nativo de 50 ms, retiro de todas las obligaciones, secuencia real del core y nueva sesión operativa; reiniciar el banco no satisface este criterio |

Cada tanda conserva fuentes congeladas, resultados terminales y respaldo
local/NFS. Una prueba unitaria del secuenciador será evidencia parcial hasta
integrar el ledger y los productores reales. Siguen vigentes DIMM exacto,
512 GiB/cuatro DDR, inferencia con pesos reales y cierre físico de la placa.
