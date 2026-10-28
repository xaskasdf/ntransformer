# Modelo del adaptador PCIe y AXI para NTRPU

Implementación funcional local del 2026-10-04 UTC, sobre el
[modelo de dos nodos](TRANSPORT_MODEL.md). Añade traducción de ventanas,
fragmentación de escrituras, buffers de recepción, eventos AW/W/B y drenaje
antes de revocar accesos. Es un banco Python dirigido por eventos, no RTL
ni una ejecución del IP PCIe de AMD.

Pasaron **18 pruebas del adaptador y 25 regresiones del protocolo**.
Cada suite incluye un ensayo con 20 semillas de intercalado bidireccional.
La comprobación de visibilidad usa un modelo explícito de memoria final;
su equivalencia con DDR/HBM y el IP elegido sigue pendiente.

## Archivos y ejecución

- [Adaptador y planificador](../../tools/ntrpu_pcie_axi_model.py).
- [Pruebas del adaptador](../../tests/test_ntrpu_pcie_axi_model.py).
- [Modelo de protocolo](../../tools/ntrpu_transport_model.py), con el hook
  interno `_complete_packet` para devolver crédito después de escribir
  todos los bytes de un fragmento en el destino final.

Desde la raíz, con Python 3.10 o posterior:

```bash
python -B tests/test_ntrpu_pcie_axi_model.py -v
python -B tests/test_ntrpu_transport_model.py -v
```

Ambos comandos deben terminar en `OK` y devolver cero. No requieren VM,
CUDA, compilación, conexión de red ni instalación de paquetes. El banco
permanece independiente de CMake y de la cola Vivado.

## Evidencia local

La [tanda verificada](../../artifacts/ntrpu-pcie-axi-model/20261004T023657194851Z/result.json)
registra 43 pruebas PASS, comando, Python, tiempos y hashes antes/después de
ejecutar. Incluye una [traza de escritura parcial](../../artifacts/ntrpu-pcie-axi-model/20261004T023657194851Z/partial_write_trace.json)
con cinco comprobaciones negativas: confirmación prematura, revocación con
AXI activo, rearme antes de B, lectura de datos fallidos y lectura de una
región nueva todavía inválida.

El [respaldo ZIP](../../artifacts/ntrpu-pcie-axi-model/20261004T023657194851Z.zip)
contiene fuentes, documentación de entrada, salidas de ambas suites y la
traza. Se verificaron sus 14 miembros por CRC y contenido contra el manifiesto.
El [registro de verificación](../../artifacts/ntrpu-pcie-axi-model/20261004T023657194851Z/VERIFIED.json)
conserva SHA-256
`9328d20ee86cbf0874fbdf63f5f0be772b5653e8cffa21b6bdcdd85d50583360`.

## Perfil inicial y traducción

El perfil propuesto usa escrituras completas de 64 bytes, AXI de 512 bits,
ráfagas INCR, ID fijo y una sola ráfaga activa por receptor. Es una referencia
conservadora de orden y recuperación; no intenta representar el caudal final
ni la concurrencia que necesitará una tarjeta con varios canales de memoria.

`plan_writes` divide un fragmento según el menor límite entre MPS, máximo de
beats AXI, bytes restantes y fronteras de 4 KiB en **ambos** espacios de
direcciones. Un plan admite como máximo 1 MiB. MPS es un parámetro suministrado
por el banco, no un valor negociado con un dispositivo real. MRRS y las
lecturas DMA de origen no se modelan.

La traducción conserva direcciones de 64 bits:

```text
PCIe = base_BAR + offset_destino + offset_fragmento - offset_ventana
AXI  = base_region + offset_destino + offset_fragmento
```

Por ejemplo, 768 bytes desde PCIe `0x100000fc0` hacia AXI `0x200000f80`,
con MPS 256, se dividen en `64, 64, 256, 256, 128` bytes. Las fronteras
de página de origen y destino no tienen por qué coincidir. Otra prueba
mapea los últimos 256 bytes de una región virtual de 512 GiB mediante una
ventana pequeña. Los índices/tamaños BAR del hardware siguen abiertos;
se mantienen BAR0 CSR y BAR1 MSI-X del diseño existente.

## Cuándo una escritura está confirmada

AMD documenta que una respuesta del puente al iniciador AXI puede reflejar
el envío ordenado de un MemWr PCIe. No acredita por sí sola visibilidad
en memoria remota. El modelo separa ese evento de la respuesta AXI del
receptor. [AMD PG194, Transaction Ordering for PCIe](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Transaction-Ordering-for-PCIe).
Esta referencia de puente no califica automáticamente el modo XDMA existente.

El perfil inicial elige `AWCACHE=0000`, Device Non-bufferable. La especificación
AXI exige obtener su respuesta de escritura del destino final; el tipo
Device Bufferable permite respuestas intermedias.
[Arm IHI0022H, A4.4.1](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf).
Esta elección debe conservarse y verificarse en toda la ruta concreta.
Los escritores actuales de activaciones/resultados usan `0010`, también
no bufferable, y no se modificaron para esta exploración.

Secuencia del banco:

1. `post` captura datos y prepara spans; `acknowledge_posted` solo marca
   la confirmación local del puente de origen.
2. `deliver` acepta spans en orden FIFO y respeta la capacidad de recepción.
3. `launch` reserva la única transacción AXI del receptor. `accept_aw` y
   `accept_w` representan handshakes independientes, en cualquiera de los órdenes.
4. `memory_step` escribe un beat de 64 bytes en la memoria final simulada.
5. `accept_b` exige AW y todos los W. OKAY solo es admisible después de la
   visibilidad completa; SLVERR/DECERR abortan y pueden dejar datos parciales.
6. `confirm_remote` requiere todos los fragmentos escritos y todas sus
   respuestas OKAY. Después, `retire` libera referencias y espacio de cola.

El chequeo EARLY_BRESP pertenece al **oráculo del banco**. Un circuito real
no conoce `memory_step`: debe apoyarse en garantías verificadas del destino
y del interconnect o incorporar un mecanismo de finalización equivalente.
Confiar solamente en un B de origen o de un buffer invalidaría esta prueba.

Los créditos de fragmento se devuelven al terminar sus escrituras físicas
modeladas. La ventana permanece retenida hasta el retiro del trabajo. Son
vidas distintas: un crédito libre no autoriza reasignar el BAR.

## Revocación y transacciones pendientes

Ante fallo, el banco exige tres condiciones antes de revocar:

| Condición | Evento del modelo | Obligación del futuro hardware |
| --- | --- | --- |
| Emisores detenidos | `source_stop_ack` de ambos nodos | Ningún motor puede emitir nuevos accesos de la sesión antigua |
| Entradas aisladas | `isolate_ingress_ack` de ambos receptores | Nuevas escrituras antiguas no alcanzan memoria; se resuelven buffers anteriores al lanzamiento AXI |
| AXI drenado | Ninguna ráfaga activa | Todo AW/W lanzado termina y su B final se consume |

La primera condición no borra escrituras publicadas en el enlace. Aislar
el ingreso tampoco permite borrar una transacción AXI lanzada: incluso si
todavía no hubo handshake AW/W, sus VALID podrían estar expuestos. El banco
conserva esa transacción y permite terminarla con los datos ya recibidos.

La revocación descarta solamente spans que nunca se lanzaron a AXI. Una
escritura activa puede completar después del timeout y del retiro de su
finalización de error; sus bytes permanecen inválidos para el consumidor.
`recover` exige las tres condiciones, finalizaciones consumidas y una nueva
sesión. La memoria conserva los bytes parciales, con registros invalidados.

Los ACK de parada y aislamiento son hipótesis explícitas del entorno.
Deshabilitar Bus Master Enable, cambiar una época o resetear un contador
no constituye aquí una prueba de drenaje físico. Si el destino deja de
responder, el modelo queda retenido: no declara un reset unilateral seguro.
Una respuesta AXI de error finaliza la ráfaga dentro del banco; esa semántica
también debe calificarse en el controlador real.

## Cobertura y alcance pendiente

Se comprueban traducción sobre 4 GiB, ventanas en 512 GiB, límites MPS/AXI,
fronteras de página distintas, overflow, AW antes/después de W, WLAST,
respuestas prematuras, backpressure, crédito sin liberación de ventana,
errores parciales y revocación con tráfico ya lanzado. Las 20 semillas del
adaptador intercalan entrega, AW, W, memoria y B en ambos sentidos, con
pausas y comparación contra payloads independientes.

Los eventos representan handshakes aceptados, no ciclos. Todavía faltan
estabilidad VALID/READY bajo pausa, codecs TLP, requester IDs, CDC, tiempos,
lecturas, gestión del IP y la confirmación remota por un transporte real.
No se ha unido el banco al RTL FFN, a MSI-X ni al supervisor del principal.

El [control RTL aislado de recepción y drenaje](PEER_RX_GUARD.md) ya conserva
AW/W hasta aceptación, retiene la ventana hasta B/finalización, cierra
admisiones al parar y rechaza rearme con tráfico pendiente. Pasaron 74 casos
de señales en Icarus: 62 trazas contrastadas con este modelo y 12 rechazos
locales sin AXI. La [síntesis y replay funcional de netlist](PEER_RX_SYNTHESIS.md)
ya están revisados; integración con PCIe/DDR, visibilidad física, timing
integrado y cruces de reloj siguen pendientes.
