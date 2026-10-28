# Integración del retorno HBM con el protocolo peer

Contrato P0/P1 del 2026-10-07. Ya existe un [subperfil RTL de un span](PEER_READ_LEDGER.md), con validación de contexto y retención hasta ACK; el contrato completo permanece abierto. Parte del [retorno probado](HBM_READ_RETURN.md), del [modelo de transporte](../../tools/ntrpu_transport_model.py) y de las fronteras ya existentes. No modifica la ABI ni los bloques del principal.

## Contexto antes de reservar memoria

El modelo distingue `RegionHandle(session, node, slot, generation)`, una ventana con generación propia y `TransferId(session, source, sequence)`. Esos campos no caben por definición en el tag opaco de 64 bits del lector. Un adaptador debe conservar el contexto completo en una tabla y usar `cmd_tag` como identificador local de esa entrada, sin equipararlo a AWID/RID o a una dirección BAR.

Perfil vigente F1: dos entradas activas por canal, cuatro en total, coherentes con sus cuatro reservas de lectura. El gestor de regiones valida y fija la entrada antes del handshake del comando; durante su vida no puede cambiar destino, permiso ni generación.

| Verificación P0 | Acción requerida antes de `cmd_valid` |
| --- | --- |
| Sesión, nodo y generación | Comparar todos los campos con el registro vigente |
| Permiso de lectura y región publicada | Rechazar región cerrada, revocada o bloques de pesos aún no válidos |
| Offset y longitud | Comprobar `offset <= size` y `length <= size-offset`; verificar también overflow de base+offset |
| Identidad duplicada | Consultar comandos activos y protección contra replay de comandos retirados |
| Span local | Alineación de 64 bytes, 1–64 beats, mismo bloque de 4 KiB, rango HBM de 34 bits |
| Reserva de retorno | Asegurar contexto y capacidad de la cola de salida antes de emitir AR |

Dividir una transferencia mayor en spans no crea transferencias independientes: conservar identidad original, índice, cobertura y estado agregado. El contador de bytes útiles no debe incluir fragmentos fallidos ni duplicados.

## Conexión con el lector

| Frontera del lector | Responsabilidad del adaptador |
| --- | --- |
| `cmd_addr`, `cmd_len`, `cmd_tag` | Dirección local autorizada, `span_bytes/64-1`, token de contexto |
| `data_valid/ready`, `data_tag`, `data_last` | Relacionar payload con la entrada fijada; mantener orden y destino bajo backpressure |
| `cpl_valid/ready`, `cpl_status`, `cpl_bytes` | Registrar fin local; contrastar identidad y bytes esperados |
| `fault`, `stopped`, `drained` | Estados distintos; ninguno reemplaza confirmación remota o certificado de aislamiento |

Los ocho bancos del escritor y las cuatro reservas del lector F1 tienen ciclos de vida diferentes. No compartir su lista libre sin demostrar ownership, créditos y exclusión entre lectura, escritura y consumo. El subperfil actual ya retiene dos contextos por canal hasta la condición remota, aunque el frame local pueda haber sido liberado antes.

## Qué significa terminar una lectura remota

El lector acredita únicamente que entregó datos al consumidor local. Si ese consumidor es un DMA, su aceptación todavía puede dejar descriptores, TLP, respuestas o payload en colas intermedias. Mantener la región y el contexto retenidos hasta la condición final exigida por el transporte.

Para una operación peer que copia los datos a otro nodo, enlazar explícitamente: lectura local → aceptación del DMA → envío → visibilidad de memoria remota → confirmación de la transferencia. Una respuesta B del puente de origen no reemplaza esa confirmación, como ya establece el [contrato de escritura](PEER_RX_INTEGRATION.md).

Al parar o perder el enlace, cerrar nuevas admisiones. Los VALID ya ofrecidos deben conservarse y las obligaciones aceptadas deben drenarse o permanecer en cuarentena. El consumidor puede necesitar retirar y descartar payload de una época invalidada, pero no puede convertirlo en éxito. Un timeout no libera identidad, región ni buffers pendientes.

## Reutilización y límites de los bloques existentes

`ntrpu_pcie_fragment_peer.sv` ya enlaza una lease preautorizada con ingreso NTRPUF01 y escritura de una línea. Conserva contexto hasta drenaje y retorno. Es una referencia para la gestión de vida útil, **no un adaptador de lectura HBM de 4 KiB**.

El modelo Python conserva regiones, permisos, generaciones y transferencias; servirá como oráculo del nuevo adaptador. La integración PCIe debe usar la XCI y las interfaces efectivas de la variante principal: no conectar el payload directamente a un supuesto BAR/DMA ni cambiar sus resets para acomodar este bloque.

## Evidencia de aceptación pendiente

El 2026-10-07 se ejecutaron los **25 tests del modelo Python** desde una copia de fuentes verificada. Pasaron permisos, generaciones, replay, retención de créditos, completions, revocación y confirmación remota. Esto valida el oráculo existente; no conecta ni califica el nuevo RTL de lectura. Evidencia: `D:/AI/artifacts/ntrpu-read-return/20261007T103522780001Z/20261007T130949443841Z-transport-model/`.

```bash
python3 -B tests/test_ntrpu_transport_model.py
```

La composición RTL debe reproducir las siguientes observaciones del modelo y añadir fallos específicos de HBM/DMA:

1. Rechazos sin AR: rango, overflow, permiso, revocación, generación/sesión obsoleta, duplicado activo y replay tras retirar CPL.
2. Comando retenido mientras se intenta actualizar la región; ninguna traducción cambia entre admisión y último consumidor.
3. Dos canales con regiones y destinos distintos; stalls de comando, datos y finalización, sin mezclar identidades.
4. Fallo en el último fragmento, error del DMA y confirmación remota errónea; ningún éxito parcial presentado como transferencia completa.
5. Stop/pérdida de enlace en cada frontera; demostrar exactamente qué obligaciones siguen retenidas antes de rearmar.
6. Traza única entre descriptor, spans, HBM, DMA y ACK remoto; comparación numérica y de bytes antes de medir caudal.

Cerrar P0 exige RTL compuesto y estas pruebas; cerrar P1 exige además el endpoint real en simulación. La comprobación sobre dos tarjetas físicas permanece como etapa posterior.
