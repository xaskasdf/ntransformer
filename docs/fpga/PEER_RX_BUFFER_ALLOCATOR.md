# Asignador RTL de bancos propios y reserva

Tanda local del 2026-10-04: **24 simulaciones y 14.682 ciclos contrastados
con un modelo independiente**, con cuatro mutaciones incorrectas detectadas.
Es un bloque de control aislado de la [propuesta multibuffer](PEER_RX_MULTIBUFFER.md).
Todavía no contiene RAM de payload, emisores AXI ni conexión PCIe/HBM.

## Contrato de asignación

El [RTL](../../fpga/rtl/ntrpu_buffer_pool_allocator.sv) parametriza canales,
bancos propios por canal, reserva compartida, cuota de préstamo por canal
y ancho de generación. Por defecto: dos canales, dos bancos propios cada
uno, cuatro de reserva, cuota cuatro y generación de 64 bits.

- Una asignación total por ciclo. Cada canal solicita con `alloc_valid`;
  `alloc_ready` selecciona como máximo uno. Al coincidir ambos en el flanco,
  el canal captura `alloc_bank` y `alloc_generation` compartidos.
- Se prefieren bancos propios. Si no hay uno disponible, se presta reserva
  dentro de la cuota. Round robin selecciona entre canales elegibles;
  la elegibilidad requiere banco disponible y petición activa.
- Una devolución por canal y ciclo. El puerto identifica al propietario;
  el banco y su generación deben coincidir con una asignación vigente.
- Devoluciones válidas actualizan disponibilidad y cuota para ese mismo
  flanco. Un banco devuelto puede reasignarse inmediatamente, incluso a
  otro canal si pertenece a la reserva, con una generación nueva.
- `stop` cierra admisión y conserva las devoluciones. No revoca asignaciones
  ni implementa por sí mismo el stop completo del protocolo de transporte.

`alloc_ready` es una aceptación de petición, no un canal de respuesta con
VALID pendiente. Cada handshake representa una asignación distinta. El
solicitante debe disponer de descriptor y espacio de resultado antes de
pedirla, y registrar el propietario y la generación en ese mismo flanco.
Es una interfaz interna de petición de recursos, no AXI: una solicitud
todavía no aceptada puede retirarse. La garantía de reparto supone que
el canal mantiene su petición mientras espera y sigue siendo elegible.

## Protección y recuperación

Una devolución con índice inválido, banco libre, propietario ajeno o
generación antigua produce `return_error` y un `fault` persistente.
Bloquea nuevas asignaciones incluso en el flanco que detecta el error.
Las devoluciones válidas simultáneas se conservan; las posteriores siguen
aceptándose para drenar. No existe recuperación automática ni timeout que
libere memoria por su cuenta.

La generación aumenta al asignar. Cuando llega a su máximo, el banco se
puede devolver pero queda retirado, visible en `retired_mask`; el contador
nunca vuelve a cero. El ensayo de agotamiento usa tres bits para recorrer
el límite; los perfiles normales usan 64 bits.

El reset es síncrono y frío/coordinado. Solo se admite tras drenar y
aislar productores y devoluciones antiguas. El bloque no conserva una
época a través del reset: la integración debe impedir que una devolución
de la sesión anterior pueda reaparecer con un identificador reutilizado.
`drained` indica ausencia de bancos asignados, **no drenaje de AXI ni de
resultados**. El banco de pruebas drena antes de cada nuevo reset.

El asignador tampoco conoce WLAST: quien devuelve el banco debe certificar
que todos sus datos se aceptaron y ningún registro de salida depende de
ellos. Los descriptores pendientes de B y referencias de ventana tienen
vida separada. Retener esas referencias sigue siendo responsabilidad del
receptor completo, según el [contrato de integración](PEER_RX_INTEGRATION.md).

## Validación realizada

Icarus 12.0 en WSL local, tres semillas (`1`, `7`, `20261004`) por perfil:

| Perfil | Canales | Propios/canal | Reserva | Cuota/canal | Generación |
| --- | ---: | ---: | ---: | ---: | ---: |
| Referencia | 2 | 2 | 4 | 4 | 64 bits |
| Reserva ampliada | 2 | 2 | 8 | 8 | 64 bits |
| Cuota limitada | 2 | 2 | 4 | 2 | 64 bits |
| Nueve bancos | 3 | 2 | 3 | 3 | 64 bits |
| Pool compartido | 2 | 0 | 3 | 3 | 64 bits |
| Banco único | 1 | 1 | 0 | 0 | 64 bits |
| Agotamiento | 1 | 1 | 0 | 0 | 3 bits |
| Sin reserva | 2 | 2 | 0 | 0 | 64 bits |

El [banco de pruebas](../../fpga/tb/tb_ntrpu_buffer_pool_allocator.sv) genera
solicitudes simultáneas, saturación, cuotas, devolución/reasignación en el
mismo ciclo, pausa de admisión, doble devolución, propietario ajeno,
generación antigua, índices fuera de rango cuando son representables y
agotamiento. Añade 512 ciclos aleatorios por ejecución y drenaje final.

El [revisor independiente](../../tools/check_ntrpu_buffer_pool_allocator.py)
reconstruye asignaciones con conjuntos y diccionarios desde las entradas;
compara READY, banco, generación, errores y máscaras antes/después de cada
flanco. Comprueba secuencia de ciclos, casos presentes y finalización.
Resultados agregados: 5389 asignaciones y devoluciones válidas, 1575
reutilizaciones en el mismo ciclo y 99 devoluciones inválidas rechazadas.
Se aceptaron 712 devoluciones durante stop y 42 durante fault.

Cuatro variantes alteradas exclusivamente en la carpeta de evidencia
ignoraban la generación, perdían el banco reasignado, ignoraban la cuota
o permitían volver a generación cero. Todas produjeron una discrepancia
de señales con el revisor; no se contó un fallo de compilación como detección.
Los avisos de Icarus sobre sensibilidad de arrays están conservados.

## Reproducción y evidencia

Con Icarus accesible en PATH, desde la raíz del repositorio en Linux/WSL:

```bash
python3 -B tools/run_ntrpu_buffer_pool_allocator.py artifacts/ntrpu-buffer-allocator/nueva-tanda
```

La salida debe ser una carpeta nueva. El ejecutor admite `--iverilog`,
`--vvp` y `--ivl-base` para la herramienta local extraída. Esta ejecución
usó `artifacts/ntrpu-peer-rx-toolchain/extracted/usr/`; no instaló paquetes.

- [Revisión de la tanda](../../artifacts/ntrpu-buffer-allocator/20261004-allocator-r1/REVIEW.json):
  perfiles, cobertura, mutaciones y hashes de entradas.
- La misma carpeta contiene fuentes exactas, comandos, salidas completas,
  diagnósticos y ejecutables de simulación. El ejecutor registró 4,24 s
  para compilaciones, simulaciones y replay.
- [Respaldo local](../../artifacts/ntrpu-buffer-allocator/20261004-allocator-r1-handoff.zip),
  con manifiesto SHA-256 y comprobación de integridad del archivo.

## Límites y siguiente integración

Esto valida el reparto de identificadores de banco, no los datos que
contendrán. Faltan RAM con sus puertos/pipeline, colas por destino,
seguimiento de varias escrituras AXI, créditos de finalización y pruebas
del stop/rearme completo. El planificador debe respetar la preferencia y
las cuotas implementadas; no hay préstamo dinámico de cuota en este bloque.

Las 5389 asignaciones no son una medida de caudal. No se ejecutó síntesis,
colocación/ruteo, análisis de CDC ni simulación de netlist de este asignador;
faltan coste y timing del camino combinacional de arbitraje/devolución.
Las generaciones añaden estado propio además de la RAM de payload.

Los resultados de caudal del [modelo exploratorio](PEER_RX_MULTIBUFFER_STUDY.md)
siguen dependiendo de sus supuestos; aún no se han reproducido en RTL.
El receptor previamente validado, su banco de pruebas, la cola Vivado
y los trabajos del principal se mantuvieron sin cambios.
