# Parada H2C con dos descriptores adyacentes

Tanda V4 `20261006T042709Z-V4-ffn-supervised-xdma-chain-stop`, iniciada
el 2026-10-06 UTC. Observación funcional revisada y respaldada. Conserva el ejemplo XDMA nativo
y el [ensayo de respuesta única y BAR](XDMA_STOP_INVENTORY.md).

## Hipótesis y estímulo

Observar qué trabajo termina cuando `Run` se borra con la primera
respuesta B retenida y dos descriptores ya entregados por el host. No se
supone que el segundo se descarte ni que `Run=0` vacíe las colas protegidas.

| Descriptor | Host | Fuente de datos | Destino AXI | Longitud | Siguiente |
| --- | --- | --- | --- | --- | --- |
| Primero, no terminal | `0x100` | `0x400` | `0x1000` | 64 B | `0x120` |
| Segundo, terminal | `0x120` | `0x500` | `0x1100` | 64 B | 0 |

`Dsc_Adj=1` solicita el bloque inicial de 64 bytes. Ambos descriptores
tienen `Nxt_adj=0`; solo el segundo tiene Stop. El bloque está alineado
a 32 bytes y no cruza 4 KiB. Los patrones son `(i*13)^0xa5` y
`(i*7)^0x3c`, truncados a un byte. Se comparan independientemente.
[Formato y reglas de descriptores, PG195](https://docs.amd.com/r/en-US/pg195-pcie-dma/Descriptors).

El control documentado permite terminar el descriptor actual al borrar
Run. El experimento distingue esa operación del bit Stop del descriptor,
que termina la lista de fetches. No transforma esa documentación en una
afirmación sobre el instante interno en que avanza el motor.
[Control H2C, PG195](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Control-0x04).

## Secuencia y aceptación

1. Consultar reposo y contador inicial cero; cargar dirección inicial,
   adyacencia y Run mediante peticiones PCIe reales.
2. Retener la primera B en la aplicación pública. Observar Busy y contador
   cero. Borrar Run mediante W1C y verificar dos rondas de consultas de
   control/estado/contador, conservando B pendiente.
3. Liberar B. Esperar 1.024 ciclos y clasificar cuántos descriptores terminan.
   Exigir coincidencia entre AW, W, B y contador nativo, con uno o dos
   descriptores completos; rechazar respuestas erróneas o inventario parcial.
4. Verificar Busy cero, repetir consultas tras otros 1.024 ciclos y exigir
   que no aparezcan más transferencias en el intervalo observado.

La clasificación no es una garantía de cancelación. Si termina uno, siguen
sin observarse directamente las colas internas que recibieron el segundo.
Si terminan ambos, se documentarán los tiempos de fetch, AW/W/B y parada.
Una respuesta de éxito exige reconstruir cada byte de los dos descriptores,
los datos solicitados y las transferencias aceptadas, no solo contar logs.

## Resultado y evidencia

El bloque de **dos descriptores** fue entregado por completo antes de
retener la primera B. El ensayo terminó a **152.660000 µs**;
se completaron **2 descriptores y 128 bytes**. Todos los bytes
y las 24 consultas de registros se cotejaron con VCD. La
primera B se retuvo durante **25.140000 µs** impuestos por el banco.

| Descriptor | Fetch de datos terminado | AW | W | B |
| --- | --- | --- | --- | --- |
| 1 | 98.640000 µs | 98.908000 µs | 98.932000 µs | 124.076000 µs |
| 2 | 98.652000 µs | 98.920000 µs | 98.944000 µs | 124.084000 µs |

La escritura de Run=0 se emitió a **103.012000 µs** y
su lectura de confirmación terminó a **105.460000 µs**.
Al emitirla ya se habían aceptado 2 AW,
2 W y 0 B. Esta tanda
no aísla un segundo descriptor todavía no emitido a AXI.

Se reconstruyeron los fetches y las AW/W/B, incluidos los eventos posteriores
a la escritura de Run=0. El reposo final concuerda con el número observado
de respuestas completas; no demuestra por sí solo ausencia de otros estados
protegidos. No se califica cancelación general ni reutilización de propiedad.

El revisor rechazó **52 alteraciones de evidencia** y
contrastó 139 avisos heredados con sus fuentes compiladas. Respaldo
nativo SHA-256: `81d93ece41f8feb35bf22930f2bbbfa1fd5b55b3f10fc91d3f84bcb9b962a408`. Paquete de revisión:
`b2623550baf403ef229926b633fb132efd9ec5bef47b3e5f07284e2b4341b0bf`. Ambos están verificados localmente y en NFS.
Son tiempos funcionales simulados, sin medición de rendimiento físico.

## Recuperación del respaldo local

El archivo nativo contiene **841 archivos** y ocupa **114.751.140 bytes**.
Su transferencia terminó, pero materializar otra copia completa agotó C:.
Se verificaron nuevamente el archivo completo, cada miembro del manifiesto
y el ZIP de fuentes frente a NFS. La extracción incompleta quedó identificada
como `.partial-extraction`; el `.tar.gz` verificado es el respaldo completo.
El archivo de control del ejecutor se recuperó del archivo sellado y el
comprobador se restauró y volvió a pasar la revisión y los 52 controles negativos.

La compresión NTFS de cachés cerradas conserva sus hashes. Las siguientes
tandas deben usar `seal_recoverable_archive_only.py` y
`resume_sealed_archive_only.py`: verifican el contenido dentro del archivo,
sin crear automáticamente otra copia del árbol extraído. Para esta tanda se
ejecutó y verificó la recuperación específica `resume_sealed_stream.py`.

## Implementación y límites

- Banco: `fpga/tb/ntrpu_xdma_chain_stop.sv`.
- Emisor público acotado: `fpga/tb/ntrpu_xdma_chain_completion.vh`.
- Ejecutor: `tools/ntrpu_board_xdma_chain_stop.tcl`.
- Revisión independiente: `tools/check_ntrpu_xdma_chain_stop.py`.

El emisor responde únicamente a lecturas nativas de 64 bytes en las tres
direcciones previstas. Conserva TC y atributos; no modifica lógica protegida.
Se trazan CQ/CC, consultas RQ/RC y toda la interfaz AXI pública. Los tags de
CQ/CC se cotejan; las consultas de registros son estrictamente seriales y
los puertos de asignación automática de tags RQ no están trazados.

Dos hilos, 1.800 segundos de ejecución y 180 µs simulados. El Completion
Timeout permanece en 50 ms. Sin C2H, reset, reutilización de propiedad,
DDR ni calificación física. Los tiempos de retención son estímulos del banco.
Conservar archivo nativo, diagnóstico o positivo, revisión y hashes localmente
y en NFS antes de avanzar por la [matriz de cierre](XDMA_DRAIN_ACCEPTANCE.md).
