# Dos colas por destino: prototipo de control

Tanda validada: `20261005T050909966057Z-dispatch`, Vivado 2026.1.
El módulo aislado `fpga/rtl/ntrpu_destination_dispatch.sv` implementa
**dos colas en total, una por destino**, con ocho descriptores cada una.
Pasó 24 casos de protocolo y síntesis OOC. Esa tanda se limitó al control.
La integración posterior con R4 y payload se documenta en
[Reserva atómica R4](PEER_RX_RESERVED_DISPATCH.md); DMA y MIG siguen pendientes.
El RTL previamente validado se conserva.

## Qué implementa

Ambos productores pueden publicar simultáneamente. Las cabeceras compiten
mediante arbitraje circular entre destinos con recursos disponibles.
Una cola llena o un destino sin reserva no impiden servir la otra.
Se conserva FIFO dentro de cada destino; los trabajos entre destinos
deben haber sido declarados independientes por el productor.

Cada descriptor ocupa 232 bits, empaquetados de mayor a menor:

| Campo | Bits | Función |
| --- | --- | --- |
| Época | 231:200 | Identidad opaca; se contrasta al confirmar la trama |
| Tag | 199:136 | Identidad del trabajo |
| Destino | 135:72 | Dirección de escritura |
| Origen | 71:8 | Referencia al payload retenido aguas arriba |
| LEN | 7:0 | Beats de 512 bits menos uno; prevalidado entre 0 y 63 |

Las colas almacenan referencias, no payloads. Sus 16 entradas suman
3.712 bits, **464 bytes lógicos**, más el descriptor seleccionado,
la barrera y el control. No se aumenta la capacidad del pool R4.

## Reserva y secuencia de transferencia

`reserve_can[1:0]` significa que el receptor ofrece de forma atómica
banco, descriptor y espacio de resultado para la cabecera correspondiente.
`reserve_take` selecciona como máximo uno en ese flanco. El receptor debe
retener esa concesión hasta consumir el comando; una señal informativa
de disponibilidad no satisface este contrato.

La secuencia es reserva → comando aceptado → solicitud de fetch aceptada
→ confirmación de trama. El descriptor y el destino quedan registrados;
no cambian durante esperas de comando o fetch. No se selecciona otra
trama hasta terminar la actual. Esta serialización introduce ciclos de
control cuyo coste debe medirse al integrar el datapath.

El productor de payload informa `frame_done` únicamente después de que
el receptor acepte los LEN+1 beats. Tag, época y cantidad deben coincidir.
Una confirmación incorrecta activa fallo y conserva la concesión; el
banco de pruebas verifica que una confirmación posterior correcta pueda
cerrar ese compromiso. Esta interfaz confía en el contador del productor:
el prototipo no observa ni comprueba los bytes transferidos.

## Stop, fallo y barrera

`stop` cierra nuevas publicaciones y drena los descriptores ya aceptados.
Un fallo enclavado impide nuevas reservas y pone las colas pendientes en
cuarentena. El comando/fetch ya comprometido se mantiene y puede terminar;
el receptor que ofrece la concesión debe respetarla incluso tras el fallo.

Una petición de barrera tiene prioridad sobre publicaciones simultáneas.
Cierra ambas entradas, deja avanzar el trabajo anterior y espera colas
vacías, ninguna concesión activa y `sink_drained`. Después conserva el
resultado y su identificador hasta que el consumidor los acepte. La
barrera ordena lo ya admitido; el productor debe publicarla después de
haber entregado todos los trabajos que quiera incluir.

`drained` incluye las dos colas, la concesión, el receptor y la barrera.
Reset es síncrono y requiere una época coordinada ya aislada. Los resets
del banco separan escenarios independientes; no validan recuperación
en caliente ni autorizan descartar transporte pendiente. Validación de
ventanas, direcciones y época al publicar sigue siendo responsabilidad
de la capa anterior.

## Evidencia y coste

Se probaron profundidades 1, 3 y 8; cada una ejecuta ocho casos:
progreso con la otra cola llena, arbitraje/envoltura de índices,
esperas con stop, barrera, tres confirmaciones incorrectas y fallo externo
con comando expuesto. Se revisaron **222 publicaciones y 210 secuencias
completas de control**; 12 trabajos quedaron expresamente en cuarentena
en los escenarios de fallo. No son transferencias de payload a DDR.

El revisor reconstruye colas, reservas, metadatos, fases y barreras de
1.140 muestras de ciclo. Rechaza 15 corrupciones deliberadas de trazas:
comando inestable, reserva duplicada, fetch prematuro, falso drenaje y
barrera anticipada, en las tres profundidades.

Síntesis de profundidad ocho en `xcvu5p-flva2104-2-i`:

| Recurso | Uso |
| --- | ---: |
| LUT totales | 1.061 |
| LUT de lógica / RAM distribuida | 789 / 272 |
| Registros | 323 |
| CARRY8 | 5 |
| BRAM / URAM / DSP | 0 / 0 / 0 |

Sin latches ni cajas negras. El informe interno estima setup +3,201 ns
y hold +0,051 ns con periodo de 5 ns, añadido después de síntesis.
**No es cierre de timing:** faltan placement/ruteo, fuente física de reloj
y retardos de 646 entradas/1.016 salidas. Este coste aislado no equivale
al incremento final dentro del receptor.

Se preservaron dos intentos anteriores: uno encontró una espera fija
insuficiente en el banco; otro pasó el protocolo pero falló al inferir
un array con dos escritores. La versión final usa un array independiente
por cola y reproduce exactamente todas las trazas del segundo intento.
`SYNTHESIS_REVIEW.json` documenta además las advertencias de entorno,
genérico aplicado a `glbl`, consultas vacías y timing OOC.

## Integración pendiente y reproducción

El **adaptador de reserva atómica para R4** ya pasó una prueba aislada con
contador real de beats y barrera de completions; véase
[evidencia y limitaciones](PEER_RX_RESERVED_DISPATCH.md). Su oferta utiliza
copias generadas del receptor y consume conjuntamente banco y metadatos
mediante el handshake real de comando. Aún falta resolver el timing de
la composición e incorporar el supervisor de época y el transporte real.

Comparar esa composición contra R4 ordenada con las mismas pausas y
payloads, y verificar bytes, drenaje, errores, CDC y timing ruteado antes
de sustituir la referencia. Este banco no acredita PCIe/P2P ni HBM.

```bash
source /ruta/Vivado/settings64.sh
cd /snapshot/inputs
python3 -B tools/run_ntrpu_destination_dispatch.py /snapshot/output --timeout 300
cd /ruta/ntransformer
python3 -B tools/check_ntrpu_destination_dispatch.py /snapshot
```

El snapshot debe contener las fuentes en `inputs/`; el runner y el
revisor exigen salidas nuevas. Evidencia y respaldo local verificado:
`D:\AI\artifacts\ntrpu-destination-queues\20261005T045730084371Z`.
Incluye los tres intentos, scripts, checkpoint, informes y manifiestos.
La tanda final tardó 111,701 s con dos CPU y prioridad baja; no quedan
procesos de esta prueba en la VM.
