# Presión y reutilización de las cuatro reservas

Fecha: 2026-10-07. Estado: **dos perfiles de preflight aprobados** con CDC y
conversores AMD reales y un sustituto AXI de memoria. El perfil corto también
pasó después con el [modelo HBM nativo de AMD](PEER_RX_HBM_NATIVE_RESERVES.md).
Los resultados tabulados aquí pertenecen a los preflights.

La [prueba nativa anterior](PEER_RX_HBM_NATIVE_TRANSPORT.md) aprobó tráfico
y lectura de comprobación, pero alcanzó solo dos bancos (`0x05`). Esta
extensión conserva sus nueve módulos generados, las trece fuentes RTL de
la referencia asíncrona y las configuraciones de IP calificadas.

## Estímulo y criterio de aceptación

El banco intercala una compuerta en W, en la frontera pública AXI3 de ambos
destinos. Antes de abrirla, mantiene WREADY hacia el transporte en cero y
WVALID hacia la memoria en cero; datos, strobes y WLAST permanecen estables
cuando corresponde. AW y B conservan sus conexiones. No fuerza señales
internas ni modifica el DUT.

Cada perfil ejecuta dos tandas en una sola época de reset frío:

1. Ofrece doce descriptores por destino mientras ambas salidas W están
   bloqueadas. Espera `banks_used=0xff` y el fin del llenado en curso.
2. Mantiene los ocho bancos ocupados durante cien ciclos de fuente
   (aproximadamente 0,5 µs), sin completions prematuras ni sobrescritura.
3. Abre solo el canal cero, verifica una completion y conserva el canal uno
   detenido. Después abre ambos, drena y comprueba todos los datos por lectura.
4. Cierra las compuertas únicamente con el transporte vacío y repite con
   tags nuevos. Todos los bancos deben reutilizarse con generaciones nuevas.

Los bancos se devuelven después del último **W interno** aceptado: las colas
siguientes ya poseen esos datos. No se exige conservar el banco hasta B ni
hasta la completion. El revisor registra por separado asignación, último W,
devolución validada y finalización, y reconstruye la propiedad ciclo a ciclo.

## Resultados

| Medida | Marcos de 512 bytes | Marcos de 4 KiB |
|---|---:|---:|
| Duración de la tanda Vivado | 56,541 s | 58,685 s |
| Descriptores completados | 48 | 48 |
| Ráfagas AXI3 escritas / leídas | 48 / 48 | 384 / 384 |
| Beats de 256 bits escritos / leídos | 768 / 768 | 6.144 / 6.144 |
| Bytes únicos verificados | 24.576 | 196.608 |
| Máximo de bancos ocupados | 8 | 8 |
| Ventanas de saturación | 2 | 2 |
| Reservas utilizadas | 4, 5, 6, 7 | 4, 5, 6, 7 |
| Reutilizaciones por banco, incluidas asignaciones iniciales | 14, 4, 12, 6, 4, 2, 4, 2 | 14, 4, 14, 6, 4, 2, 2, 2 |
| Pausas de lectura comprobadas | 424 | 3.470 |

Cada perfil pasa 27 controles negativos sobre trazas: retirar eventos o
corromper datos/generaciones debe provocar rechazo. Son controles del
revisor, no inyecciones de fallos en RTL. El revisor interpreta `TE S` como
estado anterior a la actualización del flanco para compararlo correctamente
con las concesiones y devoluciones de ese mismo ciclo.

El perfil largo ejercita las 64 palabras de 512 bits de los bancos de 4 KiB.
La máscara llena acredita ocho reservas de almacenamiento concurrentes;
no implica que todo su payload siga dentro de la RAM, porque una parte
puede haber pasado a las colas de salida. Ambos perfiles terminan sin
bancos, créditos o descriptores pendientes y con `drained=1`.

## Reproducción y evidencia

Herramientas: [generador](../../tools/prepare_ntrpu_hbm_reserve.py),
[estímulo](../../fpga/tb/ntrpu_hbm_reserve_body.svh),
[runner](../../tools/run_ntrpu_hbm_reserve.py) y
[revisor](../../tools/check_ntrpu_hbm_reserve.py). El runner requiere Linux,
Vivado 2026.1, la referencia asíncrona exportada y `fixture/native_hbm.veo`
del enlace nativo aprobado. Genera su Tcl en un directorio nuevo y conserva
el Tcl y las fuentes originales. Usa dos CPU, prioridad baja y límite de 900 s.

```bash
python3 -B tools/run_ntrpu_hbm_reserve.py /tanda/output \
  --reference /referencia-async/output --beats 64
python3 -B tools/check_ntrpu_hbm_reserve.py /tanda/output > /tanda/REVIEW.json
```

`--beats 8` selecciona el perfil corto. Las primeras ejecuciones quedan
congeladas con sus versiones de fuentes, hashes y revisor correspondientes.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-hbm-reserve\20261007T033021827486Z`.
Tandas: `20261007T033208335055Z-preflight` y
`20261007T033647813786Z-preflight`. Cada una tiene revisión independiente,
transferencia verificada y respaldo incremental SHA-256/gzip en
`D:\AI\artifacts\ntrpu-content-store`; no se retienen ZIP de transferencia.

## Alcance y próximos pasos

Esto acredita presión, integridad, propiedad y reciclado con un destino
AXI controlado. No mide rendimiento HBM, ancho de banda PCIe, P2P, cierre
temporal ni tolerancia térmica; la lectura de comprobación sigue siendo
local al banco de pruebas.

El [perfil corto nativo](PEER_RX_HBM_NATIVE_RESERVES.md) ya tiene evidencia
independiente, con idéntico banco, lógica y configuraciones. Su lectura
completa necesitó una ventana mayor que la inicialmente prevista. El perfil
largo nativo sigue pendiente y exige planificar más tiempo después del
arranque; su PASS de preflight no se traslada al controlador nativo.

La [propuesta de entradas independientes](PEER_RX_MULTI_INGRESS_PROPOSAL.md)
define el siguiente cambio arquitectónico. Esta tanda no altera la cola
Vivado ni el diseño de placa del agente principal.
