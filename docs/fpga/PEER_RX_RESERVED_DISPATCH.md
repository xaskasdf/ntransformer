# R4 con reserva atómica y colas por destino

Estado: **integración funcional validada con payload; timing pendiente**.
El experimento posterior de [cabeceras registradas](PEER_RX_REGISTERED_DISPATCH.md)
conservó el protocolo y mejoró la síntesis, pero su ruteo quedó incompleto
por el límite de ejecución; todavía no cualifica esta composición a 200 MHz.
La composición aislada `fpga/rtl/ntrpu_reserved_dispatch.sv` conecta las
[dos colas de entrada, una por destino](PEER_RX_DESTINATION_DISPATCH.md), con el receptor R4
y su salida AXI elástica. La prueba usa un productor y un destino AXI
de comportamiento, con un solo reloj. No incorpora DMA real, MIG, CDC,
PCIe/P2P ni HBM. Se conserva el receptor de referencia.

## Reserva y recorrido de los datos

Cada destino tiene ocho descriptores de entrada. R4 mantiene dos bancos
dedicados por destino y cuatro compartidos: ocho bancos de 4 KiB,
**32 KiB lógicos**, además de 16 descriptores por destino y hasta cuatro
escrituras pendientes de respuesta por destino. El almacenamiento de
payload conserva la implementación replicada por lector de R4.

`tools/prepare_ntrpu_reserved_receiver.py` genera copias del receptor
que exponen `reserve_can`, con manifiesto SHA-256 y diferencias exactas.
La oferta exige banco elegible libre, descriptor disponible y ausencia
de llenado/fallo. El descriptor permanece ocupado hasta consumir su
completion, por lo que también reserva capacidad para el resultado.

La toma de la oferta ejecuta **el handshake real de comando de R4** y
asigna banco y descriptor en ese flanco. La fase posterior de comando
del despachador comprueba la concesión capturada; no vuelve a asignar.
Solo entonces se permite el fetch. El adaptador cuenta los beats de
512 bits realmente aceptados por R4 y confirma la trama tras LEN+1.

La oferta es conservadora: una devolución en ese mismo flanco no basta
por sí sola para habilitarla. Si ya existe otro banco libre, el asignador
puede preferir el banco que se devuelve; la prueba verifica dos de estas
reutilizaciones mediante propietario y generación. El banco se libera
cuando el receptor entrega su último beat a la salida elástica, que puede
conservarlo mientras el AXI externo espera.

Se comprueban alineación de 64 bytes, máximo de 64 beats y límite de
página de 4 KiB. La autorización de ventanas y la validación de época
siguen aguas arriba. La dirección de origen es un metadato: este banco
produce un patrón verificable y no lee memoria mediante DMA.

## Stop, errores y barrera

`stop` impide publicar nuevos trabajos y permite drenar los ya encolados.
Un fallo bloquea reservas nuevas y conserva los trabajos pendientes en
cuarentena. Una concesión ya asignada mantiene comando, fetch y llenado
incluso si el fallo ocurre antes del primer dato. La barrera espera ambas
colas, el payload, las respuestas AXI y el consumo de las completions.

Los resets del banco separan escenarios independientes. No acreditan
recuperación en caliente ni permiten descartar tráfico de una época activa.

## Validación

La simulación final `20261005T055109915312Z-reserved` pasó ocho escenarios:

| Escenario | Evidencia |
| --- | --- |
| Tráfico alternado de 64 B, 512 B y 4 KiB | 24 trabajos completos |
| W0 detenido | Seis bancos comprometidos; 12 trabajos del destino 1 terminan antes de liberar W0 |
| Completions de destino 0 retenidas | 16 descriptores ocupados y bancos libres; ocho trabajos del destino 1 terminan |
| Barrera con B0 y completion 1 detenidas | Espera ambas condiciones y conserva su resultado |
| Fallo externo después de reservar | La trama comprometida termina; la siguiente queda en cuarentena |
| BRESP con error y otra reserva activa | Propaga SLVERR, termina la otra trama y conserva la pendiente |
| Stop antes del primer beat | Mantiene el banco y después drena el trabajo aceptado |
| AW0 detenido | Acepta W antes de AW sin completar prematuramente |

El revisor independiente reconstruye **4.719 ciclos, 87 publicaciones,
85 reservas/devoluciones y 85 completions**. Contrasta 2.171 beats de
entrada y salida, **138.944 bytes**, incluyendo datos, orden, direcciones,
longitudes y generaciones. Hay 84 completions OKAY y una SLVERR inyectada;
los tags 301 y 402 quedan expresamente en cuarentena. La comprobación
de bytes ocurre en el AXI simulado y no demuestra persistencia en DDR.
Ocho controles negativos rechazan datos corruptos, reservas/confirmaciones
ausentes, conteos incorrectos, falsa disponibilidad y devoluciones inválidas.

## Síntesis y trabajo pendiente

La tanda `20261005T054038651289Z-reserved` sintetizó la misma lógica en
`xcvu5p-flva2104-2-i`, con reloj de 5 ns aplicado antes de síntesis:

| Recurso | Uso |
| --- | ---: |
| LUT totales / RAM distribuida incluida | 6.131 / 272 |
| Registros / CARRY8 | 12.548 / 249 |
| BRAM equivalentes de 36 Kib | 15 (14 RAMB36 y dos RAMB18) |
| URAM / DSP / latches / cajas negras | 0 |

**No cumple todavía el objetivo de 200 MHz:** WNS −0,810 ns, TNS
−2.048,179 ns y 5.652 extremos con incumplimiento de setup. Hold estimado
+0,051 ns. El camino crítico tiene 22 niveles desde la lectura de la
cabecera hasta la habilitación del descriptor receptor, atravesando
validación de página, selección de destino y admisión de R4.

Faltan placement/ruteo, fuente física de reloj y retardos de 1.069 entradas
y 1.641 salidas. Estas cifras son del conjunto sintetizado, no el coste
incremental respecto a una referencia ruteada.

El siguiente experimento debe registrar cabeceras y su validación para
acortar ese camino, manteniendo la reserva efectiva en el handshake real
de R4. No basta registrar disponibilidad: podría quedar obsoleta. Repetir
los casos de pausa/fallo y exigir setup/hold internos no negativos tras
ruteo antes de integrar CDC, capacity guards, supervisor de época y MIG.
Después comparar contra R4 ordenada con idénticos payloads y pausas;
esta tanda no mide ganancia de ancho de banda.

## Reproducción y respaldo

```bash
source /ruta/Vivado/settings64.sh
cd /snapshot/inputs
python3 -B tools/run_ntrpu_reserved_dispatch.py /snapshot/output --timeout 600
# Para repetir solo simulación: otra salida nueva y --simulation-only
cd /ruta/ntransformer
python3 -B tools/check_ntrpu_reserved_dispatch.py /snapshot
```

El runner y el revisor requieren salidas nuevas. Evidencia local:
`D:\AI\artifacts\ntrpu-reserved-dispatch\20261005T053346302256Z`.
Incluye tres tandas, snapshots, trazas, checkpoint, informes y respaldo
`20261005-reserved-dispatch-handoff.zip` con SHA-256, CRC y hashes por archivo.

La primera simulación omitió separadores en tres campos de traza. Las dos
repeticiones corrigieron el formato y añadieron devoluciones explícitas
para auditar la reutilización en un mismo flanco. No cambiaron la lógica
RTL; `PROVENANCE_REVIEW.json` verifica su equivalencia y la identidad de
todas las trazas comunes. El comentario de la oferta se precisó.

Se usaron dos CPU, prioridad baja, durante 334,421 s de simulación/síntesis
y 35,465 / 35,416 s de repetición. Las salidas están descargadas y verificadas;
no quedan procesos de esta prueba en la VM. No se modificó la cola principal.
