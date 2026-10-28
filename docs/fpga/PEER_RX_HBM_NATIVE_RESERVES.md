# Reservas bajo presión con el modelo HBM nativo

Fecha: 2026-10-07. Estado: **`NATIVE_HBM_RESERVE_PRESSURE_PASS`**.
El perfil corto de [presión sobre reservas](PEER_RX_HBM_RESERVE_PRESSURE.md)
pasó con el controlador HBM nativo de AMD, conservando exactamente el banco
de pruebas aprobado. Esta composición mantiene una entrada compartida de
payload; el [prototipo de doble entrada](PEER_RX_DUAL_INGRESS_PROTOTYPE.md)
se valida por separado y todavía no participa en esta prueba nativa.

## Composición y estímulo

Se conservan las nueve piezas de lógica generada de la referencia asíncrona,
sus trece fuentes RTL calificadas, las configuraciones de CDC, registro y
conversión AXI, y el enlace HBM de dos stacks/16 GB. Fuente a 200 MHz,
memoria a 450 MHz, destinos AXI00/AXI16 y lectura de comprobación local.

La compuerta de prueba bloquea W en la frontera pública de cada destino.
Cada ronda reserva los ocho bancos, termina el llenado en curso y conserva
la ocupación durante cien ciclos de fuente. Después abre primero el canal
cero, comprueba progreso mientras el otro permanece detenido y libera el
canal uno. Tras completar escrituras y lecturas, repite sin reset, con
tags nuevos. No fuerza inicialización ni señales internas del controlador.

## Resultado revisado

| Medida | Resultado |
|---|---:|
| Rondas / descriptores completados | 2 / 48 |
| Ráfagas AXI3 escritas / leídas | 48 / 48 |
| Beats de 256 bits escritos / leídos | 768 / 768 |
| Bytes únicos comprobados | 24.576 |
| Pico de bancos / máscara utilizada | 8 / `0xff` |
| Asignaciones de bancos 0–7 | 14, 4, 10, 6, 6, 2, 4, 2 |
| Pausas R con estabilidad comprobada | 384 |
| Muestras anteriores al arranque con admisión cerrada | 8.734 |
| Tiempo simulado final | 58,0002 µs |

Todos los bancos se reutilizaron con generaciones nuevas. El replay
independiente reconstruye asignación, propiedad, último W interno,
devolución validada y los bytes escritos/leídos, además del orden de B y
completion. Dieciocho controles negativos sobre las trazas detectan eventos
ausentes y datos/generaciones corruptos; no son inyecciones de fallos en HBM.

Las sesenta muestras de progreso terminan con inicialización pública e
interna de ambos stacks en uno, `drained=1` y fault, créditos, bancos,
colas y pendientes en cero. Ambos `CATTRIP` siguen en X; se contabilizaron
4.325 ciclos de memoria con estado térmico desconocido. Temperatura,
paridad y comportamiento de hardware siguen sin calificar.

## Ejecución y diagnóstico conservado

La compilación de una copia privada de la caché revisada terminó en
842,396 s. Se verificaron el banco congelado del preflight, configuraciones,
fuentes, entradas del modelo y ausencia de cambios en el donante.

La primera ejecución terminó a 56 µs por la guarda Tcl de 850 s: había
completado todas las escrituras y completions, pero solo 448 de los 768
beats de lectura. Aunque XSIM devolvió cero y el transporte estaba vacío,
el revisor rechazó la tanda por faltar la comprobación completa.

La repetición cambió únicamente la ventana de ejecución: límite exterior
de 1.080 s y guarda Tcl de 1.050 s. Usó el mismo snapshot y estímulo,
terminó en **925,122 s** y pasó la revisión completa. La tanda anterior
permanece como diagnóstico rechazado. No se recompiló ni se modificó RTL
para ampliar esa ventana.

## Reproducción y evidencia

[Compilador aislado](../../tools/run_ntrpu_hbm_reserve_native.py),
[runner de ventana ampliada](../../tools/run_ntrpu_hbm_reserve_window.py) y
[revisor](../../tools/check_ntrpu_hbm_reserve_native.py):

```bash
python3 -B tools/run_ntrpu_hbm_reserve_native.py /compile/output \
  --stage compile --cache /cache-nativa/output \
  --cache-review /cache-nativa/REVIEW.json \
  --preflight /reservas-cortas/output --preflight-review /reservas-cortas/REVIEW.json
python3 -B tools/check_ntrpu_hbm_reserve_native.py /compile/output > /compile/REVIEW.json
python3 -B tools/run_ntrpu_hbm_reserve_window.py /traffic/output \
  --prior /compile/output --review /compile/REVIEW.json
python3 -B tools/check_ntrpu_hbm_reserve_native.py /traffic/output > /traffic/REVIEW.json
```

Las salidas deben ser nuevas. Cada tanda usa dos CPU y prioridad baja.
Las quince entradas del snapshot y su copia de ejecución se verifican
antes y después. Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-reserve-dual\20261007T035149419987Z`.
La ejecución aprobada es `20261007T042528687713Z-traffic`; su predecesora
incompleta es `20261007T040904950212Z-traffic`. Hay respaldos incrementales
verificados en `D:\AI\artifacts\ntrpu-content-store`, sin ZIP retenidos.

## Límites

La prueba nativa usa marcos de 512 bytes; los bancos completos de 4 KiB
se ejercitaron en los preflights, no en esta ejecución HBM. La lectura
sigue siendo del banco de pruebas, sin retorno CDC/PCIe ni inferencia.
Esto no acredita ancho de banda de tarjeta, coherencia entre NTRPU,
capacidad de 512 GB, diseño de placa ni integración nativa de doble entrada.
