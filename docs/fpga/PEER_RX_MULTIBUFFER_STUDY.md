# Estudio de planificación del pool de buffers

Tanda local del 2026-10-04: **199 escenarios completados** en 16,60 s,
con conservación de marcadores de datos y drenaje de todas las transacciones.
El modelo tiene además **13 pruebas unitarias aprobadas**, incluyendo 12
semillas de tráfico y pausas. Es un estudio de planificación, **no una
simulación RTL ni una medición de PCIe, HBM o frecuencia alcanzada**.

## Supuestos reproducibles

- Dos destinos; una entrada de 64 bytes/ciclo, no interrumpible dentro de
  una transferencia. Cada destino tiene una secuencia W ordenada.
- Dos bancos propios por destino, más reserva `S=0/4/8/16/32`. Cada banco
  aloja una transferencia de hasta 4096 bytes, sin empaquetar las pequeñas.
- Arbitraje ideal, sin coste de asignación, pipeline RAM, CDC ni cabeceras.
  Un banco liberado puede reutilizarse ese mismo ciclo.
- Referencia de 200 MHz para convertir ciclos a GB/s; no valida ese reloj.
- AW se acepta antes o junto al primer W; como máximo un B por destino y
  ciclo. El retraso B de referencia es 32 ciclos desde WLAST. AW, W, B y
  el consumidor de resultados pueden pausarse independientemente.
- Se reservan descriptor y resultado antes del payload. Normalmente el
  banco se devuelve tras WLAST; el descriptor espera B y el resultado
  espera consumo. También se compara retener el banco hasta B.
- Cada caso ofrece 256 transferencias desde el inicio. La entrada
  `eligible` elige destino antes de comenzar; `ordered` no puede saltar
  una cabecera bloqueada. No se presupone que el PCIe real permita elegir.

Las transferencias usan marcadores únicos por beat para detectar pérdida,
reordenamiento y reutilización incorrecta. No se simulan bytes de pesos,
direcciones, TLP, ventanas BAR, errores, épocas, stop ni reset.

## Efecto de la reserva

Esta comparación usa tráfico 50/50, dos salidas independientes y ambas
pausadas durante los primeros 96 ciclos de cada periodo de 256. El tráfico
mixto elige longitudes de 256/512/1024/4096 bytes con semilla `20261004`.
Cada fila añade reserva a los cuatro bancos propios. El caudal incluye
llenado inicial y drenaje final: no es una estimación de régimen permanente.

| Reserva | Capacidad total | Mixto, 4 pendientes/puerto | 256 B, 4 pendientes/puerto | 256 B, 8 pendientes/puerto |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16 KiB | 9,613 GB/s | 7,511 GB/s | 7,844 GB/s |
| 4 | 32 KiB | 11,861 GB/s | 7,634 GB/s | 8,838 GB/s |
| 8 | 48 KiB | 12,398 GB/s | 7,634 GB/s | 9,230 GB/s |
| 16 | 80 KiB | 12,465 GB/s | 7,634 GB/s | 10,886 GB/s |
| 32 | 144 KiB | 12,465 GB/s | 7,634 GB/s | 11,329 GB/s |

En el caso mixto, ocho bancos de reserva dejan solo 45 ciclos de bloqueo de
entrada; dieciséis eliminan esos bloqueos. Pasar de ocho a dieciséis mejora
el caudal un 0,55%; el máximo ocupado de reserva es once. Treinta y dos no
mejoran ese caso. Son resultados de esta secuencia, no una cota universal.

Con transferencias de 4096 bytes, esta misma pausa ya queda cubierta por
los bancos propios: 12,651 GB/s para todas las reservas, sin usar ninguna.
Con transferencias de 256 bytes, cada banco usa solo el 6,25% de su capacidad.
Más bancos absorben más transferencias, pero la respuesta B y el límite de
pendientes pueden seguir determinando el caudal. La reserva también puede
adelantar el fin de la entrada sin cambiar el tiempo final de completado.

## Otros límites observados

- **Interconexión:** con 32 bancos de reserva y transferencias de 4 KiB,
  reducir el servicio agregado de dos lecturas a una baja de 12,651 a
  7,972 GB/s bajo la pausa periódica. El pool compartido con una sola
  lectura reproduce el mismo límite; capacidad no equivale a puertos.
- **Cabecera bloqueada:** con 256 B, reserva cuatro y el destino cero
  detenido 1024 ciclos, la entrada ordenada obtiene 4,067 GB/s frente a
  6,074 GB/s al poder elegir destino. Cambiar esta política requiere
  capacidad real de planificación aguas arriba.
- **Vida del payload:** con dos bancos del destino activo, B demorado
  128 ciclos y 32 escrituras pendientes, liberar tras WLAST obtiene
  11,127 GB/s; retener hasta B obtiene 0,758 GB/s. Ambos respetan orden
  y créditos en el modelo. No autoriza liberar antes de aceptar los datos.
- **Aislamiento:** las pruebas dejan un destino detenido durante todo su
  presupuesto. Con entrada elegible, el otro completa su trabajo; el
  resultado global permanece incompleto y no se declara drenaje exitoso.
- **Cuotas y resultados:** cuotas de préstamo limitan la reserva retenida
  por un destino. Pausar el consumidor cierra admisión al agotar sus
  créditos; al reanudarlo se completa el drenaje sin perder respuestas.

El barrido también compara bancos totalmente compartidos a igual
capacidad, reparto 90/10 y 100/0, tamaños mezclados y respuestas lentas.
Las políticas de préstamo probadas son cuota fija y préstamo sin cuota;
no se ha diseñado todavía una cuota dinámica con garantía de espera máxima.

## Decisión provisional y trabajo pendiente

Conservar `2*C+S` parametrizable. Para el siguiente prototipo aislado,
comparar reserva cuatro y ocho; dieciséis sirve para contrastar cobertura
de pausas. Las cargas de transferencias pequeñas justifican explorar más
descriptores y sub-bloques de almacenamiento antes de fijar una reserva
grande de bancos de 4 KiB. Subdividir cambia el control y requiere estudio.

Falta comprobar puertos RAM reales, latencia de lectura, arbitraje,
préstamos simultáneos, fallos/drenaje y coste LUT/BRAM mediante RTL y
síntesis. También falta integrar créditos PCIe y medir trazas reales de
pausas. Esta tanda no satisface los criterios completos de integración de
la [propuesta](PEER_RX_MULTIBUFFER.md), ni modifica la cola Vivado principal.

## Reproducción y evidencia

Desde la raíz del repositorio, con Python 3.10+ y sin dependencias externas:

```powershell
python -B -m unittest discover -s tests -p test_ntrpu_buffer_pool_model.py -v
python -B tools/run_ntrpu_buffer_pool_study.py artifacts/ntrpu-buffer-pool/nueva-tanda
```

La carpeta de salida debe ser nueva. No ejecutar con `-O`: el modelo
requiere sus invariantes. Las pruebas tardaron 0,405 s en esta ejecución.

- [Modelo](../../tools/ntrpu_buffer_pool_model.py),
  [pruebas](../../tests/test_ntrpu_buffer_pool_model.py) y
  [ejecutor](../../tools/run_ntrpu_buffer_pool_study.py).
- [Resultados de los 199 casos](../../artifacts/ntrpu-buffer-pool/20261004T042329767Z/RESULTS.json):
  configuración, pausas, semilla, métricas, contadores y hashes de entradas.
- La misma carpeta conserva entradas exactas y 21 trazas de transacciones
  representativas; no son trazas de señales RTL.
- [Respaldo local](../../artifacts/ntrpu-buffer-pool/20261004T042329767Z-handoff.zip):
  entradas, resultados, trazas y documentación actualizada, con manifiesto
  de hashes y comprobación de integridad del ZIP.

El código usa una sola tarea local. No se accedió a la VM, no se ejecutó
Vivado y no se modificaron RTL, CMake ni trabajos del principal.
