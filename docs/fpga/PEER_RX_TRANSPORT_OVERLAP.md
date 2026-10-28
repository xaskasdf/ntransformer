# Solapamiento entre bloques del transporte NTRPU

## Resultado y alcance

Tandas del 6 de octubre de 2026 UTC, posteriores al
[estudio del transporte](PEER_RX_TRANSPORT_STUDY.md). Dos variantes aisladas
pasan simulación RTL y revisión independiente. La referencia que cerró timing
interno a 200 MHz conserva sus fuentes y sus resultados.

En tráfico continuo, la separación entre fetches pasa de **N+7** ciclos a
**N+6** con preparación anticipada (`ahead`) y a **N+4** con la combinación
(`compact`), donde N es el número de beats de 64 bytes del bloque.
El pool sigue teniendo ocho bancos de 4 KiB: dos dedicados por destino y
cuatro compartidos. No se amplían las colas ni se habilitan dos llenados activos.

## Cambios evaluados

1. **Preparar metadatos durante el bloque activo.** El candidato puede existir
   sin crédito. Si hay destinos elegibles se priorizan; en su ausencia se
   prepara una cabecera disponible. La reserva sigue exigiendo IDLE, ausencia
   de fallo y oferta vigente del destino en el instante de aceptación.
2. **Confirmar la reserva al emitir el fetch.** La comprobación del eco y el
   fetch pueden aceptarse en el mismo ciclo. El eco reconoce la reserva ya
   capturada; nunca emite un segundo comando al receptor. Si hay contrapresión,
   se recuerda la confirmación y se conserva el fetch.
3. **Cerrar el llenado al aceptar su último beat.** La finalización usa la
   aceptación efectiva por el motor interno, no la entrada a la cola externa.
   Se libera la reserva y se vuelve a IDLE en ese flanco. Una nueva reserva
   puede ocurrir desde el ciclo siguiente, nunca simultáneamente con ese beat.

`ahead` aplica solamente el primer cambio; `compact` aplica los tres.
Stop drena trabajo aceptado; fault deja en cuarentena trabajo sin reserva.
La barrera sigue esperando las finalizaciones consumidas por el exterior.

## Rendimiento observado

Cada fila incluye 128 transferencias, 64 por destino. El caudal abarca desde
la primera publicación hasta la última finalización, incluidos ambos ciclos.
GB/s es una conversión de ciclos de 5 ns, **no una medición de placa**.

| Bloque | Ciclos referencia / ahead / compact | GB/s referencia → compact | Ganancia compact |
| --- | ---: | ---: | ---: |
| 64 B | 1.039 / 912 / 657 | 1,5769 → 2,4938 | 58,14 % |
| 512 B | 1.942 / 1.815 / 1.560 | 6,7493 → 8,4021 | 24,49 % |
| 4 KiB | 9.166 / 9.039 / 8.784 | 11,4398 → 11,9373 | 4,35 % |

Reserva → fetch externo baja de tres a dos ciclos en `compact` sin pausas.
Ambas variantes alcanzan un máximo de dos bancos ocupados en ese perfil.
Con W aceptado cada dos ciclos por destino, `compact` alcanza tres bancos
para bloques de 512 B y 4 KiB, sin aumentar la capacidad física del pool.

No mejora el perfil que acepta fetch solamente cada ocho ciclos: esa pausa
oculta la reducción. Con entrada de datos cada dos ciclos, `compact` mejora
32,65 / 9,91 / 1,51 % para los tres tamaños. Con destino 0 bloqueado, el
destino 1 completa sus 64 trabajos antes de liberar el bloqueo; el pool
alcanza ocho bancos. No se deduce de ello ancho de banda PCIe o HBM.

## Validación

Por variante pasan:

- Los 13 casos originales: 91 publicaciones, 88 trabajos completos,
  tres cuarentenas y 127.040 bytes comprobados.
- Cinco casos de crédito: preparación sin oferta, retirada del destino
  elegido, stop, fault, retirada durante FRAME y fetch retenido sin repetir
  su confirmación.
- Tres casos integrados adicionales: fallo con candidato preparado durante
  el llenado, stop y barrera en esa ventana, y fallo exactamente en el último
  beat interno. Completan cuatro trabajos y dejan dos en cuarentena.
- Quince perfiles de rendimiento: 1.920 trabajos y 2.990.080 bytes.
  `compact` observa 1.920 confirmaciones simultáneas con fetch y 1.920 cierres
  en el último beat. Ninguna reserva coincide con otra reserva activa.

Los revisores reproducen datos, identidades, orden y causalidad a partir de
trazas, y rechazan 23 alteraciones por variante. Los hashes de las fuentes
de referencia coinciden con la tanda previa utilizada para la comparación.

## Reproducción y respaldo

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_transport_overlap.py /tanda/output --variant compact
python3 -B tools/check_ntrpu_transport_overlap.py /tanda/output \
  --review /tanda/REVIEW.json \
  --baseline /tanda_base/output/suite/simulation.log
# Repetir con --variant ahead y un directorio de salida nuevo.
```

El revisor necesita también `output/result.json` de la tanda base.
El generador produce copias y diferencias legibles; no modifica RTL fuente.
Cada ejecución usa dos CPU, prioridad baja y límite de 900 s.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-transport-overlap\20261006T010843420367Z`.
Tandas `20261006T011439219793Z-ahead` (100,516 s) y
`20261006T011649068046Z-compact` (97,374 s).
Manifiestos verificados en `D:\AI\artifacts\ntrpu-content-store`:
`transport-overlap-ahead-20261006`, `transport-overlap-compact-20261006` y
`transport-overlap-handoff-20261006`. El almacén reutiliza objetos por hash;
las transferencias ZIP no se conservan como archivos.

## Siguiente criterio de aceptación

La [evaluación física posterior](PEER_RX_OVERLAP_PHYSICAL.md) completa
síntesis, placement, ruteo y comparación funcional de `compact`: los
16 escenarios de netlist pasan, pero el setup del ruteo original queda en
−0,323 ns a 5 ns. Dos [pasadas de optimización física](PEER_RX_OVERLAP_PHYSOPT.md)
lo mejoran a −0,244 ns y vuelven a pasar los 16 escenarios, con los mismos
recursos. Esas tandas todavía no acreditaban 200 MHz. La referencia N+7
mantiene su cierre independiente.

El [ensayo de distribución física](PEER_RX_OVERLAP_FLOORPLAN.md) posterior
termina la colocación, pero agota 900 s de ruteo con congestión residual;
no produce un nuevo resultado físico o funcional acreditado; en esa
tanda se conservó la variante anterior de −0,244 ns.

La [distribución posterior por conectividad de bancos](PEER_RX_OVERLAP_BANKPLAN.md)
sí cierra `compact` N+4 a 200 MHz: setup +0,054 ns, hold +0,006 ns y
los 16 escenarios funcionales de netlist pasan. Usa 6.789 LUT, 16.022
registros y 15 bloques BRAM. Se conserva el contrato funcional sin
modificar el RTL fuente. Los caudales de la tabla siguen siendo
conversiones de simulación; ahora hay cierre interno del bloque a ese
reloj, pero todavía no son mediciones de una tarjeta PCIe/HBM.

El próximo criterio es integrar este checkpoint y sus restricciones en
un banco de alcance mayor y volver a validar timing y comportamiento.
El [primer montaje con IP AXI registradas](PEER_RX_OVERLAP_AXI.md) ya pasa
los 16 casos funcionales y 15 perfiles, preservando N+4; suma dos ciclos
por lote sin pausas. Su ruteo agota 900 s, por lo que el cierre físico del
montaje integrado sigue pendiente. Se conserva el resultado acreditado
del bloque sin esas IP y una propuesta nueva de distribución por conexiones.
`ahead` queda como alternativa de menor alcance, todavía sin evaluación
física. Continúan fuera de estas tandas la IP PCIe/HBM, autorización de
direcciones BAR, CDC, recuperación entre épocas y hardware real.
