# Reservas frente a selección por destino

Estudio local del 2026-10-05. **504 casos modelados, 18 pruebas unitarias
aprobadas y 18 comparaciones de compatibilidad exactas.** Este barrido
complementa la [medición dual R4](PEER_RX_DUAL_PERFORMANCE.md); no ejecuta
RTL, MIG ni Vivado y no consume capacidad de la VM.

## Alcance de la comparación

Se cruzaron reservas R4/R6/R8; entrada ordenada, ventanas de dos/ocho
descriptores y selección ideal sin límite; 16/128 transferencias de 4 KiB;
reparto equilibrado, 3:1 y agrupado por destino; ausencia de pausa o W0/B0
detenido durante 4/8/16 µs. Cada destino conserva dos bancos dedicados,
cuatro escrituras pendientes y 16 reservas de descriptor/resultado.

La nueva opción `ingress_window` limita la selección a los primeros N
trabajos todavía no admitidos, conservando el orden dentro de cada destino.
Una entrada reproduce exactamente la política ordenada; omitir la opción
conserva el modelo anterior. Los descriptores están disponibles desde el
inicio; el payload debe permanecer accesible selectivamente aguas arriba.
Esto no permite saltar datos ya serializados en una misma entrada.

El modelo supone arbitraje y reutilización de recursos en el mismo ciclo,
dos servicios de lectura independientes y B demorado 32 ciclos. Omite el
coste de comandos, RAM, CDC, precarga de descriptores y colas elásticas
adicionales del RTL. La conversión a 200 MHz solo expresa sus ciclos.
Por ello sus GB/s y esperas **no sustituyen** los 11,136 GB/s y 975 ns
medidos en el banco ruteado anterior.

## Resultados que orientan el diseño

Caso equilibrado, 128 transferencias; W0 pausado al principio. Las cifras
son caudal modelado incluyendo llenado y drenaje final, en GB/s decimales.

| Variante | Pool lógico | Pausa 4 µs | Pausa 8 µs | Pausa 16 µs | Completions del destino libre antes de 16 µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4, ordenada | 32 KiB | 12,507 | 11,417 | 9,723 | 6 |
| R6, ordenada | 40 KiB | 12,650 | 11,745 | 9,960 | 8 |
| R8, ordenada | 48 KiB | 12,650 | 12,092 | 10,208 | 10 |
| R4, ventana 2 | 32 KiB | 12,603 | 11,498 | 9,781 | 7 |
| R4, ventana 8 | 32 KiB | 12,650 | 12,003 | 10,145 | 13 |
| R8, ventana 8 | 48 KiB | 12,650 | 12,650 | 10,674 | 17 |
| R4, selección ideal | 32 KiB | 12,650 | 12,650 | 12,650 | 42 |

Sin pausa, estas variantes obtienen el mismo 12,650 GB/s. Con 4 µs,
R4 ordenada acumula 475 ns de bloqueo; ventana dos reduce a 155 ns y
ventana ocho lo elimina, igual que R6 ordenada. Con 16 µs, R4/ventana ocho
todavía acumula 10.235 ns: no elimina una pausa arbitrariamente larga.
La selección ideal tiene visibilidad ilimitada y no representa una cola
de ocho entradas implementada.

En el caso agrupado de 128 trabajos, los primeros 64 pertenecen al
destino 0. Ventanas de dos y ocho no alcanzan los descriptores del destino
libre: ambas completan cero trabajos allí durante la pausa, incluso con
R8. La selección ideal sí puede acceder a su cola. Esta diferencia
justifica separar las colas de envío desde el productor, no solo ampliar
la mirada sobre una cola global.

## Candidato para el siguiente prototipo

Conservar R4 como referencia y probar **dos colas de descriptores, una por
destino, inicialmente de ocho entradas cada una**, con arbitraje circular
entre cabeceras elegibles. Esta profundidad es una propuesta para medir;
los casos de ventana global ocho no validan esa arquitectura.

El contrato del candidato debe resolver estos puntos antes de conectarlo:

1. El productor publica cada descriptor en la cola de su destino; la
   precarga puede avanzar en ambas sin depender de drenar la otra.
   El descriptor referencia payload persistente en origen. Solo se trae
   ese payload después de seleccionar y reservar su destino.
2. La elegibilidad debe cubrir banco, descriptor y resultado, con una
   concesión atómica para la entrada compartida. `banks_used` por sí solo
   no acredita todos esos recursos. El RTL actual no expone esta concesión
   por destino: hay que diseñarla, registrarla y verificarla.
3. Una selección publicada se mantiene hasta su handshake. Después, el
   llenado de su trama no se interrumpe. Solo se arbitra entre tramas;
   no se retira un comando bloqueado para sustituirlo por otro.
4. Conservar FIFO por destino, identidad/época del trabajo y las barreras
   entre operaciones dependientes. Reordenar únicamente trabajos que el
   contrato marque como independientes; una barrera espera sus anteriores.
5. Stop, error y pérdida de reloj deben cerrar nuevas admisiones y mantener
   el drenaje del transporte comprometido. Las colas nuevas forman parte
   del inventario de recursos que impide declarar un rearme seguro.

R6 agrega 8 KiB lógicos y R8 agrega 16 KiB frente a R4. No se ha medido su
incremento de BRAM/LUT/ruteo en esta composición. Las nuevas colas también
cuestan memoria y control. Si se almacenaran ocho payloads completos en
vez de referencias, serían otros 32 KiB por cola: esa alternativa debe
contabilizarse por separado. Ninguna de estas capacidades eleva el techo
de la entrada compartida de 512 bits.

El siguiente banco RTL debe comparar R4 ordenada, R4 con colas separadas
y R6 ordenada, usando las mismas trazas y pausas. Exigir conservación de
datos, estabilidad de comandos bajo bloqueo, orden/barreras, progreso del
destino libre y drenaje ante fallos. Medir además el coste de precarga y
el tiempo desde disponibilidad del descriptor, evitando ocultar esperas
al medir solo desde la admisión. Repetir síntesis, ruteo, STA y CDC antes
de reemplazar el receptor validado. R8 sigue como punto de contraste.

## Evidencia y reproducción

El revisor comprueba 36.288 transferencias, correspondientes a 148.635.648
bytes modelados, sin sustituir la comprobación interna de marcadores por
beat. Revisa visibilidad, exclusión de bancos, conservación, orden,
créditos, pausas y aritmética de las métricas; rechaza cinco corrupciones
deliberadas. Se conserva el primer revisor y `REVIEW.json`; la revisión
vigente `REVIEW_V2.json` añade configuración exacta y una prueba de
visibilidad que mantiene tags únicos. No fue necesario repetir el barrido.

```powershell
python -B -m unittest discover -s tests -p test_ntrpu_pool_lookahead.py -v
python -B -m unittest discover -s tests -p test_ntrpu_buffer_pool_model.py -v
python -B tools/run_ntrpu_dual_reserve_study.py artifacts/ntrpu-reserve-lookahead/nueva-tanda
python -B tools/check_ntrpu_dual_reserve_study.py artifacts/ntrpu-reserve-lookahead/nueva-tanda
```

Las salidas deben ser nuevas. Evidencia local:
`D:\AI\artifacts\ntrpu-reserve-lookahead\20261005T035929990241Z`.
Incluye entradas exactas, todas las trazas, resultados, pruebas,
compatibilidad, comparación y respaldo ZIP con SHA-256/CRC por miembro.
El RTL y la cola Vivado del principal permanecen fuera de estos cambios.

## Avance posterior: prototipo de control

El [selector RTL con dos colas](PEER_RX_DESTINATION_DISPATCH.md) ya pasó
24 casos de protocolo y síntesis aislada. Conserva la referencia R4;
falta el adaptador que haga efectiva su reserva atómica en ese receptor
y la validación con payload y controladores reales.
