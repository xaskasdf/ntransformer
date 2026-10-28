# Doble entrada con bancos agrupados

Fecha: 2026-10-07. Experimento aislado derivado del
[prototipo de doble entrada](PEER_RX_DUAL_INGRESS_PROTOTYPE.md).
Estado: simulación funcional revisada y temporización interna OOC positiva
tras colocación y ruteo.
El transporte calificado y el asignador anterior permanecen intactos.

## Organización y compromiso

[El RTL](../../fpga/experimental/ntrpu_dual_ingress_packed.sv) conserva
ocho bancos lógicos de 4 KiB: dos dedicados por canal y cuatro reservas
prestables a cualquiera. Agrupa esos bancos en dos arrays de 256 × 512 bits:

| Grupo físico | Bancos dedicados | Reservas compartidas |
|---|---|---|
| 0 | 0, 1 — canal 0 | 4, 6 |
| 1 | 2, 3 — canal 1 | 5, 7 |

Cada grupo dispone de una lectura y una escritura por ciclo. Sus árbitros
son independientes y alternan entre solicitantes cuando ambos compiten.
Se mantienen dos beats de entrada y dos lecturas por ciclo cuando utilizan
grupos distintos. Si ambos canales solicitan el mismo puerto de un grupo,
uno espera; mientras ambos mantienen la petición, el siguiente turno es suyo.
Esto no garantiza progreso si el productor o el consumidor dejan de participar.

El arbitraje de escrituras se aplica a `data_ready`; el de lecturas, antes
de avanzar el contador y marcar la lectura pendiente. Una concesión conserva
la latencia de lectura de un ciclo. El banco regresa al pool tras el último
W aceptado y la validación posterior de su generación. El estado de B y
completion permanece separado. Se conservan stop con drenaje, fence global,
dieciséis descriptores y cuatro escrituras pendientes por canal.

Las reservas siguen siendo compartidas: su grupo físico limita los puertos,
no cambia su propietario lógico ni las reglas del asignador. Más capacidad
de reserva por sí sola no aumenta esos puertos. A 200 MHz, un puerto de
512 bits tiene un techo aritmético de 12,8 GB/s; dos grupos independientes,
25,6 GB/s. Son límites de esta interfaz, no tasas PCIe/HBM medidas.

## Evidencia funcional

[El banco](../../fpga/tb/tb_ntrpu_dual_ingress_packed.sv) conserva los ocho
escenarios anteriores y añade préstamos asimétricos en ambos sentidos:
un canal ocupa seis bancos y el otro dos. Se solapan llenados que compiten
por el mismo grupo y luego se provocan conflictos de lectura al drenar.

Pasaron diez escenarios en modo doble y en control serial: **105 descriptores
y 319.104 bytes verificados por modo**. Incluyen dos saturaciones completas
sin reset entre ellas, reutilización de todos los bancos, pausas asimétricas,
independencia de entradas, fence, stop y errores de comando/BRESP.

El [revisor independiente](../../tools/check_ntrpu_dual_ingress_packed.py)
reconstruye datos, orden, leases, generaciones, devoluciones y concesiones
de cada puerto. Contrasta las escrituras concedidas con las entradas y
cuenta una lectura por beat emitido. Veintiún controles negativos de traza
detectan eventos ausentes, datos o generaciones corruptos y concesiones
incorrectas. Estos controles no son inyecciones de fallos del silicio.

Además se ejecutaron dos copias con mutaciones reales de RTL. Usar datos
del canal equivocado falla por `OUTPUT_DATA`. Mantener siempre la prioridad
del canal 0 conserva los datos y termina la simulación, pero el revisor de
puertos lo rechaza por incumplir la alternancia. Ninguna copia mutada se
considera un resultado aprobado del diseño.

| Escenario de préstamos, modo doble | Conflictos de escritura | Conflictos de lectura |
|---|---:|---:|
| Canal 0 ocupa seis bancos | 126 | 118 |
| Canal 1 ocupa seis bancos | 126 | 122 |

Cada cifra cuenta ciclos con dos solicitudes al mismo puerto de un grupo,
no ciclos de bloqueo de toda la unidad. Ambas cargas terminan con datos
correctos; el replay comprueba la alternancia durante esos conflictos.

## Caudal y recursos

Carga equilibrada de 64 KiB, reloj simulado de 200 MHz:

| Medida | Ocho arrays anteriores | Dos grupos compactos |
|---|---:|---:|
| Ciclos de llenado | 536 | 536 |
| Ciclos hasta drenaje | 608 | 608 |
| Caudal durante aceptación de payload | 24,545 GB/s | 24,545 GB/s |
| Caudal hasta drenaje completo | 21,558 GB/s | 21,558 GB/s |
| Ciclos con dos beats aceptados | 504 | 504 |

El control serial compacto conserva 1.105 ciclos hasta drenaje. La mejora
doble/serial de esta carga sigue siendo 1,817×. Esta carga utiliza bancos
dedicados y no mide la penalización de préstamos entre grupos.

Síntesis OOC, Vivado 2026.1, `xcvu47p-fsvh2892-2-e`:

| Recurso | Ocho arrays anteriores | Dos grupos compactos |
|---|---:|---:|
| LUT | 10.068 | 5.176 |
| Registros | 9.913 | 9.854 |
| RAMB36 / RAMB18 | 56 / 8 | 14 / 2 |
| BRAM equivalentes de 36 Kib | 60 | 15 |
| Latches / blackboxes / URAM | 0 / 0 / 0 | 0 / 0 / 0 |

Son **75 % menos BRAM y 48,6 % menos LUT** en síntesis, conservando 32 KiB
lógicos. El reloj de 5 ns se aplica después de sintetizar, igual que en la
referencia: +1,388 ns de setup y +0,073 ns de hold son estimaciones internas
posteriores a síntesis. No califican por sí mismas el funcionamiento físico.

El mismo RTL completó colocación, optimización física y ruteo en 314,374 s.
La revisión posterior registra **5.128 LUT, 9.854 registros y 15 BRAM
equivalentes**, con **+0,588 ns de setup y +0,042 ns de hold** a 5 ns.
Las 12.695 redes informadas están completamente ruteadas, sin errores de
ruteo, DRC, latches ni blackboxes. No hay endpoints internos sin reloj.

Permanecen **1.388 entradas y 1.402 salidas sin presupuesto externo**.
Este cierre cubre caminos internos del núcleo aislado; no incluye simulación
de la netlist ruteada, floorplan conjunto PCIe/HBM ni hardware. El
[runner de ruteo](../../tools/run_ntrpu_dual_ingress_packed_route.py) trabaja
con una copia privada del checkpoint de síntesis y el
[revisor](../../tools/check_ntrpu_dual_ingress_packed_route.py) comprueba
la identidad de fuentes, checkpoint, informes y límites del resultado.

## Reproducción y respaldo

```bash
python3 -B tools/run_ntrpu_dual_ingress_packed.py /nuevo/doble --stage sim
python3 -B tools/run_ntrpu_dual_ingress_packed.py /nuevo/serial --stage sim --serial 1
python3 -B tools/run_ntrpu_dual_ingress_packed.py /nuevo/synth --stage synth
python3 -B tools/check_ntrpu_dual_ingress_packed.py /nuevo/doble
python3 -B tools/check_ntrpu_dual_ingress_packed.py /nuevo/synth > /nuevo/SYNTH_REVIEW.json
python3 -B tools/run_ntrpu_dual_ingress_packed_route.py /nuevo/route \
  --prior /nuevo/synth --review /nuevo/SYNTH_REVIEW.json
python3 -B tools/check_ntrpu_dual_ingress_packed_route.py /nuevo/route
```

Cada salida debe ser nueva. Los runners limitan cada tanda a dos CPU,
prioridad baja y 900 s; guardan hashes, informes y trazas. Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-packed-dual\20261007T044843897319Z`.
El respaldo usa objetos incrementales SHA-256/gzip en
`D:\AI\artifacts\ntrpu-content-store`; las transferencias ZIP viven en memoria.

## Límites y siguiente decisión

La geometría implementada es fija: dos canales, ocho bancos y 64 beats por
banco. No incluye PCIe, CDC, controlador HBM, ECC de payload ni recuperación
coordinada. Requiere marcos de exactamente `LEN+1` beats completos. El éxito
HBM del receptor anterior no valida automáticamente este núcleo nuevo.

La [comparación de preferencia local](PEER_RX_RESERVE_LOCALITY.md) incorpora
un asignador configurable que busca reservas del grupo propio antes de
pedir prestadas las del otro. Registra caudal, conflictos, préstamos y coste
físico con llegada tardía y pausas periódicas. Conserva la posibilidad de
usar las cuatro reservas desde un solo canal; su beneficio depende de la carga.

También merece un ensayo separado duplicar la profundidad física a 512
palabras: AMD documenta el modo SDP de 512 × 72 para RAMB36E2 en
[UG573](https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources/Address-Bus-ADDRARDADDR-and-ADDRBWRADDR).
Esto sugiere que 64 KiB lógicos podrían caber en una cantidad similar de
BRAM. Es una hipótesis de organización; no se han implementado ni medido
dieciséis bancos, su asignador, ECC ni su temporización.
