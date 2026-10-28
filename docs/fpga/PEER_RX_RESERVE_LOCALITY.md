# Preferencia de reservas por grupo físico

Fecha: 2026-10-07. Comparación experimental sobre el
[receptor compacto](PEER_RX_PACKED_DUAL_INGRESS.md). Simulación funcional
aprobada para ambas políticas y cierre de temporización interna OOC tras
ruteo del selector final con preferencia local.

Continuación: [receptor doble con CDC y HBM](PEER_RX_DUAL_HBM.md), con
inventario de operaciones en vuelo y pruebas de retención entre relojes.

## Política y alcance

[El receptor](../../fpga/experimental/ntrpu_dual_ingress_locality.sv) mantiene
los dos grupos RAM, ocho bancos de 4 KiB y dos entradas independientes.
El parámetro `PREFER_LOCAL` cambia únicamente el orden de selección del
[asignador experimental](../../fpga/experimental/ntrpu_locality_allocator.sv):

| Canal | Orden ascendente, `0` | Preferencia local, `1` |
|---|---|---|
| 0 | 0, 1, 4, 5, 6, 7 | 0, 1, 4, 6, 5, 7 |
| 1 | 2, 3, 4, 5, 6, 7 | 2, 3, 5, 7, 4, 6 |

Primero se buscan bancos dedicados disponibles; después, reservas del grupo
propio; finalmente, cualquier reserva restante. Los bancos ocupados o
retirados siguen excluidos. La cuota de cuatro reservas por canal se conserva:
un canal puede tomar las cuatro y ocupar seis bancos mientras el otro
utiliza sus dos dedicados. La preferencia no convierte reservas en propiedad
exclusiva ni garantiza que estarán libres para su canal afín.

El selector final recorre una sola lista de índices reordenados; el orden
local está definido para esta geometría de dos canales y cuatro reservas.
No cambia el arbitraje alternado entre canales, la validación diferida de
devoluciones, la generación de leases ni las reglas de stop/fence. El núcleo
de memoria y sus árbitros de lectura/escritura conservan el comportamiento
del compacto. Los archivos anteriores y el RTL calificado quedan intactos.

## Comparación funcional

Se ejecutaron **las mismas fuentes y el mismo banco** con `PREFER_LOCAL=0`
y `=1`, manteniendo doble entrada. Cada configuración aprobó **14 escenarios,
249 descriptores y 796.288 bytes**. Además de los diez escenarios anteriores,
se añadieron dos llegadas tardías del segundo canal y dos cargas sostenidas
con pausas periódicas, invirtiendo cuál de los canales recibe más pausas.

Las pruebas tardías son cargas dirigidas de 32 KiB: precargan bancos de un
canal, solapan el siguiente llenado con el segundo marco del otro y completan
la ocupación de los ocho bancos antes de liberar ambas salidas. Las cargas
periódicas transfieren 64 marcos de 9 o 64 beats, 205.824 bytes por escenario;
pausan W, AW, completion e input con calendarios relativos al inicio del caso.
No se ajustan los calendarios en función de la política seleccionada.

| Carga | Ciclos hasta drenaje, ascendente | Preferencia local | Cambio de caudal |
|---|---:|---:|---:|
| Equilibrada, 64 KiB | 608 | 608 | 0 % |
| Llegada tardía, canal 0 precargado | 931 | 868 | +7,26 % |
| Llegada tardía, canal 1 precargado | 931 | 868 | +7,26 % |
| Pausas periódicas, canal 0 más detenido | 6.118 | 6.097 | +0,34 % |
| Pausas periódicas, canal 1 más detenido | 6.097 | 6.097 | 0 % |

La mejora tardía equivale a 63 ciclos menos, **6,77 % menos tiempo**; el
llenado baja de 526 a 463 ciclos. La tasa de 24,545 GB/s en la ventana de
payload de la carga equilibrada permanece igual. Son medidas de simulación
RTL a 200 MHz, sin PCIe, CDC ni HBM nativo.

## Conflictos: beneficio condicionado por la carga

| Carga | Escrituras en conflicto, antes → después | Lecturas en conflicto, antes → después |
|---|---:|---:|
| Llegada tardía, canal 0 precargado | 126 → 0 | 0 → 0 |
| Llegada tardía, canal 1 precargado | 126 → 0 | 0 → 0 |
| Pausas periódicas, canal 0 más detenido | 454 → 524 | 330 → 297 |
| Pausas periódicas, canal 1 más detenido | 445 → 520 | 328 → 297 |

Un conflicto cuenta un ciclo con dos solicitudes al mismo puerto del mismo
grupo. No equivale a un ciclo perdido por toda la unidad. La preferencia
local elimina los conflictos de las llegadas tardías ensayadas, pero con
pausas sostenidas puede **aumentar conflictos de escritura** y reducir los
de lectura, con poco efecto sobre el tiempo total. No demuestra una mejora
universal ni sustituye una medición con el tráfico del sistema integrado.

## Verificación de la política

El [revisor](../../tools/check_ntrpu_dual_ingress_locality.py) reconstruye
datos, completions, ocupación, generaciones y concesiones RAM. Un modelo
separado evalúa cada decisión del asignador: disponibilidad, orden de bancos,
turno de canal y bloqueo durante devoluciones. Los casos de préstamos
comprueban el uso de los ocho bancos y de reservas de ambos grupos.

Cada revisión aplica **24 controles negativos de traza**. Además, una copia
con una mutación real desactiva la preferencia dentro del asignador mientras
la configuración declara que está activa. La transferencia de datos termina
correctamente, pero el revisor la rechaza: concede banco 5 cuando correspondía
6. Ese resultado se conserva como detección esperada de la mutación.

## Coste físico

Síntesis con Vivado 2026.1, `xcvu47p-fsvh2892-2-e`, mismas fuentes del
selector final y cambio exclusivo de `PREFER_LOCAL`:

| Recurso | Control ascendente | Preferencia local |
|---|---:|---:|
| LUT | 5.176 | 5.246 |
| Registros | 9.854 | 9.851 |
| BRAM equivalentes | 15 | 15 |
| RAMB36 / RAMB18 | 14 / 2 | 14 / 2 |
| Latches / blackboxes / URAM | 0 / 0 / 0 | 0 / 0 / 0 |

El coste adicional de la política en esta síntesis es **70 LUT, +1,35 %**,
sin aumentar BRAM. Este resultado no es una medida de consumo eléctrico.

La primera implementación exploraba reservas locales y después repetía una
búsqueda general. Aunque pasaba las pruebas, producía 7.812 LUT y 9.982
registros, frente a 5.176 y 9.854 del control con las mismas fuentes. Su
sustitución por una sola lista de prioridad conservó las medidas funcionales
y redujo el coste sintetizado. Las trazas completas DI/LP/CP son idénticas
entre implementaciones para cada una de las dos políticas ensayadas; esto
comprueba esos estímulos, no equivalencia formal para todo estado posible.
Se guarda la primera versión y su ruteo para
comparación; ese efecto de síntesis no se considera un coste inevitable de
la política ni se atribuye exclusivamente a unas puertas adicionales.

El reloj de 5 ns se aplica después de sintetizar, como en la referencia.
Los márgenes internos posteriores a síntesis del selector final son +1,541 ns
de setup y +0,073 ns de hold.

Tras colocación, optimización y ruteo del mismo RTL: **5.202 LUT, 9.851
registros y 15 BRAM equivalentes**, con **+0,753 ns de setup y +0,044 ns de
hold** para el reloj de 5 ns. La tanda tardó 321,362 s. Las 12.778 redes
informadas quedaron ruteadas, sin errores de ruteo, DRC, latches ni
blackboxes. No hay endpoints internos sin reloj.

Quedan **1.388 entradas y 1.402 salidas sin presupuesto externo**. Es cierre
de caminos internos del núcleo aislado: no incluye simulación de la netlist
ruteada, floorplan conjunto con PCIe/HBM ni hardware. La referencia compacta
anterior ruteada ocupaba 5.128 LUT; esa comparación no equivale a medir el
consumo eléctrico ni el rendimiento de una tarjeta.

## Reproducción y evidencia

```bash
python3 -B tools/run_ntrpu_dual_ingress_locality.py /nuevo/control --stage sim --prefer-local 0
python3 -B tools/run_ntrpu_dual_ingress_locality.py /nuevo/local --stage sim --prefer-local 1
python3 -B tools/check_ntrpu_dual_ingress_locality.py /nuevo/local
python3 -B tools/run_ntrpu_dual_ingress_locality.py /nuevo/synth --stage synth --prefer-local 1
python3 -B tools/check_ntrpu_dual_ingress_locality.py /nuevo/synth > /nuevo/SYNTH_REVIEW.json
python3 -B tools/run_ntrpu_dual_ingress_locality_route.py /nuevo/route \
  --prior /nuevo/synth --review /nuevo/SYNTH_REVIEW.json
python3 -B tools/check_ntrpu_dual_ingress_locality_route.py /nuevo/route
```

Cada salida debe ser nueva. Las tandas tienen dos CPU, prioridad baja y
límite de 900 s; usan directorios privados. La evidencia se encuentra en
`D:\AI\artifacts\ntrpu-locality\20261007T050622046995Z`, con fuentes,
configuraciones, trazas, revisiones y hashes. El respaldo incremental está
en `D:\AI\artifacts\ntrpu-content-store`; no se conservan ZIP de transferencia.

## Decisión de integración

Mantener la política configurable durante la integración: ayuda a algunas
secuencias de llegada, pero no elimina la competencia por los puertos RAM.
Antes de elegirla como configuración del transporte, repetir la comparación
con CDC, HBM y pausas del productor real. La
autorización de regiones PCIe, visibilidad remota, retorno de lectura,
recuperación coordinada y ECC siguen fuera de este ensayo.
