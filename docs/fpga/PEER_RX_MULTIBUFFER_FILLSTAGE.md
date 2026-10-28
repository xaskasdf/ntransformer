# Captura compartida de generación durante el llenado

Experimento aislado del 2026-10-04 sobre la variante
[decoded](PEER_RX_MULTIBUFFER_DECODED.md). Busca reducir la carga de
`alloc_generation`, cuya ruta física terminaba en los registros de
generación de 32 descriptores. El objetivo sigue siendo R4, con dos
canales, cuatro bancos dedicados y cuatro de reserva: 32 KiB lógicos.

**Resultado físico: R4 cumple STA a 200 MHz en el arnés registrado,
con +0,032 ns de setup y +0,010 ns de hold.** El incremento posterior a
enrutado es de 6 LUT y 66 FF respecto a decoded, sin aumentar BRAM ni
CARRY8. La netlist ruteada conserva las trazas de referencia. El margen
es estrecho y no acredita cierre del sistema PCIe/HBM.

## Cambio y contrato

El [RTL fillstage](../../fpga/rtl/ntrpu_peer_multibuffer_fillstage.sv)
añade `fill_generation`, un registro compartido de 64 bits. Captura la
generación en `accept_cmd` y la escribe en el descriptor al aceptar el
último beat de entrada, en el mismo flanco que lo marca READY.

Sólo puede existir un llenado activo. El comando y su primer beat no se
aceptan en el mismo ciclo; el envío observa READY en un ciclo posterior
al último beat. Por ello, incluso con un solo beat, la generación ya está
disponible antes de `start_write`. No se cambia el handshake público ni
se retrasa la devolución del banco en el último W.

El registro se limpia con reset. Stop y fault cierran la admisión pero
permiten completar el llenado y comprometer su generación. El asignador
decoded, los registros de devolución, la RAM, las cuotas y las reglas de
propiedad permanecen iguales. La generación del descriptor puede conservar
su valor anterior mientras está FILL; sólo es válida para envío desde READY.

## Regresión y casos límite

El [ejecutor](../../tools/run_ntrpu_multibuffer_fillstage.py) conserva los
bancos y oráculos originales. Compara decoded y fillstage:

- 36 simulaciones normales: nueve perfiles y dos semillas, con el oráculo
  completo de transacciones, bytes y leases.
- 12 simulaciones dirigidas: R4, R8 y bancos de un beat, dos semillas.
  Interrumpen un llenado con reset, mantienen fault durante un llenado
  pausado y mantienen stop hasta aceptar el último beat.
- Las 24 comparaciones conservan sus trazas ciclo por ciclo.
- Cinco mutaciones detectadas: captura perdida, uso de generación viva
  al comprometer, compromiso bloqueado por stop o fault y reset omitido.

El monitor registra la generación aceptada independientemente de los
registros nuevos y la comprueba durante FILL, READY/WRITE y envío activo.
La regresión normal de la variante verificó 1336 trabajos, 600.320 bytes,
10.712 comprobaciones de llenado, 226.229 de descriptor, 16.182 de envío
y 444 de reset. La tanda completa r2 tardó 21,82 s.

Los casos dirigidos reutilizan el BFM de payload y respuestas del banco
original, añaden assertions y comparan trazas con decoded. No usan el
oráculo Python de drenaje seguro para el reset que descarta un llenado.
Esta prueba local de reset no acredita abort/rearm del transporte real.

La tanda r1 detectó un hueco del banco: el stop original terminaba antes
del último beat y una mutación escapó. Se conserva como fallo de cobertura;
r2 añadió casos explícitos y pasó sin cambiar el RTL. No es prueba formal
exhaustiva.

```bash
python3 -B tools/run_ntrpu_multibuffer_fillstage.py /ruta/nueva/rtl
```

Con Icarus extraído, añadir `--iverilog`, `--vvp` y `--ivl-base`.

## Medición física

La síntesis R4 en VU5P `xcvu5p-flva2104-2-i`, Vivado 2026.1, terminó en
253,766 s, sin errores ni avisos críticos:

| Síntesis | Decoded | Fillstage |
| --- | ---: | ---: |
| LUT | 4328 | 4353 |
| FF | 9521 | 9587 |
| Tiles BRAM / CARRY8 | 15 / 206 | 15 / 206 |
| WNS estimado a 5 ns | +1,401 ns | +1,425 ns |

La RAM conserva 14 RAMB36 y dos RAMB18. No hay latches, cajas negras,
hallazgos DRC ni endpoints internos sin reloj o restricciones. La peor
ruta estimada va del contador FIFO al enable de estado de un descriptor;
los informes de síntesis no sustituyen el resultado físico.

La implementación terminó en 644,287 s con el arnés registrado anterior,
dos CPU, prioridad `nice 10`, placement `Default` y route
`RuntimeOptimized`. El arnés conserva 673 FF de entrada, 1486 de salida
y un BUFGCE real, reloj de 5 ns e incertidumbres setup/hold adicionales
de 0,100/0,050 ns. Se confirmó identidad del arnés salvo el nombre del
núcleo. Se realizó una sola colocación/enrutado, con límite de 1200 s.
No hubo reintentos ni cambios en excepciones, incertidumbres o severidades.

| Después de P&R, 200 MHz | Decoded | Fillstage |
| --- | ---: | ---: |
| LUT / FF del núcleo | 4317 / 9521 | 4323 / 9587 |
| Tiles BRAM / CARRY8 | 15 / 206 | 15 / 206 |
| WNS setup | −0,409 ns | +0,032 ns |
| WHS hold | +0,003 ns | +0,010 ns |
| TNS setup | −150,162 ns | 0 ns |
| Endpoints con violación setup | 1605 | 0 |

La mejora de WNS es de 0,441 ns en estas implementaciones. No constituye
una estimación estadística sobre múltiples colocaciones. El arnés añade
2159 FF y un BUFGCE: total de 11.746 FF. `all_registers` devuelve 9603
celdas del núcleo; el informe de utilización cuenta sus 9587 FF.

Las 13.012 nets enrutables están completas; no hay errores de ruteo,
GAP, latches, cajas negras ni hallazgos DRC. Setup y hold cubren 31.502
endpoints; el chequeo de ancho de pulso cubre 11.779, con margen mínimo
de +1,958 ns. No hay violaciones ni endpoints internos sin reloj o
restricciones. Las fronteras registradas de entrada/salida también
cumplen: setup +0,317/+1,198 ns y hold +0,025/+0,025 ns.

Se revisaron dos avisos críticos `Project 1-486`, ambos anteriores a la
importación EDIF del núcleo, mientras el arnés contenía su caja negra.
Después de importar no hay avisos críticos; los recursos coinciden
antes/después y el recuento final de cajas negras es cero. Se conserva
un mensaje informativo `Route 35-443` de congestión local. Persisten los
avisos OOC, LOC fijo del BUFG, RAM sin registro de salida adicional y
puertos externos sin delays. No se aplicaron waivers.

El objetivo de 5 ns se cumplió directamente en la implementación. No se
hizo búsqueda de Fmax ni otra tanda de auditorías a frecuencias menores.
La netlist ruteada pasó en 63,109 s: 81 trabajos, 80.832 bytes y 3869
ciclos, con cuatro pendientes por puerto, 253 ciclos W simultáneos y
los casos B erróneo/ajeno. Sus 2881 líneas TRACE/CASE/PASS coinciden
con la referencia. XSIM usó un reloj de 10 ns, sin SDF y sin observar
leases internos. Conserva los avisos conocidos `glbl`/`LIBRARY_PATH`.
Las pruebas dirigidas de reset, stop y fault descritas arriba son RTL.

La referencia decoded cumple STA a 175 MHz en ese arnés. Los nuevos
200 MHz describen análisis temporal, no rendimiento medido en hardware.
El arnés no acredita timing de placa ni integración PCIe/HBM o DMA entre
tarjetas. R8 conserva sólo la validación funcional de esta variante.

## Siguiente integración

La peor ruta física va de
`allocator/bank_state[7].generation_reg[7][9]` a `status_reg[10][1]/CE`:
13 niveles y 4,620 ns de datos, con 3,422 ns de cableado (74,07 %).
Ya no termina en el dato de generación de los descriptores. La
distribución de admisión y enables sigue siendo una zona sensible.

Conservar esta variante como candidata de integración y comprobarla con
fronteras AXI elásticas y el transporte real antes de aumentar el pool.
El margen de 32 ps de setup obliga a repetir STA al añadir interconexión,
cruces de reloj y floorplan del dispositivo final. Las fronteras de prueba
actuales son registros de medición; no implementan backpressure AXI.
La adopción y cualquier cambio del receptor integrado quedan pendientes.

La variante lateral [con fronteras AXI elásticas](PEER_RX_MULTIBUFFER_ELASTIC.md)
añade dos entradas por canal AW/W/B y valida por separado la propiedad de
BRAM y de los datos encolados. Su medición física y sus límites se registran
en ese documento; fillstage se conserva como referencia.

## Evidencia

- [Regresión completa r2](../../artifacts/ntrpu-multibuffer-fillstage/20261004-rtl-r2/REVIEW.json).
- [Resumen funcional](../../artifacts/ntrpu-multibuffer-fillstage/RTL_SUMMARY.json).
- [Cambios RTL](../../artifacts/ntrpu-multibuffer-fillstage/RTL_CHANGE.diff).
- [Referencias protegidas](../../artifacts/ntrpu-multibuffer-fillstage/SOURCE_BASELINE.json).
- [Síntesis R4](../../artifacts/ntrpu-multibuffer-fillstage/20261004T151621243630Z-fillstage-synth-r4/REVIEW.json).
- [Comparación y compatibilidad del arnés](../../artifacts/ntrpu-multibuffer-fillstage/SYNTH_COMPARISON.json).
- [Implementación R4](../../artifacts/ntrpu-multibuffer-fillstage/20261004T152146435788Z-fillstage-route-r4/REVIEW.json).
- [Comparación física](../../artifacts/ntrpu-multibuffer-fillstage/ROUTE_COMPARISON.json).
- [Revisión de avisos](../../artifacts/ntrpu-multibuffer-fillstage/IMPORT_WARNINGS_REVIEW.json).
- [Netlist posterior a P&R](../../artifacts/ntrpu-multibuffer-fillstage/20261004T153402513263Z-fillstage-netlist-r4/REVIEW.json).
- [Revisión consolidada](../../artifacts/ntrpu-multibuffer-fillstage/FINAL_REVIEW.json).
- [Respaldo local](../../artifacts/ntrpu-multibuffer-fillstage/20261004-fillstage-handoff.zip),
  [verificación](../../artifacts/ntrpu-multibuffer-fillstage/20261004-fillstage-handoff.VERIFIED.json)
  e [instrucciones](../../artifacts/ntrpu-multibuffer-fillstage/HANDOFF.md).
- [Primera tanda con hueco de cobertura](../../artifacts/ntrpu-multibuffer-fillstage/20261004-rtl-r1/REVIEW.json).

Las tandas se guardan en `artifacts/ntrpu-multibuffer-fillstage/` y en una
carpeta lateral independiente de la cola principal en la VM.
