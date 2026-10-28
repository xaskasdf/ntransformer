# Receptor dual R4: rendimiento y reservas

Tanda UTC del 2026-10-05: `20261005T031606329162Z-dual-netlist`.

## Estado y configuración

Se integró la corrección de [arranque frío](PEER_RX_DUAL_STARTUP.md) en
`fpga/rtl/ntrpu_dual_mig_cold_start.sv`: el contador permanece limpio
mientras existe reset UI o falta calibración. Es el único cambio funcional
al RTL protegido. `INTEGRATION.json` registra la identidad con la candidata
validada, normalizando únicamente CRLF/LF.

Se reutilizó su núcleo ruteado, acompañado de dos modelos MIG originales,
con lecturas periódicas activas. La entrada compartida es de 512 bits a
200 MHz: techo bruto de 12,8 GB/s. Ambos puertos UI funcionan a 200 MHz
con fases distintas. R4 tiene dos bancos dedicados por destino y cuatro
compartidos, todos de 4 KiB: 32 KiB lógicos. Conserva 16 descriptores por
destino y cuatro créditos de escritura pendientes.

## Método y resultados

Después de calibrar y completar dos transferencias de preparación, se
ejecutaron seis casos de 16 transferencias cada uno. Cada caso se drenó
y se leyó íntegramente antes del siguiente. La ventana de caudal incluye
comandos, llenado y drenaje final; excluye calibración y lectura de
verificación. GB/s usa unidades decimales. Se muestreó cada ciclo de
entrada; la fuente ofrece las tramas secuencialmente.

| Caso | GB/s | Uso de entrada | Bancos máx. | Reservas máx. | Bloqueo de comandos (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equilibrado, 64 B | 2,626 | 20,5 % | 3 | 0 | 0 |
| Equilibrado, 512 B | 8,402 | 65,6 % | 2 | 0 | 0 |
| Equilibrado, 4 KiB | 11,136 | 87,0 % | 4 | 0 | 0 |
| Reparto 3:1, 4 KiB | 9,892 | 77,3 % | 5 | 2 | 0 |
| W0 pausado, 4 KiB | 7,372 | 57,6 % | 8 | 4 | 975 |
| B0 pausado, 4 KiB | 10,648 | 83,2 % | 6 | 2 | 0 |

Las pausas duraron 4.002,798 ns. Ningún caso bloqueó datos ya ofrecidos
(`data_valid && !data_ready`). Los ciclos sin datos también incluyen
control, espera de comandos y drenaje: no son todos contrapresión.
Las latencias p50/p95/p99 desde aceptación hasta completion fueron
1.065/1.435/1.435 ns en el caso equilibrado de 4 KiB y
1.415/5.140/5.140 ns con W0 pausado. Con 16 muestras, estos percentiles
describen el caso; no estiman colas estadísticas.

## Qué limita el pool

Los 195 ciclos de bloqueo corresponden al comando 79, destino 0.
Durante todos ellos estaban ocupados sus dos bancos dedicados y las
cuatro reservas; quedaba libre al menos un banco dedicado al destino 1.
Las máscaras fueron `0xf7` durante 66 ciclos y `0xf3` durante 129.
El pico global de ocho bancos ocurrió en otros ciclos del mismo caso.

El destino 0 alcanzó dos escrituras pendientes en la fuente y siete
descriptores, por debajo de sus límites. `pending_counts` cuenta
escrituras pendientes; la ocupación de descriptores se reconstruye
entre aceptación y completion. La propiedad de cada reserva por destino
no se observa directamente.

El destino 1 completó cinco transferencias antes de liberar W0. Después,
la fuente serial encontró el comando bloqueado y demoró nuevas ofertas,
incluidas las de destino 1. Esto identifica bloqueo por el comando al
frente de la entrada; no demuestra que el segundo MIG estuviera detenido.
`STALL_REVIEW.json` conserva tiempos y máscaras verificadas.

## Validación y alcance

Pasaron 13 casos de arranque y 98 comandos/completions: 271.488 bytes
escritos y otros tantos leídos, con comprobación de contenido, orden,
direcciones, clocks y drenaje. El revisor rechazó ocho corrupciones
deliberadas. La ejecución tardó 798,975 s usando dos CPU y prioridad baja.

La auditoría verificó la procedencia de los modelos y la generación del
banco. Se drenaron 29/30 retornos periódicos. Los 720 avisos DDR del
puerto 0 pertenecen a cinco lecturas periódicas previas a la primera
escritura; se comprobó que no contaminaran el payload. Tres advertencias
de declaración adelantada de `scenario` en el banco están justificadas
en `WARNING_REVIEW.json`; las seis identidades se verifican en las trazas.

El núcleo conserva la evidencia física y CDC de la tanda de arranque:
7.158 LUT, 17.951 FF y 15 BRAM, con STA interno positivo. No se volvió a
rutear. Continúa abierta la advertencia CDC-26 del reset externo.
Esta simulación usa primitivas sin SDF; no califica placa/PHY, PCIe/P2P,
HBM, coherencia entre tarjetas ni caudal sostenido.

## Reproducción, respaldo y siguiente comparación

Herramientas: `tools/prepare_ntrpu_dual_perf.py`,
`tools/ntrpu_dual_perf.tcl`, `tools/run_ntrpu_dual_perf.py` y
`tools/check_ntrpu_dual_perf.py`. Ejecutar desde un snapshot con
`evidence/routed_core_funcsim.v` de la tanda física validada:

```bash
python3 -B tools/run_ntrpu_dual_perf.py /tanda-nueva/output \
  --example /ruta/ntrpu_ddr_geometry_ex --timeout 1800
python3 -B tools/check_ntrpu_dual_perf.py /tanda-nueva \
  --physical /tanda-fisica
```

Runner y revisor exigen destinos nuevos. El launcher aislado, entradas,
salidas, revisores y documentos están en
`D:\AI\artifacts\ntrpu-dual-mig-performance`.
El respaldo `20261005-dual-mig-performance-handoff.zip` incorpora la
procedencia física/CDC y el ejemplo MIG; su `.VERIFIED.json` registra
SHA-256, CRC y hashes de todos los miembros.

Mantener R4 como referencia. Comparar pausas de 4/8/16 µs y una entrada
que pueda elegir otro destino antes de ampliar a seis u ocho reservas.
Medir caudal, bloqueo, equidad y coste físico con la misma carga;
cualquier cambio RTL requiere repetir implementación, STA y CDC.
Esta tanda no compara tamaños de reserva: más reservas pueden reducir
esperas, pero no demuestra una mejora del caudal total.

El [estudio posterior de reservas y selección](PEER_RX_RESERVE_LOOKAHEAD.md)
compara R4/R6/R8 con ventanas acotadas en un modelo local. Orienta el
próximo prototipo; sus supuestos ideales no sustituyen estas mediciones
con MIG ni califican nuevas variantes RTL.
