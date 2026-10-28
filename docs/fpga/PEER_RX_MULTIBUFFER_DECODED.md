# Validación por banco y concesiones one-hot

Experimento aislado del 2026-10-04 sobre la variante
[con registro de devolución](PEER_RX_MULTIBUFFER_CONTROL.md). El objetivo
es acortar la devolución y reasignación de un banco, conservando el
protocolo y los 32 KiB lógicos de R4. No sustituye al receptor integrado.

**Resultado: R4 cumple STA a 175 MHz nominales en el arnés registrado,
con +0,306 ns de setup y +0,003 ns de hold.** La netlist ruteada conserva
las trazas de referencia. A 200 MHz todavía incumple setup. La variante
queda como candidata para continuar el diseño, sin integración automática.

## Cambio propuesto

El [asignador decoded](../../fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv)
compara cada devolución contra bancos con índices constantes: banco,
propietario, ocupación y generación. Una devolución sin coincidencia
produce error, incluidos índices fuera de rango. Cada banco combina las
coincidencias de los puertos y habilita su estado mediante una concesión
one-hot. Los identificadores codificados se generan sólo para la salida.

El [datapath decoded](../../fpga/rtl/ntrpu_peer_multibuffer_decoded.sv)
conserva los registros `active_bank`/`active_generation` de la variante
anterior. Cambia únicamente su asignador predeterminado. Se mantienen
64 bits de generación, retiro sin rollover, prioridad de bancos dedicados,
cuotas de reserva, round-robin, stop/fault y drenaje. La reasignación gana
frente a una devolución válida en el mismo flanco. Un retorno incorrecto
bloquea nuevas asignaciones, pero no impide liberar otro banco válido.

La hipótesis física es que evitar multiplexores antes del comparador y
del enable compense replicar comparaciones y distribuir sus señales.
No se añade una etapa de pipeline ni se presupone una mejora de frecuencia.

## Validación funcional

El [ejecutor local](../../tools/run_ntrpu_multibuffer_decoded.py) usa copias
de los bancos y oráculos existentes, sin cambiar sus criterios:

- 48 simulaciones del asignador: ocho perfiles, tres semillas y dos
  implementaciones, `parallel` y `decoded`.
- 36 simulaciones del datapath: nueve perfiles, dos semillas y dos
  implementaciones, `control` y `decoded`.
- Coincidencia exacta de TRACE/CASE/PASS en las 42 comparaciones.
- Cuatro mutaciones detectadas: ignorar generación, perder la reasignación
  simultánea, ignorar cuota y permitir rollover.

La tanda `20261004-rtl-r2` pasó en 21,53 s. Cubre R4/R8, reserva cero,
un canal, bancos y descriptores no binarios, devolución obsoleta o ajena,
retornos conflictivos, pausas, errores y drenaje. Es regresión dinámica,
no equivalencia formal. La primera tanda conserva un error del ejecutor
al combinar metadatos (`seed` duplicado); se corrigió y repitió completa.
La variante verificó 14.682 ciclos del asignador, 5389 devoluciones
válidas, 99 inválidas y 1575 reutilizaciones en el mismo ciclo. El
datapath verificó 1336 trabajos y 600.320 bytes.

```bash
python3 -B tools/run_ntrpu_multibuffer_decoded.py /ruta/nueva/rtl
```

Con Icarus extraído, añadir `--iverilog`, `--vvp` y `--ivl-base`.
Nunca reutilizar una carpeta de salida.

## Medición física

La síntesis R4 terminó en 297,724 s, sin errores ni avisos críticos:

| Síntesis | Registro de devolución | Decoded |
| --- | ---: | ---: |
| LUT | 4304 | 4328 |
| FF | 9528 | 9521 |
| Tiles BRAM | 15 | 15 |
| CARRY8 | 164 | 206 |
| WNS estimado a 5 ns | +1,326 ns | +1,401 ns |

La RAM conserva 14 RAMB36 y dos RAMB18, repartidos en dos réplicas.
No hay latches, cajas negras, hallazgos DRC ni endpoints internos sin
reloj o restricciones. Persisten los avisos OOC, RAM sin registro de
salida adicional, puertos constantes y tamaño de celda aplanada.

Colocación y enrutado terminaron en 708,406 s con el mismo arnés registrado
VU5P `xcvu5p-flva2104-2-i`, reloj de 5 ns e incertidumbres setup/hold
adicionales de 0,100/0,050 ns. Se confirmó que el arnés es idéntico salvo
el nombre del núcleo: 673 FF de entrada, 1486 de salida y un BUFGCE real.
La tanda usó dos CPU, `nice 10`, placement `Default`, route
`RuntimeOptimized` y límite de 1200 s. Hubo una sola implementación, sin
reintentos ni cambios en las excepciones temporales. La referencia control
también partió de síntesis con esos ajustes; no se estudiaron múltiples
colocaciones para estimar variabilidad.

| Después de P&R / STA | Registro de devolución | Decoded |
| --- | ---: | ---: |
| LUT / FF del núcleo | 4313 / 9528 | 4317 / 9521 |
| Tiles BRAM / CARRY8 | 15 / 164 | 15 / 206 |
| WNS a 200 MHz, 5 ns | −0,781 ns | −0,409 ns |
| WNS a ~175 MHz, 5,715 ns | No auditado | +0,306 ns |
| WNS a ~166,7 MHz, 6 ns | +0,219 ns | +0,591 ns |
| WNS a ~150 MHz, 6,667 ns | +0,886 ns | +1,258 ns |
| WHS en los periodos auditados | +0,005 ns | +0,003 ns |

El coste físico incremental es de 4 LUT, −7 FF y 42 CARRY8. No cambia
la memoria. El arnés añade 2159 FF y un BUFGCE: total de 11.680 FF.
Las 13.002 nets enrutables están completas, sin errores, GAP, latches,
cajas negras ni hallazgos DRC. El recuento `all_registers` devuelve 9537
celdas del núcleo; usar el informe de utilización para contar sus 9521 FF.
A 200 MHz quedan 1605 violaciones setup, TNS −150,162 ns. A los tres
periodos auditados no hay violaciones setup, hold ni ancho de pulso.

La auditoría de periodos tardó 63,907 s. Verificó identidad de LOC, BEL y
ROUTE y el hash del DCP original, manteniendo incertidumbres y cobertura
temporal interna. Es STA sobre una implementación fija, no búsqueda de
Fmax ni rendimiento medido en hardware. El pequeño margen hold exige
revalidación al integrar el padre real.

La netlist ruteada pasó 81 trabajos, 80.832 bytes y 3869 ciclos en
62,904 s; conserva cuatro pendientes por puerto, 253 ciclos W simultáneos
y los casos de B erróneo/ajeno. Las 2881 líneas TRACE/CASE/PASS coinciden
con la netlist compacta anterior. La simulación fue funcional a 10 ns,
sin SDF y sin observación interna de leases.

Se revisaron dos avisos críticos `Project 1-486`, ambos anteriores a la
importación EDIF cuando el arnés aún tenía su núcleo vacío. Después de
importar no hay avisos críticos; los recursos coinciden antes/después y
el recuento final de cajas negras es cero. No se aplicaron waivers.
El router emitió dos mensajes informativos `Route 35-443` de congestión
local, pero completó todas las conexiones. Persisten los avisos OOC,
LOC de reloj fijo, RAM y timing a 5 ns, así como los avisos conocidos
`glbl`/`LIBRARY_PATH` en XSIM.

El arnés no acredita timing de placa, integración PCIe/HBM, CDC ni DMA
entre tarjetas. R8 sólo tiene evidencia funcional en esta tanda.

## Próxima mejora a estudiar

La [captura compartida durante el llenado](PEER_RX_MULTIBUFFER_FILLSTAGE.md)
ya fue implementada como variante aislada: R4 cumple STA a 200 MHz en
el mismo arnés, con margen estrecho. Sus pruebas y costes se documentan
por separado; los resultados de decoded de esta página se conservan.

La peor ruta física va de
`allocator/bank_state[2].generation_reg[2][14]` a
`generation_reg[30][22]/D`: 13 niveles y 5,324 ns de datos, de los que
3,802 ns corresponden a cableado (71,41 %). Pasa por la validación,
la concesión `grant_4` (fanout 131) y `alloc_generation[22]` (fanout 32).
La mejora del enable del banco expone ahora el reparto de generación
hacia los descriptores como siguiente límite.

La propuesta que motivó esa variante fue capturar la generación aceptada en un registro
compartido del llenado y escribir el descriptor durante ese llenado,
antes de marcarlo READY. El diseño actual permite un llenado a la vez;
esto podría reducir la carga inmediata de la salida del asignador.
Sus criterios eran conservar el primer beat, bancos de un beat,
stop/fault/reset, leases y trazas públicas antes de medirla. Consultar
el documento de la variante para la evidencia obtenida.

## Evidencia local

- [Regresión completa](../../artifacts/ntrpu-multibuffer-decoded/20261004-rtl-r2/REVIEW.json).
- [Respaldo de la regresión](../../artifacts/ntrpu-multibuffer-decoded/20261004-rtl-r2.zip).
- [Hashes de referencias protegidas](../../artifacts/ntrpu-multibuffer-decoded/SOURCE_BASELINE.json).
- [Síntesis R4](../../artifacts/ntrpu-multibuffer-decoded/20261004T144124546031Z-decoded-synth-r4/REVIEW.json).
- [Comparación y compatibilidad del arnés](../../artifacts/ntrpu-multibuffer-decoded/SYNTH_COMPARISON.json).
- [Implementación R4](../../artifacts/ntrpu-multibuffer-decoded/20261004T144708756394Z-decoded-route-r4/REVIEW.json).
- [Comparación física](../../artifacts/ntrpu-multibuffer-decoded/ROUTE_COMPARISON.json).
- [Netlist posterior a P&R](../../artifacts/ntrpu-multibuffer-decoded/20261004T150029107639Z-decoded-netlist-r4/REVIEW.json).
- [STA de 175/166,7/150 MHz](../../artifacts/ntrpu-multibuffer-decoded/20261004T150029102136Z-decoded-period-r4/REVIEW.json).
- [Avisos de importación y congestión](../../artifacts/ntrpu-multibuffer-decoded/IMPORT_WARNINGS_REVIEW.json).
- [Revisión consolidada](../../artifacts/ntrpu-multibuffer-decoded/FINAL_REVIEW.json).
- [Respaldo local](../../artifacts/ntrpu-multibuffer-decoded/20261004-decoded-handoff.zip),
  [verificación](../../artifacts/ntrpu-multibuffer-decoded/20261004-decoded-handoff.VERIFIED.json)
  e [instrucciones](../../artifacts/ntrpu-multibuffer-decoded/HANDOFF.md).

Las entradas y resultados de cada tanda se conservan bajo
`artifacts/ntrpu-multibuffer-decoded/`, con hashes SHA-256 y verificación
de los archivos comprimidos. La cola principal permanece independiente.
