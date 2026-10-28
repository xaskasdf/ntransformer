# Retorno HBM con fragmentos concurrentes

Candidato experimental W1 del 2026-10-07, posterior a la [prueba de los conversores](HBM_READ_CONCURRENCY.md). **Se conserva profundidad ocho para el siguiente experimento**, con tráfico HBM nativo revisado, timing interno positivo y regresión de netlist. Conserva un frame de 4 KiB por canal y añade seguimiento parametrizable de 1/2/4/8 fragmentos. El adaptador y retorno previamente calificados permanecen intactos; el candidato se compone en directorios privados mediante `prepare_ntrpu_read_window.py`.

## Contrato del seguimiento

`ntrpu_hbm_read_tracker.sv` distingue reservas, AR ofrecidas, AR aceptadas, beats aún por capturar y respuestas validadas. Una cola guarda las longitudes reales; un único slot registrado conserva la dirección ofrecida hasta su aceptación. Las respuestas comparten ID cero y avanzan en orden.

El crédito vuelve **al validar el último beat**, no al capturarlo ni al aceptar AR. Las dos colas de respuesta siguen contando para `pending` después de liberar ese crédito. Por tanto, retirar una entrada de seguimiento no acredita entrega al consumidor ni finalización remota.

Se conservan los seis bits de RID y las comprobaciones RLAST/RRESP. RREADY depende exclusivamente de estado registrado, capacidad y reset. Los fallos de protocolo quedan retenidos; las respuestas ya ofrecidas al conversor conservan VALID y pueden drenarse. Las direcciones ofrecidas también se conservan aunque aparezca un fallo. Solo un reset frío coordinado descarta las obligaciones en cuarentena.

El nuevo almacenamiento explícito de longitudes ocupa 4 bits por entrada, más punteros y contadores; no son frames adicionales. Las reservas de payload continúan sumando 8 KiB entre ambos canales, además de las colas y del almacenamiento de las IP.

## Pruebas funcionales revisadas

Las profundidades 1/2/4/8 pasaron los 20 escenarios con relojes independientes; profundidad ocho pasó además con reloj común. Cada tanda contiene 46 comandos, 41 escrituras de preparación, 936 beats entregados de 64 bytes y 41 completions. Los cinco comandos inválidos no emiten AR; dos errores de lectura no publican DATA; cinco fallos de protocolo permanecen en cuarentena.

El ledger reconstruye cada ciclo de reserva, aceptación, captura, validación y retirada. Observó los máximos 1/2/4/8 de ocupación correspondientes; con ocho créditos se llenaron ambas colas de dos slots. Cada regresión completa rechaza **32 corrupciones de evidencia**, incluidas alteraciones de los créditos y eliminación de eventos internos.

Los perfiles cortos de cuatro y ocho pasaron cuatro escenarios y 30 controles negativos cada uno. Incluyen 14 lecturas, 60 AR, 876 beats nativos y 438 beats de salida. Siguen siendo prepruebas con respondedor sintético; seleccionar el perfil corto no activa HBM nativa.

El banco directo `tb_ntrpu_read_tracker.sv` pasó seis escenarios con cuatro y ocho créditos. Comprueba retención de dos datos validados ante RID ajeno, RLAST temprano/ausente y EXOKAY; también una AR retenida durante una respuesta no solicitada. Tres copias mutadas del RTL fueron rechazadas por su causa esperada:

| Alteración deliberada | Comprobación que la detectó |
| --- | --- |
| Liberar crédito al capturar RLAST | `EARLY_CREDIT_RELEASE` |
| Ignorar bits altos de RID | `QUARANTINE_FAULT` |
| Hacer RREADY dependiente de RID | `READY_DEPENDENCE` |

El auditor se corrigió en profundidad uno: un fragmento malformado puede estar completamente capturado y seguir pendiente de validación en cuarentena. Se separaron ambos registros; se mantienen la identidad, los datos, la ausencia de publicación y la contabilidad interna. Las fuentes y revisiones iniciales quedaron archivadas.

## Ventanas de servicio

En el modelo sintético, dos lecturas simultáneas de 4 KiB desde aceptación hasta CPL tardaron 1.600/1.045/835/835 ns con 1/2/4/8 créditos. Cuatro y ocho empatan en ese modelo; son ventanas finitas, con memoria preparada y consumidor disponible, no caudal sostenido ni mediciones de PCIe.

Con HBM nativa, misma configuración de IP y caso de servicio, la comparación revisada es:

| Configuración | Dos lecturas de 4 KiB: aceptación→CPL | GB/s agregados en esa ventana | Separación máxima entre beats nativos |
| --- | ---: | ---: | ---: |
| Referencia serial, revisión 3 | 1.740 ns | 4,708 | 144,430 ns |
| W1, cuatro créditos | 940 ns | 8,715 | 35,552 ns |
| W1, ocho créditos | 920 ns | 8,904 | 4,444 ns |

Ocho reduce esa latencia un 47,1 % frente a la referencia y un 2,1 % frente a cuatro. No es una mejora uniforme: un beat sigue tardando 220 ns y ocho beats, 285 ns; nueve beats pasan de 425 a 295 ns con ambas profundidades. La publicación de 64 beats aún ocupa 320 ns sin solaparse con el llenado del siguiente frame.

Ambas variantes pasaron los cuatro escenarios nativos: 14 escrituras de preparación, 14 lecturas, 60 AR, 876 beats nativos, 438 beats entregados —28.032 bytes— y 14 CPL. Incluyen consumidor asimétrico, CPL retenida y stop con lecturas pendientes. Cada auditor rechazó 30 corrupciones de evidencia; el seguimiento alcanzó cuatro/ocho AR aceptadas, y ocho llenó ambas colas de dos slots. Los errores inyectados y la cuarentena se validan en los bancos funcionales anteriores, no en estos cuatro escenarios nativos.

Las elaboraciones tardaron 492,820/495,705 s y el tráfico 1.045,145/1.046,173 s, con ocho CPUs por variante y prioridad reducida. Ambos stacks terminaron su inicialización antes del arranque a 48,3902 μs, sin forzarlo. CATTRIP sigue X/X; el estado térmico queda sin calificar. Los snapshots están descargados y verificados.

## Implementación física aislada

Parte `xcvu47p-fsvh2892-2-e`, relojes de 200/450 MHz e incertidumbres setup/hold de 0,100/0,050 ns, sin relajar presupuestos:

| Créditos | Setup origen | Setup memoria | Hold mínimo | LUT | Registros | Resultado |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | +0,950 ns | −0,077 ns | +0,013 ns | 4.849 | 8.103 | Timing abierto: 326 endpoints de setup |
| 8 | +1,037 ns | +0,005 ns | +0,006 ns | 4.522 | 8.089 | Timing interno positivo |

Ambas usan cero BRAM36; esto no significa memoria gratuita: los frames, colas e IP consumen LUT/registros. Los recursos y el ruteo corresponden a estas implementaciones concretas; más créditos no garantizan siempre menor área. Cuatro queda como comparación funcional/nativa, con timing abierto; ocho es el candidato conservado.

El checkpoint de ocho tiene 10.839 nets ruteadas y cero errores de ruteo, latches, blackboxes o endpoints internos sin restricción. Hay 11 CDC-3 y ocho CDC-9 informativos; ocho restricciones de bus skew con margen mínimo +1,524 ns. RREADY tiene 22 orígenes registrados y ninguna entrada combinacional.

La regresión del netlist de ocho coincide ciclo a ciclo con RTL en los 20 casos: 6.384 muestras de origen, 7.236 de memoria y doce resets fríos. Rechaza 32 corrupciones de evidencia y las dos alteraciones PAYLOAD/READY del comparador. No usa SDF y no constituye equivalencia formal.

Las siete advertencias DRC siguen visibles, sin waivers: seis pines LUT sin efecto según INIT y RTSTAT-10 con 1.331 nets, clasificadas en 1.229 puertos externos sin carga interna y 102 nets sin carga lógica. Cero objetos sin explicar no equivale a DRC limpio.

**El margen de 5 ps es mínimo y solo interno.** Quedan 815 entradas y 1.398 salidas sin retardos externos. Este checkpoint no contiene hard HBM ni PCIe y no califica placa, interfaces externas ni cierre del sistema integrado.

## Siguiente experimento: dos frames por canal

**Avance posterior del 2026-10-07:** el [candidato F1](HBM_READ_FRAMES.md) completó este experimento con dos frames y ocho créditos. En una ventana comparable tras calentamiento, HBM nativa pasó de 8,445 a 14,711 GB/s (+74,2 %). F1 conserva evidencia funcional, nativa y de transacciones del netlist, con timing interno de solo +1 ps; no declara equivalencia exacta por ciclo. Los resultados siguientes describen el plan original y esta página conserva la referencia W1.

Mantener inicialmente ocho créditos y aumentar únicamente las reservas de lectura de 8 a 16 KiB totales: llenar un frame mientras se entrega el otro. Hace falta admisión y contabilidad por frame, conservar el orden y no publicar un frame hasta validar toda su respuesta. Los créditos de fragmento y la liberación del frame tras CPL son obligaciones distintas; una futura confirmación peer añade otra frontera de propiedad.

Antes de aceptar ese candidato, probar dos reservas ocupadas, error en el segundo frame, CPL retenida, consumidores asimétricos, stop y cuarentena. Medir una secuencia sostenida tras calentamiento, incluidos los huecos de admisión, y repetir recursos, CDC, timing, netlist y HBM nativa. Esta tanda no implementa todavía el doble frame ni determina el tamaño óptimo del pool de reserva.

## Evidencia y reproducción

Raíz de la tanda: `D:/AI/artifacts/ntrpu-read-window/20261007T153534477246Z/`. Las referencias `full1/2/4/8`, `common8`, `short4/8`, `unit`, `compile4/8`, `traffic4/8`, `physical4/8`, `netlist8` y `drc8` contienen fuentes, logs, hashes y revisiones. `VARIANT.json` registra las transformaciones privadas; `COMPARISON.json` identifica los resultados y la configuración HBM compartida. `WINDOWS_*.json` guarda las ventanas calculadas por `analyze_ntrpu_read_window.py`.

Los recibos `BACKUP_*.json` apuntan al almacén por SHA-256 `D:/AI/artifacts/ntrpu-content-store/`: cada tanda añade solo objetos nuevos, sin retener ZIP de transferencia. `RESTORE_REPLAY.json` acredita restauración y repetición de los auditores de las 16 tandas de evidencia; las copias temporales de esa comprobación se eliminaron. Los dos snapshots nativos tienen respaldos propios con verificación de contenido.

```bash
# SALIDA es nueva; binding y stub proceden de la referencia revisada.
python3 -B tools/prepare_ntrpu_read_window.py SALIDA --depth 8 \
  --binding REFERENCIA/native_binding.sv --stub REFERENCIA/native_stub.sv
# Ejecutar el runner del árbol privado con Vivado 2026.1 configurado.
python3 -B SALIDA/tools/run_ntrpu_read_return.py TANDA \
  --binding SALIDA/fixtures/native.sv --stub SALIDA/fixtures/stub.sv
python3 -B tools/check_ntrpu_read_window.py TANDA
```

El generador no sobrescribe directorios ni fuentes calificadas. Los estados `READ_WINDOW_*` distinguen la evidencia del candidato de la referencia anterior. Esta validación no integra permisos peer, DMA PCIe, cómputo ni hardware físico.
