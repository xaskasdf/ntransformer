# Retorno HBM con dos frames por canal

Candidato experimental F1 del 2026-10-07, construido sobre el [seguimiento W1 de ocho créditos](HBM_READ_WINDOW.md). **Se conservan dos frames por canal y ocho créditos de fragmento** para el siguiente paso. Aumenta las reservas de lectura de 8 a **16 KiB totales**: dos frames de 4 KiB en cada uno de los dos canales. La validación funcional, HBM nativa, el timing aislado y las transacciones del netlist están revisados dentro del alcance indicado abajo.

**Avance posterior del mismo día:** el [estudio C1](HBM_READ_CUT.md) conserva este mismo RTL con un nuevo ruteo, +87 ps de setup y +7 ps de hold en memoria, sin aumentar recursos. Esta página conserva las cifras y evidencia de la tanda F1 original; el checkpoint vigente se identifica como `refined_reference` en el estudio posterior.

## Organización y propiedad

`ntrpu_hbm_read_frames.sv` separa admisión, llenado y publicación. Cada canal tiene punteros independientes para reservar, llenar y publicar, y un contador de reservas ocupadas. El motor de lectura mantiene un frame en recepción y puede comenzar el siguiente mientras el publicador entrega el anterior. Los ocho créditos nativos siguen controlando fragmentos; no representan ocho frames.

La reserva permanece ocupada desde la aceptación del comando hasta **la aceptación de CPL**. Una CPL ofrecida y retenida no libera capacidad. La salida conserva el orden de los comandos de cada canal y publica únicamente frames completos y validados. La memoria de payload no se borra con reset; sus metadatos impiden publicar contenido antiguo.

SLVERR/DECERR producen una completion de error sin payload para ese frame. Los fallos de protocolo retienen las obligaciones en cuarentena. Un primer frame ya validado puede terminar de entregarse mientras el segundo queda en cuarentena; su completion no acredita salud de la época global. Stop y los fallos externos cierran nuevas admisiones y conservan el trabajo aceptado. Solo un reset frío coordinado descarta obligaciones.

Este almacenamiento pertenece al retorno de lectura. No consume automáticamente el pool de reservas de escritura ni constituye una confirmación de recepción remota por PCIe.

## Pruebas funcionales

La regresión anterior de 20 escenarios pasó con el nuevo módulo de dos frames. El banco adicional pasó doce escenarios con relojes independientes y comunes: 60 comandos y escrituras de preparación, 56 CPL y cuatro comandos en cuarentena. Incluye las dos reservas llenas, consumidores asimétricos, CPL retenidas, stop, reloj de memoria pausado, fallo externo y errores tardíos o malformados en el segundo frame mientras el primero está retenido.

El auditor independiente reconstruye comandos, fragmentos, payload nativo, llenado, sellado, entrega y liberación de cada frame. Comprueba el contador de reservas en cada ciclo y rechaza 24 corrupciones de evidencia. El perfil de cuatro escenarios se ejecutó con una y dos reservas para comparar bajo el mismo motor y ocho créditos.

Tres copias mutadas del RTL fueron rechazadas por su causa esperada: liberar la reserva al ofrecer CPL (`CPL_STABILITY`), ocultar un error de lectura (`PUBLICATION_BEFORE_FULL_FRAME`) y leer el banco incorrecto (`RETURN_DATA`). Ninguna detección dependió de agotar el watchdog.

Las dos primeras tandas nuevas completaron sus bancos, pero el lanzador esperaba la etiqueta antigua `RR ALL_PASS` y las rechazó. Se conservan como diagnóstico. Las tandas finales se repitieron después de corregir el marcador y el número esperado de escenarios del lanzador privado.

## Medición después de calentamiento

La secuencia contiene doce lecturas de 4 KiB por canal, con memoria preparada. Se excluyen dos frames de calentamiento y se miden los ocho siguientes, desde la CPL del último frame de calentamiento hasta la CPL del último medido; quedan dos frames finales. Se incluyen huecos de admisión, conversión, recepción, publicación y CPL.

| Reservas por canal | Ventana para 64 KiB agregados | Caudal agregado, modelo sintético |
| --- | ---: | ---: |
| Una | 6.800 ns | 9,638 GB/s |
| Dos | 4.080 ns | 16,063 GB/s |

Dos reservas mejoran un 66,7 % esta ventana tras calentamiento. Se observaron 672 ciclos de llenado y entrega simultáneos, frente a cero con una reserva. Estas cifras no son mediciones de hardware, PCIe ni del modelo HBM nativo. Tampoco deben compararse directamente con la latencia de servicio de un único par de lecturas de W1.

Con reloj común de 200 MHz, el mismo candidato pasó las comprobaciones funcionales y midió 8,111 GB/s, sin ciclos de llenado y entrega simultáneos. El auditor inicialmente exigía solapamiento en todos los perfiles de dos frames; se corrigió esa suposición para el perfil común, conservando ambas versiones. La comprobación de solapamiento positivo sigue siendo obligatoria para el objetivo de 200/450 MHz. Las reservas por sí solas no eliminan la latencia ni el límite de la interfaz de memoria.

## Comparación con HBM nativa

Ambas variantes pasaron los cuatro escenarios con el modelo HBM de AMD, la misma configuración de IP, ocho créditos y relojes de 200/450 MHz. Se conserva la secuencia y definición de ventana anteriores: ocho frames medidos por canal después de dos de calentamiento.

| Reservas por canal | Ventana para 64 KiB agregados | Caudal agregado, HBM nativa |
| --- | ---: | ---: |
| Una | 7.760 ns | 8,445 GB/s |
| Dos | 4.455 ns | 14,711 GB/s |

Dos reservas aumentan el caudal de esta ventana un **74,2 %** y reducen su duración un **42,6 %**. Se observaron 440 ciclos de llenado y publicación simultáneos, frente a cero con una reserva. Es una ventana acotada de simulación tras calentamiento; no acredita rendimiento de PCIe, de una placa ni de una inferencia completa.

Los bancos completos registraron, respectivamente, 36/40 escrituras y comandos, 3.912/4.424 beats nativos, 1.956/2.212 beats entregados y 36/40 CPL. La cantidad varía porque los escenarios de capacidad y stop llenan todas las reservas disponibles; la ventana comparada contiene los mismos 64 KiB. Ambos auditores rechazaron 24 corrupciones de evidencia. La inyección de errores y cuarentena corresponde a los doce escenarios funcionales, no a este perfil nativo.

Las elaboraciones tardaron 488,030/483,394 s y el tráfico 1.579,474/1.562,679 s, con ocho CPUs por variante y prioridad reducida. Ambos stacks terminaron su inicialización antes del arranque a 48,3902 μs, sin forzarlo. CATTRIP sigue X/X: el estado térmico queda sin calificar. Los dos snapshots nativos están descargados y verificados; requieren la instalación Linux de Vivado 2026.1 correspondiente.

## Implementación física aislada

El checkpoint de dos frames pasó timing interno en `xcvu47p-fsvh2892-2-e`, manteniendo 200/450 MHz e incertidumbres setup/hold de 0,100/0,050 ns:

| Dominio | Setup | Hold |
| --- | ---: | ---: |
| Origen | +0,754 ns | +0,015 ns |
| Memoria | +0,001 ns | +0,002 ns |

Ocupa **6.399 LUT, 8.386 registros y cero BRAM36**. Frente al checkpoint W1 de ocho créditos, añade 1.877 LUT y 297 registros para otros 8 KiB de reservas; son costes observados de estas implementaciones, no una fórmula lineal para ampliar el pool. El payload utiliza memoria distribuida y sigue teniendo coste aunque BRAM sea cero.

Hay 13.519 nets ruteadas, cero errores de ruteo y ningún endpoint interno sin restricción. CDC informa 11 CDC-3 y ocho CDC-9 informativos; las ocho restricciones de bus skew conservan margen mínimo +1,266 ns. La inspección de RREADY encuentra 22 orígenes registrados y ninguna entrada combinacional. La auditoría clasificó las siete advertencias DRC, sin waivers: seis pines LUT sin efecto según INIT y RTSTAT-10 con 1.335 nets —1.233 puertos externos sin carga interna y 102 nets sin carga lógica—. No es un informe DRC limpio.

**Un margen de 1 ps no ofrece una reserva práctica para integrar el resto del sistema.** Esta implementación excluye hard HBM, PCIe y restricciones externas de la placa. Una pasada adicional `Explore` conservó el setup en +1 ps y mejoró el hold de memoria a +7 ps; quedó archivada como exploración. El checkpoint de la tabla sigue siendo el asociado a la regresión y DRC finales.

## Regresión del netlist y límites de comparación

El netlist ruteado, como único DUT de lectura, pasó los doce escenarios y coincidió con el RTL en las transacciones: 60 comandos, 56 completions, 2.558 beats de salida y cuatro comandos en cuarentena. El digest de resultados por comando coincide; se rechazan 18 corrupciones de evidencia pública. También conserva la ventana sintética de 4.080 ns. El auditor deriva las obligaciones de comandos y CPL; no observa contadores internos del netlist ni acredita con esa ejecución sus ciclos internos de solapamiento.

Dos intentos anteriores de comparación exacta por ciclo quedaron rechazados. El primero difirió en `started` tras un reset: el reset de memoria del RTL cambió 33 ps antes del flanco de origen, mientras el modelo FDRE de AMD tiene un retardo de salida de 100 ps incluso sin SDF. El archivo instalado, su hash y la evidencia temporal están registrados en `RESET_PHASE_DIAG.json`. Una repetición con fases de liberación separadas de ese intervalo difirió después en `h_rready`, durante el flujo.

Por tanto, **no se declara equivalencia exacta por ciclo**. La regresión final usa el ready/valid propio del netlist y comprueba cada transacción y el payload completo, sin modificar sus bytes, las primitivas AMD, los relojes ni la fase original de reset. Conserva ambas tandas fallidas y elimina el ajuste de fase de la ejecución final. No usa SDF y no constituye equivalencia formal.

## Siguiente paso y criterios pendientes

F1 queda conservado como candidato experimental de dos frames, con W1 como referencia anterior. Antes de componerlo con el endpoint, revisar la ruta crítica de memoria y conseguir margen de temporización útil sin relajar frecuencias ni incertidumbres. Cualquier cambio del candidato deberá repetir los escenarios de errores, la contabilidad de reservas, la comparación nativa y la implementación física; conservar la regresión de transacciones del netlist y estudiar por separado las diferencias por ciclo todavía abiertas.

El [estudio C1 de frontera registrada](HBM_READ_CUT.md), realizado después de esta tanda, comparó dos colas adicionales y optimización física. Conservó F1 optimizado con +87 ps, mayor margen y menor área que las alternativas. Pasaron las transacciones del nuevo netlist y la auditoría DRC; la evidencia nativa se vincula por identidad de RTL y parámetros de IP. Sigue siendo margen interno aislado, sujeto a una nueva validación al integrar el sistema.

El resultado integrado seguirá requiriendo restricciones de E/S, endpoint PCIe, permisos/épocas peer, cómputo y validación de placa. El contrato peer debe distinguir CPL local de recepción remota y conservar obligaciones ante pérdida de enlace. Esta tanda no decide la arquitectura eléctrica DDR4/HBM ni demuestra que más reservas sigan mejorando el caudal.

## Evidencia

Raíz: `D:/AI/artifacts/ntrpu-read-frames/20261007T164158535082Z/`. Cada `repo/VARIANT.json` conserva la composición privada, sus fuentes y hashes. Las referencias `legacy2`, `full2`, `common2` y `short1/2` distinguen regresión anterior, nuevos escenarios y prepruebas sintéticas. `compile1/2` y `traffic1/2` identifican elaboración y tráfico nativos; `COMPARISON.json` registra su configuración HBM compartida y las ventanas comparables.

`physical2`, `drc2` y `routed2` corresponden al checkpoint conservado. `refined2` es solo una exploración de timing; los dos `netlist2_*` son diagnósticos fallidos de comparación por ciclo. Los tres `mut_*` documentan rechazos esperados y los dos `pilot_*`, el marcador inicial del lanzador. `netlist2.json` es un alias de `netlist2_guarded`, no otra ejecución.

Los recibos `BACKUP_*.json` identifican **20 tandas y dos snapshots** en el almacén por SHA-256 `D:/AI/artifacts/ntrpu-content-store/`. Se deduplican archivos completos; no se retienen ZIP de transferencia. `RESTORE_REPLAY.json` acredita la restauración de 1.506 archivos y repetición de los auditores de las veinte tandas, incluidos controles negativos y diagnósticos. Las copias temporales se eliminaron. Los snapshots tienen verificación adicional de contenido.

`REMOTE_FINAL_AUDIT.json` confirma hashes remotos/locales, fuentes estables y ausencia de procesos de este experimento al cierre. `HANDOFF/README.md` explica la recuperación y `HANDOFF_BACKUP.json` identifica su respaldo. El generador `prepare_ntrpu_read_frames.py` compone copias privadas; los bloques previamente calificados y la cola del agente principal no se modificaron. Los estados `READ_FRAMES_*` separan F1 de las evidencias anteriores.
