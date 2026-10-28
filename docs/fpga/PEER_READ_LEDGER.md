# Retención de contexto peer sobre el lector F1

## Estado y alcance — 2026-10-07

Primer subperfil experimental de P0: dos contextos por canal, un span de 64–4096 bytes por transferencia. Está compuesto con F1 de dos frames por canal y ocho créditos de fragmentos nativos. No incorpora la cola C1 descartada. **P0 completo, P1 y la placa permanecen abiertos.**

Fuentes: `fpga/experimental/ntrpu_peer_read_{pkg,lane,pair}.sv`. Los tipos empaquetados son interfaces internas de laboratorio, no una ABI PCIe. La composición con F1 se genera en una copia privada mediante `prepare_ntrpu_read_frames.py`; compilar directamente el `ntrpu_hbm_read_return.sv` original conserva su perfil anterior de un frame.

## Admisión y propiedad

Cada canal fija una región de origen, una región remota y una ventana. Conserva sesión, nodos, slots, generaciones, época del modelo, offsets y longitud. Antes del comando comprueba permisos/apertura, publicación en un intervalo contiguo, overflow de dirección, límites de región/ventana, alineación, rango HBM de 34 bits y frontera de 4 KiB. Rechaza destinos solapados mientras los contextos estén retenidos.

La secuencia de transferencia crece estrictamente por origen y no se reinicia al retirar una transferencia ni al reconfigurar las regiones. El `cmd_tag` usa esa secuencia como token local dentro de una sesión fija; no sustituye la identidad completa, almacenada por separado. Un rechazo tiene su propia salida con backpressure.

Una configuración nueva se acepta solamente sin contextos, con la misma sesión y generaciones crecientes. Tiene prioridad explícita frente a un descriptor simultáneo, evitando espera mutua. Si coincide con `revoke`, prevalece la revocación. `revoke` cierra admisiones y conserva los compromisos existentes. El propietario externo debe impedir escrituras concurrentes sobre los pesos y mantener la autorización remota; estos registros locales no implementan ese gestor distribuido.

## Fin local, DMA y confirmación remota

El payload sale con contexto y dirección de destino estables. `dma_address` es la base del span; el futuro DMA debe generar el avance de dirección por beat. El adaptador contrasta tag, cantidad de bytes y `last`, y distingue tres hechos:

1. El lector entregó todos los bytes y su CPL es correcto.
2. El DMA informó finalización correcta para la identidad y ventana esperadas.
3. El destino confirmó visibilidad remota con identidad, ventana y bytes correctos.

Los dos últimos avisos pueden llegar en distinto orden, una vez aceptado todo el payload. El contexto se libera al aceptar el resultado satisfactorio. Un resultado ya ofrecido es inmutable; si todas las condiciones se cumplieron antes de un stop posterior, ese resultado permanece válido.

Las reservas de contexto y los buffers de payload tienen costes distintos: un frame puede quedar libre tras CPL mientras su contexto sigue esperando ACK. Este perfil conservador limita también esos contextos a dos por canal. Antes de aumentar memoria de payload, medir la duración de esa retención y dimensionar por separado la tabla de metadatos; aproximadamente, `contextos × bytes_por_transferencia / tiempo_de_retención` limita el caudal por canal. Aquí no se ha medido caudal sostenido del transporte completo.

Stop, pérdida de enlace, error de lectura/DMA o aviso inválido detienen nuevas admisiones. Los VALID ofrecidos se mantienen. Se puede informar el error con cero bytes útiles, pero **retirar ese error no libera la región**. El drenaje tampoco lo hace. Solo se modela reset frío coordinado; un integrador debe garantizar aislamiento real y una sesión nueva antes de reutilizar identidades. No hay recuperación automática por timeout.

La parada del lector F1 es global: `|stop` también llega a ambos adaptadores. Desde `pair5`, parar cualquiera de los canales cierra la admisión de los dos y conserva todos sus contextos pendientes. Los checkpoints de esta tanda corresponden a `pair4`; la [caracterización posterior de `pair5`](PEER_READ_CUT.md) incluye el candidato con corte registrado y su netlist probado.

## Evidencia funcional

Raíz local: `D:/AI/artifacts/ntrpu-peer-read/20261007T205936932898Z/`.

| Tanda | Alcance | Resultado |
| --- | --- | --- |
| `unit2` | Contextos + frames F1, reloj común | 20 escenarios; 29 rechazos; 17 cuarentenas |
| `full1` | F1, CDC y convertidores AXI reales; HBM conductual | Mismos escenarios; pausa explícita de comando incluida |
| `pair1` | Módulo compuesto destinado a síntesis | Mismos escenarios; conexión directa de comando |
| `pair2` | Composición y reutilización de contextos | 20 transferencias satisfactorias, 17 resultados de error; 94.720 bytes comparados en el banco |
| `pair4` | Configuración concurrente; base de caracterización física | Mismos resultados, más prioridad configuración/descriptor y configuración/revocación |
| `pair5` | Revisión vigente; parada global con ambos canales activos | 21 escenarios, 29 rechazos, 18 escenarios de cuarentena; 20 resultados satisfactorios y 19 de error; 94.720 bytes comparados |

La ampliación reutiliza ambos slots con longitudes de 1, 8, 9 y 64 beats. Incluye ACK fuera de orden entre transferencias, pausas de datos/resultados, intento de reconfiguración retenida, replay activo/retirado, error en el último fragmento y stop/pérdida de enlace en cinco fronteras. El banco directo `pair` no introduce la puerta artificial de pausa de comando usada en `unit/full`.

El auditor independiente reproduce payload y obligaciones desde la traza pública y rechaza seis alteraciones de evidencia. Seis mutantes RTL fallaron por la causa esperada: omitir ACK (`DMA_IS_NOT_ACK`), liberar tras error (`FAILED_LEASE_RELEASED`), desactivar replay (`REJECT_STATUS`), bloquear configuración/descriptor simultáneos (`CONFIG_PRIORITY_DEADLOCK`), perder una revocación concurrente (`CONFIG_REVOKE_PRIORITY`) y conservar parada individual frente al lector global (`GLOBAL_STOP_ADMISSION`). El primer piloto tuvo una carrera del driver del banco y terminó por watchdog; está archivado, sin crédito funcional. El lanzador exige la marca de éxito además del código de salida de XSIM.

`check_ntrpu_peer_read_oracle.py` reproduce las 20 transferencias satisfactorias de `pair2`, `pair4` y `pair5` mediante el modelo de transporte existente: 37.376 bytes exactos, confirmación remota y retirada de leases. Los 25 tests del modelo también pasan. Este mapeo de fixtures no demuestra equivalencia general, ni incorpora los fallos RTL al modelo.

## Caracterización física: timing abierto

Vivado 2026.1, `xcvu47p-fsvh2892-2-e`, implementación fuera de contexto, periodos de 5 ns y 2,222 ns; incertidumbre nominal de setup 0,100 ns y hold 0,050 ns. **Estos resultados pertenecen a `pair4`; los de `pair5` se registran por separado en [PEER_READ_CUT.md](PEER_READ_CUT.md).**

| Tanda | Setup transporte | Setup memoria | Hold transporte / memoria | Endpoints con fallo de setup |
| --- | ---: | ---: | ---: | ---: |
| `resume3` | +45 ps | −172 ps | +10 / +5 ps | 1.585 |
| `refined3` | +70 ps | −146 ps | +10 / +10 ps | 881 |

Ambas rutas usan 16.452 LUT y 18.083 registros, sin BRAM36; tienen 25.926 redes completamente ruteadas, cero fallos de hold/pulse width y cero endpoints internos sin restricciones. Pasan los ocho controles de bus skew y las comprobaciones CDC/READY registrado. Permanecen 7.235 entradas y 5.875 salidas sin retardos externos: no es cierre de placa ni se ha simulado este netlist posterior al ruteo.

`refined3` endureció temporalmente la incertidumbre de setup de memoria a 0,200 ns durante optimización/ruteo y restauró 0,100 ns antes de evaluar y guardar el resultado nominal. El auditor verifica esa restauración; no se relajaron periodos ni hold. Mejora 26 ps, pero **ninguna ruta cumple timing**.

El peor camino sigue entre `first_word_reg` del convertidor de ancho AXI y la habilitación de escritura de la FIFO de retorno CDC: seis niveles lógicos y aproximadamente 72 % del retardo de datos en interconexión tras el ajuste. Conviene evaluar un corte registrado en esa frontera sobre la composición completa, conservando créditos y backpressure. El resultado previo de C1 sobre F1 aislado no decide su utilidad en esta composición.

La comparación posterior, ya sobre la revisión `pair5` con parada global, está en [PEER_READ_CUT.md](PEER_READ_CUT.md). Sus tandas distinguen control sin cola y candidato con corte registrado; no atribuirles los checkpoints `pair4` de esta tabla.

La auditoría detallada `drc3` se aplica únicamente al checkpoint `resume3`: seis advertencias PDCN-1569 por entradas LUT sin uso lógico y una RTSTAT-10. Clasificó sus 5.933 redes (5.831 puertos externos sin carga interna y 102 sin cargas lógicas hoja), sin casos pendientes ni waivers. El diseño conserva esas advertencias; no se declara DRC limpio. La auditoría de objetos no se transfiere al checkpoint refinado.

Evidencia vinculada por SHA-256: resultado `pair5` `1883f53ce8145b29c5fb7f2437fe6feaf4a9ffe0bebca1a5cba3cf2481f64445`; DCP `resume3` `b84143bdbde4346c3621d41c62b9ccc757ae738ae8bc2e13d549a0f3cb6e53a9`; DCP `refined3` `8863ddcdccfbb186cb5a05f2106dbe9b45bdffcf0095effd91656890365a0f49`. Las revisiones y manifiestos de respaldo están bajo la raíz local indicada arriba.

## Pendientes antes de cerrar P0/P1

- **Identidad compartida entre canales:** este perfil histórico asigna orígenes lógicos 0 y 1. La nueva [admisión compartida](PEER_READ_SHARED.md) ya valida funcionalmente un watermark por origen, conflictos cruzados y cuatro reservas atómicas. Su composición aún requiere timing/HBM nativa, secuenciador y selección por localidad. Instanciar dos ledgers con el mismo origen sin ese control sigue permitiendo replay cruzado.
- **Gestor de regiones:** autorización remota efectiva, publicación por bloques, exclusión con escritores y revocación/aislamiento coordinados. Las leases de este banco proceden de un controlador confiable.
- **Transferencias mayores:** dividirlas en spans manteniendo una identidad y un resultado agregado; cubrir fallos tardíos sin éxito parcial. Actualmente se rechazan tamaños mayores de 4 KiB.
- **Endpoint real:** adaptar estos eventos al DMA y a las interfaces/XCI de PCIe efectivas. No existe aquí un BAR unificado, coherencia automática, ni una confirmación remota derivada de AXI B.
- **Ampliación del cierre físico:** el subperfil posterior con corte registrado pasa timing interno y transacciones del netlist; falta validarlo con HBM/PCIe nativos y restricciones de placa. El margen actual es estrecho y debe revisarse al ampliar la composición. La prueba conductual no acredita caudal en hardware ni reutiliza los +87 ps del lector aislado.

La siguiente composición debe recibir descriptores en una cola ordenada por nodo, comprobar y reservar contexto de forma atómica, y después seleccionar el canal HBM. Así, la secuencia pertenece al nodo y el puerto solo expresa localidad de memoria. La retención de metadatos hasta ACK se separará del retorno de los buffers locales; será necesario probar replay cruzado, dos descriptores simultáneos, conflictos con escritores y progreso bajo pausas antes de ampliar el pool.

Los respaldos por tanda usan el almacén SHA-256 compartido; no retienen ZIP ni copias completas de Vivado. Cada tanda se restaura y se verifica; las aprobadas repiten el auditor sobre la restauración.

La entrega consolidada usa el manifiesto `peer-read-20261007T205936932898Z-handoff` en `D:/AI/artifacts/ntrpu-content-store`. Incluye fuentes, copia privada reconstruible, revisiones y manifiestos de las tandas. Los DCP y reportes completos se recuperan mediante esos manifiestos; el almacén de objetos debe conservarse junto con ellos. `FINAL_REMOTE_REVIEW.json` registra 22 tandas con resultado idéntico al local y ninguna ejecución propia pendiente. No se modificaron la cola principal ni los RTL de referencia.

## Reconstrucción aislada

Usar Linux con Vivado 2026.1 inicializado. `native_binding.sv` y `native_stub.sv` son fixtures de la tanda R0/R1 conservada, no archivos creados a mano. Todas las carpetas de salida deben ser nuevas.

```bash
python3 -B tools/prepare_ntrpu_peer_read.py /tmp/peer-src \
  --binding /ruta/evidencia/native_binding.sv \
  --stub /ruta/evidencia/native_stub.sv
python3 -B /tmp/peer-src/tools/run_ntrpu_peer_read.py /tmp/peer-sim --full --pair
python3 -B /tmp/peer-src/tools/check_ntrpu_peer_read.py /tmp/peer-sim > /tmp/peer-review.json
python3 -B /tmp/peer-src/tools/check_ntrpu_peer_read_oracle.py /tmp/peer-sim/suite/simulation.log
python3 -B /tmp/peer-src/tools/run_ntrpu_peer_read_physical.py /tmp/peer-route \
  --prior /tmp/peer-sim --review /tmp/peer-review.json
python3 -B /tmp/peer-src/tools/check_ntrpu_peer_read_physical.py /tmp/peer-route
```

El auditor físico puede devolver `OPEN` aunque las evidencias sean válidas. Un código cero del auditor no equivale por sí solo a timing cerrado. El flujo fija dos CPU y prioridad baja; no modifica la cola principal.

La caracterización compara también una copia con la directiva de síntesis `RuntimeOptimized`, consultada en la instalación antes de usarla. AMD indica que reduce trabajo de optimización para acortar la síntesis; por eso sus resultados se registran como una estrategia distinta, conservando RTL y restricciones de reloj. [Referencia AMD de síntesis](https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Using-Synthesis-Settings).


La síntesis normal y `RuntimeOptimized` excedieron el límite global de 20 minutos durante implementación. Se conservaron sus checkpoints de síntesis; no se consideran resultados ruteados. `run_ntrpu_peer_read_resume.py SALIDA_NUEVA --prior SALIDA_ANTERIOR` permite continuar desde un checkpoint de síntesis verificado, con cuatro CPU y nuevos checkpoints tras optimización y colocación. Mantiene restricciones y verifica que el donante no cambie.
