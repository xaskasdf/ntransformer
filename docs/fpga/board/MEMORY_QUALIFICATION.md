# Calificación del primer canal DDR4

Estado 2026-10-02: **V2 bloqueado y V3 bloqueado por dependencia**. La ficha legible por herramientas es [MEMORY_CONFIG.json](MEMORY_CONFIG.json). Se conserva el objetivo de dos RDIMM de 64 GiB, x4, dos ranks cada uno, x64 útil más ECC y 128 GiB por canal.

## Resultado reproducible

Vivado 2026.1, DDR4 2.2 rev30, rechaza `MTA36ASF8G72PZ-3G2F1` con `IP_Flow 19-3461`. La ejecución `20261002T091508Z-V2-exact-dimm-r2` aísla ese rechazo usando `NO_DM_NO_DBI` y un reloj de referencia exploratorio de 4000 ps. El primer intento también tenía dos selectores incorrectos; ambos intentos se conservan. La IP puede restaurar valores anteriores tras fallar: **sus XCI de diagnóstico no representan un canal de 128 GiB aprobado**.

```bash
vivado -mode batch -source tools/ntrpu_board_ddr_candidate.tcl \
  -tclargs xcvu5p-flva2104-2-i build/NUEVA_TANDA_DDR
```

La tabla instalada ofrece `M393AAK40B41-CTC`, de 128 GB con chips de 8 Gb, dos ranks y stack height 4. Es otra topología, pendiente de calificación propia; no reemplaza al Micron. No se ha demostrado que el objetivo sea físicamente imposible.

## Datos necesarios para una parte personalizada

Obtener la ficha exacta con revisión/proceso, SPD completo y modelo de chips/módulo/RCD. Completar filas/columnas, bancos y grupos, mirroring, ranks/slots, señales CS/CKE/ODT, timings, registros RCD, condiciones eléctricas, reloj y reset. Los datos generales de la [lista Micron AIC](https://my.micron.com/partners/odm-partners/aic) no completan esos campos. Las consultas públicas de la [ficha](https://www.micron.com/products/memory/dram-modules/rdimm/part-catalog/part-detail/mta36asf8g72pz-3g2f1) y del [SPD](https://www.micron.com/products/memory/dram-modules/rdimm/part-catalog/part-detail/spd-data/mta36asf8g72pz-3g2f1) devolvieron `Invalid Partnumber` en sus API; las respuestas quedaron guardadas como evidencia, sin inventar bytes SPD.

[PG150, Basic Tab](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Basic-Tab?contentId=MGveBVBHGBtHBQNV7JSeVA) permite CSV personalizados, pero exige incorporar manualmente el modelo y adaptar el banco para su simulación. Partir del esquema de `mem_v1_4/csv/ddr4_sdram/`, incluyendo timings y registros; no editar la instalación AMD ni rellenar campos desconocidos copiando otro DIMM.

## Evidencia documental y exploración de geometría

El [addendum Micron, Rev. F 08/2021](https://www.rxelectronics.si/datasheet/33/mta36asf8g72pz-3g2b2.pdf), páginas 1–2, confirma 18 bits de fila, 10 de columna, dos de banco y dos de grupo; chips MT40A4G4 de 16 Gb x4 y dos ranks. Cita explícitamente el sufijo `-3G2F1` como ejemplo. El PDF es de Micron, obtenido de un espejo de distribución; SHA-256 `e71fd83003c6857b31fed567a663c361113808817702510694f159c99800d8eb`. No contiene SPD ni modelo. Con x64 útil, esta geometría da 32 GiB/rank, 64 GiB/módulo y 128 GiB con dos slots.

`tools/ntrpu_board_ddr_geometry.tcl` y `ddr4_geometry_exploratory.csv` exploran **solo el soporte de esa geometría**, conservando ECC, dos slots y 37 bits AXI. El nombre `NTRPU_RDIMM64_GEOMETRY_ONLY` evita atribuirle calificación al Micron. La base de timings es la fila AMD `16Gb:sg125E:x4` en `mem_v1_4/csv/ddr4_sdram/timingparams.csv`; los cuatro parámetros exclusivos de 3DS son cero para stack height 1. La tanda inicial usó CL11/CWL9 y tRCD/tRP=13,75 ns. Para la simulación actual se elevaron **CL a 12 y tRCD/tRP a 15 ns** de forma conservadora frente al modelo instalado; CWL9, CA mirror=1, CS/CKE/ODT=2 y CK=1 por módulo siguen siendo supuestos genéricos. La tasa es 1250 ps. **No son timings ni registros certificados para F1**; no usar este CSV para fabricar ni cerrar V2/V3. `ddr4_custom_part_rc.csv` contiene reglas de validación, no registros RCD.

La tanda `20261002T100234Z-V2-generic-geometry` probó inicialmente 833 ps con la fila `16Gb:sg083E:x4`, CL16/CWL12. Vivado rechazó esa configuración: `Slot=Dual` y `DataWidth=72` exigen 1250–1600 ps; la parte elegida exige 938–1600 ps. Esto invalida DDR4-2400 para **ese perfil exploratorio en esa parte y versión**. Se abrió otra tanda a 1250 ps (DDR4-1600), sin reducir capacidad, slots ni ECC. La compatibilidad del DIMM físico sigue pendiente. El techo analítico de cuatro canales x64 a 1600 MT/s sería 51,2 GB/s antes de sobrecargas; no es ancho de banda medido.

`20261002T100600Z-V2-generic-1600` terminó con salida 0 y targets generados en 49,167 s. El wrapper contiene `ROW_WIDTH=18`, `COL_WIDTH=10`, cuatro ranks, `ECC="ON"`, slots `0x03/0x0c`, AXI de 37/512 bits y 72 DQ/18 pares DQS. `tools/check_ntrpu_ddr_geometry.py` deriva 137438953472 bytes útiles y rechaza cuatro mutaciones de fila, dirección, slot y ECC. Es una comprobación del RTL generado; no prueba calibración ni ausencia de alias en transferencias. Respaldo verificado: SHA-256 `e305a1589ba5091920782e1ea14af3bbad1a0310a66abbbeef276041d269403c`, 32.442.559 bytes.

## Condiciones de entrada y salida de V3

**Comprobación previa del banco:** `20261002T102039Z-V2-generic-example` generó el ejemplo, pero `imports/sim_tb_top.sv` conserva `CONFIGURED_DENSITY=_4G`; otro wrapper conserva `DDR4_4G_X8` y `DDR4_750_Timing`. No ejecutarlo sin revisar y adaptar densidad, x4, timings y modelo RDIMM/RCD en una copia nueva. El paquete incluye `_16G`, pero su presencia no prueba correspondencia con F1. No se inició simulación con esos valores incorrectos.

La revisión adicional confirma un `$finish` inmediato con `SIMULATIONS NOT SUPPORTED FOR CUSTOM PART` (líneas 403–407 del banco). Un proceso que termine correctamente con ese ejemplo **no constituye PASS**. El reloj de entrada real sí es correcto: el `always` usa `4000/2.0 ps`, coherente con la IP; `SYSCLK_PERIOD=tCK` es un parámetro sin uso y no prueba un error de reloj. Antes de una exploración genérica, revisar CKE/ODT y clocks por slot: el banco conecta los dos bits inferiores de CKE/ODT y el clock del slot 0 a ambos módulos. No quitar el bloqueo y simular sin resolver estos puntos y añadir comprobación independiente.

Antes de generar: comprobar dos slots, cuatro ranks totales, ECC activo, 137438953472 bytes útiles y AXI de 37 bits de dirección/512 de datos. Los rangos estáticos de `component.xml` no son definitivos: el Tcl de la IP calcula dinámicamente el ancho según geometría, slots y ECC. No forzar 36 bits ni truncar direcciones.

Después: generar el ejemplo/modelo exactos, probar primero bancos 44–46 y luego 64–66. Exigir calibración, lectura/escritura comparada, máscaras y ECC, cambio de slot/rank, backpressure, direcciones superiores a 4 GiB y cercanas al final de 128 GiB, y ausencia de alias. Revisar DQ/DQS x4, clocks, reset y todas las advertencias. Coordinar AM17/AP17 con V4 antes de expandir. La exploración siguiente cubre parte de estos criterios con memoria genérica; no cierra la dependencia exacta ni la colocación.

## Simulación genérica de 128 GiB

`20261002T104404Z-V3-generic-capacity` terminó con PASS funcional en 215,211 s de VM. `tools/ntrpu_board_ddr_capacity_sim.tcl` copia el ejemplo a una carpeta nueva, regenera la IP con el CSV actual y utiliza `fpga/tb/ntrpu_ddr_capacity_tb.sv`. Cada módulo recibe su propio CK y los bits correspondientes de CS/CKE/ODT. El modelo usa `_16G`, x4 y dos ranks por slot. Los selectores `DDR4_16G_X4`, `DDR4_1250_Timing` y `FIXED_1600` siguen la plantilla AMD `model_wrapper.ttcl`; no se modifica el modelo cifrado ni se desactivan sus comprobaciones.

Se escribieron 36 direcciones con patrones distintos y se leyeron en orden inverso, incluyendo cada bit 6–36 y los extremos de 4/32/64/128 GiB. Después se aplicaron 12 escrituras con máscara alternada de bytes y se volvieron a comparar las 36 ubicaciones. Dos ráfagas de ocho beats terminan en 4 GiB y 128 GiB. El banco comprueba datos, respuestas e IDs AXI, RLAST, estabilidad bajo backpressure y actividad de los cuatro ranks. Resultado: **64 beats escritos, 88 leídos, 5632 bytes comparados**, calibración a 28,051098 µs y final a 46,788498 µs. No se recorre cada byte de la capacidad ni se mide throughput sostenido. La calibración acelerada no mide el arranque físico.

`tools/check_ntrpu_ddr_sim.py` exige marcas, conteos, ranks, pausas y tiempos consistentes; rechaza errores de modelo y finales prematuros. Pasó el log real y rechazó 15 mutaciones negativas. El wrapper generado volvió a pasar el comprobador de geometría. Esta primera tanda ejercita ECC mediante actualizaciones parciales; la inyección se ejecutó después, como se describe a continuación. La recuperación de reset sigue pendiente.

Límites conservados: el modelo genérico `_16G` usa tRFC1=350 ns, mientras el controlador mantiene 550 ns; por tanto, el modelo no certifica todos los márgenes del componente. XSIM avisa de dos assertions AMD no soportadas, índices fuera de rango en ramas del modelo/calibrador y reemplazo del MicroBlaze por el modelo de simulación. La rama `rank[2]` corresponde al caso RANKS>4, no seleccionado aquí; los avisos internos del modelo cifrado quedan sin resolver. No hay errores de comparación ni diagnósticos de violación de timing en esta ejecución. Los avisos completos, plantillas, parámetros, fuentes y hashes se conservan junto al resultado.

## Inyección ECC por registros

`20261002T110203Z-V3-generic-ecc` repitió el banco con `ECC_TEST=1` y terminó **PASS en 207,339 s**. Usa el mapa público de `ddr4_v2_2_axi_ctrl_top.sv`: estado W1C en `0x000`, IRQ en `0x004`, habilitación en `0x008`, contador CE en `0x00c` e inyección XOR de los bits 31:0 de la próxima escritura en `0x300`. Las fuentes AMD y sus hashes se incluyen en el respaldo.

- Un bit inyectado en `0x100001000`: datos corregidos, respuesta OKAY, estado CE=2, contador=1 e IRQ activa.
- Dos bits inyectados a 1024 bytes del extremo de 128 GiB: respuesta SLVERR=2, estado UE=1 e IRQ; el contador CE sigue en 1.
- Después de cada caso: limpiar estado/IRQ mediante W1C, reescribir datos y comparar una lectura sin error. También se verificó el borrado del contador CE.

No se fuerza ninguna señal interna ni se modifica el modelo cifrado. El registro de habilitación se comprueba antes del tráfico y el estado inicial debe estar limpio. Totales, incluyendo la capacidad dispersa: **68 beats escritos, 92 leídos, 5824 bytes comparados**, 4071 ciclos UI; final a 48,408498 µs simulados. La lectura con error no corregible no cuenta como datos comparados. `check_ntrpu_ddr_sim.py --ecc` exige estas pruebas y rechazó siete mutaciones adicionales. Pendientes: reset/recalibración con tráfico, más posiciones y patrones de fallos, calificación exacta del DIMM y colocación de pines.

## Reset con lectura en curso: revisión rechazada

`20261002T112359Z-V3-generic-reset` y `20261002T113227Z-V3-generic-reset-r2` interrumpen una lectura de ocho beats después de comparar dos. El banco bloquea las respuestas de la época anterior, espera recalibración, comprueba registros ECC y reescribe/lee datos. Sus marcas propias son correctas, pero **ambas ejecuciones contienen 558 errores del modelo DDR y son FAIL** según `check_ntrpu_ddr_sim.py --ecc --reset`. No aceptar el código de proceso 0 como validación de recuperación.

La segunda tanda alarga el reset externo a 1 µs. Ambos RCD entregan reset bajo desde 48,706098 hasta 51,303098 µs (2,597 µs); el primer error aparece a 52,652686 µs, con bancos abiertos antes de LMR. Recalibración a 77,663098 µs y fin a 78,200498 µs. Los conteos observados son 84 beats W, 102 R y 6464 bytes comparados; se registran como diagnóstico, no como resultado aprobado. No se modifica ni fuerza el modelo protegido. Se investiga mediante trazas de señales públicas.

La síntesis y la exploración de bancos se documentan en [PIN_PLANNING.md](PIN_PLANNING.md); no califican el DIMM ni resuelven el fallo de reset.

`20261002T115825Z-V3-reset-component-trace` reutiliza el ejecutable elaborado y registra las interfaces públicas `iDDR4` de los 72 chips. Todos reciben reset bajo durante 2,597 µs, 1470 cambios de reloj (735 flancos positivos) y CKE bajo de 48,718186 a 52,080186 µs. La pausa máxima entre flancos durante reset es 1,678875 µs; el reloj reanuda antes de liberar reset. Por tanto, un reset desconectado o la falta completa de reloj durante el pulso no explican el fallo observado. Queda pendiente aislar la secuencia y el comportamiento del modelo; no se declara defecto del proveedor sin esa comprobación.

`tools/inspect_ntrpu_ddr_reset_trace.py --components` exige 72 interfaces y conserva medidas y hash del VCD. Cuatro comprobaciones negativas rechazan unidades incorrectas, falta de reset del sistema, pulso sin liberación y cobertura de chips ausente. La ventana termina deliberadamente a 55 µs; su `INCOMPLETE` es propio del diagnóstico, no una excepción para el verificador funcional.

## Reproducción aislada del reset

La traza `20261002T122011Z-V3-reset-command-trace-r2` añade comandos públicos de un chip y permite extraer la secuencia MRS/ZQ mediante `tools/extract_ntrpu_ddr_commands.py`. No interpreta una ventana diagnóstica como validación funcional.

`fpga/tb/ntrpu_ddr_model_reset_tb.sv` reproduce esa secuencia sobre un único modelo genérico x4 de 16 Gb, sin MIG, RCD ni AXI y con CK continuo de 1250 ps. `20261002T123042Z-V3-model-reset-widths` ejecuta cuatro procesos independientes: con/sin fila abierta y reset bajo de 2,6/200 µs. Los cuatro arranques iniciales no presentan errores. Ambos casos con fila abierta producen siete errores LMR y uno ZQ después del reset; los otros dos no presentan esos errores. Alargar el pulso no elimina la reproducción.

El respaldo compara los hashes de todos los archivos del modelo con su origen y conserva logs, marcas y errores en `review.json`. No se modifica ni fuerza el modelo. El resultado acota el fallo a la secuencia/modelo de chip, pero no demuestra un defecto de silicio ni acredita reinicialización completa o conservación de datos en los casos silenciosos. Resolver el contrato de reset con un modelo correspondiente y soportado sigue siendo requisito; el ensayo completo continúa **FAIL**.

`20261002T130332Z-V3-model-reset-full-delays` elimina las esperas abreviadas: reset inicial de 200 µs y espera RESET_n→CKE de 500 µs tanto al arrancar como al reiniciar; pulso de reinicio de 2,6 µs. La secuencia se contrasta con la ficha Micron `16gb_ddr4_dram.pdf`, Rev. H 8/2021, páginas 37–40 ([documento del fabricante en espejo](https://device.report/m/92f06ed489b68537693753a3d634a7c26f183810e7ebb89a07b651d8b41e94da)). La descarga directa del PDF respondió 403; no se afirma tener una copia local.

Ambos casos completan la inicialización inicial a 700,728125 µs. El caso con fila abierta conserva siete errores LMR y uno ZQ tras reset; el caso sin fila abierta no los produce. Finalizan a 1208,655/1208,635 µs, respectivamente, sin una segunda marca de inicialización del modelo. Duración de VM: 76,202 s. La aceleración de RESET→CKE no explica por sí sola el fallo. Los archivos del modelo mantienen sus hashes originales; no se añadió precharge para ocultarlo. El [catálogo de Micron](https://in.micron.com/products/memory/dram-components/ddr4-sdram/part-catalog/part-detail/mt40a2g16tbb-062e-f) lista un modelo SystemVerilog con acceso restringido; no sustituye la calificación del módulo objetivo.

### Fase de reset y separación de CKE

La tanda `20261004T075538Z-V3-model-reset-phase` ejecuta ocho procesos XSIM
independientes en 231,535 s. Conserva CK de 1250 ps, reset frío de 200 µs,
espera RESET→CKE de 500 µs y reset posterior de 2,6 µs. Compara reset en
el flanco descendente y 313 ps después, con CKE bajando cero o 20 ns antes.
En el caso de cero tiempo hay un ciclo delta entre ambas asignaciones;
el intento original con asignaciones contiguas sigue conservado.

| Fase después de CK↓ | CKE antes de RESET↓ | Sin fila abierta | Con una fila abierta |
| --- | --- | --- | --- |
| 0 ps | 0 ns | 0 errores | 7 LMR + 1 ZQ |
| 0 ps | 20 ns | 0 errores | 7 LMR + 1 ZQ |
| 313 ps | 0 ns | 0 errores | 7 LMR + 1 ZQ |
| 313 ps | 20 ns | 0 errores | 7 LMR + 1 ZQ |

El revisor reconstruye tiempos y flancos desde muestras de los pines
públicos, comprueba los dos trenes MRS/ZQ y la presencia o ausencia de ACT.
Cuenta explícitamente las muestras `$strobe` duplicadas del ciclo delta.
Compara ambos logs por proceso y rechaza 96 alteraciones de evidencia.
Los 35 archivos copiados del modelo conservan sus hashes originales.
La tanda completa está respaldada localmente y en NFS, con 101 archivos.

Mover las señales de los flancos y adelantar CKE no elimina esta reproducción.
Los ocho procesos muestran una sola marca de inicialización; los controles
silenciosos **no califican reinicialización ni retención de datos**. El reset
DDR completo mantiene su estado FAIL. Se necesita resolver el contrato con
un modelo correspondiente y soportado; estos resultados no prueban un defecto
de silicio ni autorizan a suprimir diagnósticos del modelo.

Para la futura conexión de MIG, [PG150: Reset Sequence](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Reset-Sequence)
distingue `sys_rst`, `ui_clk_sync_rst` y la finalización de calibración.
La prueba de arranque frío del camino FFN→DDR puede avanzar por separado;
debe mantener bloqueado el tráfico hasta la calibración y conservar este
fallo como dependencia abierta de recuperación.

```bash
vivado -mode batch -source tools/ntrpu_board_ddr_model_phase.tcl \
  -tclargs MODELO_ORIGINAL NUEVA_CARPETA
python3 tools/check_ntrpu_ddr_phase.py NUEVA_CARPETA/row1_phase313_lead20000.log \
  --row 1 --phase 313 --lead 20000
```

El estado `DIAGNOSTIC_TRACE_VALID` acredita la reconstrucción de la traza.
El campo separado `WARM_RESET_MODEL_ERRORS_PRESENT` conserva el fallo funcional.

### RESET antes de bajar CKE

La tanda `20261004T142430Z-V3-model-reset-lag` añade cuatro procesos aislados
en 129,017 s. RESET baja 313 ps después de CK↓ y CKE baja 12,5 ns o 2 µs
después; el pulso RESET conserva 2,6 µs. CKE queda bajo durante 2,5875 µs
o 600 ns antes de liberar RESET. Se conservan el arranque de 200 µs, la
espera de 500 µs y los dos trenes MRS/ZQ. Es una variación del orden de
señales, no una reproducción completa del controlador o del RCD.

La ficha Micron de 16 Gb, Rev. H, página 40, exige CKE bajo al menos 10 ns
antes de liberar RESET con alimentación estable; no exige bajarlo antes
de iniciar RESET. [Documento del fabricante en espejo](https://device.report/m/92f06ed489b68537693753a3d634a7c26f183810e7ebb89a07b651d8b41e94da).

Ambos casos con fila abierta mantienen **7 errores MRS + 1 ZQ**; los dos
controles sin fila abierta no producen errores. Los cuatro conservan una
sola marca de inicialización. Este orden tampoco elimina la reproducción;
el ensayo completo de recuperación DDR sigue **FAIL**.

La auditoría posterior de conexiones públicas coteja los tres bancos
aislados con el `ddr4_rank.sv` original: ambos dejan `model_enable` abierto
(el wrapper lo identifica como inout), mantienen PWR/ZQ/VREF altos y usan
el mismo orden CK_t/CK_c. Los hashes del modelo coinciden con la tanda
respaldada. No hay evidencia para corregir esa conexión; este cotejo no
explica el estado interno del modelo ni califica reset. No se lanzó otra
simulación variando arbitrariamente esos pines.

`check_ntrpu_ddr_lag.py` reconstruye siete transiciones públicas de pines,
16/17 comandos y el final real de cada proceso. La revisión compara los
dos logs de cada caso y rechaza 50 alteraciones. Los 35 archivos del modelo
mantienen sus hashes originales. Respaldo local/NFS verificado: 93 archivos,
SHA-256 `2ebd1b3caf88b7de2329f3db52564ff49810ac97df09509a00e29a8f2d04e342`.
No se añadieron precharge, fuerzas de señales ni supresión de diagnósticos;
los casos silenciosos no prueban reinicialización ni conservación de datos.

## Reconsulta pública de SPD con control positivo — 2026-10-06

Se consultó la ruta que publica la [herramienta SPD de Micron](https://www.micron.com/sales-support/design-tools/serial-presence-detect).
La búsqueda de `MTA36ASF8G72PZ-3G2F1` devolvió una lista vacía. La ruta
de datos declarada por su [página exacta](https://www.micron.com/products/memory/dram-modules/rdimm/part-catalog/part-detail/spd-data/mta36asf8g72pz-3g2f1)
respondió HTTP 200 con `Invalid Partnumber`, sin bytes SPD.

Como control, la misma herramienta encontró `MTA18ASF1G72PZ-2G3B1` y
su [exportación](https://www.micron.com/products/obsolete/obsolete-rdimm/part-catalog/part-detail/spd-data/mta18asf1g72pz-2g3b1)
entregó tablas para 13 códigos de proceso. Ese módulo es distinto:
sus bytes no se usan como parámetros del candidato F1. La comparación
acota el fallo observado al acceso público de esa parte; no demuestra
ausencia de datos en el fabricante.

Las respuestas, rutas y hashes están en
`build/cloud-setup/20261002-board/micron-spd-public-recheck-20261006.json`
y sus ocho archivos referenciados. No se modificó la configuración MIG.
V2 sigue requiriendo SPD, revisión y modelo funcional del DIMM exacto.
