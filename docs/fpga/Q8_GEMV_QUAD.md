# GEMV experimental con cuatro MAC

Actualizado el 2026-10-04 UTC. Variante aislada motivada por el
[perfil de la FFN](FFN_PERFORMANCE.md): los dos MAC originales permanecían
ocupados durante 86–88% de cada etapa matricial.

## Arquitectura y selección

`ntrpu_q8_gemv_quad.sv` asigna `fila mod 4` a cuatro instancias del mismo
MAC HLS calificado. Conserva layout 2, orden K y acumulación binary64 por
fila. Dos copias de activaciones, con escrituras replicadas y dos puertos
de lectura cada una, alimentan los cuatro motores. No se resetea el
contenido BRAM; la máscara de validez impide reutilizar datos antiguos.

`ntrpu_q8_gemv_quad_memory.sv` conserva la interfaz de lectura AXI del
wrapper dual. La finalización exige drenar todos los MAC iniciados ante
error/cancelación. El reset requiere detener/resetear también al par AXI.
El diseño dual sigue siendo la configuración predeterminada.

## Evidencia funcional

| Tanda | Resultado revisado |
| --- | --- |
| `20261003T220450Z-V7-gemv-{quad,dual}-extended` | Numérica exacta; cobertura de reset insuficiente. Conservadas como diagnóstico. |
| `20261003T220934Z-V7-gemv-{quad,dual}-started-reset` | Cada variante: 123 casos, 474 binary64 exactos, 12 resets y 39 alteraciones de evidencia rechazadas. |
| `20261003T221203Z-V7-gemv-quad-brandon` | 174 trabajos, 5.376 binary64 exactos y 33 alteraciones rechazadas; 599,385 s de Vivado. |
| `20261003T230624Z-V7-gemv-quad-mac-faults-r3` | 105 trabajos, 216 binary64 exactos y 66 alteraciones rechazadas; errores HLS reales, cancelación, timeout y reset con cuatro motores ocupados. |

El primer banco podía resetear cuando el último motor tenía el arranque
reservado, antes del flanco `ap_start` de HLS. El banco corregido exige
observar todos los motores en ejecución y contar cada arranque antes del
fallo. Cubre filas 1/2/3/4/5/31/32, distintos K, error de lectura de pesos,
cancelación, reset y recuperación. El banco adicional `sim_quad_faults`
inyecta una escala inválida en la entrada del HLS y observa 30 errores
reales `INVALID_SCALE`, siempre con otro motor ocupado. Un wrapper de
simulación retiene las completions reales: nueve casos alcanzan el
watchdog de 32 ciclos, seis cancelan y seis resetean antes de liberarlas.
Cada fallo lleva un trabajo sano posterior. No se fuerza el contador ni
el detector de fallos; tampoco se prueba recuperación de un reloj físico
detenido. El watchdog de producción permanece en 65.536 ciclos.

Tres intentos diagnósticos están conservados. Los dos primeros muestran
que una escala Q8 externa inválida se rechaza antes del HLS; con el
watchdog reducido, el drenaje del decoder termina clasificado como timeout.
Esa clasificación queda pendiente de estudio. El tercero falló porque
el estímulo no solapaba otro MAC para cinco filas; se corrigió la selección
del bloque, sin cambiar el RTL sintetizable.

## Ciclos y recursos medidos

Comparación con `20261003T043551Z-V7-gemv-memory-fixture`, mismas imágenes
y semillas 1/7/42. Las pausas aleatorias pueden diferir al cambiar la
duración. Se compara latencia simulada; no tiempo de ejecución de XSIM.

| Trabajo | Factor de reducción de ciclos dual/quad |
| --- | ---: |
| 54 proyecciones Brandon por semilla | 1,753–1,865× |
| Forma sintética 32 × 2048 | 1,869–1,874× |
| Forma sintética 32 × 8192 | 1,877–1,881× |
| Forma sintética 1 × 8192 | 0,997–1,001× |
| Forma sintética 31 × 2048 | 1,875–1,877× |

Son proyecciones independientes con entradas sintéticas y pesos Brandon,
no una cadena FFN ni inferencia Llama. La síntesis
`20261003T220808Z-V7-gemv-quad-synth` conserva cuatro MAC de 26 DSP cada
uno: 25.040 LUT, 17.784 FF, 104 DSP, 46 RAMB36 y dos RAMB18; nueve RAMB36
por copia de activaciones, cero LUTRAM/latches/blackboxes. WNS +0,148 ns y
WHS +0,046 ns son estimaciones anteriores a colocación, a 5 ns. Faltan
1.103 retardos de entrada y 281 de salida: no es cierre de placa.

## Reproducción y siguiente integración

`tools/ntrpu_board_gemv_memory.tcl` acepta `sim_quad_extended`,
`sim_dual_extended`, `sim_quad` con fixture y `synth_quad`. Usa una carpeta
nueva por ejecución y el MAC HLS bitcast ya calificado. Revisa con
`tools/check_ntrpu_gemv_lanes.py`, además de la referencia numérica.

Ruteo exploratorio: `tools/ntrpu_board_gemv_quad_route.tcl`, checkpoint de
síntesis, sitio padre supuesto `BUFGCE_X0Y60` y carpeta nueva. No agrega
excepciones ni retardos inventados. La tanda `20261003T222310Z-V7-gemv-quad-route`
cumple las restricciones internas a 5 ns: WNS +0,263/WHS +0,010 ns,
25.168 LUT, 17.592 FF, 104 DSP y la misma BRAM. Sus 43.552 nets enrutables
figuran completas, con cero errores y un gap del reloj padre.

La auditoría inicial agotó 300 s consultando rutas de toda la jerarquía.
`20261003T224026Z-V7-gemv-quad-route-audit-r2` termina en 66,351 s al
consultar ROUTE solo para segmentos PARTIAL, contando todos los estados
y conservando el informe global. Identifica 824 segmentos con la misma
ruta de reloj incompleta. Puertos y reloj padre siguen sin calificar.

La FFN admite `sim_quad_fp32`/`synth_quad_fp32` mediante el define explícito
`NTRPU_GEMV_QUAD` del wrapper de escritura. Quad y dual aprobaron cada
uno 63 casos con 71.232 bytes exactos y 24 alteraciones rechazadas:
`20261003T222527Z-V7-ffn-quad-errors` y
`20261003T222559Z-V7-ffn-dual-regression`. El wrapper predeterminado
recupera el texto original al quitar únicamente la rama condicional quad.
`20261003T223550Z-V7-ffn-quad-brandon` aprobó tres cadenas Brandon D256/H720:
42.048 bytes intermedios/finales exactos, incluidos 768 FP32, y ocho
alteraciones de evidencia rechazadas. Ciclos por muestra:
207.430 / 207.366 / 207.389, equivalentes a 1037,150 / 1036,830 /
1036,945 µs simulados a 5 ns. El control dual
`20261003T224122Z-V7-ffn-dual-brandon-control` también está aprobado y
respaldado: 307.247 / 307.382 / 307.387 ciclos. Fuentes, HLS y fixture
son idénticos; solo cambia el modo. Las 657 escrituras de resultados
coinciden bit a bit. La reducción de ciclos es **32,49–32,54%**
(factor **1,481–1,482×**); SwiGLU ocupa ahora **42,43–42,45%** del total.
Las secuencias de pausas aleatorias pueden divergir al cambiar la latencia.

La síntesis completa `20261003T223656Z-V7-ffn-quad-synth` conserva los
cuatro MAC y SwiGLU: 50.628 LUT, 39.991 FF, 145 DSP, 48 RAMB36 y tres
RAMB18. WNS +0,057/WHS +0,039 ns son estimaciones anteriores al ruteo;
2.470 entradas y 845 salidas siguen sin retardos externos.

El ruteo completo `20261003T224908Z-V7-ffn-quad-route` **falla setup**:
WNS −0,215 ns, 610 endpoints y TNS −26,640 ns; WHS +0,010 ns. Conserva
50.767 LUT, 39.775 FF, 145 DSP y la misma BRAM. Sus 82.691 nets enrutables
figuran completas, sin errores de ruteo. La auditoría
`20261003T230708Z-V7-ffn-quad-route-audit` identifica 1.459 segmentos del
mismo reloj padre incompleto. Ambos resultados tienen respaldo verificado.
El aviso `Route 35-328` se registra como fallo, sin exenciones de timing.

La ruta crítica va del estado de un MAC a `weights/unpacker/reservoir`:
diez niveles lógicos, 4,989 ns de datos, 84,7% en interconexión. Se prueba
una cola de dos bloques Q8 con `NTRPU_QUAD_BLOCK_FIFO` y sufijo `_fifo`
en los modos quad. Su disponibilidad depende del contador registrado;
la finalización exige vaciarla. El banco coteja cada payload/metadato,
ocupación, descartes por reset y pausas. Esta variante requiere nueva
calificación numérica y ruteo; no convierte el resultado anterior en PASS.

Primer filtro aprobado y respaldado:
`20261003T231448Z-V7-gemv-quad-fifo-extended`, 123 casos, 474 binary64,
12 resets y 54 alteraciones de evidencia rechazadas. Por semilla se
observan 10.063 entradas, 10.058 salidas y cinco descartes por reset;
ocupación máxima dos, 42.102 pausas con cola llena y 7.640 ciclos con
entrada/salida simultáneas. Cada bloque se coteja con un scoreboard
independiente de payload, orden y metadatos.

`20261003T231448Z-V7-gemv-quad-fifo-off-control` aprueba los mismos 123
casos con la cola desactivada y 39 negativos. Todas las líneas funcionales
y contadores coinciden con `20261003T220934Z-V7-gemv-quad-started-reset`.
Se conserva también la equivalencia léxica del core al retirar la rama
condicional. La cola automática exige revisión y respaldo de cada
dependencia; un fallo detiene los pasos dependientes.

El banco `20261003T232011Z-V7-gemv-quad-fifo-faults` también pasa y está
respaldado: 105 trabajos, 216 valores exactos y 81 negativos. Cada semilla
observa 7.084 bloques encolados, 7.080 retirados y cuatro descartados por
reset. La regresión independiente `20261003T232548Z-V7-gemv-quad-fifo-fixture`
también pasa: proyecciones Brandon y formas hasta 32 × 8192, 174 trabajos
y 5.376 binary64 exactos. Cada semilla encola y retira 29.888 bloques,
sin descartes ni pendientes; ocupación máxima dos.

La FFN pequeña `20261003T232011Z-V7-ffn-quad-fifo-errors` aprueba 63 casos.
La tanda `20261003T232820Z-V7-ffn-quad-fifo-brandon` aprueba tres FFN,
42.048 bytes exactos y 768 FP32 finales: 207.531 / 207.523 / 207.559 ciclos.
Ambas tienen respaldo verificado. Esta cola busca cortar la ruta de
backpressure, sin prometer una mejora de latencia de cálculo.

La síntesis `20261003T232820Z-V7-ffn-quad-fifo-synth`, también respaldada,
da 50.605 LUT, 40.014 FF, 145 DSP y 48 RAMB36 + 3 RAMB18;
WNS +0,378/WHS +0,039 ns son estimaciones. El ruteo
`20261003T234000Z-V7-ffn-quad-fifo-route` vuelve a **fallar setup**:
WNS −0,416 ns, TNS −1088,500 ns y 7.517 endpoints; WHS +0,010 ns.
Conserva 50.451 LUT, 39.718 FF, 145 DSP y la misma BRAM, con 82.718 nets
enrutables completas y cero errores de ruteo. La auditoría
`20261003T235811Z-V7-ffn-quad-fifo-route-audit` identifica 1.482 segmentos
del reloj padre incompleto. Ambos resultados tienen respaldo verificado.

El primer revisor se detuvo ante cuatro avisos nuevos `Timing 38-242`
en `m_axi_arready`, `m_axi_awready`, `m_axi_rvalid` y `m_axi_wready`.
La revisión separada conserva esos avisos pendientes y el fallo temporal:
son entradas sin contexto externo; no se añaden clocks ni excepciones.
Sigue habiendo un único clock declarado y cero latches/endpoints internos
sin restricciones. Esto no cierra las fronteras OOC.

La ruta crítica ahora va del comparador de un MAC a un acumulador:
14 niveles, 5,048 ns de datos, 78,2% de interconexión. Pasa por `error_now`,
compartido por los cuatro motores. La siguiente hipótesis es separar
la captura local de completions de la propagación global de errores,
conservando prioridad del primer error, drenaje y publicación atómica.
Es una propuesta pendiente de implementación y bancos de fallo; no basta
con eliminar una condición de control para declarar resuelto el timing.

La optimización física del DCP FIFO anterior
(`20261004T000957Z-V7-ffn-quad-fifo-physopt`, auditoría `001636Z`)
mejora WNS a −0,343 ns, con WHS +0,010 ns: sigue fallando setup.
Conserva 145 DSP y 48 RAMB36 + 3 RAMB18; utiliza 50.455 LUT/39.718 FF.
Ambas tandas están respaldadas; no se modificaron RTL ni restricciones.

En paralelo, [SwiGLU con pipeline](SWIGLU_Q8_BLOCK.md) pasa la FFN nativa
con 34,02–34,09% menos ciclos. Su ruteo separado `20261003T235718Z`
termina con WNS +0,046/WHS +0,009 ns y 153 DSP. La auditoría y los
respaldos están completos. La ruta crítica pasa ahora por el unpacker Q8;
esto no aprueba el resultado FIFO anterior ni las fronteras/reloj de placa.

Todavía no se ha integrado el quad con PCIe/DDR real ni calificado su
energía, frecuencia física o rendimiento de inferencia completa.

## Respaldo

Las tandas terminadas anteriores tienen `VERIFIED.json`, fuentes y todos los
miembros verificados en `artifacts/ntrpu-board-vivado/<run_id>/`, con
almacenamiento en D: y copia NFS. Las dos tandas con cobertura insuficiente
se conservan separadas. SHA-256 de la síntesis:
`88b0fe2cefd91b8ae2f837dd2a05ac1e21466535a68afa6a942231431113bef9`;
del fixture quad:
`fb162f3e3ee3550864a8c179bde71c764662905274186883bbb8299f2d2874c5`.

La FFN Brandon quad también está respaldada: 91.304.977 bytes, SHA-256
`32ffc6eea6aa956f59ebcd1fac60267172e2464f7ba1a94709d4b4baf262827e`.
La síntesis completa: 83.719.817 bytes, SHA-256
`c26e797c3a9d38866e05c4e209cede72dec94a4e0b28e9db91feeb731437ec0e`.
