# Variante de validación de rango HBM

## Cambio aislado

Esta tanda optimiza la ruta PCIe identificada en [HBM_CLOCKTREE.md](HBM_CLOCKTREE.md).
La fuente original conserva SHA-256
`0454df4e72a9007b59274a0eb814b756e8b8bc90d581f4df39fee900d8d6a9f6`.
`prepare_hbm_range.py` genera una copia privada de la composición seleccionada;
solo reemplaza el predicado de dirección final en `source_join`. No cambia
registros, handshakes, retención, etiquetas, conversores AMD ni resets.

La variante tiene SHA-256
`f088fb91df5a124bc33fe6593d74367d3f735aed3f6821e24821b402d3fa78d1`.
El archivo original del repositorio y las tandas calificadas permanecen intactos.
La tarjeta prioritaria sigue siendo VU47P/HBM; este mapa no corresponde al
nodo DDR4 de 512 GiB.

## Equivalencia del predicado

Antes se sumaban una dirección de 64 bits y una longitud de 13 bits en 65 bits
para comparar el final con 16 GiB. La variante comprueba:

```systemverilog
wire range_bad = src_req_address >= 64'h400000000
                 && src_req_bytes > 13'd0;
```

Conserva las comprobaciones de longitud 1–4096, alineación de 64 bytes y
prohibición de cruzar una página de 4 KiB. Como 16 GiB termina en un límite
de página, una petición válida cuya dirección inicial está dentro del rango
no puede excederlo; terminar exactamente en 16 GiB sigue siendo válido.

Se usan comparaciones relacionales deliberadamente. Una reducción de los
bits altos puede ocultar un `X` en otros bits de dirección. El término de
longitud positiva evita convertir una longitud desconocida en un rechazo
definitivo cuando la dirección conocida está fuera del rango. La rama de
longitud cero continúa siendo rechazada por la comprobación original.

`prove_hbm_range.py` modela anchos y comparaciones sin signo mediante
[bitvectors de Z3](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/).
Con Z3 5.1.0, la búsqueda de diferencias binarias es UNSAT para los 77 bits
de entrada, sin suposiciones. Un modelo manual de expresiones de cuatro
estados también resulta UNSAT. Este modelo no es un parser HDL ni una
prueba secuencial del módulo completo.

Cinco mutaciones producen contraejemplos SAT: eliminar el límite de página,
usar el límite de un solo stack, rechazar el final exacto y dos alternativas
que alteran `X/Z`. Los cinco contraejemplos se reproducen sobre HDL con
Icarus, comparando el módulo original y la variante real.

## Validación funcional

El banco de predicado compara ambos módulos con igualdad de cuatro estados:
380.448 vectores binarios y 28.480 vectores con `X/Z`. Incluye todos los
valores de longitud en doce direcciones de borde y todos los offsets de la
última página combinados con longitudes alineadas válidas.

Se reutilizan sin cambios los bancos de frontera HBM y CDC: 40 y 44 escenarios,
respectivamente. La revisión reconstruye 21.760 bytes por simulador y exige
retención en 76 escenarios de fallo. Icarus completó todos los casos en
8,71 s. XSIM completó los mismos bancos en 803,08 s, compartiendo las dos CPU
con la implementación. Ambos simuladores pasan los 84 escenarios y todas
las comparaciones del predicado.

Cinco controles de evidencia rechazan un veredicto ausente, un resultado
perdido, payload alterado, propiedad de contexto perdida y una fuente editada.
No se repite el banco de supervisión de relojes como evidencia de este cambio:
ese banco no instancia el predicado modificado.

## Implementación y criterios

La VM usa el espacio privado `ntrpu-side/axi-bridge-hbm-range/`, dos CPU
compartidas por las tandas propias y prioridad reducida. Cada proceso tiene
límite y grupo propio. La síntesis conserva el perfil exacto de los tres IP
AMD; el ruteo usa los cuatro árboles y las restricciones específicas de la
tanda anterior. No se añaden excepciones para ocultar rutas negativas.

La primera síntesis de esta variante tiene 11.608 LUT y 14.401 FF; la unión
de fuente ocupa 559 LUT frente a 584 en la misma composición original
(741 → 740 FF). La estimación PCIe antes del ruteo es
−0,006 ns frente a −0,530 ns del donante. Estas cifras no certifican cierre.

Para seleccionar la variante se exige volver a verificar cobertura CDC,
skew, topología de sincronizadores, reset canónico y rutas físicas completas.
Setup, hold y ancho de pulso se informan incluso si fallan. Permanecen fuera
del alcance los relojes reales de placa, los IP nativos HBM/PCIe, el ACK
físico y la calificación PCB.

## Primer ruteo y ajuste posterior

La tanda `physical-20261011T115212Z` terminó en 905,04 s: síntesis 184,65 s e
implementación 720,12 s. PCIe mejoró de −0,172 a +0,132 ns, eliminando sus
1.281 destinos fallidos. El nuevo peor margen quedó en memoria: −0,022 ns
en 27 destinos, dentro del conversor de ancho AMD. Lector y gestión tienen
+0,481 y +8,776 ns; hold global +0,010 ns, sin violaciones de hold o pulso.

Los cruces mantienen 1.501 rutas con margen mínimo +1,613 ns; los 16 límites
de skew pasan y el Gray privado máximo es 0,400 ns. Las 88 conexiones de
sincronizadores y los 17 consumidores de reset canónico pasan. Se conservan
las advertencias CDC, PDCN-1569, RTSTAT-10, LUTAR-1 y de fronteras sin tiempos
externos de la prueba anterior; no hay waivers.

Se ejecutó una única pasada posterior al ruteo sobre una copia del checkpoint,
limitada a 600 s. Usa `phys_opt_design -placement_opt -routing_opt
-critical_pin_opt -path_groups mem_clk`, opciones descritas en
[UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/phys_opt_design).
No cambia el RTL ni añade restricciones. Otra ejecución abrió el resultado
para repetir la auditoría completa; los XDC antes/después conservan exactamente
las mismas líneas no comentadas.

## Candidato seleccionado y pendientes

`refine-20261011T121033Z` terminó en 246,05 s, incluyendo reapertura y auditoría.
Su checkpoint tiene SHA-256
`cb110a36e78b08860b041443d7ba85fb913d66ee0c5e4f896d064f0767170ce6`.
La selección es un **candidato funcional y físico con setup pendiente**:

| Dominio | Setup mínimo original | Setup mínimo seleccionado | Destinos fallidos |
| --- | ---: | ---: | ---: |
| PCIe, 250 MHz | −0,172 ns | +0,132 ns | 0 |
| Memoria, período 2,222 ns | +0,015 ns | −0,007 ns | 5 |
| Lector, 200 MHz | +0,759 ns | +0,481 ns | 0 |
| Gestión, 100 MHz | +8,726 ns | +8,776 ns | 0 |

Las 24.900 redes que requieren ruteo están completas, sin errores ni gaps.
Hold global mínimo +0,010 ns; cero fallos de hold o ancho de pulso. La
auditoría CDC/reset conserva sus resultados; 22 controles físicos rechazan
evidencias alteradas, además de los cinco controles funcionales y cinco
contraejemplos HDL.

Quedan cinco rutas de cinco niveles lógicos desde la cola de comandos del
conversor de ancho AMD hasta entradas R de `S_AXI_RDATA_II_reg`, en el canal 0.
Sus márgenes son −7, −5, −5, −5 y −4 ps; el TNS es −0,025 ns. `OPEN_PATHS.json`
registra sus destinos concretos. No se considera cerrado el timing ni se
relaja el período para convertir este resultado en un pase.

El siguiente trabajo debe centrarse en ese camino interno del conversor y en
su integración con el reloj nativo HBM. No se infiere que añadir un registro
externo arregle una ruta interna del IP. Cualquier cambio de perfil de IP
requiere validar otra vez conversión, latencia, reset y CDC.

## Reproducción y respaldo

```bash
python -B prepare_hbm_range.py COMPOSICION_CONGELADA NUEVA_PREPARACION
python -B prove_hbm_range.py NUEVA_PREPARACION NUEVA_PRUEBA --deps Z3_LOCAL
python -B run_hbm_range.py NUEVA_TANDA --prepared NUEVA_PREPARACION \
  --engine xsim --settings VIVADO_SETTINGS
python -B run_hbm_range.py NUEVA_TANDA_FISICA --prepared NUEVA_PREPARACION \
  --engine physical --settings VIVADO_SETTINGS
python -B review_hbm_range.py TANDA_RESTAURADA
python -B run_hbm_range.py NUEVA_TANDA_AJUSTE --prepared NUEVA_PREPARACION \
  --engine refine --donor CHECKPOINT_RUTEADO --donor-sha SHA256 \
  --settings VIVADO_SETTINGS
```

La raíz de evidencia es `D:/AI/artifacts/ntrpu-axi-bridge-hbm-range/`.
Cada tanda remota produce un archivo transferible con fuentes, logs, resultados
y manifiesto SHA-256. `20261011-hbm-range.zip` reúne las tres tandas, la
validación local, las consultas SMT reproducibles, los controles negativos,
la dependencia Z3 para Windows y el respaldo físico anterior como procedencia.
`BUNDLE_BACKUP.json` registra su SHA-256 y la revisión de una restauración nueva.
`review_hbm_range_bundle.py` vuelve a resolver las consultas SMT y coteja las
comparaciones de timing contra los informes originales.

Las tres tandas remotas terminaron y sus 95 grupos de procesos propios no
siguen activos. Los originales del repositorio y las simulaciones del hilo
principal no se modificaron.
