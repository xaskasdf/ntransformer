# Inventario PCIe con validación y publicación completas

## Cambio funcional

La integración usa el productor canónico inicial, su tabla y sus contextos
retenidos. El nuevo validador comprueba identidad, tamaños, terminal,
byte-enables y payload RQ/RC. El estado y las fuentes de fallo se publican
juntos un ciclo después del commit canónico. La referencia completa sólo
existe en el banco de pruebas; ninguna salida suya alimenta al candidato.

El [contrato de ciclos y vacío](PCIE_INVENTORY_JOIN_CONTRACT.md) identifica
las interfaces. La secuencia es diagnóstica; no sustituye la propiedad del
tag. Cold reset cancela la publicación pendiente. Los errores de enlace
no borran solicitudes pendientes. El bloque sigue siendo un observador
pasivo y no acredita recepción de posted writes ni drenaje del núcleo.

## Banco completo — 2026-10-08

`20261008T055055Z-V4-ffn-supervised-pcie-inventory-join-rtl` pasó en
**118,304 segundos**, seguido de revisión independiente y respaldo:

- **22.339 filas y 68 escenarios**, incluidas todas las filas posteriores
  a errores y las pruebas X/Z existentes del banco.
- **5.718.784 comparaciones** de palabras de tabla en RTL.
- **22.136 publicaciones y 67 cancelaciones** por cold reset: todas las
  filas capturadas quedan contabilizadas; se publica también la última.
- **71 retiradas inmediatas de vacío**, 3.609 filas vacías y una vuelta
  del contador diagnóstico. No se mantiene ocupado el bloque por
  capturar ciclos quietos.
- Capacidad ejercida: **256 tags, 257 lecturas y 1.052.672 bytes**, RQ2/RC4,
  sin introducir pausas del observador.

Se detectaron siete mutantes reales: omitir RQ (fila 4), omitir RC (133),
omitir validación (4.132), forzar vacío (4), alterar secuencia (129),
ignorar cold reset (197) y corromper metadatos canónicos (133).
Los **17 controles de corrupción de evidencia** también fueron rechazados,
incluida la alteración conjunta de salida y referencia dentro del log.
La referencia archivada se cotejó por separado y se regeneraron los vectores.

La única clase de advertencia del banco fue `XSIM 43-3431`, una por modo.
El archivo de respaldo tiene SHA-256
`9d1645fec19550d1bff94878596a87143c9294214e603fdabec6d76a11de391c`;
la revisión independiente,
`a88ecf9bafe379652b5d981282f99df5d1892ecf974a17e4cfef80824f18ef1b`.
Ambos están identificados en `build/cloud-setup/20261002-board/` y
`artifacts/ntrpu-board-vivado/<ID>/`, con copia certificada en D: y NFS.

## Síntesis por defecto: diagnóstico conservado

`20261008T055539Z-V4-ffn-supervised-pcie-inventory-join-synth` alcanzó el
límite de **1.800,015 segundos**, código 124, durante `Timing Optimization`.
No exportó DCP, netlist ni informes finales; no se le atribuyen recursos
ni timing. Archivo certificado:
`443074e7017e5e62bca29a9a7198e4390419b58f99cedcafa1c320919a2cb362`.
La revisión independiente comprueba ese alcance y la ausencia de la marca
real de exportación; el eco del código Tcl no cuenta como ejecución.

Se preparó una tanda distinta con `RuntimeOptimized`, que AMD describe como
una estrategia con menos optimizaciones para reducir duración
([UG901](https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Using-Synthesis-Settings)).
Conserva RTL, capacidad, reloj objetivo de 4 ns e I/O de 0 ns, sin excepciones.
La estrategia utilizada debe acompañar a sus medidas; no representa por sí
misma una mejora arquitectónica ni un cierre de timing.

El primer intento `20261008T063523Z` completó `synth_design`, pero terminó
con código 1 a los **240,454 segundos**: `report_property` recibió tres
objetos y exige uno. No se habían exportado DCP/netlist ni medidas finales.
Se conserva el diagnóstico, SHA-256
`8229eca7de26b0c12680428bb7c37d4e83e8f9b425129e0412fe6d4d5e07016f`.
La revisión Tcl posterior genera propiedades por archivo y guarda primero
los artefactos compilados; mantiene idénticos los tres módulos RTL.

## Síntesis completa y estructura

`20261008T064320Z-V4-ffn-supervised-pcie-inventory-join-synth-runtime-r1`
completó exportación y revisión en **317,846 segundos**, con estrategia
`RuntimeOptimized`, VU5P y caracterización fuera de contexto a 4 ns.

| Medida del monitor completo | Resultado |
| --- | ---: |
| LUT | 48.189 |
| FF | 10.729 |
| CARRY8 | 260 |
| Latches / BRAM / URAM / DSP | 0 / 0 / 0 / 0 |
| WNS / TNS setup | −9,877 / −109.536,984 ns |
| Extremos setup fallidos | 15.851 / 29.739 |
| WHS / THS hold | −0,075 / −139,020 ns |
| Extremos hold fallidos | 1.936 / 29.739 |

**No cumple setup ni hold.** La peor ruta sigue dentro del productor:
`address_next_reg[51][0] → remaining_bytes_reg[20]`, con **13,860 ns y
56 niveles**; 3,350 ns son lógica y 10,510 ns ruteo estimado. No se ha
colocado/ruteado el diseño. El cambio de estrategia y de top impide atribuir
una diferencia frente al productor aislado exclusivamente al RTL.

Las doce comprobaciones de `check_timing` no reportan omisiones ni ciclos;
no hay excepciones temporales. La auditoría del DCP comprobó **69.149
primitivas y 75.627 nets**, sin ciclos combinacionales, y rechazó ocho
controles de corrupción. Persisten advertencias `TIMING-16` y `XDCH-2`.

Se revisaron cinco salidas diagnósticas del productor sin conectar, bits
superiores de sideband no usados por este validador y tres registros de
captura con el mismo D/reset que registros públicos. Los mensajes de
igualdad X/Z y puertos sin carga alcanzaron el límite de 100 mensajes por
clase; la equivalencia mapeada se evalúa en el dominio binario del banco.

Archivo SHA-256:
`11f801a3cd2934bf86bb48d4721fe4177565d9be37d46409ee0d9c131f5131fc`.
Netlist funcional SHA-256:
`efad67c175c239a6c091325c6d51455c4d80692fe22502d777ee1f8ea179f804`.

## Replay de la netlist propia

Los bancos `20261008T062713Z` (RTL, **51,585 s**) y `20261008T065021Z`
(netlist, **106,245 s**) pasaron **22.330 filas y 67 escenarios** cada uno.
Coinciden todas las salidas públicas: estado, bitmap, fuentes completas
de fallo, secuencia, validez de publicación, pendiente y vacío, incluidas
las filas posteriores a errores. Cada revisor rechazó doce corrupciones.

En ambos hay **22.130 publicaciones, 66 cancelaciones por cold reset,
70 retiradas inmediatas de vacío, 3.608 filas vacías y una vuelta de
secuencia**. La comparación de las 256 palabras internas se limita al RTL.
La netlist compilada tiene el hash indicado arriba; no se compilaron como
DUT los módulos RTL del monitor, productor o validador. Se comprobó que
ambas definiciones `glbl` tienen el mismo cuerpo. No se aplicó SDF ni se
extiende el resultado binario a todos los valores X/Z posibles.

Los IDs completos usan `-V4-ffn-supervised-pcie-inventory-join-binary-rtl`
y `-V4-ffn-supervised-pcie-inventory-join-binary-netlist`. Archivos certificados:

- RTL: `861d7699c395f02634eed8daa38fafd2103338a487423e108f747e4b6ed5aceb`.
- Netlist: `21f33208a632f82ec7314a54b8e7c2a0d08f35f0b0c0d9ce531336621f50fc19`.

Queda completada esta integración funcional aislada. La siguiente tarea
es mejorar la ruta crítica medida manteniendo la capacidad, el orden de
actualizaciones, el comportamiento tras errores y las garantías de vacío.
Cada candidato debe pasar nuevamente el banco completo y su propia netlist;
el objetivo de 4 ns y la integración nativa continúan abiertos.

## Conservación de cachés locales

Ante falta de espacio en C:, se trasladaron **37 ZIP antiguos de fuentes**,
912.026.113 bytes, a
`D:\AI\ntransformer\artifacts\ntrpu-board-vivado\source-cache-relocated\20261008T063408Z-inventory-join\`.
No se descartaron archivos. Los ZIP actuales `054937Z`, `055454Z` y `062712Z`
permanecen en C:. Se verificaron SHA-256, tamaños y CRC de cada caché y su
copia certificada; después del traslado se revalidaron ambas copias.

El plan y resultado, con prefijo
`pcie-inventory-join-cache-relocation-20261008T063408Z`, conservan la relación
`source → destination → retained` y los hashes. Están en
`build/cloud-setup/20261002-board/` y en la carpeta de destino. Si una
reproducción histórica requiere una caché en su ruta anterior, copiar el ZIP
desde `destination` o `retained` a `source` y comprobar el hash del plan.
No repetir el traslado. El espacio libre observado en C: pasó de 271 MB
a 1.183 MB; la actividad concurrente puede cambiarlo.

Al retomar la revisión, C: volvió a bajar de 100 MB. Una segunda tanda
trasladó **186 cachés certificadas del 2 al 6 de octubre**, 902.856.497
bytes, con las mismas comprobaciones y sin descartar archivos. Su plan
usa el prefijo `pcie-inventory-join-cache-relocation-20261008T101922Z` y
el destino termina en `20261008T101922Z-inventory-join-extra`. El margen
observado pasó de 120 MB a 1.023 MB. Las dos tandas reubicaron en total
223 ZIP; la recuperación de una ruta histórica sigue el plan correspondiente.

## Límites conservados

El banco verifica los casos enumerados; no es una prueba formal para todas
las entradas posibles. El replay nativo histórico cubre sólo tres lecturas
de 32/64/32 bytes y no ejercita la capacidad máxima del endpoint.

Siguen abiertos la identidad host/BDF independiente, integración y CDC
nativos, reset completo de 50 ms, SPD exacto del DIMM, cuatro canales DDR4
y 512 GiB, inferencia con pesos reales, ECAD y pruebas físicas. La identidad
`01a0` es un fixture del banco, no una configuración de producción.
