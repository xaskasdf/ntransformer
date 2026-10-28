# E/D con habilitación por fila: síntesis pendiente

## Entrada y alcance

Esta tanda espera el cierre revisado y respaldado del banco aislado E/D
y su monitor completo en la cola R5. No combina las variantes de sumas E,
límites B o conteo C. Conserva el control RQ 984 y cambia únicamente el
productor E de habilitaciones y su consumidor D.

Antes de sintetizar, repite los bancos `monitor` y `range_return`: 529
filas, 31 perturbaciones en quiet y comparación de 447 bits públicos y
21.951 bits internos de estado/alineación. Compila las mismas 28 fuentes
del monitor completo. La síntesis usa 14 fuentes, sin el par E/D adicional
que sirve de referencia en el banco.

## Ejecución y criterios

Parte `xcvu5p-flva2104-2-i`; reloj de 4 ns, retardos E/S cero,
`RuntimeOptimized`, jerarquía `rebuilt`, dos hilos y límite de 1.800 s.
Son las condiciones de la referencia RQ 984. La comparación reportará
LUT, FF, CARRY8, setup/hold y cobertura de restricciones; no se exige una
cantidad de FF derivada solamente de los 512 bits declarados en RTL.

Se exportan DCP, EDIF, netlist funcional, XDC y reportes. El revisor
comprueba cada aviso emitido, ausencia de latches y cajas negras,
restricciones, inventario de primitivas y auditoría del grafo combinacional.
La cobertura interna debe coincidir con los dos bancos de la regresión
completa. No se relajan límites de avisos para obtener aprobación.

Solo una síntesis revisada, comparada y respaldada habilita el replay
binario propio. Las estimaciones de interconexión no prueban mejora
física de E→D; colocación/ruteo y calificación temporal siguen pendientes.

## Archivos reproducibles

En `build/cloud-setup/20261002-board/`, el manifiesto
`pcie-symbolic-monitor-predecode-row-enable-synth-r1-preparation.json`
registra las seis recetas de helpers y la composición exacta del Tcl.
SHA-256: `f1cfc950e12c1187fb64a2f98a4c33164ead6f1f82fc4692c65af1c896476130`.
El generador R2 conserva el intento R1 que falló antes de escribir productos.
El runner transmite las comprobaciones SSH por entrada estándar en bytes
UTF-8, evitando tanto el límite de comandos de Windows como la conversión
de saltos de línea a CRLF. Cada resultado terminal se respalda en D: y NFS.
