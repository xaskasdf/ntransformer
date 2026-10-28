# R12: caminos por familia de registros

Revisado el 2026-10-09 UTC. El DCP original conserva sus recursos,
restricciones y resumen global: **WNS −7,068 ns**, WHS −0,075 ns.
No se ejecutó síntesis, optimización, colocación ni ruteo. La medición
confirma que hay varias cadenas combinacionales largas; no cierra 250 MHz.

## Cobertura y resultados

Los 10.729 FDRE coinciden con el inventario original y se clasifican una
sola vez, sin huecos ni solapamientos. Se verificaron 56 consultas de timing,
con hasta veinte rutas por consulta. Los conteos cubren todos los registros;
las muestras no enumeran todos los caminos posibles del circuito.

| Destinos | Registros | Setup peor, ns | Setup interno, ns | Hold peor, ns |
| --- | ---: | ---: | ---: | ---: |
| Bytes/rango | 23 | -7.068 | -7.039 | 0.107 |
| Tabla de metadatos | 7680 | -5.795 | -5.766 | 0.113 |
| Bitmap/tags/lecturas | 278 | -4.407 | -4.407 | 0.105 |
| Contexto RC/RQ y stalls | 1421 | -4.180 | -4.151 | -0.075 |
| Metadatos canónicos | 176 | -3.708 | -3.679 | -0.029 |
| Control del productor | 126 | -1.884 | -1.884 | -0.075 |
| Captura | 559 | 3.488 | 3.521 | -0.075 |
| Retirada/validación pública | 466 | -3.126 | -3.126 | -0.075 |

Setup interno significa registro→registro; la columna setup peor permite
también fuentes de entrada. Los grupos son filtros de reportes, no excepciones.
Registro→salida tiene margen de setup +2,007 ns y entrada→salida +2,414 ns.
El peor hold interno es +0,066 ns; el hold global negativo proviene de las
consultas entrada→registro con estos presupuestos OOC. No trasladar este
resultado al ruteo Quick, donde el hold interno también falló.

La ruta de bytes va de `rc_user[76]` a `producer/remaining_bytes_reg[20]/D`:
11,079 ns y 43 niveles. La interna va desde `producer/allocated_tags_reg[51]/C`
al mismo destino: 11,022 ns y 45 niveles. La peor ruta de tabla termina en
`producer/remaining_reg[0][6]/D`: 9,805 ns y 38 niveles.

## Integridad y límites

DCP `293ee2f048b1f8c6624d871e1cdb5a1d8ded88f1c58b717a39f89903e57b437a`; 52.234 LUT, 10.729 FF y 1.902 CARRY8.
El XDC antes/después conserva 4 ns, retardos I/O cero y ninguna excepción
válida o ignorada. Las doce categorías de `check_timing` dan cero, sin
eliminar los fallos de setup/hold. Se conservan los avisos de jerarquía,
origen externo del reloj y DRC OOC incompleto. Vivado usó dos CPUs de timing.

El primer revisor local exigía el rótulo Setup/Hold también a puertos de
salida; Vivado usa Max/Min allí. R1 distingue ambos formatos y pasa cinco
controles de corrupción. Se conservan el revisor original y su diagnóstico;
no se modificó evidencia ni se repitió Vivado. Python 3.14 emitió dos avisos
de escape de expresión regular en R1; la expresión se ejecutó y los
controles de tipo/destino pasaron. Las revisiones quedan identificadas por hash.

## Próximo experimento

Preparado [R14: sumas registradas](PCIE_INVENTORY_REGISTERED_SUM_CONTRACT.md).
Conserva la reducción por carriles de R12 y registra sus sumas finales RC/RQ
antes de acumular bytes. Aún no está simulado. El ensayo previo R3 usaba
otra disposición de deltas y empeoró otra versión; no valida esta candidata.
La tabla ya falla por separado, así que no se espera cierre completo con
este cambio aislado. Mantener R12 hasta obtener evidencia comparable.

Siguen abiertos segmentación completa, integración nativa, reset/CDC,
DIMM exacto, cuatro DDR/512 GiB, cómputo y calificación física.

## Respaldo

Tanda `20261009T002132Z-V4-ffn-supervised-pcie-inventory-path-census`, salida 0, **195.477 s**.
Archivo `306ccc84389986ad9e84f0d8d27e022b62bfe0699d9af386fdfa68acd12afbd7`, 12690769 bytes, 92 miembros de manifiesto.
Fuente `c022cd3b2b896991ad8498b52766df5b013577c942012b45e5482a3cd0b05416`. Revisión `dc7a00aa32f3d1e088c884555c2dec58152c35ab4669645bb853ff883813839d`.
Fuentes, reportes, entrada DCP y archivo completo están verificados en
`artifacts/ntrpu-board-vivado/<tanda>/` sobre D: y en NFS.
R103 conserva la decisión y el diagnóstico local del primer revisor.
