# Unidad de composición de metadatos: contrato inicial

Preparada el 2026-10-09. Es una pieza de la arquitectura segmentada;
no sustituye el monitor ni clasifica tráfico PCIe. Requiere revisión RTL
archivada antes de síntesis y replay propio antes de integrar.

## Formato y orden

El registro de 57 bits contiene: reemplazo `[56]`, semilla completa de
metadatos `[55:26]`, cantidad de consumos saturada a dos `[25:24]`, primer
count de 11 bits `[23:13]`, suma limitada `[12:1]` e indicador de count
desconocido `[0]`. La semilla usa el formato existente: remaining `[12:0]`,
dirección `[24:13]`, TC/atributos `[29:25]`.

Componer siempre anterior→posterior. Un reemplazo posterior elimina los
efectos anteriores sobre esa fila. No elimina la pertenencia, obligaciones
ni contexto de un payload RC terminal retenido; estos pertenecen al
planificador y siguen pendientes de integración/versionado.

El encoder interpreta `valid` y `replace` con `if` procedural: un control
X/Z no satisface esa condición. Los resúmenes producidos tienen controles
binarios, cantidad 0/1/2 y campos inactivos normalizados. Resúmenes externos
malformados no forman parte de esta interfaz. Esto no establece equivalencia
con los controles de tráfico crudo, que debe probarse al construir el selector.

## Anchos y cuatro estados

Para datos binarios, limitar la suma de counts a **2049** es suficiente:
`4*2049 - offset >= 8193 > 8191`, para cualquier offset de dos bits y
remaining de 13 bits. Counts mayores retiran exactamente el mismo remaining.
2048 no basta con offset tres. La suma intermedia de dos valores de 12 bits
usa 13 bits antes de limitarse. Conservar la excepción primer count cero
y dirección desalineada; no reinterpretarla como una resta con signo ordinaria.

Cero consumos conserva la semilla, incluso X/Z. Un consumo ejecuta las
asignaciones R7 con available/delivered de 15 bits y truncamientos originales.
Dos o más consumos conservan el comportamiento de contaminación aritmética:
un bit desconocido de remaining/dirección o de un count usado produce X en
los 25 bits aritméticos. TC/atributos se copian sin aritmética. El banco debe
comparar esa regla con la recurrencia de asignaciones, no asumirla válida.

## Frontera registrada y aceptación

La unidad recibe dos resúmenes ordenados y la semilla de la misma consulta.
En cada flanco captura su composición y aplicación cuando `valid_in` vale
uno; `valid_out` identifica ese resultado. En burbuja retiene datos y baja
valid; reset frío domina y limpia todo. El banco comprueba estabilidad antes
del flanco y el resultado después del flanco. No hay backpressure.

Exigir límites de suma, reemplazos, TC/atributos, direcciones con wrap,
0/1/muchos consumos, control/datos X/Z activos y enmascarados, pausa y reset.
Rechazar un DUT real con el orden de composición invertido. Mantener separadas
la comprobación finita de esta unidad, síntesis OOC a 4 ns, replay de netlist
y validación del monitor completo. Selección cruda, generaciones, payload,
vaciado, CDC/reset nativo y DDR/512 GiB siguen sujetos a la cola global.
