# Contrato del productor RC con prefijos efectivos

El candidato R5 extiende R4 sin combinar el contador registrado R3. Está
pendiente de ejecución y medición. Conserva la interfaz del monitor, sus
256 tags, RQ2/RC4, 257 lecturas, 1.052.672 bytes y latencia pública.

## Cambio

El selector R1 adjunta a cada encabezado una máscara de encabezados anteriores
cuyas escrituras efectivas forman su cadena, más la selección de semilla.
Un SOP sin EOP no se incorpora a esa máscara. Se conserva la prioridad por
lane y el comportamiento de punteros repetidos/fuera de orden, terminales,
propiedad y contexto retenido. Las máscaras y la selección son binarias por
construcción: se inicializan a cero y se copian o activan por decisiones
procedurales; no contienen resultados de aritmética de bytes.

La semilla es la tabla propia o el resultado literal de la escritura retenida:
`rc_original-rc_delivered` truncado a 13 bits y `rc_address+rc_delivered` a 12.
`rc_prefix_state` calcula directamente el estado previo al encabezado desde
esa semilla y los counts de su máscara. Ya no encadena los valores de bytes
de las escrituras previas. El cálculo del fragmento actual, su delta, el
commit RC antes de RQ y los contadores globales conservan la aritmética anterior.

## Cuatro estados

Una máscara vacía conserva los bits originales; una escritura utiliza la
recurrencia mixta con y sin signo original, incluyendo la captura de 13 bits
del delta. Con dos o más escrituras, cualquier X/Z en semilla o count usado
produce dos palabras almacenadas desconocidas en esa recurrencia: una dirección
parcialmente desconocida primero contamina la dirección y luego el disponible.
El candidato representa explícitamente ese resultado. Los counts fuera de
la máscara no se observan. Esta regla debe comprobarse con el banco real;
no cambia el tratamiento de errores ni elimina filas desconocidas.

## Validación obligatoria

- Selector y ascendencia: 89.632 vectores de un modelo de lista de eventos,
  regenerados y vinculados a la ejecución del módulo real.
- Función real: 45.056 comparaciones contra la recurrencia secuencial original,
  incluidas 6.624 inyecciones X/Z en los 69 bits de semilla/count bajo las
  16 máscaras conocidas. No se afirma equivalencia formal universal.
- Integración adicional: 544 filas/60 escenarios desde entradas crudas,
  incluyendo cuatro fragmentos del mismo tag, ceros, agotamiento, counts
  inválidos, continuación retenida, reutilización RQ y punteros malformados.
  Comparar todas las palabras de tabla y estados posteriores a fallos contra
  el RTL R7 congelado; las expectativas Python dejan de ser fiables tras
  tráfico malformado y no sustituyen esa comparación.
- Banco completo intacto: 22.339 filas/68 escenarios, siete mutantes y todos
  los controles de evidencia.
- Síntesis completa, auditoría de ciclos y netlist propia: 22.330 filas/67
  escenarios binarios más las 544 filas adicionales, coincidentes con RTL.

Medir la misma parte, directiva RuntimeOptimized y restricciones OOC de 4 ns,
I/O cero y sin excepciones. No añadir pausas. Respetar cold reset, cancelación,
fallos y vacío inmediato/eventual. Comparar con la referencia original y R4;
una mejora OOC no cierra conexión nativa, CDC/reset, DIMM exacto, cuatro DDR/
512 GiB, pesos reales, ECAD o timing físico.
