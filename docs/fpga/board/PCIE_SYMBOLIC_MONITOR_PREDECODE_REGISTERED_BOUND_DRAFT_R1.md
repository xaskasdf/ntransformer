# Borrador de límites de payload registrados en B

## Estado

Hay cuatro módulos RTL preparados; **no están compilados, simulados ni
sintetizados**. La cola activa sigue usando el contexto por prefijo ya
aprobado. Este borrador investiga retirar la suma del límite de payload
de la etapa B→R; la mejora de timing no está demostrada.

## Frontera propuesta

La variante `ntrpu_pcie_symbolic_delivery_predecode_bound_r1` conserva el
registro B original y añade cuatro límites de consulta y uno de contexto.
Cada límite es una suma unsigned explícita de quince bits:
`{13'b0, address[1:0]} + {2'b0, delivered}`.
Se calcula desde `next_b` y se captura en el mismo flanco que ese registro.
Son **75 bits declarados**; su coste físico lo determinará la síntesis.

La validación tardía selecciona el límite con los mismos SOP y contexto
que la dirección y `delivered`. Conserva la comparación inferior de
dirección y las demás condiciones. El retiro recibe estos campos junto
al registro B; no añade etapas, punteros, backpressure ni capacidad nueva.

La revisión textual confirmó puertos públicos idénticos, tipo B original
intacto, función `delivery` idéntica, lógica secuencial de retiro intacta
y expresiones de `pipeline_pending`/quiet idénticas. Esto no verifica aún
alineación temporal ni semántica SystemVerilog de cuatro estados.

## Gates antes de promover

1. Comparar B completo con su referencia y verificar los cinco límites
   desde el mismo origen, antes/después del flanco, con reset y X/Z.
2. Comparar las 32 contribuciones de fallo por lane y el resultado agregado;
   incluir anchos, alineación, desbordamiento y mutantes de selección/desfase.
3. Ejecutar la integración completa de 183.058 filas y sus controles.
4. Medir A→B y B→R, recursos, DRC y la netlist propia. Cargar otra etapa
   puede desplazar el cuello de botella; no basta con mejorar un tramo.

El [diagnóstico nuevo del prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_CRITICAL_LUTS_R1.md)
ya difiere del anterior. No hay tanda de este borrador encolada: primero
se completa el replay activo y se decide qué operación abordar.

## Procedencia

El generador R2 deriva exactamente los cuatro padres de la tanda 965;
sus recetas se pueden invertir sin diferencias. R1 detectó un esquema
de manifiesto distinto antes de escribir archivos; se conserva ese fallo
y su corrección. Manifiesto SHA-256
`98290bb59c09111fec0d5b9e91adc8056e55b5f2728d435c661e7c903879e0d5`.
Comprobación estática SHA-256
`fa1856120c4fc021b9b8ea7d17c911e98d530321f5f3594de4e06103723de5d2`.
