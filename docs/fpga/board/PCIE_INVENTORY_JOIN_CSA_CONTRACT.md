# Inventario PCIe: compresión del contador de bytes

El candidato `ntrpu_pcie_inventory_scheduler_r2.sv` cambia solamente la
aritmética final del contador de bytes del monitor integrado. Conserva el
productor canónico, la validación, los registros y la publicación alineada.
No añade ciclos, pausas, restricciones de tráfico ni supuestos posteriores
a un error. Mantiene 256 tags, RQ2/RC4, 257 lecturas y 1.052.672 bytes.

## Identidad aritmética

La referencia calcula `current + rq_sum - (rc0 + rc1 + rc2 + rc3)`.
Los cuatro términos RC de 13 bits suman como máximo 32.764: su suma de
15 bits no pierde información. `rq_sum` conserva su suma con signo de
16 bits, incluido cualquier truncamiento previo.

En módulo 2²³, el nuevo árbol calcula
`current + (sign_extend(rq_sum) + 4) + ~rc0 + ~rc1 + ~rc2 + ~rc3`.
Cada RC se extiende a 23 bits antes del complemento. Compresores de tres
operandos producen palabras de suma y acarreo sin propagar acarreo entre
bits; una suma final restaura el resultado. Los 23 bits completos siguen
alimentando los guardas de rango existentes.

La suma final conserva la propagación de X de la expresión aritmética.
El banco suplementario llama a la función real del DUT y contrasta
20.000 casos deterministas, 64 extremos y X/Z en cada uno de los 89 bits
de entrada. Esto complementa el banco completo; no constituye equivalencia
formal universal.

## Criterios de ejecución

Primero deben pasar todas las 22.339 filas y 68 escenarios del banco RTL,
las comparaciones internas y los siete mutantes reales. Después se mide el
monitor completo con el mismo dispositivo, reloj de 4 ns, restricciones y
directiva `RuntimeOptimized` de la referencia. Se conserva cualquier fallo.

Exigir replay binario RTL y de su propia netlist sobre 22.330 filas y
67 escenarios. Una mejora OOC no acredita timing colocado/enrutado, SDF,
conexión nativa, la placa de 512 GiB ni inferencia completa.
