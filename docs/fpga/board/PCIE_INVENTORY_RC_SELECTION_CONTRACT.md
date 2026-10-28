# Contrato del selector de actualizaciones RC

El candidato `ntrpu_pcie_inventory_scheduler_r4.sv` parte del productor original.
Extrae la elección de actualizaciones a `ntrpu_pcie_rc_update_select.sv`, que
recibe exclusivamente entradas RC crudas, bitmap y contexto retenido propios.
Todavía requiere ejecución y medición. No incluye el contador registrado R3
ni cambia la suma, resta, saturación o latencia de las salidas.

## Dependencias separadas

El selector reproduce el recorrido de SOP/EOP por lane, liberación terminal
en SOP y limpieza del propietario en EOP. Para cada encabezado devuelve si
existe una escritura previa de su tag y el slot de la de mayor lane. Para
cada tag devuelve la última escritura del beat. Conserva los mapas con
punteros repetidos/fuera de orden y el último slot asignado a un EOP repetido.
El contexto retenido se utiliza literalmente, incluso después de fallos.

El parser continúa calculando los datos de cada escritura en el orden original.
Lee el registro seleccionado de sus actualizaciones anteriores y después
aplica RQ, conservando su prioridad. Ninguna decisión del selector depende
de los bytes calculados; no recibe respuestas de un oráculo externo.

## Cobertura exigida

El banco del selector instancia el módulo real contra 89.632 vectores de un
modelo independiente de lista de eventos: 65.536 combinaciones de cuatro
punteros EOP, 4.096 combinaciones SOP/contexto y 20.000 muestras binarias.
Incluye liberación, reutilización, contexto parcial, máscaras inválidas,
solapamientos y discontinue. El banco completo mantiene 22.339 filas/68
escenarios de cuatro estados, comparación de tablas y siete mutantes reales.
No se afirma equivalencia universal de X/Z por pasar ese banco.

Exigir síntesis del monitor completo, auditoría del DCP y reproducción de
la netlist propia sobre las 22.330 filas/67 escenarios binarios. Medir con
la misma parte, directiva RuntimeOptimized, reloj de 4 ns, I/O cero y ninguna
excepción. No promover el candidato sólo porque pase el selector aislado.

## Relación con el siguiente paso

El estudio binario `check_ntrpu_pcie_rc_prefix_math.py` pasó 67.108.864 casos
de un fragmento, 2.097.152 pares límite y 1.000.000 de prefijos muestreados.
Es evidencia aritmética de modelo; **este candidato aún no implementa prefijos**.
Primero se mide la separación de controles, según el
[plan de actualizaciones RC](PCIE_INVENTORY_RC_FORWARDING_PLAN.md).

Se mantienen 256 tags, RQ2/RC4, 257 lecturas, 1.052.672 bytes, cold reset,
estados posteriores a fallos, publicación alineada y vacío inmediato/eventual,
sin pausas del observador. Native/CDC/reset, DIMM exacto, cuatro DDR/512 GiB,
pesos reales y cierre físico siguen pendientes.
