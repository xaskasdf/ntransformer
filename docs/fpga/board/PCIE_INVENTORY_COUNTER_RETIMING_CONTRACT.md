# Contrato del contador de bytes con deltas registrados

El candidato `ntrpu_pcie_inventory_scheduler_r3.sv` parte del productor original,
sin la aritmética CSA. Instancia `ntrpu_pcie_inventory_byte_accumulator.sv` para
separar los eventos RC/RQ de su acumulación global. Está pendiente de ejecución,
síntesis y replay propio; el [plan anterior](PCIE_INVENTORY_COUNTER_RETIMING_PLAN.md)
se conserva como registro de la decisión.

## Estado y alineación

El acumulador retiene una base de 21 bits, una suma RQ con signo de 16 bits y
cuatro deltas RC de 13 bits. Todos inicializan a cero. En cada flanco habilitado
guarda el total anterior truncado a 21 bits y los deltas del beat actual.
La salida combinacional de 23 bits depende exclusivamente de esos registros.

Tras el flanco N, `candidate` reproduce el total posterior al beat N, incluidos
sus bits altos. El productor publica sus 21 bits bajos y comprueba el rango
con los 23 bits completos. El wrapper existente captura estado y fallos en
N+1; no cambia su latencia ni recibe datos de una referencia externa.

El habilitador es el registro `observing` del productor, que sólo toma cero
o uno. El armado conserva cero; cold reset limpia base y deltas. Un ciclo
inactivo habilitado captura deltas cero. Reset del enlace y fallos mantienen
las obligaciones, igual que en la referencia. Se conserva el truncamiento
entre beats incluso después de un desbordamiento.

## Verificación requerida

El banco secuencial instancia el acumulador real y compara 20.414 ciclos
contra la recurrencia original: 20.000 estímulos deterministas, reset,
pausa, valores negativos, desbordamientos y 136 inyecciones X/Z. Comprueba
que un total desconocido persista con deltas cero y se recupere con cold reset.
La pausa debe conservar un total conocido pese a entradas desconocidas.

El banco completo sigue comparando 22.339 filas/68 escenarios, todas las
palabras internas, el estado posterior a fallos y siete mutantes reales.
Exigir además 22.330 filas/67 escenarios binarios en RTL y en la netlist
propia, DCP sin ciclos y síntesis OOC completa a 4 ns sin excepciones.

No reducir capacidad, RQ2/RC4 ni añadir pausas del observador. Examinar las
rutas de tabla, acumulación y fallo de rango. El experimento no acredita
conexión nativa, SDF, cierre físico, cuatro DDR/512 GiB ni inferencia real.
