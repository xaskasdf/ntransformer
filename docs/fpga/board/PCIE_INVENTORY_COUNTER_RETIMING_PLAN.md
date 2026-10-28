# Siguiente experimento: registrar los deltas de bytes

Estado: **encolado; todavía sin RTL ni validación**. El monitor integrado
conserva su referencia original como base. Cambiar solamente la forma de
sumar no resolvió su ruta crítica; la revisión CSA produjo WNS −10,883 ns
frente a −9,877 ns, con idéntico reloj OOC de 4 ns y directiva de síntesis.

## Separación propuesta

El total global de bytes no decide la propiedad de tags ni las actualizaciones
de sus tablas. Probar una separación entre el cálculo de eventos RC/RQ y la
acumulación global, aprovechando el ciclo de publicación ya existente:

1. En el flanco N, registrar los cuatro deltas RC de 13 bits y la suma RQ
   con signo de 16 bits, sin añadirles todavía el contador global.
2. En ese mismo flanco, guardar en una base de **21 bits** el total visible
   anterior. Conservar ese truncamiento incluso después de un fallo: guardar
   23 bits cambiaría el comportamiento original al desbordar.
3. Durante el ciclo N, calcular el candidato completo de 23 bits a partir de
   la base y los deltas registrados. Sus 21 bits bajos representan el total
   posterior al beat N; sus bits altos siguen alimentando el control de rango.
4. El wrapper publica ese total en N+1, igual que antes. La tabla, el bitmap,
   las consultas canónicas y los demás contadores mantienen su actualización.

La identidad a demostrar por ciclo es
`total_N = zero_extend(total_(N-1)[20:0]) + sign_extend(rq_N) - sum(rc_N)`.
La implementación inicial debe usar la suma original; CSA queda como
experimento separado, con su regresión conservada.

## Riesgos y criterios

Revisar cuidadosamente la alineación del fallo de rango: debe aparecer con
el total correspondiente y persistir en `fault_latched` como en la referencia.
La aritmética combinacional de salida debe depender únicamente de registros;
no introducir realimentación combinacional desde eventos o validadores.

Cold reset limpia base y deltas y cancela la publicación pendiente. El armado
conserva el total cero. Los ciclos quietos registran deltas cero; resets de
enlace y errores retienen obligaciones. Mantener vacío inmediato y eventual,
secuencia diagnóstica, todas las filas posteriores a fallos y X/Z probados.

Exigir el banco completo 22.339/68, siete mutantes, comparaciones internas,
síntesis del monitor completo, auditoría del DCP y replay RTL/netlist 22.330/67.
Medir las nuevas rutas de tabla, acumulación y fallo de rango con las mismas
restricciones. No cambiar latencia pública, capacidad, RQ2/RC4, pausas ni
excepciones temporales. Esto es una hipótesis de timing, no una mejora probada.
