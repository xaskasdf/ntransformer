# Siguiente experimento: acortar las actualizaciones RC encadenadas

Estado: **diseño propuesto; sin candidato RTL ni validación de esta transformación**.
El [experimento del contador](PCIE_INVENTORY_COUNTER_RETIMING_RESULTS.md) deja
como peor ruta `allocated_tags_reg[51]` → `remaining_reg[50][10]`, con 55 niveles
y 13,941 ns estimados. Su WNS es −9,959 ns; la referencia original sigue siendo
la preferida entre las medidas OOC, con −9,877 ns. Ninguna cierra 4 ns.

## Separar decisiones de eventos y aritmética

El productor original recorre los 16 DWORD del beat. Cada encabezado puede
consultar una actualización RC anterior; encadenar selección, resta, dirección
y nueva selección alarga la ruta. El próximo candidato debe conservar ese
orden observable mientras calcula primero qué eventos dependen de cuáles.

1. Decodificar SOP/EOP y sus posiciones con las reglas actuales. Mantener
   termómetros inválidos, punteros repetidos y fuera de orden, sobrescritura
   de `rc_end_slot` y prioridad de la actualización con mayor lane anterior.
   No ordenar ni corregir paquetes malformados.
2. Resolver propietario, tag, normal/terminal y contexto de cada EOP sin
   depender de los bytes calculados. Un encabezado terminal libera el tag
   en SOP, aunque su payload continúe; un EOP limpia `co` incluso con fallo.
3. Obtener el estado inicial desde las tablas propias. Asociar a cada
   encabezado solamente los EOP anteriores que realmente actualizan su tag.
   Mantener separados el contexto retenido de un paquete parcial y la tabla.
4. Calcular los valores canónicos que necesita cada consulta y cada escritura.
   Explorar sumas de prefijos para cadenas del mismo tag; conservar selección
   explícita para los casos que no cumplan la identidad aritmética propuesta.
5. Confirmar RC antes de RQ. La reutilización RQ del mismo beat conserva la
   última escritura, su TC/atributos y su dirección; una finalización de un
   paquete terminal antiguo no puede sobrescribir el tag reutilizado.

## Identidad candidata y límites

Para un fragmento binario normal, propio y no terminal, sea `R` el restante
de 13 bits, `A` la dirección de 12 bits y `W` el count de 11 bits. El productor
actual calcula `available = 4*W - (A & 3)` y compara con `R` mediante la mezcla
de operandos con y sin signo de su RTL. Por ello, **W=0 con dirección
desalineada entrega R**, incluso en estados posteriores a un error; no entrega
un delta negativo ni cero.

Para una cadena no vacía de esas actualizaciones, se propone:

- Si el primer fragmento tiene `W=0` y el desplazamiento inicial no es cero,
  el delta total es `R`.
- En los demás casos, el delta total es
  `min(R, 4*sum(W) - (A & 3))`; el restante es `R-delta` y la dirección
  `(A+delta) mod 4096`.

La justificación a comprobar es que un fragmento positivo alinea la dirección
si no agota R; una vez agotado, los siguientes entregan cero. Aplicar la fórmula
a cada prefijo permitiría obtener también las consultas y deltas individuales,
sin encadenar restas saturadas. La suma necesita ancho suficiente para cuatro
counts de 11 bits. Una cadena vacía conserva el estado sin aplicar la fórmula.

Esta identidad **no está validada aún**. No cubre por sí sola X/Z, selección de
eventos, terminales, huérfanos, discontinue ni contexto parcial retenido. Este
último guarda `rc_original`, `rc_delivered` y `rc_address`, que pueden diferir
de la tabla después de fallos; debe consumirse literalmente antes de formar
una cadena nueva. No reconstruirlo desde el count o la tabla.

## Orden de implementación y aceptación

Primero comprobar la identidad binaria y sus límites con un modelo independiente,
incluyendo cero, agotamiento, wrap de dirección y todo el rango posterior a
fallos. Después probar la decodificación de eventos manteniendo la aritmética
actual; sólo entonces introducir prefijos en una revisión separada. Registrar
los contraejemplos y conservar las fuentes de cada intento.

Cada candidato integrado debe usar entradas crudas y estado propio. Exigir
22.339 filas/68 escenarios RTL, todas las palabras de tabla y filas posteriores
a fallos, siete mutantes y controles de evidencia; añadir casos que distingan
las nuevas prioridades y prefijos. Exigir después síntesis completa, auditoría
de ciclos y replay de la **netlist propia** con 22.330 filas/67 escenarios
binarios. Un modelo aritmético aislado no califica el monitor.

Conservar 256 tags, RQ2/RC4, 257 lecturas, 1.052.672 bytes, latencia pública,
cold reset, cancelación pendiente y vacío inmediato/eventual. No añadir pausas
del observador ni excepciones temporales. Comparar ambos candidatos previos
con la misma parte, directiva y restricciones OOC de 4 ns; no acumular cambios
de contador y parser sin medir su contribución.

Las puertas nativas, identidad/BDF, CDC/reset, reset nativo de 50 ms, DIMM
exacto, cuatro DDR/512 GiB, pesos reales y diseño físico permanecen abiertas.
Este experimento no cambia los criterios de [la cola principal](VIVADO_QUEUE.md).
