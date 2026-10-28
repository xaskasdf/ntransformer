# Siguiente candidato: propiedad RC sin liberaciones encadenadas

Partir de R6 una vez aprobada su propia netlist, conservando R5 como
referencia del mejor WNS OOC. La rama R6 reduce LUT de 44.857 a 42.403 y
TNS de −127995,078 a −90778,977 ns, pero empeora WNS de −8,806 a
−8,903 ns. No promoverla como mejora global de timing. Esta elección de
rama permite trabajar sobre RC después de retirar la cadena RQ de cuatro
posiciones. [Resultados R6](PCIE_INVENTORY_RQ_ADMISSION_RESULTS.md).

## Evidencia que guía el cambio

La peor ruta revisada de R6 es `allocated_tags_reg[179]` →
`remaining_bytes_reg[20]`: **12,886 ns/53 niveles**. Recorre control hasta
`selected_prefix_mask[9]`, aritmética del estado previo, dirección,
`canonical_delivered` y suma global de bytes. La llegada a esa máscara es
4,996 ns en el informe OOC; los nombres del netlist orientan el diagnóstico,
sin demostrar por sí solos la causalidad de cada ecuación RTL.

La primera hipótesis es predecodificar en paralelo propiedad y liberación
por SOP RC, evitando que cada encabezado dependa del bit de liberación
calculado por el encabezado anterior. Mantener intacta inicialmente la
aritmética de prefijos R5 y la admisión RQ R6. Si el camino sigue dominado
por aritmética, investigar después el delta del fragmento directamente;
no mezclar ambos cambios en una única medición.

## Semántica que debe demostrarse

En entradas binarias, un encabezado obtiene propiedad desde la tabla
registrada, salvo que un SOP anterior del mismo tag sea terminal. Si
existen varios, sólo el primero puede liberar ese tag. El antecedente se
puede calcular desde SOP, tag, terminal y propiedad inicial, sin leer bytes
ni los resultados de liberación de otros encabezados.

Para cuatro estados, preservar la propiedad inicial X y las decisiones
procedurales originales. Una propiedad inicial desconocida no genera por
sí sola una liberación; terminal X tampoco la genera. Una comparación de
tags desconocida no debe borrar propiedad. Probar la equivalencia de la
predecodificación contra la recurrencia, sin asumir que una expresión
booleana equivalente en dos estados conserva estos efectos.

Compartir la decisión validada entre el productor y el selector de
ascendencia RC. Conservar el orden físico de lane, SOP duplicados, punteros
invertidos/desconocidos y máscara malformada. El código actual libera en
SOP incluso con descriptor erróneo: no añadir una condición de normalidad.
Mantener separado el contexto retenido y su EOP; no liberar otra vez un
tag que fue liberado en el SOP de un beat anterior. Preservar discontinue,
reutilización RQ posterior y estado después de fallos.

## Criterios de avance

1. Modelo independiente por eventos y referencia secuencial: coincidencia
   de propietario por encabezado, liberaciones, selección de escritura,
   raíz retenida y máscara de ascendencia. Cubrir varios terminales del
   mismo tag, propiedad 0/1/X/Z, terminal y tag desconocidos, punteros y
   contexto retenido. Conservar las 89.632 entradas existentes del selector
   y ampliar salidas/escenarios sin reemplazar sus comprobaciones.
2. Banco principal 22.339/68, siete mutantes y todos los controles.
   Conservar los suplementos RQ 615.536/5.712 X/Z y 8.850/240, más los
   suplementos RC 45.056/6.624 X/Z y 544/60. Comparar las tablas internas,
   estados posteriores a fallos, retiro, cancelación y vacío.
3. Síntesis completa y auditoría DCP bajo la misma parte/directiva/4 ns,
   seguida de replay de la netlist propia contra 22.330/67 y ambos bancos
   de integración. Añadir al replay los nuevos casos binarios aplicables.
4. Registrar área, WNS, TNS, endpoints setup/hold y la ruta nueva frente
   a R6, R5 y el original. No aceptar menos capacidad, pausas o cambios
   de latencia, ni sustituir la validación completa por una función aislada.

Esto no cierra identidad/BDF, conexión nativa, CDC/reset de 50 ms, DIMM
exacto, cuatro DDR/512 GiB, inferencia real, ECAD o timing físico. Mantener
las dependencias y criterios completos de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
