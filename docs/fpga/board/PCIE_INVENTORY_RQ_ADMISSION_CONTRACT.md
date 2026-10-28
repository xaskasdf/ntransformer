# Contrato R6: dos admisiones RQ efectivas

R6 parte del productor R5 revisado y conserva sus prefijos RC, contador
global, interfaz, capacidad y latencia. Implementa el
[plan de admisión](PCIE_INVENTORY_RC_PREFIX_FOLLOWUP.md); todavía requiere
ejecución RTL, síntesis y replay de su propia netlist.

## Transformación

La función real `rq_admission_flags` decodifica como máximo dos SOP,
descarta los índices desconocidos como las escrituras procedurales a
`sop_map`, combina punteros duplicados y ordena por lane. Resuelve sólo
dos decisiones de propiedad/capacidad. Devuelve admisión/rechazo por
posición y el count final firmado de 11 bits.

El parser físico permanece: conserva framing, payload, EOP, fallos,
byte enables y prioridad de slot. En duplicados, el último slot sigue
controlando los datos de la escritura. La función ve la tabla propia
después de liberaciones RC y la primera reserva efectiva al decidir la
segunda. No consulta resultados de referencia ni cambia el tratamiento
procedural de condiciones desconocidas. No recorta counts malformados.

## Bancos obligatorios

- Función instanciada: **615.536 comparaciones** contra una exploración
  independiente por lane con los dos registros de actualización originales.
  Incluye 262.144 combinaciones de control, 327.680 casos de capacidad
  (todos los counts almacenados 0–511 y cero a cuatro liberaciones),
  20.000 muestras y 5.712 inyecciones X/Z: cada uno de los 952 bits de
  entrada, en tres contextos. No constituye equivalencia formal universal.
- Integración nueva: **8.850 filas/240 escenarios** desde entradas crudas:
  todos los pares de punteros, máscaras, read/write, tags iguales/distintos,
  capacidad 254/255/256 y liberación RC/reutilización RQ en el mismo beat.
  Comparar todos los estados y palabras internas con R7 congelado,
  incluyendo tráfico posterior a errores. Python sólo conserva expectativas
  cuando su modelo no rechazó el tráfico malformado.
- Mantener el banco completo 22.339/68, sus siete mutantes y controles,
  y todos los suplementos R5: selección 89.632, aritmética 45.056/6.624 X/Z,
  e integración RC 544/60.
- Tras el RTL, sintetizar el mismo monitor/parte a 4 ns, RuntimeOptimized,
  I/O cero y ninguna excepción; auditar DCP. Ejecutar la netlist propia
  contra 22.330/67 filas binarias más ambos suplementos de integración.

Registrar WNS, TNS, endpoints setup/hold y recursos frente a R5 y al
original. Conservar regresiones y archivos fallidos. No introducir pausas,
reducir capacidad, cambiar cancelación cold reset o semántica de vacío.
Conexión nativa, identidad/BDF, CDC/reset de 50 ms, DIMM exacto, cuatro DDR/
512 GiB, inferencia real, ECAD y cierre físico siguen abiertos.
