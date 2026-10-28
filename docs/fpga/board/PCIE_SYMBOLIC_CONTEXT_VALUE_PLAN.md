# Contexto simbólico por valor: candidato para el modelo

Propuesta del 2026-10-09; no implementada ni validada. Complementa
[el diseño pendiente](PCIE_SYMBOLIC_CONTEXT_DESIGN.md). Busca eliminar
la necesidad de referencias duraderas a slots pendientes antes de elegir
el pipeline. No cambia los criterios del contrato ni acredita timing.

## Representación autocontenida

El álgebra revisada permite representar una secuencia de consumos con
tamaño fijo, incluso si la secuencia es larga. Una fila puede almacenar:

| Campo | Bits | Función |
| --- | ---: | --- |
| Remaining y dirección de la semilla | 25 | Estado anterior a los consumos resumidos |
| Clase de consumos | 2 | Cero, uno o varios; clase 3 fuera del contrato |
| Primer count | 11 | Conserva el caso de count cero con dirección desalineada |
| Suma limitada a 2049 | 12 | Cota revisada para remaining de 13 bits |
| Count utilizado desconocido | 1 | Conserva las reglas X/Z del álgebra |

La descripción suma **51 bits**. TC/atributos ocupan otros cinco bits
por separado: **56 bits por tag** frente a los 30 actuales. La diferencia
lógica sería 6.656 bits para 256 tags, antes de optimizaciones y registros
adicionales. No es una estimación de FF después de síntesis.

Una admisión RQ fija semilla y clase cero, y reemplaza TC/atributos.
Un consumo concatena el resumen anterior con el nuevo count. No requiere
aplicar remaining/dirección para producir la siguiente descripción.
Se mantienen normalización, orden, saturación y reglas de desconocidos
del paquete de efectos; no sustituirlos por suma modular.

## Consultas, contexto retenido y escritura

Una cabecera copia su descripción propia de 51 bits después de seleccionar
la semilla de tabla/contexto retenido y componer el prefijo crudo de R3.
También captura el count y las condiciones necesarias para calcular
delivered con la expresión exacta existente. TC/atributos de esa consulta
se capturan desde la fila correspondiente, independientemente del origen
de remaining/dirección. No derivar delivered restando dos resultados
evaluados: puede alterar la propagación de X/Z.

El contexto retenido conserva esa descripción **por valor**. Su duración
no reutiliza un puntero a una tabla circular. EOP no terminal, normal y
con propiedad escribe la descripción tras ese consumo; la prioridad de
slots y la selección de commits siguen siendo las de R3. La escritura RC
conserva TC/atributos de la fila actual. El EOP terminal solo produce el
descuento global de la semilla consultada; no escribe la fila liberada.
RQ conserva prioridad final sobre RC, incluso en tráfico posterior a fallo.

La evaluación numérica diferida recibe copias completas y **no escribe de
vuelta** el contexto vivo ni la tabla de descripciones. Esto podría evitar
generaciones para esos destinos: los datos pendientes no poseen referencias
a almacenamiento reutilizable. Es una hipótesis que el modelo debe probar,
no permiso para retirar una comprobación existente. Si el diseño termina
guardando tickets o punteros en cualquier lugar, vuelve a ser obligatorio
acotar su vida y demostrar el comportamiento al envolver versiones.

## Fronteras a evaluar

1. Control crudo y commit simbólico: propiedad, selección, descripción de
   cada consulta y estado de framing. Cuatro RC y dos RQ por ciclo.
2. Registro de las descripciones y del estado correspondiente al beat.
3. Aplicación numérica, delivered y validación desde esas copias.
4. Retiro ordenado de resultados y deltas, con cancelación solo por reset
   frío y revocación inmediata de vacío ante actividad/obligaciones.

Son responsabilidades candidatas, no cuatro ciclos comprometidos. El
modelo debe determinar qué señales necesitan más registros. La recurrencia
de composición de 51 bits y la selección de 256 filas también necesitan
medición; quitar aplicación de esa ruta no garantiza 4 ns.

## Prueba exigida antes de adoptar la representación

Mantener estado simbólico propio desde entradas crudas, sin alimentarlo con
resultados de R7. Comparar consultas, tabla evaluada, contexto retenido,
delivered, deltas y retiro por secuencia en **todos** los bancos existentes,
incluidos estados tras fallo y las filas históricas 4131/4140/4291/4300.
Añadir retención prolongada, semilla con atributos distintos, reemplazo de
contexto incompleto, prioridad RC/RQ, count cero desalineado, saturación 2049,
X/Z activos/enmascarados y reset con todas las etapas ocupadas.

Un mutante que vuelva a consultar la fila actual al EOP debe fallar ante
reutilización del tag; también uno que copie TC/atributos antiguos en un
commit RC. Solo después de estas comparaciones fijar formato RTL,
profundidad, latencia y procedimiento de retiro. R12 sigue como referencia.
