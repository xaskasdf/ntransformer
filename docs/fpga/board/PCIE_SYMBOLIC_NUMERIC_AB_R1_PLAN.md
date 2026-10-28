# Aplicación y entrega numérica A/B R1

Candidata preparada el 2026-10-09; **todavía sin ejecución ni calificación**.
Su lanzamiento requiere el replay propio aprobado de C/E/D con control R3.
Implementa las dos fronteras del [plan numérico](PCIE_SYMBOLIC_NUMERIC_STAGES_PLAN.md).

## Circuito

`ntrpu_pcie_symbolic_apply_r1` registra nueve valores de 25 bits mediante
`apply_effect`: cuatro consultas, cuatro EOP y el contexto RC previo.
Transporta el control de 391 bits y el registro descriptivo de 2.107 bits;
su salida completa tiene 2.723 bits. No consulta tablas ni propiedad actual.

`ntrpu_pcie_symbolic_delivery_r1` registra las entregas y contribuciones EOP
usando únicamente esa copia, con salida de 2.892 bits. Conserva el `if`
procedural del enable, el operador ternario terminal y la expresión mixta
signed/unsigned original. Solo el reset frío cancela orígenes. Las latencias
respecto a S1 son tres y cuatro ciclos, respectivamente.

## Banco y criterios

La comparación conserva todas las comprobaciones S1/R7/probe/modelo y
C/E/D. Coteja todos los bits A/B por origen, las consultas y el contexto
contra R7, dieciséis contribuciones por lane y el candidato de bytes.
Los estados del candidato nunca alimentan las referencias.

Se prepararon 177.833 filas positivas: 80.120 crudas, 97.496 con vaciado y
217 dirigidas. Los bancos con vaciado contienen todos los registros de
sus predecesores e insertan dos ciclos adicionales antes de cada reset:
cuatro ciclos permiten retirar también B. Se exige cero tráfico original
cancelado en esos bancos, contabilidad separada A/B y cuatro flancos finales
comparados. Los bancos crudos conservan los resets con etapas ocupadas.

Siete módulos mutados prueban orden de campos, descarte del contexto sin
propiedad, omisión de origen, fallo desalineado, pérdida del enable,
omisión del terminal y falta de reset frío. Deben fallar en comparaciones
numéricas, no por errores de compilación o watchdog.

## Límites

Esta tanda no implementa retiro R, validación tardía ni certificado de vacío;
el contador de bytes todavía se comprueba como expresión en el banco.
Quedan las pruebas focalizadas adicionales de cuatro estados, medidas de
síntesis, replay propio y conexión al monitor completo. Se conservan R12
y las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
