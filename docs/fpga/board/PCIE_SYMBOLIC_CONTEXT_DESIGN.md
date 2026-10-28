# Próximo corte: contexto simbólico entre consulta y aplicación

Propuesta del 2026-10-09, pendiente de modelo e implementación. Se concreta
una alternativa autocontenida en [contexto por valor](PCIE_SYMBOLIC_CONTEXT_VALUE_PLAN.md).
R15 conecta
el álgebra de efectos con las consultas crudas del productor, conservando
su frontera actual. Este documento identifica el cambio de estado necesario
para introducir registros; no acredita una profundidad ni timing nuevos.

## Dependencia que debe desaparecer

Un EOP del beat siguiente puede necesitar la dirección y los bytes del
header anterior antes de que una etapa aritmética nueva los registre.
Insertar un registro solamente en `rc_prefix_state` produciría ese problema:
la siguiente consulta y el contexto retenido aún dependen del resultado.
No resolverlo leyendo la fila actual del tag: una cabecera terminal pudo
liberarlo y permitir una solicitud nueva mientras sigue llegando payload.

La alternativa a modelar es retener una **descripción de la consulta**:
semilla propia registrada más resumen de consumos ordenados. El EOP y un
header dependiente seleccionarían/compondrían esa descripción antes de
aplicar su aritmética, en vez de esperar el resultado numérico del header
previo. Las admisiones RQ reemplazan la semilla según el orden existente.

## Dos clases de estado

La asignación de tags, la propiedad, SOP/EOP, longitud/posición y la selección
de ancestría dependen del control crudo. Deben conservar su ritmo de un beat
RQ y RC por ciclo. Los valores aritméticos y su validación pueden calcularse
más tarde, siempre que cada resultado y delta conserve el evento de origen.

La parte remaining/dirección necesita su semilla y efecto simbólicos.
TC/atributos se validan desde la fila que corresponde a la consulta, pero
una escritura RC no los reemplaza. El modelo debe expresar esa independencia:
copiar un contexto antiguo completo de 30 bits podría restaurar atributos
equivocados. No asumir que ningún caso posterior a fallo puede distinguirlo.

Separar las liberaciones de cabecera de las escrituras de metadatos y de
los descuentos globales en EOP. Retener las reglas de R3 para alias de
punteros/slots y casos malformados; no reemplazarlas por una lista idealizada
de seis eventos que ignore prioridad o pérdida de contexto en R7.

## Referencias en vuelo y wrap

Una referencia temporal a una consulta pendiente debe expirar al copiar su
descripción/resultado al contexto retenido. El payload puede persistir mucho
más que la profundidad del pipeline; no debe seguir apuntando a un slot
circular que ya fue reutilizado. Un header nuevo puede reemplazar un contexto
incompleto y provocar fallo: el resultado tardío del header anterior tampoco
puede sobrescribir el contexto nuevo.

Fijar primero la máxima edad de cada referencia y el máximo de nuevas
asignaciones por ciclo; derivar después los bits de slot/versión. Probar wrap
con referencias antiguas todavía presentes, reemplazo de contexto, reset frío
y todos los slots ocupados. El contador de secuencia de diagnóstico no basta
por sí solo como identificador de propietario.

## Entregable y criterio para pasar a RTL

Construir un modelo de descripciones pendientes desde los selectores crudos
con estado propio y compararlo por secuencia con R7, incluyendo cada fila
posterior a fallo. Debe producir las mismas consultas, escrituras y deltas
en RC4→RQ2, tags repetidos consecutivos y terminal con payload retenido.
Un contraejemplo obliga a revisar la representación, sin filtrar el tráfico.

Después congelar formato, profundidad, dueño de cada registro y reglas de
retiro/cancelación; medir por separado consulta/composición y aplicación.
El banco de integración completo sigue siendo obligatorio. Ver
[el contrato de pipeline](PCIE_INVENTORY_PIPELINE_CONTRACT.md) y
[los requisitos de integración](PCIE_METADATA_EFFECT_INTEGRATION_PLAN.md).
