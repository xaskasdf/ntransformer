# A/B R1: prueba focalizada de cuatro estados

Preparada el 2026-10-09. **Pendiente de ejecución y revisión.** Los trece
bancos positivos de [A/B R1](PCIE_SYMBOLIC_NUMERIC_AB_R1_PLAN.md) pasaron,
pero el banco de contexto no detectó `ignore_enable`; los controles terminal
y cold quedaron sin ejecutar. Esta prueba se habilita para investigar esa
cobertura, sin habilitar síntesis hasta completar también los controles crudos.

## Alcance

Se instancian los módulos A y B completos, sin cambiar sus fuentes.
El banco compara los 2.723 y 2.892 bits registrados y comprueba estabilidad
entre flancos. Las expectativas usan las tres funciones originales del
modelo, copiadas con un cambio de nombre verificable. Esta prueba
complementa la comparación independiente contra R7; comparte `apply_effect`
con el modelo y no sustituye esa referencia.

## Cobertura prevista

- 262.144 combinaciones: todos los conteos de palabras de 11 bits, cuatro
  alineaciones bajas, ocho valores frontera de remaining y enable 0/1/X/Z.
  Los cuatro slots EOP ejercitan simultáneamente terminal 0/1/X/Z.
- 3.672 inyecciones individuales X/Z en los 51 bits de cada una de las
  nueve descripciones, sobre cuatro semillas/resúmenes especificados.
- 792 inyecciones X/Z en cada bit de palabras de esos nueve consumidores.
- 64 combinaciones de validez 0/1/X/Z del registro, consultas y EOP.
- Dos casos dirigidos iniciales, incluido `words=0` con dirección baja 3,
  y seis flancos de reset frío; ambas etapas están ocupadas al primer reset
  posterior al llenado.

Son 266.681 flancos comparados: 266.626 orígenes válidos capturados,
266.622 retirados y cuatro cancelados. Los siete módulos mutados del banco
completo deben fallar en una comparación A/B/reset, no en compilación.
Se comprueban también los casos inválidos, las copias canónicas cero y la
retención del contexto sin propiedad.

Las inyecciones incluyen descripciones fuera del conjunto canónico para
caracterizar la transformación local. No prueban toda la combinación de
51 bits de cuatro estados ni la composición algebraica arbitraria.
Quedan la síntesis/replay propios y la validación y retiro del monitor
completo, incluidos los temporales de 15 bits de cabecera.
