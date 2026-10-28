# Integración numérica: validación y retiro pendientes

Inspección de fuentes del 2026-10-09. **Notas de diseño; sin circuito ni
prueba de equivalencia nuevos.** Complementan el [plan A/B](PCIE_SYMBOLIC_NUMERIC_AB_R1_PLAN.md).

## Anchos antes de truncar

En `ntrpu_pcie_request_inventory_r7.sv`, `available` y `delivered` tienen
15 bits signed durante la validación de cabecera. El contexto almacena
`cdelivered` en 13 bits. A/B R1 reproduce esas copias de 13 bits, las consultas
observadas y las contribuciones EOP; no demuestra equivalencia de los
predicados de validación que usan el temporal más ancho.

La siguiente revisión debe transportar el resultado guardado completo o
reproducir exactamente su cálculo con `words` y el valor capturados. No
deducir equivalencia bajo X/Z de que los casos binarios caben en 13 bits.
Conservar signedness, tamaño de expresiones, comparación mixta, ternario
y comportamiento del `if` cuando el predicado es desconocido.

## Retiro de bytes

R7 calcula un candidato signed de 23 bits desde los 21 bits almacenados,
los spans RQ y la reducción RC. Publica sus 21 bits bajos y conserva los
dos altos para detectar rango fuera de `0..1052672`. El resultado debe
pertenecer al mismo origen que tags, lecturas, parciales y fallos publicados.
El fallo de rango visible en ese flanco debe incorporarse después al sticky;
no debe desaparecer si el siguiente candidato vuelve al rango permitido.

Solo `pre_observing` autoriza el avance de estado del origen: armar el
monitor y procesar un beat son transiciones distintas. Reset frío borra
contador, guardas, sticky y etapas pendientes conjuntamente. El vaciado
de cuatro ciclos actual solo alcanza B; al añadir R hay que extender el
banco y volver a comprobar toda la contabilidad.

## Validación tardía y vacío

El control temprano no cubre toda la validación de requester/atributos,
dirección/remaining, terminal, máscaras, longitud o payload desconocido.
Reconstruir el recorrido RQ/RC desde el registro del origen y su contexto
previo, manteniendo los casos malformados y la prioridad original.
Las consultas numéricas usan su lane capturada; no se recarga una fila
por tag ni se usa la propiedad del momento del retiro.

El certificado de vacío requiere además revocación inmediata por actividad
actual, fallos tempranos y obligaciones pendientes hasta R. A/B conserva
`effect_pending`, pero todavía no ofrece ese certificado. R12 sigue siendo
la referencia del monitor completo.
