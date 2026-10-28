# Validación tardía: siguiente revisión propuesta

Inspección del 2026-10-09. **Diseño pendiente de implementar y probar**;
no cambia la evidencia A/B R1 ni habilita el monitor completo. Su ejecución
queda detrás del replay propio de A/B y conserva R12 como referencia.

## Interfaz y origen

El consumidor recibirá `b_record_t` completo. Todos los datos proceden
de `origin`: beat RQ/RC, requester esperado, parciales previos, contexto
retenido, consultas numéricas y control. No consultará la tabla actual ni
reconstruirá propiedad desde un tag reutilizado. `record.pre_observing`
autoriza la validación del beat; `record.valid` indica que existe un origen.

Mantener A/B R1 inmutables. Una nueva revisión de
`ntrpu_pcie_inventory_validation.sv` adaptará el recorrido existente de
cabeceras y payload a estas copias. Las señales de referencia solo llegan
al banco de pruebas, nunca al candidato.

## Temporal de 15 bits

R7 valida longitud y terminal con `available` y `delivered` signed de
15 bits. El validador existente reconstruye `delivered={2'b0,cdelivered}`;
esa transformación todavía no está demostrada para todos los casos X/Z.

La candidata recalculará los cuatro temporales de cabecera desde los
valores A y `words` capturados, conservando literalmente tamaños,
comparación mixta y ternario de R7. El payload retenido y la contribución
EOP seguirán usando sus copias canónicas de 13 bits. No extender los
13 bits como sustituto de la operación original sin una prueba específica.

## Retiro y fallos

R validará y publicará control, bytes y fallos del mismo origen. El
candidato de bytes conserva 23 bits; solo los 21 bajos forman la base
del beat siguiente, incluso después de un fallo. Los dos altos sirven
para validar rango y alimentar el fallo persistente. Conservar la reducción
RC y sus anchos originales; no sumar dos veces EOP que comparten lane.

Los registros válidos de ciclos ociosos no equivalen a obligaciones.
Propagar `effect_pending` y el control pendiente hasta R, y mantener la
revocación inmediata de quiet por actividad o fallo actual. Reset frío
cancela todas las etapas y el acumulador; reset de usuario o caída de link
no descartan orígenes ya capturados.

## Orden de pruebas

1. Comparar el temporal completo y los predicados de cabecera contra R7:
   extremos binarios, count cero, alineaciones y X/Z en cada operando.
2. Probar validación y retiro con todos los bancos crudos, errores y filas
   posteriores al fallo. Comparar por origen y añadir cinco flancos de
   vaciado, con contabilidad independiente de retiros y cancelaciones.
3. Rechazar cambios de truncamiento, origen, enable, doble consumo, reset
   y pérdida del fallo persistente. Verificar revocación inmediata de quiet.
4. Medir el circuito resultante y ejecutar su propia netlist. Un éxito
   aislado no cierra el déficit de C/E/D, CDC/reset, DDR, pesos ni placa.
