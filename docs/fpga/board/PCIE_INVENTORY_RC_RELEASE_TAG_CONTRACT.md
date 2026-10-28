# R13: tag RC directo con guarda de liberación

Preparación del 2026-10-08. Candidata experimental, sin aprobación funcional
ni timing hasta revisar sus propias ejecuciones.

## Cambio y alcance

Partir de R12, selector R3 intacto. Cambiar exclusivamente
`rc_release_tag[i]=selected_release_tags[i*8+:8]` por
`rc_release_tag[i]=rc_data[i*128+64+:8]`. Se conservan los dos consumidores
guardados: eliminación del bitmap y comprobación de liberaciones en
`rq_admission_flags`. Los bits inactivos del tag interno pueden cambiar;
su valor solo es relevante bajo el mismo `rc_release_valid[i]`.

Intentar retirar la dependencia ownership→tag→comparación observada en
el [ruteo diagnóstico](PCIE_INVENTORY_CLOCK_QUICK_ROUTE_RESULTS.md).
No se modifica capacidad, orden RC→RQ, latencia, contexto retenido,
guardas, contadores, fallos ni certificación de vacío. No se promete
mejora de timing por simplificar una dependencia.

## Validación previa a síntesis

El banco adicional instancia el productor real. Inyecta deliberadamente
bitmap, contador y `observing` mediante `force` solo para esta prueba
combinacional aislada. Compara liberaciones con una recurrencia secuencial
independiente, el tag válido y ambos consumidores usando los tags antiguos.
No representa una secuencia de protocolo alcanzable ni sustituye integración.

Exigir 12.832 casos, 7.680 con X/Z: todos los tags en cuatro posiciones,
los 16 patrones SOP y 256 combinaciones de punteros, controles conjuntos
de cuatro estados y envenenamiento de cada bit de las entradas del selector
en cuatro contextos. Deben observarse tags válidos y diferencias en tags
enmascarados. Un productor mutante que invierte el bit 0 debe fallar bajo
una liberación válida, en el primer caso; el revisor conserva seis controles
de corrupción de evidencia.

Conservar todos los bancos RTL R12: ownership/ancestry, selección, admisión
RQ, aritmética/máscaras, suma RC, cinco integraciones y siete mutantes reales
del monitor completo. Sus referencias R7 y vectores permanecen intactos;
en esos bancos el DUT recibe únicamente entradas públicas. El total pasa
de 17 a 19 arranques XSIM. Respaldar todos los resultados, incluso errores.

## Puertas posteriores

Solo después de la revisión completa, sintetizar con el flujo comparable
de R12 a 4 ns y sin excepciones. Registrar todos los peores caminos y
recursos, ejecutar los cinco bancos binarios RTL y del netlist propio.
Mantener R12 como preferida si el ensayo empeora. Timing físico, integración
nativa, reset/CDC, DDR/512 GiB e inferencia siguen abiertos; conservar
la [segmentación pendiente](PCIE_INVENTORY_PIPELINE_CONTRACT.md).
