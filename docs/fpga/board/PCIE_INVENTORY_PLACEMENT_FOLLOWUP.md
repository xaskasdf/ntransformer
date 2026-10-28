# Siguiente ensayo: contexto de reloj del inventario PCIe

## Motivo y alcance

La implementación R1 de R12 tiene una peor ruta interna de 22,926 ns:
2,911 ns de lógica y 20,015 ns de ruteo. El router informó 51.664 pines
críticos simultáneamente para setup y hold. La auditoría del checkpoint
confirmó un gap del reloj padre, retardos de reloj estimados y ausencia
de ubicaciones de partición en la frontera. Son observaciones distintas;
no demuestran que corregir el reloj resolverá la cadena combinacional.

Antes de otra modificación del productor, aislar el efecto de ese
contexto. Conservar R1 como diagnóstico fallido de timing y R12 como
referencia funcional; no comparar sus cifras OOC con frecuencia de placa.

## Primera tanda: preflight de un buffer interno

Depende del respaldo, revisión de frontera/DAG y replay completo de la
netlist R1. Usar una carpeta nueva y el mismo DCP de **síntesis** R12,
SHA-256 `293ee2f048b1f8c6624d871e1cdb5a1d8ded88f1c58b717a39f89903e57b437a`.
No partir del DCP ya colocado ni alterar RTL, capacidad, latencia,
secuencias de reset o bancos.

Tcl preparado, todavía sin ejecutar:
`tools/ntrpu_board_pcie_inventory_clock_preflight.tcl`, con argumentos
`SYNTH_DCP REFERENCE_OPTIMIZED_DCP NEW_OUTPUT`. Su marca final acredita
exportación y comprobaciones iniciales; exige revisión independiente de
polaridad, propiedades, DAG y equivalencia funcional antes de implementar.

1. Capturar ayuda y propiedades instaladas. Identificar la red del puerto
   `clk`, sus segmentos, sus cargas y cualquier buffer existente.
2. Solicitar `CLOCK_BUFFER_TYPE BUFGCE` sobre la **red**, conservando las
   restricciones temporales originales de 4 ns y E/S cero. No añadir el
   `HD.CLK_SRC` usado para un buffer externo supuesto.
3. Ejecutar sólo `opt_design`. Exigir un único BUFGCE, entrada desde `clk`,
   habilitación constante y todas las cargas de reloj supervivientes en
   su salida. Guardar propiedades, inventarios, restricciones, DCP, EDIF,
   netlist funcional y reportes. Comparar puertos y registros con el DCP
   `optimized.dcp` de R1, que ya eliminó registros durante optimización;
   explicar cualquier diferencia y rechazar cajas negras o cargas sin reloj.
4. Si la IP/herramienta no inserta ese buffer o la conectividad no coincide,
   respaldar el diagnóstico. No sustituirlo por una reescritura manual de
   redes dentro de la misma tanda. Aún no ejecutar colocación/ruteo.

AMD 2026.1 documenta la inserción mediante lógica de optimización y el
uso de la propiedad sobre redes después de síntesis:
[UG912: CLOCK_BUFFER_TYPE](https://docs.amd.com/r/en-US/ug912-vivado-properties/CLOCK_BUFFER_TYPE)
y [UG904: BUFG Optimization](https://docs.amd.com/r/en-US/ug904-vivado-implementation/BUFG-Optimization).
Comprobar su comportamiento real en este OOC; la documentación no valida
por sí sola el experimento. La referencia histórica UG905 sobre buffers
OOC corresponde a 2023.2, no a una guía nueva de 2026.1.

## Puertas para implementar después

Revisar explícitamente el primitivo de reloj en el auditor estructural,
con controles de polaridad, habilitación y conexiones; no declararlo
un corte secuencial arbitrario para ocultar ciclos. Exigir replay de los
cinco bancos binarios R12 de la netlist resultante, con arranque de 200 ns
y muestreo de 1 ns, sin RTL del DUT ni SDF.

Sólo después preparar una implementación separada: registrar la ubicación
del buffer, integridad de la distribución de reloj, setup/hold completos
y registro→registro, gaps, DRC y fanout. No relajar E/S, incertidumbre,
severidades ni excepciones. La frontera externa seguirá sin representar
el endpoint real; incluso una mejora mantendrá abiertos V4–V8.

Si persiste la cadena interna, el siguiente cambio de arquitectura deberá
abordar lectura de propiedad por tag, selección de prefijos y actualización
de direcciones. Un pipeline que cambie latencia requiere contrato nuevo
de aceptación, capacidad y drenaje; no hereda los resultados de R12.
