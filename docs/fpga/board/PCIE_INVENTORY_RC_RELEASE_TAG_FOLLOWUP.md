# Después de R13: caracterizar los destinos críticos de R12

R13 empeora la estimación OOC: WNS −7,268 ns frente a −7,068 ns y
74 LUT adicionales. Conservar R12 como base. La comparación archivada
`pcie-rc-release-tag-path-comparison.json` verifica restricciones idénticas
y el mismo Tcl salvo nombres de candidatas. No es una comparación física.

## Evidencia que orienta el siguiente paso

Las veinte rutas de setup de cada síntesis parten de `rc_user[76]`:
dieciocho terminan en `remaining_bytes` y dos en `byte_range_high`.
La peor ruta R12 mide 11,079 ns, con 43 niveles; R13 mide 11,279 ns,
con 45. Los nombres presentes sitúan selección de prefijo, aritmética
de dirección, bytes entregados y acumulación en la cadena; por sí solos
no prueban qué transformación RTL mejorará el circuito.

El ruteo Quick mostró otra ruta, bitmap→admisión→escritura de tabla.
No combinar sus retardos con las estimaciones sin colocar. El ensayo R3
de deltas registrados ya empeoró una versión anterior y desplazó el peor
camino a la tabla. Repetirlo sobre R12 requiere una justificación nueva.

## Tanda siguiente: lectura del DCP R12

Preparado `tools/ntrpu_board_pcie_inventory_path_census.tcl`, todavía sin
ejecutar en Vivado. La comprobación local de sintaxis y expresiones Tcl,
contra el inventario archivado, clasifica los 10.729 registros sin huecos
ni solapamientos. No valida las consultas de timing. Se conserva la
procedencia de la ayuda 2026.1 en
`pcie-inventory-path-census-preparation.json`.

1. Abrir el DCP original de síntesis R12, SHA-256
   `293ee2f048b1f8c6624d871e1cdb5a1d8ded88f1c58b717a39f89903e57b437a`.
   Verificar archivo, fuente, parte, relojes y restricciones contra su
   revisión y respaldo. No ejecutar síntesis, optimización ni implementación.
2. Consultar la ayuda instalada antes de congelar Tcl. Exportar conteos y
   rutas de setup/hold por destinos: bytes/rango; tags/lecturas; tablas de
   metadatos; contexto RC/RQ; registros de captura/validación/retiro.
   Resolver nombres contra las celdas reales. Registrar grupos vacíos,
   solapamientos y destinos no clasificados, sin excluirlos del resumen global.
   Los conteos cubren todos los registros; las rutas son muestras de hasta
   veinte por consulta, no una enumeración de todos los caminos del circuito.
3. Separar entradas→registro, registro→registro y registro→salida. Mantener
   el resumen global de 4 ns y cero excepciones; los filtros son consultas
   de reportes, nunca restricciones que eliminen caminos.
4. Respaldar y revisar el resultado antes de seleccionar otro cambio RTL.
   Comparar la cadena de bytes con las alternativas de tabla/admisión.
   Documentar el corte de registros propuesto, dependencias entre beats,
   forwarding, latencia y conservación de estados posteriores a fallos.

## Puertas de diseño

Un ensayo nuevo debe conservar los bancos completos, capacidad, RQ2/RC4,
ritmo sin pausas, contexto retenido, reset frío y revocación de vacío.
Exigir RTL revisado antes de síntesis comparable y replay del netlist propio.
No promover R13 a ruteo ni repetir Default agotado. La segmentación completa
sigue bajo el [contrato vigente](PCIE_INVENTORY_PIPELINE_CONTRACT.md);
endpoint nativo, reset/CDC, DIMM exacto, DDR/512 GiB e integración del cómputo
siguen abiertos en [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
