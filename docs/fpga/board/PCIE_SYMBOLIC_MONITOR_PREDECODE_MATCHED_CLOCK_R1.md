# Preparación física comparable: relojes RQ984 y E/D

Fecha: 2026-10-11 UTC. Extiende los
[requisitos de comparación física](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_COMPARISON_REQUIREMENTS_R1.md).
No hay resultados de estas dos optimizaciones todavía.

## Dependencias

La cola conserva las tandas existentes: síntesis/replay/diagnóstico E/D,
`mask_rows`, GEMV Llama y FFN real completa. Después de sus revisiones y
respaldos se ejecutan dos diagnósticos de reloj, primero RQ y después E/D.
Cada miembro exige su propia síntesis y replay de 174.436 filas, seis mutantes
y 34 controles de evidencia. Un fallo se conserva y detiene sus dependencias.

RQ usa exclusivamente el DCP 984 con SHA-256
`924f3c41cedf89364e30201552777ac748b6fa66d087d70f679be7f883a59631`.
E/D usa el DCP revisado por su propio diagnóstico; no sustituye la referencia
por las variantes aisladas E, B, C ni por el ruteo antiguo 955.

## Procedimiento idéntico

Cada tanda abre dos copias del mismo DCP y ejecuta `opt_design`. En una
se solicita un BUFGCE interno, con CE constante y sin cambiar RTL. Se
conservan parte `xcvu5p-flva2104-2-i`, período 4 ns, E/S cero, dos hilos
y límite de 1800 s. No se añaden excepciones temporales ni `HD.CLK_SRC`.

Los revisores exigen puertos y registros conservados, conexión exclusiva
del reloj a todos los FDRE e igualdad del grafo lógico tras contraer el
BUFGCE. Revisan restricciones, biblioteca AMD, DRC, recursos y timing.
Conservan los controles negativos del flujo anterior; las funciones de
auditoría son idénticas y las modificaciones de identidad quedan registradas.

## Alcance y continuación

`continue_symbolic_monitor_predecode_matched_clock_r1.py` crea propietarios
y snapshots nuevos, verifica archivos y respalda cada resultado en D: y NFS.
Este paso termina en DCP optimizados revisados. No autoriza implementación.

Faltan después los replays propios de ambas netlists optimizadas, la colocación
y ruteo Quick comparables (1800 s cada uno), reapertura/auditoría y replays
ruteados. Los intentos Default agotados no se repiten. La comparación final
debe incluir setup/hold global, E→D, otras fronteras y E/S; ni un PASS funcional
sin SDF ni una mejora parcial certifican timing, PCIe/DDR o placa.
