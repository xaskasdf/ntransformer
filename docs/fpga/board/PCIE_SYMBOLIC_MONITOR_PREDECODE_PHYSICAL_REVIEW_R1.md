# Revisión física de predecode R1

## Estado al 2026-10-11 UTC

La implementación Quick `20261011T093541Z` y su reapertura
`20261011T100051Z` quedaron respaldadas y aprobaron la revisión independiente.
Se comprobaron los diez grupos de caminos y la identidad de los informes
tras reabrir el DCP. Resultado: 104.500 LUT, 27.106 FF, WNS −13,498 ns y
WHS −2,972 ns. Esta revisión permitió el replay propio `20261011T101255Z`,
que después aprobó la [comparación funcional binaria](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROUTED_MAPPED_RESULTS_R1.md).
El timing sigue sin aprobar.

Certificado local: `20261011T093541Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-placement-r1-independent-monitor-predecode-physical-review-r1.json`.
Los archivos de implementación y reapertura, y esta revisión, están
respaldados en D: y NFS. [Siguiente comparación física](PCIE_SYMBOLIC_MONITOR_PREDECODE_TIMING_ROUTE_R1.md).

## Criterios

El revisor enlaza los archivos certificados de implementación y reapertura,
las fuentes, las dependencias, los DCP, los informes y la biblioteca BUFGCE.
Comprueba las 1.831 señales públicas, el reloj de 4 ns, los retardos de E/S
cero y la ausencia de excepciones temporales o una fuente `HD.CLK_SRC` añadida.
Exige rutas completas, inventarios idénticos al reabrir y conexión del BUFGCE
a todos los registros. Conserva los avisos de cobertura DRC incompleta y
de fronteras OOC; cualquier otro aviso detiene esta revisión.

Los conteos físicos y los slacks proceden del candidato. No se imponen las
26.540 FF, 19 réplicas, ubicación del BUFGCE ni retardos de la referencia.
Las nuevas celdas deben ser FDRE con un nombre de réplica asociado a una
FDRE original. Esto clasifica los cambios; **no demuestra su equivalencia**.
La simulación de la netlist resultante sigue siendo obligatoria.

## Verificación del lector

`tools/audit_ntrpu_symbolic_monitor_predecode_physical_r2.py` reprodujo los
124 caminos de los ocho grupos de la referencia archivada, sus recursos
y sus slacks: WNS −11,835 ns y WHS −3,904 ns. Rechazó 22 corrupciones
dirigidas de advertencias, restricciones, rutas, registros, reloj, recursos,
DRC e informes. La primera versión rechazaba un formato válido: el reloj
lista siete segmentos jerárquicos, todos `ROUTED`. R2 comprueba un estado
por segmento y conserva los controles de conductor, cargas y ruta.

Prueba local: `pcie-symbolic-monitor-predecode-physical-checker-r2-verification.json`,
SHA-256 `85bb7a91cdb41df4832aaa8f99724318e5cd937823c147e270c164381d2713b8`.
R1 y el registro de su corrección se conservan.

## Dependencias

Replay optimizado → colocación/ruteo → reapertura y auditoría EDIF →
comparación de diez grupos temporales, incluidos E→D → revisión física
independiente → respaldo previo → replay ruteado propio.

El coordinador de revisión espera los respaldos y publica su resultado
completo de forma atómica. Solo autoriza una simulación diagnóstica:
no concede aprobación de timing, SDF, PCIe nativo, CDC, reset, DDR ni placa.
