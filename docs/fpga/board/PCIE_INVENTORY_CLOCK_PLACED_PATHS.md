# Dependencias críticas tras colocación del inventario

Análisis provisional de la tanda `20261008T213940Z` (2026-10-08).
El router sigue en ejecución. La fuente es `placed_timing.rpt`, capturada
sin alterar el proceso; aún debe cotejarse con el archivo terminal.

## Muestra medida

Las 20 rutas de setup informadas tienen slacks entre **−6,192 y −6,156 ns**.
Son rutas correlacionadas de una muestra limitada, no un inventario completo
ni retardos finales de ruteo. Sus fuentes son tres registros de
`allocated_tags`: bit 115 (15 rutas), 119 (4) y 35 (1).

| Familia del destino | Rutas |
| --- | ---: |
| `address_next` | 15 |
| `remaining` | 1 |
| `remaining_bytes` | 2 |
| `byte_range_high` | 2 |

Hay 20 pines destino en 19 registros: el reporte incluye pines **D y CE**.
La ruta peor llega a `byte_range_high[0]/D`: 42 niveles, 10,055 ns de datos,
3,404 ns de lógica y 6,651 ns de interconexión. De sus 43 nets de datos,
33 todavía tienen retardo estimado y 10 figuran como enrutadas. Sus retardos
impresos suman exactamente los 6,651 ns reportados.

El reporte de recursos colocado contiene 51.828 LUT, 10.733 FF, 1.895 CARRY8
y un BUFGCE. El DCP optimizado tenía 10.723 FDRE; revisar los cambios de
celdas desde el DCP final antes de atribuirlos a replicación específica.

## Consecuencia para el diseño

La ruta peor atraviesa selección de propiedad, `selected_prefix_mask`,
aritmética de metadatos, bytes entregados y acumulación global. Es coherente
con las dependencias visibles del productor R12 y el selector R3; los nombres
generados no demuestran por sí solos equivalencia con cada expresión RTL.

Cambiar sólo el acumulador no aborda las otras 16 rutas de tabla de esta
muestra. El experimento anterior de deltas registrados ya desplazó el peor
destino hacia la tabla sin mejorar WNS
([resultado](PCIE_INVENTORY_COUNTER_RETIMING_RESULTS.md)). La selección
aritmética tardía también tiene un ensayo previo; no repetirlo como novedad.

Al terminar el ruteo, contrastar esta muestra con setup/hold finales, rutas
de reloj y congestión. Si persiste la cadena, retomar el
[contrato de arquitectura segmentada](PCIE_INVENTORY_PIPELINE_CONTRACT.md):
debe resolver dependencias entre beats y reutilización de tags sin frenar
el observador. Una FIFO o registros externos al bloque completo no resuelven
esa recurrencia. Esta nota no fija aún una nueva candidata ni su latencia.

## Evidencia

Observación `…-placed-phase-observation.json`, SHA-256
`a7c4918bd4d2608e88fe1b188805bf36062946489b9dc60a7b2d3ad3a987669e`.
Análisis `…-placed-path-sample-analysis-r2.json`, SHA-256
`010675c36e349d905bb5337d25e8214a811322cbc8fcca573fa0ac0b60671d9a`.
Ambos están en `build/cloud-setup/20261002-board/`, con el prefijo de tanda
completo. R0 del analizador rechazó incorrectamente un destino CE; R1 contó
pines como celdas. R2 conserva ambos tipos y separa sus cantidades. Las
versiones anteriores permanecen disponibles; no se modificó el reporte.
