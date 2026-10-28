# Diagnóstico de rutas desde el checkpoint colocado

## Motivo y dependencia

La tanda `20261008T213940Z` agotó 3.600 s durante la corrección intensiva
de hold. Está respaldada: archivo SHA-256
`980131e4dc07c74b998949ba3c3627bf5618fcf883b5bc1f7741b5684b56d169`.
No produjo DCP ni netlist finales de ruteo. Su último STA de fase completa
es el de colocación; las cifras del router son intermedias.

Después de revisar ese fallo, abrir exclusivamente `placed.dcp`, SHA-256
`bbc5657611d5bdb7987a1b4624cbf43d93ad56757946ef6c39d326037598c377`.
Parte VU5P FLVA2104 -2-i, top `ntrpu_pcie_inventory_join`, un BUFGCE.
No repetir síntesis, optimización ni colocación. Carpeta nueva, cuatro
hilos y máximo **1.200 s**, incluidos exportación y reportes.

## Pregunta y ejecución

Obtener una solución de conexiones y medir sus rutas internas con
`route_design -directive Quick`. La ayuda instalada y
[UG835 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/route_design)
describen esta directiva como prioritaria en tiempo de ejecución y sin
optimización dirigida por timing. No promete cumplir setup ni hold.

Conservar exactamente reloj de 4 ns, E/S cero, incertidumbre heredada,
ausencia de nuevas excepciones y ausencia de `HD.CLK_SRC` supuesto.
Guardar también los XDC de contexto contenidos en el DCP: su reloj de
4 ns ya existía en la entrada optimizada y permaneció idéntico.
No rebajar DRC ni introducir restricciones para ocultar incumplimientos.

Capturar ayuda, identidad de entrada, estado colocado, estado final/parcial,
EDIF, netlist funcional, XDC, recursos, DRC, metodología, setup/hold completos
y FDRE→FDRE, fanout, rutas incompletas y conexiones del buffer.
Respaldar también cualquier nuevo timeout o error.

## Criterios y límites

La conclusión esperada es diagnóstica: rutas, slacks y coste observado.
Una ejecución sin errores o una red legal no cierra el objetivo de 4 ns.
No comparar `Quick` con `Default` como si fueran ensayos de igual estrategia,
ni presentar una mejora de runtime como mejora de arquitectura.

Antes de aceptar funcionalidad, reabrir el DCP exacto, auditar EDIF/ciclos,
restricciones y reloj, y reproducir su propia netlist en los cinco bancos
R12. Si no exporta esos artefactos, esa puerta permanece cerrada.
El cierre temporal dirigido por timing y la integración con el
[BUFG_GT de XDMA](PCIE_INVENTORY_NATIVE_CLOCK_ORIGIN.md) siguen pendientes,
al igual que los demás criterios de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
