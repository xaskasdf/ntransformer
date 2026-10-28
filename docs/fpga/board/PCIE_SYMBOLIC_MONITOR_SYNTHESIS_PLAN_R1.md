# Medición conjunta del monitor PCIe

## Dependencias y alcance

El script `tools/ntrpu_board_pcie_symbolic_monitor_synth_r1.tcl` está
preparado y **no ejecutado**. Requiere aprobar y respaldar la regresión
completa de 183058 filas: los catorce bancos originales, el retorno de
rango y el caso dirigido de quiet. Los 16 controles reales deben ser
rechazados en las comparaciones y filas previstas.

La unidad a medir es `ntrpu_pcie_symbolic_monitor_r1`: C3/E2/D2/A1/B1/R1,
guardas de configuración y publicación de inventario vacío. No sumar
recursos ni frecuencias obtenidos de módulos aislados.

## Ejecución y evidencia

Repetir primero las 252 filas dirigidas y las 277 de retorno de rango con
el banco corregido R2. Congelar y comprobar las fuentes compiladas.
Sintetizar las catorce fuentes de producción en `xcvu5p-flva2104-2-i`,
OOC, reloj de 4 ns, I/O de cero, `RuntimeOptimized`, jerarquía `rebuilt`,
dos hilos y sin excepciones temporales.

Conservar DCP, EDIF, netlist funcional, XDC, primitivas, utilización,
temporización, relojes, DRC, metodología, excepciones y `check_timing`.
Revisar bucles, celdas sin correspondencia, advertencias por fuente,
caminos no restringidos y los extremos de los peores caminos.
Respaldar resultados positivos y diagnósticos en D: y NFS.

## Decisiones posteriores

Las referencias aisladas siguen incumpliendo setup: C/E/D tiene WNS
−2,546 ns y R, −3,096 ns. En R, el camino observado va de `rc_user[64]`
a `fault_latched[10]` y atraviesa 30 niveles. El recorrido secuencial
de 16 lanes en el validador tardío es un candidato a investigar;
todavía no se ha demostrado que modificarlo cierre temporización.

La composición permitirá distinguir entradas externas de conexiones
entre registros reales. Reproducir después **su propia netlist** antes
de optimizar o colocar/rutear. Toda partición nueva debe conservar
causalidad, contexto por valor, caudal, cancelación solo por frío y
publicación alineada; si añade latencia, actualizar y comprobar todos
los pendientes y orígenes. Las medidas OOC no acreditan cierre físico,
colas internas PCIe ni recepción remota de escrituras posted.
