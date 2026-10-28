# Comparación física RQ984/E/D: ejecución y reapertura

Fecha: 2026-10-11 UTC. Preparada para ejecutarse después de los bancos Llama,
los dos diagnósticos de reloj y los
[dos replays optimizados](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_CLOCK_BINARY_R1.md).
Todavía no hay colocación ni ruteo de estos dos DCP.

## Identidad y condiciones

Cada diseño parte de su `optimized_clock.dcp`, cuyo archivo y netlist deben
coincidir con su revisión estructural y replay de 174.436 filas. La referencia
parte exclusivamente de RQ984; la variante contiene únicamente el cambio E/D
ya aprobado en RTL. El ruteo antiguo 955 valida las herramientas de inspección,
pero no aporta cifras a esta comparación.

Se conservan parte `xcvu5p-flva2104-2-i`, reloj 4 ns, E/S cero, un BUFGCE,
dos hilos y restricciones originales. Se ejecutan `place_design` y
`route_design -directive Quick`, con límite de 1800 s por diseño. No se
repite optimización ni se añaden excepciones, `HD.CLK_SRC`, pines de placa
o cambios de severidad. No se repiten los intentos de ruteo Default agotados.

## Evidencia por tanda

La ejecución guarda DCP, netlist, EDIF, XDC, inventarios, reloj, fanout,
DRC/metodología y timing global. Exporta setup/hold entre registros,
entrada→registro, registro→salida, entrada→salida y el subconjunto E→D.
Los estados parciales o timeouts también se respaldan; no se convierten en PASS.

Cada resultado completo se reabre en otra tanda de hasta 900 s, sin comandos
que alteren implementación o restricciones. La auditoría exige el mismo DCP,
EDIF idéntico salvo la fecha, inventario de primitivas coincidente y controles
negativos de grafo y reloj. Registra las diferencias frente al diseño optimizado;
no las interpreta como equivalencia funcional.

## Orden y límites

`continue_symbolic_monitor_predecode_matched_physical_r1.py` implementa y reabre
RQ, luego E/D, conservando la cola anterior y respaldos en D: y NFS. Si la
implementación queda parcial, conserva la evidencia y exige inspección.

El cierre de esta cola solo acredita exportación y reapertura revisadas.
Faltan revisión independiente de restricciones/avisos/timing, comparación de
las cifras de ambos miembros y replay propio de cada netlist ruteada. Los
fallos de setup/hold siguen siendo fallos; no se aprueban PCIe/DDR, ECAD ni placa.
