# Comparación temporal por etapas de E, B y C — R1

## Objetivo y orden

Las tres variantes se comparan por separado con RQ 984. El margen global
de E sigue en −2,768 ns porque B→R domina; no permite determinar por sí
solo cuánto cambiaron C→E o E→D. El diagnóstico separará esos efectos
antes de decidir qué cambios combinar.

La cola espera el replay propio de C ya encolado. Después ejecuta, en
orden, E-sum frente a RQ, B-bound/winner frente a RQ y admission-count
frente a RQ. Cada variante debe tener su síntesis y netlist propia
revisadas y respaldadas. Cada par espera el cierre del anterior.

## Mediciones

Se reutilizan el Tcl y el parser calificados en la tanda 987. Cada
ejecución reabre ambos DCP archivados, verifica sus hashes y clasifica
todos los FDRE en siete grupos disjuntos: C, E, D, A, B, retiro y top.
Incluye los nuevos registros B por su jerarquía, sin fijar cantidades
esperadas a partir del RTL.

Se registran los 49 pares ordenados de grupos, con estado setup/hold y
hasta veinte rutas por par. La revisión contrasta los endpoints con el
inventario real de primitivas, el peor margen y el retardo. Rechaza diez
alteraciones de evidencia por ejecución. E/S permanece en el reporte
global y no se mezcla con las rutas entre registros.

## Límites y respaldo

No se optimizan ni rutean los DCP, ni se cambian sus restricciones.
Cada par tiene dos hilos y hasta 900 s. Los reportes de fanout,
utilización, relojes, restricciones y advertencias se conservan junto
con las entradas.

R16 guarda el plan, helpers y snapshots pendientes. Cada tanda terminal
queda verificada en D: y NFS. Un fallo detiene la cola sin reiniciar.
Las advertencias siguen siendo evidencia diagnóstica; estos márgenes
anteriores al ruteo no califican timing físico, throughput ni una placa.
