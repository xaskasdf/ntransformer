# Síntesis propia de la suma E

## Condiciones de entrada

Esperar el PASS revisado y respaldado de la integración completa E:
183.058 filas nuevas, 447 bits, seis controles negativos, 34 corrupciones
rechazadas y seis bancos drenados sin cancelar tráfico. Se conserva la
calificación aislada de E y la del C bitmap RQ heredado.

Congelar un snapshot nuevo y enlazar los hashes de las 26 fuentes de
simulación y las 14 de síntesis. Solo E cambia respecto al monitor 984;
C, D, A, B y retiro winner conservan sus fuentes.

## Medición comparable

Repetir el filtro de 529 filas (`monitor` y `range_return`), con 31 sondas
de quietud. Sintetizar en `xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos de
E/S de 0 ns, `RuntimeOptimized`, jerarquía `rebuilt`, dos hilos y límite
de 1.800 segundos. No se coloca ni rutea en esta tanda.

Conservar DCP, EDIF, netlist funcional, XDC, inventario de primitivas y
reportes de recursos, timing, restricciones, DRC, metodología y excepciones.
Auditar ausencia de ciclos combinacionales e inspeccionar las advertencias
reales. Una clase inesperada detiene la revisión para diagnóstico.

La comparación usa exclusivamente la síntesis padre **984**, revisada y
con replay propio aprobado: 92.913 LUT, 27.560 FF, 568 CARRY8 y WNS estimado
−2,768 ns. Verificar versiones, builds y restricciones antes de calcular
diferencias. No elegir otra referencia según el resultado.

## Cola y límites

El coordinador espera el cierre de E completo. Cada desenlace conserva
archivo terminal en D: y NFS, seguido de auditoría, revisión, comparación
y cápsula R11. No reiniciar automáticamente ante fallos de observación.

Una síntesis revisada habilita replay de su propia netlist. Un WNS mejor
no demuestra cierre físico: C→C y B→R también incumplen en el padre.
La temporización, inferencia y placa siguen siendo validaciones separadas.
