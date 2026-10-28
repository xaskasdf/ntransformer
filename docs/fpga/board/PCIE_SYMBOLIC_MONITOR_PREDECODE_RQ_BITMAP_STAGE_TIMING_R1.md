# Bitmap RQ: comparación de timing por etapas

## Propósito

Comparar la síntesis propia de RQ con su padre 979 para localizar el efecto
de anticipar las comparaciones de tags. El padre tiene WNS −3,704 ns en
C→C; C→E y B→R también incumplen setup. Un cambio de ruta crítica puede
ocultar una mejora o un empeoramiento de otro tramo.

## Orden y entradas

La tanda espera el replay binario propio RQ de 174.436 filas y el cierre
de la prueba E ya encolada. La referencia es el DCP de 979, cuyo replay
980 está aprobado y respaldado. Ambos DCP se vinculan a sus archivos,
inventarios de registros y revisiones mediante SHA-256.

La prueba E solo determina el orden de ejecución: no forma parte de
ninguno de los dos DCP comparados. Se conserva la cola en curso.

## Medición y revisión

Reutilizar sin cambios el Tcl y el parser corregido R2 ya usados en 981.
Abrir los checkpoints en lectura, sin optimizar, colocar ni rutear.
Particionar todos los FDRE en siete grupos disjuntos: C, E, D, A, B, R y
TOP. Informar setup y hold para cada uno de los 49 pares ordenados, con
hasta veinte rutas por conjunto, más timing global y fanout.

El revisor conserva exactamente sus funciones de medición y los diez
controles de evidencia de 981. Exigir identidad de dispositivo/top,
estado sintetizado, pertenencia de endpoints, cobertura de registros,
coherencia de niveles lógicos y consistencia de slack/retardo.

Presupuesto: 900 segundos y dos hilos. Guardar todo desenlace, sus
fuentes, restricciones y reportes en D: y NFS, verificando hashes y
legibilidad. La cápsula R9 conserva preparación, cola y revisión.

El generador local R1 se detuvo antes de escribir productos al buscar una
etiqueta ausente. R2 conserva el diagnóstico y aplica solo los cambios de
nombre existentes en cada helper. No hubo ejecución Vivado en ese fallo;
los nombres de configuración, owner, ejecutor y revisor permanecen R1.

## Límite de la decisión

Registrar diferencias por etapa y conservar los avisos como diagnóstico;
no considerarlos individualmente calificados por este análisis. Los
retardos siguen siendo estimaciones de síntesis. Esta tanda permite
elegir el siguiente cambio, pero no califica timing físico ni promueve
la arquitectura de placa.
