# Conteo de admisiones C: integración completa R1

## Entrada y cambio

La integración parte del monitor RQ 983 y sustituye únicamente C por
el [candidato de conteo temprano](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FOCUS_R1.md).
Conserva E original, D, A, B y retiro winner. La interfaz y los cinco
ciclos de latencia permanecen iguales; todavía no combina este cambio
con las variantes de suma E o límites B.

El congelador exige primero PASS del C real: 51.110 filas, cuatro fuentes
defectuosas detectadas, 21 controles de evidencia, archivos certificados
y hashes de las fuentes. Un diagnóstico detiene los sucesores.

## Banco integrado

Se ejecutan las **183.058 filas** originales de los 16 bancos contra
referencias sin modificar. El revisor exige los 447 bits públicos antes
y después del flanco: 183.138 muestras por fase, 31 perturbaciones de
quietud y sus 62 muestras de perturbación/restauración. Se conservan
los seis bancos drenados sin cancelar tráfico original.

El testbench añade una comparación independiente de **13 bits** entre
el C candidato y el C original: `allocate_count` y `nreads`. Se comprueba
en cada muestra positiva, antes de evaluar las salidas públicas, para
detectar divergencias aunque una alarma ya esté acumulada.

Los seis controles de entrada conservan su comparador público y sus
primeros testigos originales. El revisor mantiene los 34 controles de
evidencia existentes y añade ocho para la cobertura del contador. Se
compilan 26 fuentes; las comprobaciones internas solo están en el banco.

## Cola, respaldo y alcance

Espera el cierre focalizado, que a su vez sigue a la cadena B ya encolada.
La configuración, recetas, referencias y revisores se fijan por SHA-256.
R15 conserva las fuentes pendientes y cada resultado terminal se
respalda en D: y NFS, incluidos los fallos.

Un PASS permite medir síntesis y luego ejecutar la netlist propia.
La simulación no demuestra ahorro de recursos, mejora de timing,
inferencias correctas ni funcionamiento de la placa completa.
