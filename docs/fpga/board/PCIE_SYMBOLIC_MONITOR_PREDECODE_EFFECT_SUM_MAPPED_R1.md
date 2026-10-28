# Replay de la netlist propia de E

## Entrada y procedencia

Esperar la síntesis E revisada, su comparación con 984 y el respaldo
completo. Extraer exclusivamente su netlist funcional del archivo terminal
verificado; comprobar hash, módulos y los puertos públicos: 1.384 bits
de entrada y 447 de salida. No reutilizar la netlist RQ como candidato.

Derivar banco y herramientas del replay RQ **985**, aprobado sobre
174.436 filas. Mantener las referencias independientes con sus nombres
separados y las funciones de contabilidad, sin modificar su semántica.
La receta de enlace del banco debe reconstruirse desde el padre original
y todas las sustituciones históricas, incluida la nueva instancia E.

## Ensayo y aceptación

Ejecutar **174.436 filas binarias nuevas en 16 bancos**, con comparación
de 447 bits. Exigir diez sondas entre flancos con restauración, seis
controles negativos de entrada y rechazo de 34 corrupciones de evidencia.
Conservar los bancos, las filas esperadas y las 19 fuentes SystemVerilog
del replay padre; enlazar por separado netlist y primitivas de Vivado.

Esperar la inicialización global de simulación y comprobar cada fase
con los tiempos del banco calificado. No hay anotación SDF: el ensayo
comprueba comportamiento binario del mapeo, no propagación temporal ni
equivalencia X/Z de la netlist.

## Cola y respaldo

El coordinador espera el cierre de síntesis; congela otro snapshot,
respalda antes de ejecutar y conserva cada desenlace en D: y NFS.
Revisar la evidencia independiente y sellar la cápsula R11 antes de
habilitar la siguiente decisión. No reiniciar una tanda por un timeout
del observador ni sustituir un fallo por resultados de otro candidato.

El resultado se interpretará junto con área y timing estimado. Todavía
harán falta cierre físico, pruebas del PCIe nativo e integración de placa.
