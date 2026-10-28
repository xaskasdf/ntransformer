# Replay del netlist propio del bitmap RQ

## Dependencia

El coordinador espera síntesis propia revisada, auditoría EDIF y
comparación con 979, con respaldo de cada resultado. No reutiliza el
netlist del padre: extrae y verifica el del candidato RQ, comprueba su
interfaz completa y registra el SHA-256 antes de ejecutar XSIM.

## Prueba

La preparación deriva del replay 980 aprobado y conserva sus referencias
independientes y funciones de contabilidad. Las recetas de renombrado
de módulos se reproducen para evitar que una referencia use por accidente
el DUT. El snapshot es nuevo e incluye todo el banco de replay.

Se requieren **174.436 filas binarias nuevas** en 16 bancos, 447 bits
públicos, diez sondas de quietud y veinte muestras de perturbación y
restauración. Los seis controles de entrada deben fallar en sus testigos
originales; el revisor debe rechazar 34 alteraciones de evidencia.
Se conservan todos los resultados, incluidos diagnósticos o timeout.

## Límites y decisión

El límite de ejecución es 3.000 segundos. La salida 0 necesita revisión
independiente y respaldo verificado en D: y NFS. Los bancos con X/Z
pertenecen a la validación RTL anterior; no se atribuyen al replay binario.

No se aplica SDF ni se demuestra cierre físico. Solo después del PASS
y de comparar recursos y rutas se decide si conservar esta variante y
qué experimento físico aporta información. No repetir automáticamente
los ruteos Default que ya agotaron su presupuesto sin cerrar timing.
