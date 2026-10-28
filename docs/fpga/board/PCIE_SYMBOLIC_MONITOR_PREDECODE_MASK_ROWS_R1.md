# Máscaras de prefijo RC con índices fijos — experimento R1

## Hipótesis y cambio

Los [reportes RQ/E/B/C](PCIE_SYMBOLIC_MONITOR_PREDECODE_INPUT_PATHS_R1.md)
sitúan varias rutas críticas entre `rc_user` y las máscaras registradas
en C. El experimento parte del selector RC de RQ; no incorpora el contador
C que empeoró WNS ni cambia el contrato de la interfaz.

Cada bit de `header_prefix_mask` se forma con índices de bucle fijos,
en lugar de escribir con el origen del evento como índice dinámico.
Se conservan los cuatro candidatos, condiciones procedurales, propiedad,
retained seed, selección previous y commit. No añade registros ni etapas.
La transformación puede cambiar el mapeo; todavía no demuestra ahorro
de área ni mejora temporal.

## Comprobación local y banco RTL

El modelo finito compara **640.000 ensamblajes de máscara**, incluidos
600.000 con condiciones X/Z. Cubre cuatro orígenes internos en
`{-1,0,1,2,3}` y cuatro condiciones de cuatro estados por cabecera.
No sustituye ejecución SystemVerilog ni prueba universal de entradas raw.

Se reutilizan los bancos heredados sin modificar referencias ni vectores:
1.186.064 comprobaciones de propiedad/ancestros, incluidas 1.026.544 con
X/Z sobre 840 bits, y 89.632 vectores adicionales de 800 bits.
Tres fuentes defectuosas omiten el origen cero, invierten los bits de
origen u omiten la cabecera tres. Deben fallar en el comparador original
con testigos exactos calculados desde sus vectores; el revisor rechaza
trece corrupciones de evidencia.

## Dependencias y decisión

La tanda espera el cierre revisado y respaldado de los diagnósticos E/D
por etapas. Así conserva toda la cola E/B/C y E/D en curso. Tiene dos
hilos, límite de 900 s, transporte SSH por stdin y respaldo R18 en D: y NFS
para cualquier desenlace. R18 incluye los nuevos mutantes y snapshots
pendientes; las fuentes ejecutadas anteriormente se conservan.

Un PASS aislado habilita decidir una integración completa sobre RQ.
Solo su posterior síntesis comparable y replay propio permitirán evaluar
el cambio. No se sustituye el candidato activo ni se inicia ruteo aquí.
