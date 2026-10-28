# Conteo C: replay de netlist propia R1

## Entrada verificable

El replay espera el cierre de la síntesis del contador C, incluida su
comparación contra RQ 984. Extrae la netlist funcional del archivo terminal
verificado y comprueba tamaño, SHA-256, módulo superior y sus 1.384 bits
de entrada y 447 bits de salida.

Los módulos de referencia mantienen nombres separados para impedir
sustituciones por el candidato. El banco y la contabilidad proceden
del replay RQ 985 calificado; todas las filas se ejecutan de nuevo.

## Cobertura y revisión

Se comparan 174.436 filas binarias en 16 bancos, antes y después del
flanco. Se conservan diez perturbaciones de quietud y restauraciones,
seis controles de entrada y 34 alteraciones de evidencia. Los seis bancos
drenados deben conservar el tráfico original sin cancelaciones artificiales.

La revisión exige fuentes efectivamente compiladas, hashes de referencias,
correspondencia entre archivo y logs, cobertura exacta, contabilidad
y primeros testigos de los controles. Los tiempos de estabilización
se mantienen respecto al banco de referencia.

Este replay observa la interfaz pública de la netlist. Los 13 bits
internos de conteo se comparan en la regresión RTL y en el preflight;
su jerarquía no se presupone conservada después de la síntesis.

## Respaldo y alcance

Configuración, netlist, fuentes y revisores quedan fijados antes del
lanzamiento. R15 conserva las fuentes pendientes; cada resultado se
recoge y verifica en D: y NFS. Un fallo detiene los sucesores para inspección.

Un PASS acredita equivalencia funcional binaria dentro del banco.
No incluye SDF, cierre temporal de ruteo ni throughput medido.
La cobertura RTL X/Z y la validación física de PCIe/DDR/placa conservan
sus requisitos independientes.
