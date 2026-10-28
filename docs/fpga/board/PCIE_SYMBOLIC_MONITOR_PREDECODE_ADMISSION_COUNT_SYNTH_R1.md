# Conteo C: síntesis propia R1

## Dependencia y comparación

Esta etapa espera la [regresión integrada](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FULL_R1.md)
revisada y respaldada: 183.058 filas, seis controles negativos, 34 controles
de evidencia y cobertura adicional del contador de 13 bits en 16 bancos.
Las fuentes conservan el C focalizado y sus cuatro mutantes calificados.

Se compara contra RQ 984. Únicamente cambia la formación de
`allocate_count`; E, B y el retiro winner siguen siendo los originales.
La comparación exige igual dispositivo, compilador, estrategia y reloj.

## Trabajo y evidencia

Antes de sintetizar se ejecutan 529 filas de monitor/range_return, con
31 perturbaciones de quietud, comparación pública y contador siguiente.
El preflight usa 26 fuentes; la síntesis usa 14, en
`xcvu5p-flva2104-2-i`, con reloj de 4 ns y retardos E/S de cero.
Se conservan `RuntimeOptimized`, jerarquía `rebuilt`, dos hilos y un
presupuesto de 1.800 s.

Se guardan DCP, EDIF, netlist funcional, restricciones y reportes.
El revisor comprueba el grafo combinacional, recursos, advertencias,
cobertura de restricciones y ausencia de excepciones. La evaluación
debe informar tanto el peor margen como los endpoints incumplidos;
un cambio de ruta crítica no constituye por sí solo una mejora física.

## Secuencia y límites

El coordinador congela recetas, fuentes y revisores antes de lanzar;
R15 conserva también las fuentes pendientes. Cada resultado terminal
se respalda en D: y NFS antes de habilitar el siguiente paso.
Un error detiene la cadena sin reiniciar el ensayo automáticamente.

La síntesis revisada permite el [replay propio](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_MAPPED_R1.md).
La equivalencia binaria y el cierre físico siguen siendo gates separados.
Esta etapa no modifica capacidad DDR, frecuencia objetivo ni el contrato
PCIe para obtener un resultado favorable.
