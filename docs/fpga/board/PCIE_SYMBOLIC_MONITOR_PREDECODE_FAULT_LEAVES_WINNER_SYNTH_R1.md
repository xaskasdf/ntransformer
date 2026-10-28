# Síntesis condicional de la bandera de contexto

## Dependencias

Preparación sin resultado de síntesis. El coordinador
`continue_symbolic_monitor_predecode_fault_leaves_winner_synth_r1.py`
exige aprobación y respaldo de las 183.058 filas nuevas de integración,
precedidas por el ensayo aislado de 47.734 muestras. Las pruebas formales
del [predicado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_R1.md) no
sustituyen esos bancos.

## Ejecución y revisión

Una tanda con dos hilos y límite de 1.800 s verifica 529 filas y 31 sondas
de quiet, y sintetiza un monitor a partir de 14 unidades fuente. Mantiene
`xcvu5p-flva2104-2-i`, reloj de 4 ns, retardos de E/S de 0 ns,
`RuntimeOptimized`, jerarquía `rebuilt` y modo OOC.

El revisor comprueba fuentes, preflight, restricciones, avisos e inventario.
La auditoría EDIF verifica ausencia de ciclos combinacionales. Los resultados
terminales se respaldan en D: y NFS, incluidos errores o timeout.
Un aviso nuevo exige inspección.

## Criterio de avance

Se compararán recursos, setup, hold y profundidad con el padre por prefijo
966, verificando iguales opciones y builds. Su referencia es 105.840 LUT,
27.534 FF, 2.122 CARRY8, 24 niveles y WNS estimado −2,546 ns.
No se añaden ciclos ni registros ni se presupone mejora.

La síntesis revisada habilita el [replay de su propia netlist](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_MAPPED_R1.md).
Estos flujos no encolan optimización de reloj ni ruteo; la decisión física
requiere las medidas reales y la equivalencia funcional revisada.
