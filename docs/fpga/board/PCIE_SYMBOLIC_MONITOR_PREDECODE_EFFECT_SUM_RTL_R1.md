# Etapa E: suma única de consumos, experimento RTL R1

## Cambio propuesto

La ruta C→E de 979 sigue incumpliendo setup (−1,183 ns estimados).
`prepare_operation` compone hasta seis efectos de consumo con varias
saturaciones. El candidato `ntrpu_pcie_symbolic_effects_predecode_sum_r1`
usa tres sumas de parejas, una suma intermedia y una suma final de
14 bits, seguida de una única saturación a 2.049.

Mantiene el primer word activo, su valor X/Z original, el conteo saturado
a dos y el indicador de palabras desconocidas. Si este indicador está
activo, la suma publicada sigue en cero. Las activaciones se construyen
mediante los mismos `if` procedurales. `apply_summary` depende de pedir
el resumen, aunque no haya consumos activos. Se conservan registro C,
matches de admisión, latencia de un ciclo y reset frío.

El [modelo Z3](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_MODEL_R1.md)
prueba la identidad aritmética y el límite de 12.282. La nueva fuente
es independiente; ese modelo no la califica como RTL.

## Banco real previsto

Comparar el E original sin modificar y el candidato mediante entradas
C públicas y sus **4.554 bits de salida**, comprobando estabilidad antes
del flanco y equivalencia de cuatro estados después:

- 262.144 escenarios: todos los controles 0/1/X/Z combinados con cuatro
  perfiles de words, incluidos datos desconocidos activos y enmascarados.
- 16.896 casos de bits individuales de los seis words con 64 máscaras.
- 4.096 registros pseudoaleatorios deterministas, incluidos tags de
  admisión desconocidos; siete ciclos dirigidos y cinco de reset/cierre.

Total: **283.148 filas**. Compilar seis fuentes defectuosas: saturación
a 2.048, omisión del consumo final, primer word invertido, indicador
desconocido omitido, `apply_summary` incorrecto y prefijo X aceptado.
Exigir sus testigos exactos y rechazar 16 corrupciones de evidencia.

## Orden y límites

Se encola después del cierre revisado y respaldado de la cadena RQ;
es una prueba independiente de E, sin modificar esa variante integrada.
Límite: 1.800 segundos y dos hilos. Cada desenlace queda en D: y NFS.
La cápsula R9 añade sus bancos, mutantes y snapshots a la cobertura previa.

Un PASS permitirá decidir la integración según las métricas RQ obtenidas.
Se necesitarán regresión completa, síntesis propia y replay de netlist
antes de atribuir mejoras de timing. No se cambia aún la arquitectura
ni se declara resuelto el camino físico E→D.
