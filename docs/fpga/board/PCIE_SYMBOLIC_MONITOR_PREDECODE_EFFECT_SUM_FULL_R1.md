# Integración completa de la suma E, R1

## Cambio y referencia

Integrar el E aprobado en [986](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_RESULTS_R1.md)
en una copia del monitor RQ validado por 983 y su replay propio 985.
Solo cambia la instancia E por `ntrpu_pcie_symbolic_effects_predecode_sum_r1`;
se conservan C con bitmap RQ, D, A, B, retiro winner, puertos y cinco
relojes de latencia. Las referencias R3, numéricas y de contexto permanecen
sin modificaciones. Los límites registrados B se estudian por separado.

## Regresión y aceptación

Ejecutar de nuevo los 16 bancos y **183.058 filas RTL**, comparando
447 bits públicos. Exigir 183.138 muestras antes y después del flanco,
31 sondas de quiet con perturbación/restauración, los seis bancos drenados
sin cancelación de tráfico original y seis controles negativos de entrada.
Conservar las funciones de contabilidad calificadas y el rechazo de
34 corrupciones de evidencia. Verificar las 26 fuentes compiladas por caso.

Antes de congelar fuentes, verificar el PASS aislado de E: 283.148 filas,
4.554 bits, seis fuentes defectuosas y 16 controles de evidencia; también
el PASS original de C y la selección RC heredados. Las recetas de derivación
y los hashes deben enlazar cada candidato con su referencia y respaldo.

## Cola y respaldo

La ejecución espera el cierre revisado y respaldado del consumidor B,
después del productor y la lectura RQ/979 por etapas. Estos experimentos
solo fijan el orden; sus módulos no se incorporan a esta variante.
El congelador comprueba también los resultados E, RQ completo y netlist RQ.

Límite de 1.800 segundos y dos hilos; snapshot nuevo, archivo terminal
en D: y NFS para cualquier desenlace, revisión independiente y cápsula R11.
Si falla una dependencia, detener esta rama para inspección sin reiniciar
ni alterar las tandas anteriores.

La auditoría previa detectó un sufijo de revisión heredado de RQ.
La preparación R2 corrige ese nombre antes de cualquier ejecución,
conserva las recetas R1 y registra la enmienda; no cambia RTL ni bancos.

Un PASS habilita síntesis propia comparada con 984 y replay de su netlist.
No implica cierre de timing ni autoriza repetir los ruteos Default agotados.
