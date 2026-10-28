# Comparación de rutas por etapa: prioridad RC

Después del replay binario propio se abrirán los DCP de síntesis del
candidato RC y de su padre 970. Ambos deben contar con replay propio
revisado y respaldado. No se modifica el diseño al generar los informes.

Se conserva exactamente el Tcl usado en la tanda 972: VU5P, reloj de
4 ns, dos hilos y límite de 900 s para la lectura de ambos DCP. Se
recopilan inventarios completos de registros, utilización, restricciones,
timing global y fanout. Los siete grupos C/E/D/A/B/R/TOP son disjuntos y
cubren todos los FDRE. Cada uno de los 49 pares dirigidos registra setup
y hold, con hasta veinte rutas por subconjunto no vacío. Las rutas de
entrada/salida siguen en el informe global.

La revisión liga endpoints con las primitivas del DCP archivado y verifica
slack, retardos, niveles y sus sumas. Usa el parser R2 ya corregido y diez
controles de corrupción, sin repetir el error de interpretación de rutas
con cero niveles. Los ensayos de corrupción no dependen de que una ruta
nueva contenga un número concreto de LUT1.

Antes de encolarlo, el revisor nuevo reprodujo exactamente las medidas
calificadas de 972 sobre 220 informes archivados y rechazó sus diez
corrupciones. Registro `pcie-symbolic-monitor-predecode-commit-priority-stage-fixture-r1.json`,
SHA-256 `a6cc7ce96f7e0e1f358bb93c3bef9a498fcc15db637bbf5a0840e62c8888771e`.
Esta prueba local no aporta mediciones del candidato nuevo.

La comparación mostrará si la prioridad anticipada reduce C→C y si otra
etapa pasa a dominar. Son estimaciones de síntesis: no demuestran cierre
físico ni rendimiento. Las advertencias quedan conservadas como evidencia
diagnóstica; esta lectura no promueve la arquitectura por sí sola.

Coordinador `continue_symbolic_monitor_predecode_commit_priority_stage_timing_r1.py`
en `build/cloud-setup/20261002-board/`, detrás del replay RC R2. Cada
resultado terminal, incluso incompleto, se respalda en D: y NFS antes de
revisarlo. No se altera la cola de síntesis/replay ya congelada.
