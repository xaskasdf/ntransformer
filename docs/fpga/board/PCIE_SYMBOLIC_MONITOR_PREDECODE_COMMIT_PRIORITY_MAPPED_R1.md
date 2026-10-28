# Replay propio del monitor RC por prioridad — R1

El banco binario se encola detrás de la síntesis propia revisada, comparada
con 970 y respaldada. Usa exclusivamente el `funcsim.v` exportado por esa
síntesis, enlazado por SHA-256 con su archivo y DCP; no toma la netlist de
un candidato anterior.

## Cobertura conservada

El banco procede del replay 971: 174.436 filas nuevas en 16 bancos,
comparación de los 447 bits públicos, diez probes de quietud con restauración,
seis mutaciones de entradas y 34 corrupciones de evidencia.
La adaptación cambia los nombres del módulo y de los artefactos. Los
comparadores, vectores, referencias RTL independientes y funciones del
revisor se conservan. Las referencias mantienen sus módulos originales
con alias separados de los de la netlist.

## Ejecución y condiciones

Se crea un snapshot nuevo tras el cierre de síntesis para incluir los
archivos de replay sin modificar su snapshot anterior. Se verifican la
interfaz completa de la netlist, los hashes de todas las dependencias y
los manifiestos antes del lanzamiento. El ejecutor espera disponibilidad
real de la herramienta, usa dos hilos en Tcl y limita la tanda a 3.000 s.

Todo resultado terminal se respalda en D: y NFS antes de revisarlo.
Un PASS binario no sustituye las 183.058 filas RTL con cobertura X/Z,
ni demuestra cierre físico, comportamiento con SDF o rendimiento de
inferencia. No se lanza colocación o ruteo automáticamente.

Coordinador: `continue_symbolic_monitor_predecode_commit_priority_binary_r1.py`.
Preparación: `pcie-symbolic-monitor-predecode-commit-priority-binary-r1-preparation.json`.
Ambos están en `build/cloud-setup/20261002-board/`; las transformaciones de
fuentes quedan registradas para reproducir y revisar el banco.
