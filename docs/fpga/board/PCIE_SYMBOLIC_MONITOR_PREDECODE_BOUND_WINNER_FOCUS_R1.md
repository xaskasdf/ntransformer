# Límites de payload sobre la arquitectura winner

## Cambio aislado

El primer consumidor de límites deriva del contexto por prefijo 965.
El monitor RQ 983/984 usa winner: comparar directamente ambos mezclaría
dos modificaciones. Este candidato parte del winner calificado y añade
únicamente las cinco entradas de límites, su selección por contexto/SOP
y la comparación superior de payload. Conserva la prioridad del último
evento SOP/EOP, incluidos empates, y todos los demás predicados.

`ntrpu_pcie_symbolic_late_fault_leaves_bound_winner_r1` es una fuente
nueva; no reemplaza los módulos anteriores. El productor B calificado
en 988 permanece separado hasta validar la conexión completa.

## Banco y aceptación

Repetir **309.884 muestras nuevas**, comparando los 16 bits de fallo y
las 32 contribuciones por lane antes del OR. Conservar las 262.144
combinaciones enteras de límite, 10.240 casos SOP/EOP, 17.352 casos X/Z
y 20.000 registros pseudoaleatorios. Referencias: validador original
y winner sin modificar; el banco aporta límites combinacionales desde B.

Compilar seis fuentes defectuosas con los mismos testigos exactos:
contexto omitido, extremo inclusivo, truncamiento, consulta rotada,
desconocido tratado como máximo y extremo inferior omitido. Exigir los
21 controles de evidencia del banco anterior y sus ocho fuentes compiladas.

## Dependencias y siguiente paso

Se prepara detrás del replay propio de E, conservando toda la cola.
El congelador deberá verificar también el productor B R2 y el banco
del primer consumidor. Límite de 1.800 segundos, dos hilos, revisión
independiente y respaldo de cada desenlace en D: y NFS.

Un PASS permitirá conectar este consumidor y B al monitor RQ para una
comparación atribuible a los límites registrados. Todavía serán necesarias
regresión completa, síntesis propia y replay de netlist; no hay mejora
de timing demostrada por preparar este candidato.
