# Recuperación del observador Q/K/V por falta de espacio

Fecha: 2026-10-16, 13:46 UTC. **Supervisión recuperada; simulación original
continúa sin reinicio.** No cambia fuentes, criterios, orden ni dependencias.

## Diagnóstico y preservación

La auditoría R79 encontró muerto el observador local PID 26380. Su log
registra `OSError: [Errno 28] No space left on device` al publicar el estado
atómico; el último estado completo es de las 13:29:58 UTC. También falló
el intento de guardar el estado de error. El estado anterior permaneció legible.

Vivado/XSim continuaron en la VM: runner 952238 y XSim 954836, tanda
`20261016T124000Z-V7-qkv-memory-vivado-r3`. El fallo local no es un resultado
del banco. Estado, log, archivos parciales de publicación, configuración y
adaptador quedaron respaldados y restaurados en D:/NFS. Archivo de recuperación:
`ea9e60f3969a79e29f5f7ed5d736fca7cf99b1b421f139a69a34fdb65605c4cc`.

## Espacio y reconexión

Cuatro checkpoints históricos del 5 de octubre se copiaron a
`D:/AI/ntransformer/artifacts/ntrpu-board-vivado/relocated-c-20261016-r1/`.
Se verificó cada archivo antes de retirar su copia física en C:, y otra vez
mediante los enlaces que conservan las rutas originales. Se retuvieron
1.013 archivos y se liberaron 3.356.205.753 bytes; no se descartó evidencia.
El manifiesto `VERIFIED.json` y los inventarios individuales están en esa carpeta.

`queue_qkv_memory_observer_resume_r1.py`, PID 25464, ejecuta exclusivamente
la observación y revisión terminal del coordinador original fijado por hash.
Excluye admisión y lanzamiento. Conserva el hash del coordinador original
para sus sucesores y registra separadamente identidad y preparación del
adaptador. La auditoría R80 comprueba ambos y los PID remotos originales.

## Estado y límite

A las 13:46:48 UTC, ciclo 8.562.419: tres trabajos publicaron correctamente
3.072 resultados cada uno, con 2.410.870, 2.410.274 y 2.410.274 ciclos.
El cuarto continúa; no se declara aprobado el banco ni se libera el slot.
La revisión independiente y el respaldo terminal siguen pendientes.
El estado activo está en `qkv-memory-vivado-r3-coordinator.json`; la admisión
del adaptador está en `qkv-memory-observer-resume-r1-attached.json`, ambos en
`build/cloud-setup/20261002-board/`.
