# Recuperación R4 de la cola de predecodificación

## Motivo y alcance

El 2026-10-11 la revisión inicial de B995 detuvo la cadena por una
referencia obsoleta al nombre del módulo en la lista de avisos. La
[revisión R2 de la síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_SYNTHESIS_RESULTS_R1.md)
aprobó los archivos ya respaldados. Los nueve coordinadores R1 detenidos
conservan sus estados y hashes; se comprobó que sus procesos terminaron.

R4 cierra esa revisión en un registro nuevo y reanuda diez etapas que
todavía no tenían owner ni configuración congelada. Conserva sus nombres
R1 de ejecución, fuentes, vectores, runners y revisores. Ocho preparadores
nuevos cambian solamente dependencias y enlaces de evidencia. No repite
las tandas 993–995 ni transforma un fallo histórico en un PASS.

## Orden y condiciones

1. Replay binario de la netlist B995: 174.436 filas nuevas.
2. Conteo C: banco aislado, monitor completo, síntesis y replay propio.
3. Diagnósticos temporales E, B y C, cada DCP frente a RQ 984.
4. Habilitaciones de filas E/D: banco aislado y monitor completo.

Cada etapa exige revisión aprobada y respaldo de la anterior. Se sella
el trabajo antes de ejecutar y después de revisar; cualquier fallo detiene
la cadena y se respalda. No se califican timing físico ni arquitectura
a partir de simulación sin SDF o de estimaciones de síntesis.

## Reproducción y evidencia

En `build/cloud-setup/20261002-board/`:

- Plan: `pcie-symbolic-monitor-predecode-pipeline-recovery-r4.json`,
  SHA-256 `856d75a309c28dd43255aae465ce653143027d4f102dc6f924b3dcc3eb46595e`.
- Auditoría estática: `pcie-symbolic-monitor-predecode-pipeline-recovery-r4-audit.json`,
  SHA-256 `b7c36a6d2eb2dede0b9812831e4e84200fecfe6c7c9f92e7ec333e6a6599e97a`.
- Ejecutor: `continue_symbolic_monitor_predecode_pipeline_recovery_r4.py`.

La auditoría verificó 61 enlaces, las ocho derivaciones exactas, la ausencia
de procesos anteriores y los contratos de las diez etapas. El ejecutor
arrancó a las 21:00 UTC. Sus coordinadores `*-r4-coordinator.json` muestran
el estado vigente; los registros R1 permanecen como evidencia histórica.
