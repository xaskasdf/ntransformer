# Identidad de fuentes de la captura DDR R1

Fecha: 2026-10-16. **Revisor preparado y probado; elaboración DDR pendiente.**
Complementa la [captura R1](POWER_ACTIVITY_R1.md) y su
[análisis de actividad](POWER_ACTIVITY_DETAIL_R1.md). No cambia el paquete,
los estímulos ni el turno después de BRAM.

## Comprobaciones

El [revisor](../../../tools/review_ntrpu_ddr_activity_identity_r1.py) trabaja
sobre la tanda restaurada y terminada. Verifica el ZIP fuente R1 por SHA-256
y su inventario completo; contrasta los proyectos de compilación XSim con
los archivos conservados en el respaldo.

- Exige top, generador PRBS, startup, modelo y banco observador en la lista
  compilada. Rechaza cambios no autorizados en sus bytes, fuentes externas,
  duplicados y parámetros DDR distintos del traslado de `CustomParts`.
- Verifica que la geometría sea privada y conserve sus bytes. El banco
  derivado sólo puede añadir el bloque de observación de `SIMULATION`.
- Registra idioma, biblioteca, hash, includes y macros. Las fuentes IP
  regeneradas quedan identificadas para revisión, incluso si la ejecución
  no presenta errores.
- Busca la elaboración terminada y la observación nativa de `SIMULATION=FALSE`
  tanto en el DUT como en su generador. Conserva fallos de la invocación.

Las listas de compilación y los hashes no demuestran por sí solos la
equivalencia con un netlist. Las bibliotecas vendor precompiladas, los
includes y las macros siguen abiertos a inspección; también la asociación
de señales SAIF con el DCP original. `ready_for_power_estimate` permanece
en `false`. No se califica el DIMM físico ni la potencia de la placa.

## Aplicación automática y respaldo

El [observador de identidad](../../../tools/ntrpu_ddr_activity_identity_followup_r1.py)
espera la captura terminada y respaldada. Revalida el TAR, todos sus archivos
restaurados y el hash NFS; aplica la revisión sin modificar esa tanda y
restaura un ZIP complementario en D: y NFS. No adquiere Vivado ni escribe
la cola. La espera de doce horas no altera el límite de ejecución de R1.

Estado: `build/cloud-setup/20261002-board/ddr-activity-identity-followup-r1-monitor.json`.
El resultado se guardará en `<tanda>-activity-identity-r1/activity-identity.json`.
No duplicar el análisis mientras el observador esté procesando esa tanda.

## Evidencia de preparación

Pasaron **26 comprobaciones sintéticas del revisor y 15 del observador**
con resultados idénticos en Python 3.14 Windows y 3.10 Linux. Incluyen
cambios de ECC, fuente ajena, geometría externa, alteración del banco,
respaldo corrupto y conservación de entradas. No son simulaciones RTL.

Además, el parser leyó 122 registros de compilación del banco DDR anterior.
El banco derivado coincide con la preparación archivada:
SHA-256 `f160e2ca0492621ba53fc0bb3d7a3b8c20fab8146be708fe80f7d291ad2450cb`.
Se pudieron leer los 143 parámetros del XCI original. Estas comprobaciones
de formato no sustituyen la elaboración de R1.

Evidencia en `build/cloud-setup/20261002-board/`:
`ddr-activity-identity-r1-python-validation.json`,
`ddr-activity-identity-followup-r1-python-validation.json` y
`ddr-activity-identity-r1-archived-checks.json`.
Respaldo: `artifacts/ntrpu-board-vivado/ddr-activity-identity-handoff-20261016-r1/`.
