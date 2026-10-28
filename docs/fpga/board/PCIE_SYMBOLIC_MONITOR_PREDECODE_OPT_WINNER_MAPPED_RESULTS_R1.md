# Replay de la netlist optimizada — resultado R1

Tanda 975, `20261011T155336Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-opt-winner-binary-r1`,
aprobada y respaldada el 2026-10-11. Ejecutó la netlist del candidato
optimizado de 974, conservando testbench, Tcl, vectores y referencias de
971. Ninguna fila de otra tanda se reutilizó como resultado.

## Resultado

- **174.436 filas binarias**, 16 bancos y comparación de los 447 bits
  públicos antes y después de la captura.
- Seis controles negativos detectados y 34 corrupciones de evidencia
  rechazadas por el revisor independiente.
- Diez pruebas de quiet y veinte muestras de perturbación/restauración;
  se conservan la contabilidad de origen y los bancos drenados.
- Compilación y hash de la netlist verificados; las referencias están
  renombradas y separadas del DUT mapeado.

La ejecución terminó con código 0 en **1.446,773 segundos**. El archivo
de 92.599.930 bytes y 300 miembros está verificado en D: y NFS.

## Identidad

| Evidencia | SHA-256 |
| --- | --- |
| Netlist optimizada propia | `ce83d51ac442692ae3d4a0ced664249bf9b6505ef5f087dd974eb4db6b166e90` |
| Archivo de resultados | `7df0d27f9ca88f4a9eb9b8738747e65aa1b2b92a3792a68c25860676187fceb5` |
| Revisión independiente | `e49d15b0b7bf34105b5bf6b4cbe39db174bf61f6f754029469fc34d740e77b30` |
| Fuentes conservadas de 971 | `846ea982aeabcf541a4b1986c6556219079c028a86678de4d03e156f358392c9` |

## Alcance y siguiente paso

El resultado prueba estos bancos binarios después de la optimización
que eliminó 456 FF. No incorpora SDF ni califica X/Z o temporización
física. Permanecen los fallos de setup/hold de la
[comparación optimizada](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAIRED_OPT_RESULTS_R3.md).

Se habilitó el [ensayo aislado de prioridad RC](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_PLAN_R1.md).
Su PASS revisado y respaldado es requisito para la regresión integrada
posterior. No se inició otro ruteo.
