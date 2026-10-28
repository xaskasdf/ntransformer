# Cola de validación de la bandera, recuperación R2

## Secuencia activa

El ensayo aislado 968 está [aprobado y respaldado](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_RESULTS_R1.md).
Los coordinadores activos en `build/cloud-setup/20261002-board/` son:

| Etapa | Coordinador | Condición de salida |
| --- | --- | --- |
| RTL integrado | `continue_symbolic_monitor_predecode_fault_leaves_winner_r2.py` | 183.058 filas nuevas, quiet, seis mutantes, 34 controles de evidencia y respaldo |
| Síntesis | `continue_symbolic_monitor_predecode_fault_leaves_winner_synth_r2.py` | Preflight, auditoría EDIF, informes y archivo revisados |
| Replay propio | `continue_symbolic_monitor_predecode_fault_leaves_winner_binary_r2.py` | 174.436 filas de su netlist, controles y respaldo |
| Comparación local | `continue_symbolic_monitor_predecode_fault_leaves_winner_metrics_r2.py` | Mismas opciones y builds frente al padre 966 |
| Fronteras del pipeline | `continue_symbolic_monitor_predecode_stage_timing_r1.py` | Después del replay propio: leer ambos DCP, revisar informes y respaldar |

La primera tanda integrada `20261011T141336Z` está aprobada y respaldada;
la síntesis `20261011T143058Z` está en ejecución. Sus fuentes integradas congeladas
`source-20261011T141210Z.zip` tienen SHA-256
`2ff39f063c3c887aa948813f4844ec21d360c33c1397de86a5deb4aa44ce8085`.
Configuración: `4649346b3010353654e1f020dbdad531c44458fda07d542e9c6f663068c96ca7`.

## Recuperación y límites

R1 se detuvo por el sufijo heredado del archivo de revisión aislada;
no ejecutó una integración fallida. R2 enlaza su nombre y hash reales.
El manifiesto `pcie-symbolic-monitor-predecode-fault-leaves-winner-r2-recovery.json`
conserva estados y recetas, SHA-256
`b476a945cd36f2d34b084a1ecff9d4b826e9683f795cb53a43a9fa4e4264ae16`.
Los coordinadores R1 permanecen detenidos como evidencia histórica.

Ninguna etapa relaja comparadores ni reutiliza filas de otra variante.
Los resultados terminales, incluidos errores y timeout, se respaldan en
D: y NFS. No se encola otro ruteo: primero deben medirse los efectos del
cambio. El WNS conocido del padre sigue siendo −2,546 ns a 4 ns.
