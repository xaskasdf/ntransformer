# Síntesis aislada y simulación de netlist del receptor

Tandas del 2026-10-04 UTC en Vivado 2026.1 sobre
`xcvu5p-flva2104-2-i`. La síntesis y la simulación funcional de ambas netlists
se completaron y sus respaldos están descargados localmente. No hubo cambios
en el RTL del receptor, el banco original, el endpoint ni el supervisor del
principal. El [contrato de adaptación](PEER_RX_INTEGRATION.md) conserva las
condiciones pendientes para conectarlos.

## Resultados revisados

Restricción única: reloj de 5 ns. Los márgenes siguientes son **estimaciones
internas de síntesis**, sin colocación/ruteo ni retardos de E/S.

| Buffer | LUT totales | LUT memoria | FF | BRAM / DSP | Margen setup estimado |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 beats / 256 B | 1010 | 296 | 307 | 0 / 0 | +3,096 ns |
| 64 beats / 4 KiB | 1320 | 586 | 327 | 0 / 0 | +2,842 ns |

La memoria se implementa con recursos distribuidos; estos perfiles no usan
BRAM. Ambos tienen un reloj, cero latches, cero blackboxes y cero endpoints
internos sin restricciones. La revisión automática da
`SYNTHESIS_REVIEW_PASS_WITH_OPEN_BOUNDARIES`.

XSIM ejecutó **37 casos por netlist, 74 en total**. Las 296 y 952 líneas de
eventos respectivos coinciden exactamente con las trazas RTL previas. El
modelo Python también reproduce los 31 casos aplicables por perfil; seis
rechazos locales por perfil no exponen AXI. No se resintetizó durante replay.

El banco copiado elimina únicamente el parámetro de la instancia, ya
especializada por síntesis, y espera 200 ns antes de aplicar su reset frío
para permitir terminar GSR. Conserva estímulos, semillas y monitores de
pines. Sus duraciones son 640 y 2551 ciclos incluyendo esa espera; no
constituyen una medición de frecuencia. Es simulación funcional sin SDF,
no simulación temporal ni prueba del ingreso durante configuración.

## Advertencias conservadas

- **TIMING-18:** 858 entradas y 790 salidas carecen de retardos externos.
  La tabla de metodología alcanza su límite de 1000 entradas; `check_timing`
  conserva ambos conteos completos. Ninguno se considera resuelto.
- **Timing 38-242:** falta `HD.CLK_SRC`; no se estima retardo/skew del reloj
  padre. No se inventó ubicación de reloj para eliminar el aviso.
- **DRC 23-814:** el contexto OOC no ejecuta todas las comprobaciones de
  conectividad. Cero DRC encontrados no equivale a cierre de placa.
- **Synth 8-3917:** salidas constantes del perfil AXI, revisadas como tales;
  **Netlist 29-101:** advertencia de granularidad para floorplanning.

No se añadieron false paths, multicycle paths, waivers ni cambios de
severidad. La síntesis no valida CDC, PCIe, DDR/HBM, reset global ni integridad
de señal. Un margen positivo aquí no demuestra operación integrada a 200 MHz.

## Reproducción y límites de recursos

- [Tcl de síntesis](../../tools/ntrpu_peer_rx_synth.tcl).
- [Restricción OOC](../../fpga/constraints/ntrpu_peer_rx_ooc.xdc).
- [Tcl de netlist](../../tools/ntrpu_peer_rx_netlist.tcl).
- [Ejecutor acotado](../../tools/run_ntrpu_peer_rx_synth.py).
- [Revisor de síntesis](../../tools/check_ntrpu_peer_rx_synth.py).

Desde una copia identificada con Vivado cargado en Linux:

```bash
# Ambas carpetas deben ser nuevas.
python3 -B tools/run_ntrpu_peer_rx_synth.py /ruta/nueva/synth-output
python3 -B tools/run_ntrpu_peer_rx_synth.py /ruta/nueva/netlist-output \
  --netlist-input /ruta/nueva/synth-output
```

La segunda invocación solo lee las netlists exportadas por la primera.
Los usos de `synth_design -mode out_of_context` y netlists funcionales siguen
[UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/synth_design) y
[UG900](https://docs.amd.com/r/2024.1-English/ug900-vivado-logic-simulation/Running-Post-Synthesis-and-Post-Implementation-Simulations).

Se usó una carpeta nueva en `/home/ubuntu/vivado/ntrpu-side/peer-rx/` en el
volumen persistente. Afinidad de dos CPU, nice +10, `general.maxThreads=2`
y presupuesto de 420 s por proceso. Las síntesis tardaron 90,727 y 90,861 s;
los replays, 34,713 y 33,908 s: unos 250 s de ejecución acumulada, secuencial.
Los trabajos del principal conservaron sus procesos y directorios.

Un primer lanzamiento se detuvo al cargar el entorno AMD con `set -u` antes
de ejecutar Vivado. Su diagnóstico está conservado; el launcher corregido
carga `settings64.sh` antes de activar esa comprobación. No cambió instalación,
licencia ni variables globales.

## Evidencia local

La [tanda de síntesis](../../artifacts/ntrpu-peer-rx-synth/20261004T034042348147Z-peer-rx-synth-r2/REVIEW.json)
incluye DCP, netlists, XDC aplicado, informes, comandos y fuentes de entrada.
El [ZIP remoto descargado](../../artifacts/ntrpu-peer-rx-synth/20261004T034042348147Z-peer-rx-synth-r2/output.zip)
se verificó por SHA-256, CRC y cada miembro del manifiesto:
`7f298d539ed58ce1867cc065d6bd901dd895ad719674fd86eb920da6fa7ccb51`.

La [revisión de netlists](../../artifacts/ntrpu-peer-rx-synth/20261004T034628048280Z-peer-rx-netlist/REVIEW.json)
registra correspondencia con el RTL y resultados de las dos configuraciones.
Su [ZIP descargado](../../artifacts/ntrpu-peer-rx-synth/20261004T034628048280Z-peer-rx-netlist/output.zip)
pasó las mismas verificaciones, SHA-256:
`75ff73d3e494e08a1ecc3cceb83b53d90ec65c9e74652147dc08e64ad8b6c4f5`.

Los ZIP remotos contienen salidas. Las carpetas locales de cada tanda
añaden `inputs/`, `INPUTS.json`, comprobaciones de descarga y revisión;
el respaldo consolidado incluye ambos conjuntos y la documentación final.

[Respaldo consolidado](../../artifacts/ntrpu-peer-rx-synth/20261004T035511438838Z-peer-rx-handoff.zip):
fuentes, documentos, entradas/salidas de ambas tandas, revisiones y el
diagnóstico del primer launcher. Su registro externo `VERIFIED.json`
conserva el hash final y la comprobación de todos sus miembros.
