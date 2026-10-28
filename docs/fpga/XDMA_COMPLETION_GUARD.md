# Lector autónomo de finalizaciones C2H

Estado: **`COMPLETION_GUARD_RTL_PASS`** en un banco local con registros
modelados. Es un candidato experimental; todavía no sustituye la gestión
del [banco XDMA nativo](HBM_XDMA_NATIVE.md).

Tanda: `D:/AI/artifacts/ntrpu-xdma-completion-guard/20261009T075210867834Z/rtl1`.

## Contrato de hardware

`fpga/experimental/ntrpu_xdma_completion_guard.sv` trabaja en `axi_aclk`
y usa el maestro existente `ntrpu_xdma_local_master`. No escribe registros.
Requiere dos C2H, configuración `Control=0x00f83e7f`, contadores iniciales
cero y propiedad exclusiva de los registros durante toda la época.

Por cada canal lee **Control → Count → Status → Control**:

| Canal | Control | Count | Status sin limpieza |
| --- | --- | --- | --- |
| C2H0 | `0x1004` | `0x1048` | `0x1040` |
| C2H1 | `0x1104` | `0x1148` | `0x1140` |

Exige ocho respuestas iniciales válidas antes de admitir descriptores.
Mantiene un máximo de dos descriptores sin crédito por canal. Cuenta las
emisiones reales y los pulsos nativos Done; cada consulta toma un objetivo
de finalizaciones y pausa nuevas emisiones mientras comprueba ambos canales.
Un contador rezagado obliga a repetir la consulta. La respuesta AXI debe ser
OKAY y conservar `{epoch, transaction_serial}`.

Los créditos requieren una consulta completa y correcta posterior a Done.
Son ordenados por canal; el puente rev3 conserva las identidades de los
contextos. El lector no valida payload ni ACK, ni declara drenaje PCIe.
Los contadores no se reutilizan ni se permite su desbordamiento.

La política acepta únicamente Busy y Descriptor_completed en Status:
rechaza errores, indicaciones de parada y bits reservados. Busy puede seguir
activo. Los campos y efectos de Run proceden de
[PG195 Control](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Control-0x04),
[Status](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Status-0x40)
y [Count](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Completed-Descriptor-Count-0x48).
La secuencia de lectura y la política restrictiva son decisiones de este diseño.

## Fallos y retención

Pérdida de enlace/Run/propietario, error AXI, identidad incorrecta, inventario
imposible, timeout o `reset_request` dejan un fallo persistente. El error
hacia el puente es registrado; un fallo observado en el mismo ciclo ya
suprime el crédito. Las solicitudes ofrecidas y los AR pendientes permanecen
válidos hasta su aceptación, y las respuestas tardías se consumen sin generar
créditos. Los contadores conservan el trabajo pendiente.

`cold_rst_n` solo corresponde a una nueva configuración coordinada; no es
recuperación operacional. `reset_request` **inhibe**, no ejecuta un reset.
El límite de espera cubre arranque y periodos continuos con trabajo pendiente;
no se reinicia por cada respuesta. Un supervisor externo debe tratar la
pérdida del reloj. Este módulo, por sí solo, no protege buffers ni arbitra BAR.

## Evidencia local

Se probaron **54 escenarios × 8 variaciones de espera = 432 escenarios**,
con el maestro RTL real y un destino AXI-Lite modelado. Incluyen ambos
canales, contador rezagado/regresivo/excesivo, todos los bits de Status
prohibidos, SLVERR/DECERR, metadata writeback deshabilitado, identidades
incorrectas, Done inesperado, exceso de emisiones, timeout de AR/R y
revocación coincidente con un crédito. `reset_request` retiene inventario;
no se ensaya reset físico de XDMA.

El revisor reconstruye solicitudes, AR, R, respuestas e inventarios desde
puertos públicos de la VCD. Comprueba estabilidad bajo espera y causalidad
de cada crédito. Seis alteraciones de evidencia fueron rechazadas. Icarus
emitió seis avisos de sensibilidad a arreglos, conservados en `compile.log`;
no hubo otros avisos ni errores en esta tanda.

```powershell
python -B tools/run_ntrpu_xdma_completion_guard.py NUEVA_SALIDA --toolchain D:/AI/artifacts/ntrpu-hbm-dma-rtl/20261009T005545469986Z/restore-toolchain
python -B tools/check_ntrpu_xdma_completion_guard.py NUEVA_SALIDA/seed-00
```

El ejecutor congela fuentes, registra hashes antes/después y conserva cada
traza. Los respaldos usan `D:/AI/artifacts/ntrpu-content-store`; los recibos
de esta tanda permiten restaurar resultados, fuentes y compilador.

## Integración pendiente

1. Conectar el candidato al puente rev3 y al `s_axil` del XDMA nativo.
   Combinar `admit_descriptor` con la autorización de emisión; observar
   `dsc_load` real, nunca solicitudes aún no aceptadas.
2. Hacer efectiva la exclusión del host: `exclusive_owner` es una condición
   externa, no impide accesos BAR, cambios de Run ni lecturas de limpieza.
3. Repetir fallos con buffers/contextos del puente, ACK remoto y reloj/reset
   coordinados; comprobar retención completa, no solamente contadores.
4. Conectar HBM nativo y cerrar síntesis, timing y CDC de la composición.
   No hay calificación de PCB ni nueva ejecución de Vivado en esta tanda.
