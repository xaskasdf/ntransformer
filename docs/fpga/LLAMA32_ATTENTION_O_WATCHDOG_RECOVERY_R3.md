# Proyección O: evidencia numérica y corrección del watchdog

Estado al 2026-10-14 04:17 UTC: el banco original tiene **PASS numérico
independiente**. La revisión R3 corrigió el lazo combinacional y su síntesis
ya pasó la revisión independiente; XSim ejecuta la regresión completa.
La aprobación conjunta de R3 y el cierre físico siguen pendientes.

## Resultados conservados

La tanda `20261013T221343Z-V7-llama32-attention-projection-memory-vivado-r1`
ejecutó siete trabajos: cinco proyecciones completas con cuatro vectores
reales, rechazo de límites y cancelación de una lectura en curso, seguida
de recuperación. La revisión R4 comprobó 10.240 resultados FP64 y 40.960
bytes FP32 exactos contra las referencias congeladas de Llama 3.2 1B Q8_0.
Los 32 controles de corrupción fueron rechazados.

Cada proyección consumió 1.606.198–1.607.222 ciclos de matriz y escribió
8.192 bytes. A un período supuesto de 5 ns son 8,031–8,036 ms; se trata
de latencia RTL simulada con memoria AXI de prueba. El máximo intervalo
sin actividad fue de 11.820 ciclos. Se ejercitaron pausas AR/AW/W y burbujas
RVALID; no hubo beats R retenidos por RREADY bajo.

La síntesis original falló por entradas sin tipo de red explícito bajo
`default_nettype none`. Las variantes `vivado_explicit_nets_r2/` añaden
solamente `wire`; el cuerpo, puertos y anchuras normalizados son idénticos.

## Lazo real encontrado y cambio funcional

La tanda `20261014T040053Z-V7-attention-projection-synth-r2` terminó con
código de herramienta 0, pero la revisión **rechazó** `checking loops (1)`
y `LUTLP-1`: doce LUT conectaban el watchdog con la cancelación y la
actividad AXI. La ruta era:

`timeout_now → fault_now → child_cancel_valid → AXI READY → child_activity → timeout_now`.

El RTL `registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv`
emite cancelación por timeout después de registrar el fallo. Mantiene el
umbral de inactividad y la respuesta inmediata a cancelación externa,
cambio de época y respuesta de identidad ajena. No se añadieron excepciones
de timing ni permisos para lazos combinacionales.

El controlador corregido pasó 145 trabajos nativos y 83 controles de
corrupción: identidades completas, drenaje, timeout, cancelación, reset y
agotamiento de etiquetas. Esa prueba usa un par de protocolo y no
reemplaza el banco aritmético.

## Síntesis corregida, resultado parcial

La captura `20261014T041704Z-attention-projection-synth-preflight-r3`
verificó el top real, parte, fuentes y siete controles de identidad.
Resultado: **19.981 LUT, 17.996 FF, 38 BRAM y 54 DSP**; cero lazos,
latches, cajas negras o endpoints internos sin restricciones.
WNS de síntesis **+1,019 ns**, WHS **+0,039 ns**, reloj de 5 ns.
Las E/S externas no tienen presupuesto y no hay calificación de ruteo.

Los informes comprimidos están verificados en D: y NFS con SHA-256
`abac16819c2de4d38ad3799f8d35b430d0323f8db4967b5b5496250c84c8a1b5`.
Esta captura conserva solo el resultado de síntesis; no declara terminado
el banco numérico que continúa dentro de la misma tanda.

La captura `20261014T042932Z-attention-projection-first-tensor-r3` verificó
las fuentes realmente compiladas y comparó **todos los eventos del primer
trabajo** con la traza original revisada. Los bytes AXI, los 2.048 FP64,
los 8.192 bytes FP32 y los ciclos de cada evento son idénticos. Archivo
verificado D:/NFS: `6f3dcf03681a9d6398d379fc0f0e2a30e0e69491d391e69a35fd1106fb19b24d`.
La regresión completa continúa; este resultado no sustituye los demás casos.

## Cola y criterios

1. `20261014T041103Z-V7-llama32-attention-projection-memory-vivado-r3`:
   síntesis del top real con rechazo obligatorio de cualquier lazo;
   después, los siete trabajos completos en XSim, revisión independiente
   y respaldo antes de liberar el turno.
2. `llama32-attention-output-path-real-vivado-r3-coordinator.json`:
   cuantización real → escrituras ABI1 confirmadas → proyección O real.
   Exige PASS completo y respaldo de R3; cuatro trabajos, tres salidas y
   cancelación en el traspaso, con 38 controles previstos.
3. El [banco de raíz integrada R2](LLAMA32_ATTENTION_PROJECTED_ROOT_REAL_QUEUE_R2.md)
   está sellado y encolado detrás de cuantización → O R3, con el controlador
   corregido. Su verificador reconoce 47 eventos y prevé 67 controles;
   comprobará K/V materializado, copias FP32, ABI1 y salida O. Su compilación
   pasó; la ejecución aritmética integrada permanece pendiente.

Los coordinadores anteriores permanecen como diagnósticos. Las tandas
activas conservan directorios nuevos, dos CPU y transferencia comprimida
de evidencia; una desconexión del observador no reinicia herramientas.

## Respaldos y límites

Los archivos siguientes están verificados por SHA-256 y por cada miembro
del manifiesto en D: y NFS:

| Evidencia | SHA-256 del archivo |
| --- | --- |
| Banco O original completo | `96539d76afc4b926581e0ee726557bfcdf053dfa9ade05bef4ee437e29117896` |
| Síntesis con lazo conservada | `c66b52d17818255c36c7b4fec18debe39fd59db85585df28f02a1f402182e163` |
| Controlador corregido, protocolo nativo | `f3851d161d3588296ccbc7b9e7c572abe9446df24b1ca128ae69b0f5646a8f7b` |
| Compilación del banco de raíz completa | `23156c244afaa1950a9fd564f57b2e9f19e2034872b51a0e5ed4d8fdb83edf5d` |

La revisión numérica original tiene SHA-256
`3c32a51c2e8e68a26a78a53147217084e6355454828c5aa748734c9325842a45`.
Los certificados y coordinadores están en `build/cloud-setup/20261002-board/`;
los respaldos, en `D:/AI/ntransformer/artifacts/`.
La cápsula de trabajo R43 conserva 2.351 archivos, SHA-256
`11495aa1b2aa7067bcfd2294578ff339effc26bfc8365dcbfedc7a1ed0177b40`.

Siguen pendientes la cadena con residual/normalización/FFN, inferencia
completa, DDR física exacta, integración PCIe, ruteo y validación en placa.
