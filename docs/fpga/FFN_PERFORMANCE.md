# Medición por etapa de la FFN

Estado: 2026-10-03 UTC. Tanda `20261003T214142Z-V7-ffn-phase-profile`
aprobada y respaldada. Conserva el banco [ABI2 asíncrono](FFN_IRQ_INTEGRATION.md)
de tres FFN Brandon D=256/H=720, pesos reales y entradas sintéticas.
La integración completa [PCIe/XDMA](PCIE_FFN_INTEGRATION.md) también pasa
una FFN Brandon independiente; ninguna de las dos mide rendimiento físico.

## Resultados revisados

| Etapa | Ciclos | % del cálculo | Ocupación de cada MAC |
| --- | ---: | ---: | ---: |
| Gate | 73562–73589 | 23.81–23.83 | 86.10–86.13% |
| Up | 73490–73603 | 23.80–23.82 | 86.08–86.22% |
| SwiGLU/Q8 | 88258–88265 | 28.57–28.58 | — |
| Down FP32 | 73383–73456 | 23.76–23.78 | 88.17–88.26% |

El contador de cálculo registra 308795–308953 ciclos (1543.975–1544.765 µs simulados). El controlador y sus fronteras añaden 67 ciclos a la suma de etapas.

En las matrices, Q8 presenta datos sin poder entregarlos durante 80.61–83.31% de la etapa. Las esperas con lectura AXI pendiente y sin `RVALID` ocupan 1.70–2.00%; pueden solaparse con cómputo. En SwiGLU, los estados HLS consumen 98.82–98.83% del intervalo. Estos datos justifican ensayar más MAC antes de ampliar el tráfico de memoria del banco.

La revisión numérica aprueba 768 FP32 finales; las 12 alteraciones del perfil se rechazan. Todas las líneas funcionales y sus tiempos coinciden con la referencia: SHA-256 de traza `1f29159f213bcfb7e9a0b15dad1701e2921ccceb683297e8c189218257a0cef5`. Vivado terminó con salida 0 en 837.369 s. Avisos idénticos a la referencia, incluidos cinco de FIFO asíncrona sin retardos de sincronización modelados.

Respaldo local/NFS: `artifacts/ntrpu-board-vivado/20261003T214142Z-V7-ffn-phase-profile/`, 178519034 bytes; SHA-256 `e8580b02e7b5f5aa2e8d3b54cb7b167ea38ebe665416b20158d50cbf7b08e5c2`. Fuentes de ejecución: `206db9def4e8c785226a36809710830a3061fda07a3e19357eddc1876c29b203`.

## Qué se mide

El modo `sim_irq_async_profile_fp32` de `ntrpu_board_ffn_memory.tcl`
añade únicamente `fpga/tb/ntrpu_ffn_profile.sv`. El observador no conduce
señales, consume números aleatorios ni inserta retardos. RTL sintetizable,
HLS, fixture y parámetros de CDC se conservan.

Cada intervalo comienza en la aceptación del comando hijo y termina
en la aceptación de su completion, ambos flancos incluidos. Se separan
gate, up, SwiGLU/cuantización y down. El resto del contador de cómputo
corresponde al controlador padre y a sus fronteras.

- AXI: AR/R/AW/W/B aceptados, bytes escritos, pausas de handshake y
  ciclos con transacción pendiente sin respuesta válida.
- Matrices: ocupación por MAC, ocupación simultánea, bloques aceptados
  y terminados. Durante `RUN`, las cuatro combinaciones valid/ready
  forman una partición de ciclos.
- SwiGLU: estados de lectura, ejecución HLS, escritura y otros; inicios
  y finalizaciones de los 23 bloques de activación.

La ocupación MAC y las esperas AXI pueden solaparse: **no se suman como
causas exclusivas de latencia**. Tampoco se interpreta todo ciclo sin
`RVALID` como un MAC detenido. Los antiguos contadores `p0..p3` abarcan
programación/polling del host y no representan exactamente estos intervalos.

## Criterios de aceptación y reproducción

Usar las mismas rutas MAC/SwiGLU/fixture que la referencia
`20261003T181921Z-V7-ffn-irq-brandon`, cambiando solo el modo a
`sim_irq_async_profile_fp32` y dando una carpeta de salida nueva.
Presupuesto: cuatro hilos, 2.400 s; motor a 5 ns y host a 4 ns.

`tools/check_ntrpu_ffn_profile.py` exige 12 etapas ordenadas, coherencia
de intervalos/particiones, recuentos derivados de geometría y coincidencia
de los totales AXI con el observador independiente de memoria.
Todas las líneas funcionales `NTRPU_`, incluidos datos y tiempos, deben
coincidir con la referencia sellada. Además se ejecuta el revisor numérico
ABI2 y se alteran deliberadamente los registros del perfil para comprobar
su rechazo. El código de salida de Vivado por sí solo no aprueba el perfil.

## Comparación nativa dual/quad

El control `20261003T224122Z-V7-ffn-dual-brandon-control` y el candidato
`20261003T223550Z-V7-ffn-quad-brandon` pasan tres FFN con las mismas
fuentes, HLS, fixture y reloj nominal de 5 ns. Los 42.048 bytes escritos
coinciden exactamente, incluidos 768 FP32 finales. La latencia baja de
307.247–307.387 a 207.366–207.430 ciclos: **1,481–1,482×**.
Este banco nativo no incluye los intervalos de programación/CDC del
perfil ABI2 anterior; no se mezclan sus contadores.

SwiGLU consume ahora 42,43–42,45% de la cadena, candidato a la siguiente
optimización de cálculo. Antes hay que resolver el ruteo de la FFN quad:
WNS −0,215 ns a 5 ns. Los tiempos calculados con ese período siguen siendo
medidas de simulación, no una frecuencia física aprobada.

El informe HLS del operador conservado (`20261003T065730Z-V7-swiglu-block-cosim`)
separa `ntrpu_swiglu` (194–2.021 ciclos estimados) de la cuantización
(38–1.506). Ambos contienen bucles sin pipeline. El 42,4% medido corresponde
al conjunto SwiGLU/Q8; esos rangos de síntesis no permiten repartirlo con
precisión entre funciones. Se ensaya una planificación HLS más paralela
en una copia aislada mientras se rutea la variante FIFO, conservando
exponencial, división, redondeo y tratamiento de subnormales.

`20261003T233132Z-V7-swiglu-pipeline-both` ya pasa 27 casos C/RTL;
los 14 registros válidos y todos los estados coinciden bit a bit con
la referencia. Los bloques válidos tardan 598–722 ciclos frente a
2.934–3.794: **4,24–6,14× por caso**. Dos bucles alcanzan II=1;
HLS estima 45 DSP frente a 37. Son medidas del operador aislado:
la cola exige repetir memoria, FFN, Brandon y timing antes de sustituirlo.
Ver [fuentes, recursos y respaldo](SWIGLU_Q8_BLOCK.md#candidato-con-pipeline-en-dos-bucles).

## FFN quad/FIFO con SwiGLU optimizado

La comparación `20261003T232820Z-V7-ffn-quad-fifo-brandon` frente a
`20261003T234644Z-V7-ffn-quad-pipeline-brandon` mantiene exactamente RTL,
banco, MAC HLS y fixture. Solo cambia el artefacto HLS SwiGLU. Ambas tandas
están aprobadas y respaldadas; las 657 escrituras coinciden bit a bit:
42.048 bytes intermedios/finales, incluidos 768 FP32 finales.

| Muestra | Ciclos originales | Con pipeline | Factor |
| --- | ---: | ---: | ---: |
| 1 | 207.531 | 136.933 | 1,516× |
| 2 | 207.523 | 136.859 | 1,516× |
| 3 | 207.559 | 136.806 | 1,517× |

La reducción medida es **34,02–34,09%**. La etapa SwiGLU/Q8 pasa de
88.009–88.022 a 17.350–17.358 ciclos y ocupa 12,67–12,68% del total.
Las pausas aleatorias pueden divergir al variar el número de ciclos.
Son FFN nativas sobre RAM conductual; no tokens/s ni medidas de placa.

La síntesis `20261003T234644Z-V7-ffn-quad-pipeline-synth` utiliza 51.086 LUT,
40.426 FF, 153 DSP y 48 RAMB36 + 3 RAMB18: +481 LUT, +412 FF y +8 DSP
frente al control FIFO. WNS +0,378/WHS +0,039 ns son estimaciones.
El ruteo `20261003T235718Z-V7-ffn-quad-pipeline-route` termina con
**WNS +0,046/WHS +0,009 ns**, 50.880 LUT, 40.214 FF, 153 DSP y
48 RAMB36 + 3 RAMB18. Sus 83.702 nets enrutables están completas.
La auditoría `20261004T001638Z-V7-ffn-quad-pipeline-route-audit` confirma
timing interno a 5 ns, sin latches ni endpoints internos sin restricciones.
Persisten un reloj padre incompleto y 2.470 entradas/845 salidas sin
retardos externos: es implementación OOC, no cierre de placa. El margen
de setup de 46 ps requiere nueva revisión al integrar el resto del diseño.
Ambas tandas tienen respaldo local/NFS verificado. El banco corto
[PCIe optimizado](PCIE_FFN_INTEGRATION.md) también pasa por separado.

Respaldo Brandon: 88.373.346 bytes, SHA-256
`3be709729448364c8ca53fb6c2dd160294c3c1ab69455540e7cf70bfa0838ed7`.
El índice conserva `ffn-pipeline-paired-comparison.json` y su revisor;
SHA-256 de la traza escrita:
`b46de9e8e2d9a9c2f651720aaf69cb57305d0607b213904a8a57c29bebc9c6f3`.

## Contrato del experimento de arquitectura

Priorizar una variante aislada de cuatro MAC: el perfil confirma ocupación
alta y bloqueo Q8. Después medir de nuevo antes de decidir sobre
planificación/prefetch o paralelismo de activación. Cada variante requiere datos numéricos exactos,
recursos y timing propios. No extrapolar estos contadores a cuatro canales
DDR, 512 GiB, varias tarjetas ni tokens/s.

### Experimento candidato: cuatro MAC

Implementación y resultados vigentes: [GEMV con cuatro MAC](Q8_GEMV_QUAD.md).
El banco ampliado y las proyecciones Brandon están aprobados; la cadena
FFN y el ruteo tienen calificación independiente. Los criterios siguientes
se conservan como contrato del experimento.

El perfil completo confirma el bloqueo por MAC de la referencia dual.
Preparar una variante aislada conservando layout 2, asignar
`fila mod 4` y mantener el orden K y la acumulación binary64 de cada fila.
Ampliar el recuento de completions simultáneas a tres bits y drenar los
cuatro motores ante primer error/cancelación antes de publicar resultados.

La caché actual tiene dos puertos explícitos; no basta con ampliar arrays.
Evaluar dos copias BRAM con escrituras de activación replicadas y un par
de puertos de lectura por copia, sin resetear el contenido de RAM. La
máscara de validez por trabajo debe seguir excluyendo entradas antiguas.
Revisar inferencia de BRAM, muxes de acumuladores y timing tras sintetizar.

Orden de calificación: casos pequeños con filas 1/2/3/4/5/31/32 y distintos
tiles, cancelación/error/timeout con motores ocupados, fixture Brandon
independiente y finalmente FFN encadenada. Conservar el dual como control.
Solo después comparar ciclos, LUT/FF/DSP/BRAM y ruteo; una mejora de GEMV
no equivale al mismo factor de mejora de la FFN completa.
