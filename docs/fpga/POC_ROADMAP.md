# Prueba de concepto NTRPU: estado y criterios de cierre

Actualizado el 2026-10-03 UTC. Se recuperaron Vivado/Vitis y los resultados desde el volumen persistente en una nueva VM, y se validó una licencia para su nuevo host. Este documento distingue validación CPU, co-simulación RTL e implementación física; los resultados y bloqueos de tandas anteriores se conservan como historial.

## Lo demostrado

- [FFN por PCIe/XDMA y MSI-X](PCIE_FFN_INTEGRATION.md): una FFN sintética
  `D32/H32`, 32 FP32 exactos, H2C 3.328 B y C2H 128 B. Dos MSI-X/ACK
  reales del modelo XDMA, servicio antes/después del transporte y polling
  con dos descriptores inválidos. Cómputo: 5.424 ciclos con fases separadas
  y 5.431 con sondeo concurrente, a 200 MHz. El segundo observa MMIO
  durante ambos sentidos DMA y rechaza 29 alteraciones de evidencia.
  La separación CQ/RC del modelo host también pasa con 33 alteraciones
  de evidencia rechazadas. Usa RAM conductual; la FFN Brandon por PCIe
  está en curso. Supervisor y DDR siguen pendientes.

- [FFN real con ABI2 e IRQ](FFN_IRQ_INTEGRATION.md): 126 casos pequeños,
  incluidos 24 resets durante cómputo y 24 cancelaciones; seis FFN Brandon
  y 1.536 FP32 exactos en dos perfiles de reloj. Carga y lectura por host,
  servicio MMIO y ACK de transporte en ambos órdenes, polling e identidades
  incorrectas verificados. Una retirada IRQ prematura es detectada.
  Ruteo exploratorio: 44.472 LUT/45.219 FF/93 DSP, WNS +0,104 ns y WHS
  +0,010 ns; 17 grupos bus skew conservados e INIT IRQ auditada. Persisten
  los límites OOC; el banco PCIe pequeño de arriba añade XDMA/MSI-X.
  Faltan supervisor y DDR real.

- [Control e IRQ, ABI 2.0](FFN_IRQ_CONTROL.md): 54 casos en tres perfiles,
  27 resets y 57 retiros PASS. ACK comprueba secuencia/época y conserva
  la finalización hasta confirmación del transporte; MMIO continúa mientras
  START está bloqueado. Revisión independiente de 798 escrituras/291
  lecturas y 45 alteraciones rechazadas. Síntesis 2.093 LUT/4.927 FF;
  INIT/CDC/constraints auditados. Regresiones ABI1 y FFN real conservan
  exactamente sus trazas. ABI2 está conectada a FFN y al banco XDMA pequeño.

- [Transporte IRQ](IRQ_TRANSPORT.md): 45 casos y 24 resets coordinados
  PASS; 69 eventos admitidos, 48 retirados, 21 abortados y recuperados.
  Conserva notificación e identidad hasta atención por MMIO y ACK de
  transporte; rechaza seis identidades antiguas y detecta una retirada
  prematura inyectada. Corrige una violación XPM real durante reset.
  Síntesis 78 LUT/486 FF; 131 registros de envío/datos con INIT=0 auditados.
  CSR y FFN opcionales conectados; el banco PCIe de arriba complementa
  los pares de protocolo con XDMA. Faltan supervisor y calificación de placa.

- [FFN a dos relojes](FFN_ASYNC_INTEGRATION.md): host 4 o 6,2 ns, motor
  5 ns. Tres FFN Brandon exactas por perfil, carga AXI hacia RAM vacía,
  lectores concurrentes y lectura de 768 FP32 por perfil; IRQ sincronizada
  y confirmada por MMIO. Ambas simulaciones y síntesis tienen respaldo
  verificado. La recodificación reversible RRESP conserva ambas trazas
  completas y elimina TIMING-37/Route 35-470 sin cambiar las restricciones
  AMD. Síntesis: 44.194 LUT/44.819 FF/93 DSP. Ruteo exploratorio:
  44.038 LUT/44.607 FF/93 DSP, WNS +0,068 ns/WHS +0,010 ns;
  15 grupos bus skew con margen mínimo +3,281 ns. La auditoría verifica
  33 fuentes variables con destino único y una captura constante.
  CDC-15, dos huecos de reloj padre y fronteras pendientes impiden
  considerar estos márgenes como cierre de placa.
  La integración XDMA pequeña figura arriba; faltan DDR y supervisor.
  La revisión ABI2 añade resets
  coordinados durante cómputo en el banco pequeño.

- [Conversores de reloj host/dispositivo](HOST_CLOCK_INTEGRATION.md): IP AMD
  AXI4 512/64/4 y AXI4-Lite 32/32, modo asíncrono, tres perfiles de reloj.
  54 casos y 24 resets coordinados aprobados; 36 mutaciones de evidencia
  rechazadas y una inyección RTL de datos obsoletos detectada. Siete tandas
  respaldadas, incluidos fallos. El modelo FIFO omite retardos de
  sincronización; no mide latencia física. El enlace a FFN ya se prueba
  arriba; falta supervisor de reset sintetizable. Síntesis aislada: AXI4 998 LUT/3102 FF;
  AXI4-Lite 49 LUT/377 FF, sin DSP/BRAM. Un CDC-15 del FIFO, relojes padres
  y fronteras pendientes; cuatro tandas de síntesis adicionales conservadas.

- [Árbitro AXI host/FFN](AXI_MEMORY_FABRIC.md): revisión de dos entradas con
  54 bancos aislados y 24 resets coordinados, incluidas solicitudes anticipadas.
  La ráfaga de 64 beats pasa de 127 a 64 ciclos por dirección: un beat por
  ciclo con ambos extremos listos. Prueba negativa RTL detecta W prematura.
  Los cinco experimentos tienen respaldo verificado. La integración actual
  carga 1.777.152 bytes por AXI hacia RAM vacía, ejecuta tres FFN Brandon con
  lectores concurrentes y recupera 3.072 bytes finales exactos por el host.
  Reduce programación/cómputo de aproximadamente 316.700 a 309.400 ciclos
  por muestra, conservando exactitud. Síntesis actual: 43.146 LUT/41.364 FF/
  93 DSP, WNS estimado +0,735 ns; integración y síntesis respaldadas.
  Ruteo revisado: 43.148 LUT/41.216 FF/93 DSP, WNS +0,206 ns y
  WHS +0,010 ns. Un hueco de reloj padre y fronteras sin restricciones;
  PCIe/DDR/CDC y driver siguen pendientes.

- [Host MMIO conectado a la FFN real](FFN_HOST_INTEGRATION.md): 63/63 casos,
  71.232 bytes físicos exactos y 8.052 escrituras AXI-Lite revisadas. El host
  programa el descriptor, consulta la finalización, cancela y confirma por
  registros; el motor usa MAC/SwiGLU HLS reales. Pesos precargados en RAM
  simulada; PCIe/DMA/DDR y driver siguen pendientes. Regresión directa 63/63,
  con trazas idénticas a la referencia anterior. Tres FFN Brandon por MMIO
  escriben 42.048 bytes exactos y 768 FP32 idénticos a Q8. Síntesis del conjunto:
  41.802 LUT/38.981 FF/93 DSP; WNS estimado +0,683 ns, fronteras pendientes.
  Ruteo del conjunto: 41.536 LUT/38.775 FF/93 DSP, WNS +0,099 ns y
  WHS +0,010 ns; auditoría conserva un hueco de reloj padre y fronteras
  sin restricciones. No cierra timing de placa.
- [Control AXI-Lite del FFN](FFN_HOST_CONTROL.md): registros de 32 bits con
  direcciones de buffer de 64 bits; 27 grupos por formato FP32/binary64,
  cancelación, IRQ y recuperación de reset. Síntesis OOC: 1.729 LUT/4.372 FF,
  WNS estimado +2,280 ns; fronteras pendientes. El banco usa un par de cómputo
  simulado a nivel de protocolo; la integración real se valida por separado.
- [Ampliación de la FFN a dimensiones Llama](FFN_LARGE.md): prueba corta
  2048/96 aprobada, 243.755 ciclos y 9920 bytes correctos; forma completa
  2048/8192 PASS en 20.332.664 ciclos, con 155.648 bytes escritos exactos
  y 2.048 salidas FP32. Es un banco sintético, no inferencia con pesos Meta.
- [FFN raíz sobre memoria AXI](FFN_MEMORY.md): controlador 498/498;
  conjunto con MAC y SwiGLU reales 63/63 casos sintéticos/error/reset,
  74.176 bytes físicos exactos y 672 valores down de jobs correctos.
  Tres cadenas con pesos reales de Brandon pasan con 45.120 bytes exactos,
  307.289–307.377 ciclos y salida Q8 FP32 comprobada fuera del hardware.
  Ruteo: 38.627 LUT/31.997 FF/93 DSP, WNS +0,068 ns; fronteras pendientes.
  Son métricas de la referencia binary64, conservada sin sobrescribir.
- [FFN con salida FP32 integrada](FFN_FP32.md): 63 casos de cadena y
  71.040 bytes físicos exactos; regresión binary64 63/63. Escritor mixto
  171 casos RTL y 171 sobre netlist. Síntesis de la cadena: 40.407 LUT,
  34.617 FF y 93 DSP, WNS estimado +0,959 ns. Tres cadenas Brandon escriben
  42.048 bytes exactos; sus 768 valores down FP32 coinciden bit a bit con
  la referencia Q8 nativa. Ruteo: 40.080 LUT/34.427 FF/93 DSP,
  WNS +0,164 ns/WHS +0,010 ns; un gap de reloj padre y fronteras sin calificar.
  DDR/runtime siguen pendientes.
- [Conversor FP64→FP32](FP32_OUTPUT.md): 155.784 conversiones RTL y
  155.784 sobre netlist funcional; incluye las 768 salidas de Brandon.
  376 LUT/382 FF, cero DSP/BRAM, WNS estimado +2,755 ns; integrado al escritor.
- [SwiGLU/Q8 entre regiones AXI](SWIGLU_MEMORY.md): 105 casos con errores,
  cancelación/reset y gate variable por bloque; vector máximo de 8192
  elementos con 256 registros exactos en 980.075 ciclos. Ruteo OOC:
  17.024 LUT/37 DSP, WNS +0,567 ns/WHS +0,010 ns; reloj/fronteras pendientes.
  El escritor ABI1 pasa 159 casos RTL y 159 sobre netlist. El controlador
  raíz ya los conecta en el banco sintético indicado arriba.
- [Matrices completas sobre AXI](MATRIX_MEMORY.md): secuenciador 135/135,
  conjunto con MAC real 30/30 y tres proyecciones completas de Brandon
  por semilla, 5088 valores escritos exactos. Controla hasta 8192 filas
  sin comandos de tile desde el host. Síntesis: 19.369 LUT/54 DSP,
  WNS +1,450 ns estimado. Ruteo OOC: 19.212 LUT/54 DSP,
  WNS +0,614 ns/WHS +0,010 ns; reloj/fronteras pendientes. Las formas
  completas 8192×2048 y 2048×8192 pasan con datos sintéticos y 10.240
  valores exactos; falta la FFN encadenada a esas dimensiones.
- [Activaciones desde AXI y GEMV](ACTIVATION_MEMORY.md): cargador 96/96 y
  conjunto 60/60 casos de fallos/reset, con descriptores registrados y
  prohibición de leer pesos tras una carga fallida. Síntesis OOC del
  conjunto: 16.993 LUT, 12.438 FF, 52 DSP y 37 RAMB36 + 2 RAMB18;
  WNS estimado +0,980 ns. Las imágenes de pesos y activaciones pasaron
  174 jobs/5376 valores con diferencia Q8 cero.
- [Resultados en memoria AXI](RESULT_MEMORY.md): escritor 126/126 casos
  RTL y 126/126 sobre netlist; conjunto 105/105, más 174 jobs/5376 valores
  calculados y escritos exactos. La completion espera todas las respuestas B;
  un destino fallido permanece privado. Síntesis: 18.382 LUT, 13.744 FF,
  52 DSP, WNS estimado +0,979 ns. La ampliación pasa 111/111 casos con
  resets también durante W y esperando B. El ruteo OOC usa 18.239 LUT,
  13.644 FF y 52 DSP, WNS +0,395 ns/WHS +0,010 ns internos; el reloj
  padre y las fronteras siguen sin calificar. Faltan DMA host y composición
  con la FFN.

- Simulador CPU de dos unidades: memoria lógica, transferencias, épocas y cálculo distribuido; sin DMA ni BAR físicos.
- Co-simulación C/RTL correcta de INT8, GEMV Q8_0, cuantización y su composición FP32 → GEMV Q8_0. El top conectado pasó 17 transacciones, incluidas tres porciones de pesos reales de Brandon.
- SwiGLU aislado: 14 transacciones C/RTL correctas con la biblioteca matemática AMD, incluidos errores y extremos numéricos.
- FFN completa: 23 transacciones C/RTL correctas, incluida una muestra real de Brandon; diferencia máxima cero frente a la referencia Q8 y respaldo verificado.
- [Lector AXI de 512 bits](AXI_READER.md): 156 casos RTL correctos, incluidos límites de página, pausas y errores; sintetizado con 343 LUT y 655 FF. Es un bloque independiente, todavía sin conexión a FFN, DDR o PCIe.
- [Lector con buffers dobles](WEIGHT_TILE_BUFFER.md): 150 casos RTL correctos, publicación atómica por tile y lectura/llenado simultáneo; 973 LUT, 957 FF, 28 RAMB36 + 2 RAMB18 en síntesis. Conectado al nuevo GEMV.
- [Desempaquetador Q8 y camino integrado](Q8_UNPACKER.md): 105 casos de bloque y 30 de transporte correctos. El conjunto usa 5.037 LUT, 2.395 FF y los mismos bancos BRAM; separa errores de transporte de escalas/formato inválidos. El [GEMV conectado](Q8_GEMV_STREAM.md) pasó 72 casos de fallos/reset y 162 proyecciones reales de Brandon. Se corrigió setup segmentando comandos; su ruteo OOC usa 10.293 LUT/26 DSP y cumple timing interno bajo contexto provisional, con reloj/fronteras del padre pendientes.
- INT8 aislado ruteado a 5 ns: setup +0.721 ns y hold +0.009 ns. No se restringieron los retrasos externos; esto no certifica una placa ni el top Q8 completo.
- Referencia de FFN con las 12 capas únicas de Brandon, revisión `ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`. Las entradas son sintéticas, no activaciones obtenidas de prompts.

## Hito 1: FFN completa

El nuevo [controlador FFN](HLS_FFN.md) conecta cuantización, gate/up, SwiGLU y down. Su implementación C++ ya pasó la referencia cuantizada de las 12 FFN × ocho entradas, sin diferencia en las salidas de esas muestras, tanto con `std::exp` como con el modelo matemático AMD ejecutado en CPU. Vitis C simulation y síntesis también pasaron; la comparación RTL de la FFN pasó sus 23 transacciones, incluida una muestra real de Brandon. La licencia ya no bloquea el trabajo.

Para cerrar este hito:

1. Completado: co-simulación de SwiGLU (14 transacciones) y FFN (23), incluida una muestra real de Brandon. Ambos tops pasaron C simulation y síntesis.
2. Completado en el banco FFN: último tile parcial, padding, repetición, subnormales y fallos tardíos sin publicación parcial. La integración deberá conservar esas garantías.
3. El camino AXI → Q8 → GEMV ya pasó 72 casos RTL, con rangos registrados por job, épocas y descarte de resultados ante errores tardíos, además de 162 proyecciones con pesos reales de Brandon en memoria AXI simulada. El wrapper dual carga activaciones y escribe resultados por AXI; su composición FFN ya está validada. Faltan carga desde PCIe y registro de buffers del runtime.
4. Implementar físicamente el nuevo top, documentar todas sus restricciones y medir recursos, márgenes y ciclos por operación. Separar estimaciones HLS de resultados posteriores al ruteo.

La composición local mantiene buffers internos y reutiliza la cuantización. El protocolo de comandos con `sequence`/`model_epoch`, registro de buffers y DMA corresponde a la integración del dispositivo, todavía pendiente.

## Hitos siguientes

La [cola de diseño de placa](board/VIVADO_STATUS.md) aprobó entorno e inventario VU5P. El canal DDR genérico pasó calibración, direcciones dispersas de 128 GiB, máscaras e inyección ECC de uno/dos bits. PCIe pasó DMA de 1536 bytes por sentido con direcciones de 64 bits, cadenas, cruces de página y pausas en los cinco canales AXI. Un canal genérico DDR y el PCIe independiente están ruteados conjuntamente: 221 pines conservados y WNS +0,006 ns/WHS +0,004 ns con las restricciones presentes. V2/V3 siguen bloqueados para el DIMM exacto y el reset con fila abierta falla en el modelo; faltan contratos de placa, revisión CDC y adaptación eléctrica de PERST. No existe todavía conexión DMA–DDR, ni caudal sostenido o hardware físico medido.

V7 avanza independientemente: la FFN ya tiene síntesis HLS y síntesis Vivado
en VU5P, con 27.918 LUT, 22.961 FF y 55 DSP mapeados. La co-simulación
corta específica de ese dispositivo pasó 21 casos; los 23 casos completos
previos pertenecen al ZU7EV. Véanse [alcance y avisos](HLS_FFN.md).
El [MAC Q8 por bloque](Q8_BLOCK_MAC.md) pasó 317 operaciones C/RTL, incluida
la acumulación continua K=8192. Optimizar la conversión de escala redujo su
latencia válida de 193 a 21 ciclos. Su conexión al flujo de pesos pasó 72
casos RTL con etiquetas, activaciones, errores tardíos y reset coordinado.
Las proyecciones completas gate/up/down de Brandon compararon 5088 salidas
sin diferencia numérica frente a Q8; 0,717–0,792% de L2 relativo frente a FP32.
La ocupación medida del MAC escalar supera el 91%; motivó la versión de
[dos motores](Q8_GEMV_PARALLEL.md), con las mismas 5088 salidas y aproximadamente
1,92× menos ciclos. El caudal útil simulado pasa de 0,283 a 0,542 GB/s;
la caché dual ya infiere BRAM. La cadena FFN está integrada; falta DMA.
Las formas máximas 32×2048/8192 y el tail de 31 filas pasan con referencia
Python independiente; 32×8192 acelera ~1,94×. El [empaquetador del host](GEMV_WEIGHT_LAYOUT.md)
también pasó 174 jobs/5376 salidas leyendo imágenes inmutables en AXI
simulado, con una corrupción deliberada de un byte detectada por el banco.
El diseño de 512 GiB y la FFN completa siguen requiriendo integración adicional.

El [diseño FPGA para Llama 3.2 1B Instruct](LLAMA32_1B_HARDWARE.md) fija dimensiones, presupuesto Q8/KV y particionado de matrices. Es planificación y validación de transporte sintético; los pesos Meta y la inferencia completa siguen pendientes.

| Hito | Evidencia necesaria |
| --- | --- |
| Inferencia de Brandon | Backend NTRPU conectado al motor: normalización, atención, RoPE, KV cache y salida; ejecutar prompts y comparar logits/tokens con una referencia. |
| Una FPGA física | Seleccionar placa, integrar DDR y PCIe/DMA, cargar pesos y ejecutar el cálculo; medir ancho de banda sostenido, latencia y errores. |
| Dos NTRPU | Repartir pesos/capas sin CUDA, transferir activaciones y comprobar resultados, completions y pérdida de una unidad. Presentarlas como un backend lógico. |
| Tarjeta de 512 GB | Cerrar compatibilidad FPGA/controlador/DIMM, capacidad útil, alimentación, integridad de señal y calibración antes de diseñar la placa. |

Dos tarjetas agregan capacidad mediante el runtime y el protocolo. Sus BAR no crean automáticamente memoria coherente. El rendimiento y la compatibilidad P2P deberán medirse en la topología elegida.

## Uso de herramientas y respaldo

Desarrollar y probar localmente; reservar futuras sesiones Vivado/Vitis para trabajos preparados y acotados. La GPU A10 no fue necesaria para las herramientas FPGA. Mantener fuentes, fixtures, hashes y resultados de cada tanda fuera del disco efímero.

El respaldo anterior está indexado en `artifacts/BACKUP.json`; el instalador web y la licencia están en `artifacts/`, excluidos de Git. La instalación completa de AMD no se copió localmente, pero sobrevivió en `/lambda/nfs/vivado`. Cada host requiere validar su licencia. La FFN local está respaldada en `artifacts/ntrpu-ffn-local-20261001.zip`; la recuperación posterior en `artifacts/ntrpu-resume-20261002/`, y la nueva evidencia RTL en `artifacts/ntrpu-rtl-20261002/`. Cada tanda completa se archiva automáticamente en el volumen persistente.

El [bloque local SwiGLU/Q8](SWIGLU_Q8_BLOCK.md) ya pasa 27/27
transacciones C/RTL: convierte las salidas FP64 y produce registros ABI1.
El adaptador AXI tiene validación funcional y ruteo propios; la secuenciación
FFN con el motor de matrices ya pasa sus bancos y conserva evidencia propia.
