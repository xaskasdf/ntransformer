# Integración PCIe nativa del control NTRD

Integración V7, actualizada el 2026-10-05. La revisión de la tanda fría
está aprobada; los criterios completos de `VIVADO_QUEUE.md` siguen abiertos.

## Dependencias y composición

Partir de `ntrpu_pcie_fragment_dual_mig.sv`, con una sola instancia del
puente de reservas. La vida retenida está validada en `045125Z`; antes de
lanzar, exigir revisión aprobada y respaldo verificado de la tanda con
dos MIG `20261005T050945Z-V7-ffn-supervised-managed-dual-mig`.

El transporte aislado está en `ntrpu_pcie_fragment_dual_host_cold.sv`.
Antes de conectarlo al ejemplo XDMA, exigir también la revisión aprobada
y el respaldo de `20261005T055601Z-V7-ffn-supervised-managed-host-cold-r2`.
Consultar [contrato, drenaje y evidencia](PCIE_DUAL_HOST_CDC.md). Conservar
el intento anterior rechazado: terminaba antes del último retorno de
confirmación AXI-Lite; su PASS interno no bastaba para aprobarlo.

Conectar BAR0 → CDC AXI-Lite → control NTRD y BAR2 → CDC AXI4 de
512 bits → recolector F01. Ambos destinos usan `source_clk`; los dos CDC
hacia memoria permanecen dentro del pool. Reutilizar el transporte de
`ntrpu_axil_cdc_recode.sv` y la configuración AMD del ingreso existente.
No instanciar `ntrpu_pcie_fragment_host` o `ntrpu_pcie_fragment_async`
completos: contienen el control/recolector/camino de memoria anterior.

Crear módulos y bancos aislados. Conservar el ejemplo XDMA de referencia,
el productor `ntrpu_pcie_frame512_driver.vh`, los modelos DDR4 y las tandas
nativas anteriores. La identificación del host debe comprobar NTRD,
versión, capacidades, dos canales y 37 bits locales; mantener `0xac`
como validez del snapshot y el canal explícito en `0xb0`.
La IRQ diagnóstica permanece en el dominio fuente; MSI/MSI-X requiere su
propio cruce y protocolo de confirmación. Empezar con lectura de estado.

## Reset y barrera

Definir por separado reset de configuración, resets operativos del pool
y resets de cada extremo de los CDC del host. La prueba actual de vida
retenida no incluye estos últimos. No conectar automáticamente el reset
PCIe al registro de vida fría. La pérdida del enlace debe invalidar la
sesión y bloquear nuevas reservas; conservar diagnósticos no significa
que una respuesta atraviese un CDC reiniciado.

Primero probar arranque frío coordinado. Antes de ensayar PERST durante
tráfico, especificar el tratamiento de AW/W/B/R aceptados en cada dominio
y cómo se declara la pérdida de sus respuestas. No prometer recuperación
caliente de DDR ni reanudación de una reserva huérfana.

`source_stopped`, `ingress_isolated` y `quiesce_epoch` siguen siendo pruebas
externas de quiescencia. Exigir drenaje en ambos lados del transporte;
`transport_idle` del recolector por sí solo no prueba ausencia de escrituras
PCIe publicadas o de tráfico dentro del CDC.

## Evidencia exigida

Validar primero los dos transportes con relojes distintos y pausas AXI;
después conectar el ejemplo PCIe nativo. Correlacionar TLP → AXI host →
AXI destino → LINE → cookie → memoria → resultado → ACK, con identidad
completa y canal inmutable. Incluir errores de trama, STOP parcial y tras
exposición, lectura de diagnósticos, rearmado permitido y rechazo de ACK
sin barrera. Comparar readback DDR y observar ranks/tráfico periódico.

Una lectura directa del banco por UI solo verifica memoria: no acredita
lecturas DDR por BAR, DMA P2P, coherencia entre tarjetas ni inferencia.
Guardar modelos ejecutados, avisos, trazas y respaldo por cada nueva tanda.

El verificador nativo debe observar los IDs AXI producidos por XDMA y
correlacionar escrituras PCIe publicadas con respuestas B en el endpoint.
No reutilizar sin adaptación la convención del banco AXI que asigna el ID
según el trabajo. Para AXI-Lite, medir primera presentación y aceptación
en cada extremo: la confirmación al productor puede llegar después de
la aceptación del consumidor. Exigir drenaje completo y un intervalo
sin actividad antes del cierre; no truncar la traza al consumir la última
respuesta en el host.

## Tanda nativa revisada

`20261005T063837Z-V7-ffn-supervised-managed-native`, iniciada el 2026-10-05,
usa `ntrpu_pcie_managed_native.sv` y conserva el RTL del transporte R2.
Snapshot `source-20261005T063826Z.zip`, SHA-256
`3a399c5773cbfc7f34453ebdeadf9cb779f44f319b358d40857626d266be6abe`.
La revisión independiente aprobó `PASS_NATIVE_NTRD_DUAL_MIG_COLD`.
Enumeración, paquetes, pines AXI, propiedad y readback DDR coinciden.
Respaldo local/NFS verificado a las 07:21:59 UTC: 1.453 archivos,
246.322.457 bytes; SHA-256
`64419a87c22bb35b69c862d21e20010095047f682259d18bb67639bc17e658a4`.

El contrato de esta tanda exige 18 reservas, 35 tramas, 16 escrituras DDR
y 28 lecturas comparadas. Hay 294 escrituras de control y nueve lecturas:
ocho explícitas y una lectura de identificación durante la enumeración.
Cada retiro incluye un ACK sin barrera que debe fallar antes del ACK
válido. Los IDs AXI corresponden al perfil observado de XDMA (cero),
mientras épocas, etiquetas y cookies conservan los 64 bits.

Los verificadores `check_ntrpu_pcie_managed_native*.py` aprobaron la
evidencia y rechazaron 20 alteraciones deliberadas. El modelo de estado
conserva la lógica de la revisión anterior y cambia solo las dos
expectativas de ID AXI.
Antes de liberar reset, el endpoint puede presentar valores desconocidos;
se exige que la admisión permanezca cerrada y se comprueba con el VCD.
La colección terminal verificó el JSON de 171.789.456 bytes y su archivo
comprimido de 6.038.169 bytes. La auditoría incluye 218 entradas compiladas,
134 archivos PCIe del proveedor sin cambios y 35 archivos del modelo DDR.

## Medidas de simulación

El host opera a 4 ns, la fuente a 5 ns y los dos UI a 5 ns con fases
distintas. Se cotejan 70 señales VCD, 42.707 muestras host y 34.167 fuente.
Las 54 barreras incluyen todas las respuestas B de datos en ambos extremos.
El control drena a 170,790 us; siguen ocho ciclos fuente sin actividad y
el banco termina a 170,835 us. Vivado empleó 1.909,028 segundos.

| Intervalo observado | Mínimo–máximo |
| --- | --- |
| Primer beat RQ de trama → B aceptada por XDMA | 280–288 ns |
| RQ de escritura de control → B en XDMA | 256–272 ns |
| RQ de lectura explícita de control → RC en root | 424–428 ns |
| Línea ensamblada → B de memoria | 88,498–533,498 ns |
| Lectura DDR AR → R | 115–670 ns |
| Línea ensamblada → ACK de reserva | 1.750–2.195 ns |

Las escrituras PCIe son publicadas: B es una respuesta AXI interna, no una
finalización PCIe de escritura al root. Son intervalos de este banco con
pausas y errores deliberados, no caudal sostenido ni timing físico. El
readback compara 1.792 bytes por UI; no es una lectura DDR mediante BAR.

Los cuatro ranks de cada canal tienen actividad. Se revisan 135/129
lecturas periódicas y 14 lecturas de usuario por canal. Los 6.912/6.768
avisos del modelo corresponden a 48/47 ráfagas periódicas sin escritura
previa; se comprueban sus retornos y multiplicidad por chip. Los 169 avisos
de herramienta corresponden a la referencia nativa y los CDC/MIG añadidos.

## Próximas dependencias

La síntesis aislada `ntrpu_board_pcie_managed_host_synth.tcl` se lanzó como
`20261005T072212Z-V7-ffn-supervised-managed-host-synth` tras verificar el
respaldo nativo. Su snapshot `source-20261005T072200Z.zip` tiene SHA-256
`56889bb08708815226189c57ad59ba210581ebda11e405532878e620ade10d66`.
Preserva los 21 archivos RTL, usa cuatro hilos y un límite de 1.200 s.
Terminó con 9.939 LUT, 23.279 registros y 15 bloques RAM. Queda sin
aprobación por 12 avisos críticos de relaciones entre relojes del
supervisor; véanse [diagnóstico y respaldo](PCIE_DUAL_HOST_SYNTH.md).
PCIe y los controladores/PHY DDR quedan fuera de esa frontera de síntesis.
La auditoría del DCP precede a cualquier corrección de constraints o ruteo.

La siguiente extensión funcional sigue el
[contrato de pérdida del enlace](PCIE_DUAL_LINK_LOSS.md): distinguir
peticiones aceptadas, efectos en destino y respuestas abandonadas sin
borrar la vida retenida. No hay todavía recuperación PERST, MSI integrado,
DIMM comercial homologado, cuatro canales, placa ni inferencia calificados.
