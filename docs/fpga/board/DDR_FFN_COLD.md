# FFN supervisada sobre MIG: arranque en frío

## Alcance y estado

La tanda `20261004T082658Z-V7-ffn-supervised-ddr-cold-r2` pasa la
revisión independiente con host de 4 ns y tiene respaldo verificado local/NFS
(archivo SHA-256 `43e871914125474008796a2632f3a6d7b8fb5cadb3f48d25e89ab77adf060403`). El perfil de 6,2 ns
`20261004T083424Z-V7-ffn-supervised-ddr-cold-slow` también pasa la revisión
y tiene respaldo verificado (SHA-256
`e0d626838caf6a99e2ca5b95e0b3fecd793c4128a8d8d2c650cde6a06840c563`).
Consultar los certificados locales `artifacts/ntrpu-board-vivado/<ID>/VERIFIED.json`
antes de considerar terminado un respaldo.

Se conectan el RTL supervisado vigente, cuatro motores MAC, FIFO por bloques,
SwiGLU HLS y controlador MIG generado por Vivado 2026.1 para
`xcvu5p-flva2104-2-i`. MIG usa el modelo genérico de dos RDIMM, cuatro ranks,
128 GiB, bus físico de 72 bits con ECC, AXI de 512 bits y direcciones de
37 bits a DDR4-1600. La IP tiene `Simulation_Mode=BFM`; no es una simulación
física posterior al ruteo. No homologa el DIMM Micron objetivo.

## Conexión y secuencia

El reloj UI de MIG alimenta el motor a 5 ns. Host y gestión conservan
relojes independientes; el conversor AXI existente cruza de host a motor.
`sys_rst` de MIG se aplica al arrancar, separado de la petición de reset del
supervisor. Una confirmación registrada observa ocho flancos con interfaces
vacías bajo reset; otra exige reloj UI disponible y calibración completada
antes de permitir tráfico.

Los pesos y activaciones llegan únicamente por escrituras AXI del productor
host. MIG genera READY, R y B; el array del banco es un comparador pasivo,
sin inicialización jerárquica de la memoria. Después de cada FFN, el host
lee el resultado y completa el servicio de interrupción ABI2.

## Evidencia numérica y de transporte

El perfil de 4 ns ejecuta tres FFN sintéticas D32/H32, con estas medidas:

| Observación | Resultado |
| --- | --- |
| Resultados finales | 96 FP32 exactos |
| Transacciones host | 54: 48 escrituras y 6 lecturas |
| Bytes cargados / calculados / devueltos | 9.984 / 2.112 / 384 |
| Lecturas aceptadas de MIG | 192 beats de 64 bytes |
| Ciclos de cómputo del contador hardware | 2.224 / 2.165 / 2.154 |
| Tiempo de cómputo simulado | 11,120 / 10,825 / 10,770 µs |
| ACT observados por rank | 64 / 79 / 78 / 106 |
| Evidencias alteradas rechazadas | 36 |

El perfil de 6,2 ns conserva las fuentes RTL/HLS/modelos y produce otros
96 FP32 exactos con las mismas cantidades de datos y transacciones. Sus
contadores de cómputo miden 2.132 / 2.254 / 2.045 ciclos
(10,660 / 11,270 / 10,225 µs), y sus ACT por rank son 78 / 79 / 78 / 108.
También rechaza 36 alteraciones. Entre ambos perfiles: **seis FFN, 192 FP32,
108 transacciones y 72 alteraciones rechazadas**. Una comprobación adicional
rechaza tres inversiones de orden en eventos con igual marca temporal.

Tres rotaciones de buffers ejercitan todos los ranks, direcciones superiores
a 4 GiB, extremos exactos de 4 y 128 GiB, división de ráfagas en páginas de
4 KiB y respuestas retenidas por el host. El revisor reconstruye cada byte,
la identidad de las transacciones, contadores acumulados 1/2/3 e interrupciones
con ambos órdenes de reconocimiento y modo de sondeo.

Los ciclos anteriores delimitan aceptación y finalización del motor. El
contador del banco también incluye programación MMIO; no son intercambiables.
Estas medidas pequeñas no estiman tokens/s de Llama ni rendimiento en placa.
La calibración a 28,051098 µs pertenece al arranque acelerado de simulación;
no mide el tiempo de arranque de una placa.

## Reproducción y diagnósticos conservados

Desde Linux con Vivado configurado, usar directorios nuevos y las exportaciones
MAC/SwiGLU identificadas por sus manifiestos, además del ejemplo DDR original:

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_ddr_cold.tcl \
  -tclargs MAC_RUN SWIGLU_RUN \
  supervised_sim_irq_errors_quad_fp32_fifo NUEVA_SALIDA DDR_EXAMPLE
python3 tools/check_ntrpu_ffn_ddr_cold.py NUEVA_SALIDA/seed-1.log \
  --seed 1 --host-period-ps 4000
```

Para host lento, usar `supervised_sim_irq_errors_quad_slow_fp32_fifo` y
`--host-period-ps 6200`. La semilla es 1; las cargas y separaciones de este
banco son deterministas. Cada lanzamiento limita Vivado a cuatro hilos y
1.800 s. El revisor adicional comprueba fuentes, IP, modelos, advertencias
y alteraciones negativas antes de sellar el archivo.

`081445Z` conserva el fallo inicial de declaración del banco. `082015Z`
conserva el fallo de comparación exacta entre tiempos reales a 65,540998 µs;
la revisión siguiente compara picosegundos enteros y supera ese instante.
Ambos intentos fallidos tienen respaldo verificado.

## Límites y siguiente integración

El RTL/HLS/XPM y los conversores CDC coinciden con la revisión FFN `060223Z`.
De los 35 archivos importados del modelo, sólo cambia el wrapper de densidad,
ancho y velocidad; el modelo DRAM permanece intacto. Se conservan todas las
advertencias, incluidas dos aserciones de calibración AMD que XSIM no soporta
y modelos FIFO que omiten retardos de sincronización.

La validación del rango de este banco original pertenece al testbench.
Un [banco posterior con filtro AXI sintetizable](DDR_AXI_GUARD.md) ya pasó
ambos relojes host; conserva separado el resultado original y sus fuentes.
También falta un adaptador sintetizable de
calibración/reset con propiedad y vaciado de transacciones: las confirmaciones
de este banco sólo cubren arranque frío. El reinicio DDR con tráfico mantiene
los fallos documentados en [MEMORY_QUALIFICATION.md](MEMORY_QUALIFICATION.md).

Siguen pendientes la integración PCIe/DMA completa, cuatro canales, protección
de buffers frente a escrituras concurrentes, cierre físico con restricciones
de placa y ensayo con hardware. Este resultado avanza V7; no cierra V2/V3 ni
la [cola de diseño](VIVADO_QUEUE.md).
