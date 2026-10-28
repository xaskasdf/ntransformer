# Cruce del lector HBM hacia PCIe: 200 ↔ 250 MHz

## Alcance y contrato

`ntrpu_axi_bridge_hbm_cdc.sv` añade el transporte entre la frontera
`ntrpu_axi_bridge_hbm_source.sv`, a 250 MHz, y los dos canales del lector
congelado, a 200 MHz. Se mantiene la prioridad VU47P/HBM; la tarjeta DDR4
de 512 GiB sigue siendo un diseño separado.

Cada canal tiene una FIFO de comandos de 136 bits y una FIFO de retorno de
593 bits. Datos y CPL comparten esta última: la CPL permanece detrás de todos
los datos anteriores, incluso con contrapresión. Se reutiliza, sin cambios,
`../hbm_dma_rev2/ntrpu_bridge_async_fifo.sv`, con cuatro entradas de RAM y un
registro de salida. El tag local conserva 64 bits; la frontera anterior sigue
reteniendo y restituyendo la identidad completa de 128 bits.

Las colas conservan sus ofertas durante fallos. El escritor y la frontera
mantienen la propiedad hasta el retiro autorizado; drenar una cola no equivale
a retirar el contexto. `p_stop` queda enclavado. La admisión de solicitudes
nuevas corresponde a la frontera; este transporte puede aceptar ofertas ya
existentes durante la propagación del fallo.

## Relojes, estado y reset

Un vigilante de solicitud/eco observa el reloj del lector desde gestión a
100 MHz. Tras armarse, la pérdida del reloj, del estado iniciado o un fallo
del lector cierra la época. Reiniciar el reloj no borra ese fallo. La detección
y su propagación tienen latencia; no son un aviso instantáneo entre dominios.

Los tres puertos `*_cold_rst_n` deben proceder de las liberaciones canónicas
de **un mismo reset frío coordinado**, sincronizadas en sus respectivos
dominios. Nunca conectar reset operativo PCIe ni reiniciar un solo extremo
de las FIFO. Gestión debe permanecer activa. Este vigilante observa 200 MHz;
no prueba la continuidad del reloj interno HBM de aproximadamente 450 MHz.

## Evidencia seleccionada

| Comprobación | Icarus | XSIM en VM |
| --- | --- | --- |
| Integración: V3, escritor, frontera y lectores reales | 44/44 | 44/44; 296,2 s |
| Saturación, fases distintas y recuperación fría | 4/4 | 4/4; 40,4 s |
| Bytes auditados entre ambos bancos | 45.184 | 45.184 |
| Escenarios de integración con deuda retenida | 40 | 40 |
| Evidencias alteradas rechazadas | 16/16 | 16/16 |

Los bancos usan fixtures explícitos para memoria AXI, mapping, B y ACK.
Las trazas verifican relojes independientes, detención en alto/bajo, deuda
retenida y fallo visible en PCIe en **683,3–690,7 ns** para los casos probados.
El revisor exige límites de 800 ns en gestión y 820 ns en PCIe.

El banco de colas también prueba CPL simultánea al último dato, colas llenas,
metadatos completos, ofertas estables y reset frío con el lector detenido.
Se conserva un intento inicial fallido: el estímulo dependía del orden de
inicialización del simulador. La versión seleccionada aplica una transición
explícita de reset antes de habilitar relojes y tráfico.

## Síntesis y siguiente condición física

La síntesis del **transporte aislado** en VU47P tardó 107,5 s: 1.538 LUT,
1.598 FF, cero latches, black boxes, bucles y endpoints internos sin reloj
o restricción. No incluye los IP nativos ni constituye cierre temporal.

El inventario CDC conserva, sin waivers, **8 avisos CDC-6** sobre punteros
y **1.458 CDC-15** sobre captura RAM, además de 7 CDC-3 informativos.
Hay 1.602 bits de entrada y 1.602 de salida sin presupuestos externos.
Las fuentes físicas de reloj, el estado de entrada y la relación entre
resets quedan fuera de esta síntesis aislada.

Antes de integrar HBM y PCIe nativos se requiere:

1. Vincular y auditar restricciones de demora/skew de punteros y captura RAM,
   sin ocultarlas con excepciones generales entre relojes.
2. Comprobar sincronizadores, reset coordinado y rutas reales tras place/route.
3. Conectar los IP seleccionados y validar juntos tráfico, reloj HBM, ACK
   físico y autorización de regiones.

## Reproducción y respaldo

La evidencia reside en `D:/AI/artifacts/ntrpu-axi-bridge-hbm-cdc/`.
`SELECTION.json` fija donantes y tandas; `PROCESS_COMPLETION.json` confirma
el fin de los tres trabajos privados. Los módulos calificados anteriores
conservan sus hashes. Los runners usan dos CPU, prioridad reducida y límites
por proceso dentro de `/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-cdc/`.

```bash
python -B run_hbm_cdc.py NUEVA_SALIDA --engine xsim \
  --reader-source LECTOR --master-source MAESTRO --fifo-source FIFO_REV2 \
  --settings VIVADO_SETTINGS
python -B run_hbm_cdc_queues.py NUEVA_SALIDA --engine xsim \
  --fifo-source FIFO_REV2 --settings VIVADO_SETTINGS
python -B source/review_hbm_cdc_bundle.py PAQUETE_RESTAURADO
```

El paquete `20261011-hbm-cdc.zip` incluye fuentes, trazas, informes, checkpoint,
intento fallido y revisores. `BUNDLE_BACKUP.json` registra tamaño y SHA-256;
`BUNDLE_REVIEW.json` registra la auditoría de su restauración completa.
