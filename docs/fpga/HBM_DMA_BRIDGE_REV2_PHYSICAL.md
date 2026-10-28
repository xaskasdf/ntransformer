# Puente HBM–DMA revisión 2: implementación física aislada

Continúa [la captura FIFO registrada](HBM_DMA_BRIDGE_REV2.md) sobre el DCP
`6c5ed2814728e8aa25273cd7862f2b947dce1c9ade8489ddef48a50aa4df324a`.
Tanda: `D:/AI/artifacts/ntrpu-hbm-dma-rev2-physical/20261009T032334606453Z`.
No modifica el RTL validado ni la cola principal. El dispositivo es
`xcvu47p-fsvh2892-2-e`, con Vivado 2026.1.

## Resultado: ruteado, cierre temporal pendiente

Estado verificado: **`REV2_ROUTED_TIMING_OPEN`**. `route3` completa 10.900
nets ruteables sin errores. Los 11.133 puertos ruteados implícitamente por
el modo OOC no equivalen a conexiones físicas con otros bloques. Permanece
`RTSTAT-10`: 3.468 nets sin cargas ruteables.

| Dominio y alcance | Setup mínimo (ns) | Hold mínimo (ns) |
| --- | ---: | ---: |
| Lector, registros internos | −2,993 | −0,101 |
| DMA, registros internos | 0,735 | −0,122 |
| Lector, incluye interfaz | −3,567 | −2,297 |
| DMA, incluye interfaz | −0,252 | −2,296 |

El camino interno crítico va desde `lease_reg[source_base][1]` al enable
de un registro de contexto. Tiene 17 niveles y 7,719 ns de retardo:
1,744 ns de lógica y 5,975 ns de interconexión. Hay que volver a optimizar
timing con un contexto físico de frontera definido antes de decidir si
se añade una etapa de pipeline.

Los 1.318 caminos RAM→captura mantienen un margen mínimo de **3,040 ns**;
los 45 controles sincronizados, **3,154 ns**; los 12 buses Gray, **3,259 ns**
de skew. No hay excepciones ignoradas. La revisión reabre el DCP y reproduce
el resumen temporal. Se conservan 98 `ASYNC_REG` y los buffers ubicados en
`BUFGCE_X0Y23` y `BUFGCE_X0Y0`.

El reporte CDC conserva cinco `CDC-3`, 14 `CDC-6` y 1.318 `CDC-15`, y añade
cinco `CDC-26` sobre entradas de reset/link. Sus etiquetas y endpoints se
conservan para investigar; no se declaran resueltos. `check_timing` marca
`partial_input_delay` en `dma_link_up`, aunque el XDC exportado conserva
ambos límites de entrada. La cobertura interna no tiene endpoints sin
reloj ni sin restricción; la cobertura global sigue abierta.

## Contexto de implementación

Se insertan dos `BUFGCE` con enable constante alto y se comprueba que
conservan los consumidores originales: lector de 5 ns y DMA de 4 ns.
Las ubicaciones se eligen durante placement. No se introduce `HD.CLK_SRC`:
los buffers existen dentro del checkpoint de este experimento.

El contrato provisional reserva 1 ns de llegada máxima y 0,1 ns mínima a
las entradas; las salidas reservan 1 ns de setup externo y 0,1 ns de hold
externo (`set_output_delay -min -0.1`). Las incertidumbres conservadas son
0,1 ns para setup y 0,05 ns para hold. Estos presupuestos no provienen de
un wrapper HBM/XDMA integrado ni de una PCB. El informe conserva los
nombres, direcciones y dominios de los 11.375 bits de interfaz presupuestados.
Los retardos se referencian a los puertos de reloj anteriores al BUFG;
no modelan todavía la latencia de lanzamiento de un bloque vecino.
Tampoco hay `HD.PARTPIN_LOCS/RANGE` para las interfaces. Por ello, completar
los nets internos ruteables no demuestra ruteo de la frontera: véase
[AMD UG905, Build an Effective Floorplan](https://docs.amd.com/r/2023.1-English/ug905-vivado-hierarchical-design/Build-an-Effective-Floorplan?contentId=m~T7LXh2DsR9Tr9FfC_SbQ).

Se mantienen los 25 grupos de máximo de datos/control y los 12 límites
Gray de 4 ns. Se añade un máximo de 4 ns desde `dma_link_up` hasta
`link_s1_reg/D`; los consumidores en DMA conservan análisis síncrono.
El reset frío tiene una excepción limitada a los diez pines `CLR` de
los sincronizadores de arranque. Se verifica la inversión de polaridad,
el fanout y `ASYNC_REG`. La liberación interna sigue temporizada; la
coordinación externa de reset, aislamiento y quiescencia PCIe sigue pendiente.

## Flujo y evidencia

`tools/ntrpu_hbm_dma_rev2_physical.tcl` abre una copia del DCP, prepara el
contexto y ejecuta `opt_design`, placement, optimización física y routing
con directiva Explore. Guarda checkpoints y reportes aun si la revisión
posterior detecta un problema. El runner privado limita cada tanda a dos
CPU, prioridad reducida y 1.200 s.

`tools/ntrpu_hbm_dma_rev2_physical_audit.tcl` reabre el checkpoint archivado
sin optimizarlo. Comprueba buffers, conectividad, datos de captura y cadenas
de sincronización. Separa timing interno entre registros del conjunto
que incluye los presupuestos externos. Los informes de skew se revisan
aparte de setup/hold, como requiere
[AMD UG903](https://docs.amd.com/r/2024.1-English/ug903-vivado-using-constraints/About-Bus-Skew-Constraints).

`route1` conserva el primer rechazo del revisor: la comprobación inicial
buscaba cargas directas del reset, pero la síntesis inserta un inversor.
La corrección sigue el fanout hasta los receptores y exige un único `LUT1`
con función de inversión. No cambia la lógica del puente.

`route2` completa placement y optimización física, pero llega al límite de
1.200 s durante el ruteo inicial Explore. Su checkpoint colocado se descarga,
verifica y respalda. `route3` lo reutiliza con
`tools/ntrpu_hbm_dma_rev2_route_probe.tcl`: ruteo Quick de diagnóstico,
26 s para el router y 116,266 s para la tanda completa. Quick no optimiza
timing; mantiene las restricciones para el análisis posterior. Véase
[AMD UG835, route_design](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/route_design).
La reapertura independiente `audit1` tarda 71,397 s.

La optimización duplica dos fuentes `dma_fault`; los grupos respectivos
muestran 50 % de cobertura From porque una copia no alcanza el sincronizador.
Se exige la conexión original a cada uno de los 45 pines D y su retardo.
Los indicadores `PLACED_FULLY`/`ROUTED_FULLY` consultados como propiedades
del diseño devuelven vacío en este build: la prueba de ruteo usa el inventario
de nets y los reportes DRC, sin convertir esos vacíos en valores verdaderos.

## Reproducción y siguiente criterio

El verificador acepta evidencia de una tanda fallida por timeout únicamente
para demostrar la procedencia del checkpoint colocado. Rechaza seis
alteraciones de evidencia: omisión de datos, mezcla de FIFO, inversión de
dominios, grupos asíncronos, exclusión global y ampliación del reset.

```text
python -B tools/check_ntrpu_hbm_dma_rev2_physical.py \
  TANDA/route3 TANDA/audit1 --placement TANDA/route2
```

El siguiente paso es definir una frontera con registros vecinos y relojes
compartidos, o con puntos de partición derivados de la integración real;
resolver las advertencias de link/reset y repetir el cierre guiado por
timing desde el checkpoint conservado. Exigir setup, hold, máximos CDC y
skew no negativos antes de promover el puente a integración HBM/XDMA.

Los resultados finales se registran en `OUTCOME.json`, junto con los recibos
`BACKUP_*.json`. Esta implementación aislada no habilita fabricación ni
sustituye validación de IP nativas, reset integrado o pruebas sobre tarjeta.
