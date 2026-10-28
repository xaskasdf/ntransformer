# Revisión 3: implementación física con frontera registrada

Tanda: `D:/AI/artifacts/ntrpu-hbm-dma-rev3-physical/20261009T060206827270Z`.
Estado: `REV3_BOUNDARY_ROUTED_TIMING_PASS`. La reapertura independiente
confirma cierre temporal del banco a 200/250 MHz y conserva sus restricciones.

## Síntesis

`synth1` completa en 242,350 s, sin errores ni advertencias críticas.
El checkpoint nuevo se respalda y restaura antes de la siguiente etapa.

| Recurso | Rev2 | Rev3 | Cambio |
| --- | ---: | ---: | ---: |
| LUT totales | 8.671 | 8.469 | −202 |
| LUT de lógica | 7.903 | 7.701 | −202 |
| LUT de RAM | 768 | 768 | 0 |
| FF | 9.268 | 9.398 | +130 |

Se mantienen 98 `ASYNC_REG`. El inventario identifica 65 registros de
`window_end`, 65 de `published_end`, 65 de `source_capacity` y 64 de
`target_capacity`, todos en el lector. Estos 259 registros identificados
por nombre no equivalen al incremento neto de FF del diseño completo.

`capture1` termina en 71,993 s. Conserva 45 bits de control, 1.318 bits
capturados, 12 buses Gray, 25 máximos y ninguna excepción ignorada.
Los dominios definidos tienen cinco `CDC-3`, catorce `CDC-6` y 1.318
`CDC-15`, sin críticos. El mínimo estimado RAM→captura es +3,483 ns.
Esta medición previa a ubicación/ruteo no incluye un reloj físico real.

## Alcance y procedencia

El donante es `rtl2` de la [revisión 3 funcional](HBM_DMA_BRIDGE_REV3.md).
Se verifican sus hashes y se sintetizan únicamente el paquete, el FIFO
rev2, la lane rev1 y el wrapper rev3. La referencia rev2 y los bancos de
simulación no forman parte del diseño sintetizado.

El perfil conserva VU47P `xcvu47p-fsvh2892-2-e`, Vivado 2026.1 build
6511674, lector de 5 ns y DMA de 4 ns, con incertidumbres de setup/hold
de 0,100/0,050 ns. La [frontera anterior](HBM_DMA_BOUNDARY.md) proporciona
el método de medida: 7.662 registros de lanzamiento, 3.713 de captura,
dos BUFGCE y exclusiones limitadas al exterior de esos registros y los
diez CLR de arranque. La frontera no es un adaptador de protocolo.

## Etapas y criterios

1. `synth`: nueva síntesis y registro del coste de recursos y de los cuatro
   bancos de límites. No se reutiliza el netlist rev2.
2. `capture`: reapertura del checkpoint nuevo; identificación exacta de
   45 controles, 1.318 bits RAM→captura y 12 buses Gray. Se conservan los
   25 máximos CDC sin dispensas ni exclusiones globales.
3. `prepare`: inserción de la frontera y verificación de todas las
   conexiones originales. Link añade el máximo número 26.
4. `place`: optimización y ubicación Default, con checkpoints separados.
5. `route`: RuntimeOptimized desde el checkpoint colocado. Si necesita
   ajuste, una tanda separada puede aplicar Explore después del ruteo.
6. `audit`: reapertura independiente del checkpoint seleccionado y
   repetición de topología, restricciones, timing, CDC y ruteo.

Los cuatro grupos de límites se inventarían por registro y dominio.
Tras insertar la frontera se reportan setup y hold hacia ellos desde
registros del lector, incluidos los de lanzamiento de configuración.
La síntesis puede compartir o eliminar registros; los 260 bits declarados
en RTL no se asumen como un recuento de FF físicos.

Se conservan avisos de CDC y DRC completos. La explicación del diagnóstico
de reset usa el reproductor anterior sin RAM, cuya evidencia también debe
verificarse, junto con la polaridad real del checkpoint nuevo. No se
introducen waivers. La salida solo puede marcar timing cumplido si setup,
hold, pulso, máximos CDC y skew cumplen y no hay errores de ruteo.

`prepare1` inserta y verifica las 11.375 conexiones, pero termina con rechazo
en 239,574 s: antes de optimizar, `IS_CLR_INVERTED` se devuelve vacío en los
FDCE de arranque. Se conserva `anchors_unconstrained.dcp` y el resultado
fallido. `prepare2` reabre ese checkpoint, vuelve a comparar las conexiones
y aplica las tres exclusiones limitadas y el máximo de link.

El revisor de rev3 resuelve propiedades omitidas únicamente contra el modelo
FDCE instalado, SHA-256
`74d30d7c788e7db47f12b92d5b4b57b707974a0962708b309e4cc081d5f98b04`:
`INIT=0` e `IS_CLR_INVERTED=0`. El netlist funcional exportado contiene los
diez FDCE sin parámetro de inversión, evidencia guardada en
`RESET_DEFAULT_REVIEW.json`. Los informes conservan tanto propiedades crudas
como valores resueltos; no se cambia el netlist para normalizarlos. Una
conexión directa requiere inversión explícita, mientras que el inversor
LUT1 requiere CLR no invertido. Los diez destinatarios, dominios y estado
inicial continúan comprobándose.
Los valores por defecto y la absorción del inversor durante `opt_design`
también están descritos en
[AMD UG974, FDCE](https://docs.amd.com/r/2025.1-English/ug974-vivado-ultrascale-libraries/FDCE).

`prepare2` completa en 176,090 s y conserva exactamente los 11.375 pines
de conexión originales. Resuelve diez propiedades de inversión omitidas;
los diez INIT ya son explícitos. El verificador también reconoce la forma
literal alternativa de `write_xdc` (`[list [get_ports ...] ...]`) y compara
los mismos conjuntos exactos de puertos/pines. No evalúa código Tcl ni
modifica restricciones para facilitar la comprobación. Ocho controles
negativos de frontera/exclusiones pasan antes de ubicar el diseño.

`place1` termina en 248,674 s y guarda checkpoints optimizado y colocado.
El setup colocado es +0,573 ns; hold es −0,211 ns, con 2.684 endpoints
incumplidos. No es un resultado ruteado. Los grupos de límites tienen setup
entre +1,690 y +1,928 ns; sus valores de hold siguen negativos.

## Resultado ruteado

`route1` completa en 480,453 s desde el checkpoint colocado, usando
RuntimeOptimized dentro del límite de 1.200 s. No se aplica optimización
Explore posterior. El análisis detallado produce estos márgenes:

| Dominio | Frecuencia | Setup | Hold |
| --- | ---: | ---: | ---: |
| Lector | 200 MHz | +0,275 ns | +0,007 ns |
| DMA | 250 MHz | +0,922 ns | +0,013 ns |

El peor ancho de pulso es +1,468 ns. Setup tiene 0/36.102 endpoints
incumplidos; hold, 0/36.056; ancho de pulso, 0/22.303. Las 21.589 redes
que requieren ruteo están completas, sin errores. La implementación
incluida la frontera ocupa 8.428 LUT y 20.773 FF; este último total incluye
los 11.375 registros de medida y no sustituye el coste del núcleo sintetizado.

La ruta limitante parte de `submit_context[bytes][5]`, registrado por
`ntrpu_boundary_ff_10400`, y termina en
`lanes[1].bridge/accepted_reg[0][7]/CE`: 17 niveles, ocho CARRY8 y 4,433 ns
de datos (1,205 ns de lógica y 3,228 ns de interconexión). Frente al
resultado rev2 seleccionado, el setup global pasa de −0,339 a +0,275 ns
(+0,614 ns) en el mismo perfil de medida. Son implementaciones distintas;
la comparación no aísla el efecto del RTL del efecto de ubicación/ruteo.

| Límite precalculado | FF físicos | Setup | Hold |
| --- | ---: | ---: | ---: |
| `window_end` | 65 | +1,182 ns | +0,021 ns |
| `published_end` | 65 | +1,082 ns | +0,015 ns |
| `source_capacity` | 65 | +0,969 ns | +0,021 ns |
| `target_capacity` | 64 | +1,258 ns | +0,046 ns |

El informe CDC conserva seis `CDC-3`, catorce `CDC-6`, 1.318 `CDC-15`
y cuatro `CDC-26` asociados exclusivamente a los CLR de arranque. Los
avisos de reset requieren la reproducción anterior sin RAM y la revisión
de polaridad del checkpoint; no se descartan ni convierten en aprobación
del reset de placa. El único DRC es `RTSTAT-10`, sobre las 3.713 salidas
exteriores de captura sin carga del banco. Se revisa por separado que la
frontera interna sí esté ruteada.

`audit1` reabre el checkpoint en otro proceso Vivado y completa en
165,404 s. Timing, métricas por dominio y grupos precalculados son idénticos
a `route1`. Los 11.375 registros conservan ubicación y conexiones internas:
11.368 ruteadas y siete dentro del mismo sitio. No hay endpoints internos
sin restricciones; el único `no_input_delay` es el reset de arranque cuyo
alcance se comprueba por separado.

Los márgenes mínimos restantes son +3,041 ns para RAM→captura,
+3,416 ns para controles, +3,599 ns para link y +3,446 ns para skew Gray.
El revisor rechaza nueve alteraciones de frontera/reset y seis de captura
o inventario. Otros tres controles rechazan hash de biblioteca, polaridad
cruda e INIT crudo incorrectos. Se preservan los avisos completos y el
intento `prepare1` fallido; el PASS corresponde al banco acotado.

## Herramientas y respaldo

`tools/ntrpu_hbm_dma_rev3_physical.tcl` dirige las etapas e incorpora los
informes de configuración. Reutiliza copias congeladas de los revisores
anteriores. `check_ntrpu_hbm_dma_rev3_capture.py` verifica la cadena
RTL→síntesis→restricciones; `check_ntrpu_hbm_dma_rev3_physical.py` añade
frontera, ruteo y reapertura.

```text
python -B tools/check_ntrpu_hbm_dma_rev3_physical.py \
  REV3/rtl2 TANDA/synth1 TANDA/capture1 TANDA/prepare2 \
  TANDA/route1 TANDA/audit1 TANDA/place1 \
  --reset-probe FRONTERA/probe4 --insertion TANDA/prepare1
```

`REV3` es `D:/AI/artifacts/ntrpu-hbm-dma-rev3/20261009T052346043474Z`;
`FRONTERA` es `D:/AI/artifacts/ntrpu-hbm-dma-boundary/20261009T040422369542Z`;
`TANDA` es el directorio indicado al inicio. `OUTCOME.json` contiene la
revisión completa. El SHA-256 del checkpoint seleccionado es
`cb149e48b44d3499a0bb9593c311350ec53b537a0d2ca2eb489828bc7a1b11c9`.

Cada etapa tiene directorio propio, presupuesto de CPU/tiempo, hashes de
entrada/salida y checkpoint recuperable. Los recibos `BACKUP_*.json`
registran copias en `D:/AI/artifacts/ntrpu-content-store` y restauración
verificada. Se protegen 162 archivos previos; la cola y los trabajos del
agente principal quedan fuera de esta tanda.

`handoff/` congela revisores, dependencias, documentación y recibos,
incluidas las referencias a las pruebas funcionales y de reset anteriores.
Su `replay.py --store D:/AI/artifacts/ntrpu-content-store` verifica hashes,
objetos y copias restauradas, repite la revisión y compara el resultado
completo con `OUTCOME.json`. `--restore-to DIRECTORIO_NUEVO` permite
reconstruir toda la evidencia referenciada sin depender de sus ubicaciones
originales. La repetición local no vuelve a ejecutar Vivado.

Los resultados de este banco no califican pines externos, reset de placa,
HBM/XDMA nativos, rendimiento de inferencia ni una PCB fabricable.

## Siguiente integración

Seguir el [contrato HBM–XDMA](HBM_PCIE_BRIDGE_CONTRACT.md) en una variante
separada: confirmar el reloj efectivo del modelo XDMA frente a los metadatos
de 125 MHz, conectar ambos C2H y probar backpressure, descriptor, writeback,
errores y ACK. Después del tráfico PIPE verificado, incorporar el lector
HBM nativo y los resets coordinados. El perfil de 4 ns de este banco no
demuestra por sí solo el funcionamiento de esa IP ni cambia estos criterios.
