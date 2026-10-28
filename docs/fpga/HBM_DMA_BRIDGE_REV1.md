# Puente HBM–DMA: RAM síncrona y parada registrada

Variante aislada para corregir los dos hallazgos de la
[revisión Vivado anterior](HBM_DMA_BRIDGE_VIVADO_REVIEW.md). Conserva el
puente original como referencia. Tanda:
`D:/AI/artifacts/ntrpu-hbm-dma-rev1/20261009T021147988156Z`.

## Cambios y contrato

Las dos implementaciones sustitutas están en `fpga/experimental/hbm_dma_rev1/`.
Mantienen los nombres de módulo: seleccionar estos archivos **en lugar de**
sus homónimos originales, nunca compilar ambos conjuntos juntos. El paquete
y el wrapper superior permanecen iguales.

- **FIFO:** escritura en un proceso exclusivamente `posedge wclk`, con
  atributo `ram_style="distributed"`. `wready` inhibe escritura durante reset.
  Los punteros y sincronizadores conservan reset asíncrono; la RAM no se borra.
  Sus contenidos anteriores carecen de validez hasta una nueva escritura y
  el cruce del puntero. La codificación sigue el
  [patrón de RAM distribuida de AMD](https://docs.amd.com/r/2023.1-English/ug901-vivado-synthesis/Dual-Port-RAM-with-Asynchronous-Read-Coding-Verilog-Example).
- **Parada:** `stop_source` registra `stop || halted` en cada canal antes de
  las dos etapas DMA. `KEEP` conserva los dos orígenes por canal para revisar
  sus conexiones. El puerto `stop` debe proceder del dominio `clk`; una
  petición asíncrona requiere adaptación externa.

Una parada externa se registra en el siguiente flanco del lector. Un fallo
local recién registrado añade un ciclo del lector antes de alimentar el
sincronizador DMA. Desde `stop_source`, dos flancos DMA propagan la parada;
puede aceptarse un descriptor en el flanco que actualiza la segunda etapa.
Esto no es un certificado de drenaje ni una parada instantánea.

Una oferta AXIS existente mantiene VALID y sus datos bajo contrapresión;
puede drenarse. Los contextos afectados quedan en cuarentena aunque se acepte
su resultado. El reset frío continúa exigiendo aislamiento y coordinación
externos: el bloque no demuestra que cesaron escrituras PCIe remotas.

## Pruebas conductuales

El banco original y su revisor Python conservan 48 escenarios, 110
transferencias y 108.416 bytes reproducidos sobre el modelo. Las imágenes de
memoria coinciden con el donante; no se afirma equivalencia formal.

`fpga/tb/ntrpu_hbm_dma_rev1_cases.svh` añade cuatro casos, repetidos con 24
semillas/fases de reloj: parada con descriptores pendientes en ambos canales,
parada de una oferta AXIS retenida y drenaje, latencia de un fallo local, y
reinicio con contenidos RAM anteriores seguido de una transferencia nueva.
Son 144 escenarios por ejecución. Las pruebas adicionales usan el receptor
independiente del banco y aserciones de latencia, contenido y cuarentena.

El respaldo se restauró, recompiló y ejecutó nuevamente con el toolchain
Icarus privado restaurado. Los ocho controles negativos originales continúan
rechazando evidencia alterada.

## Reproducción aislada

`tools/prepare_ntrpu_hbm_dma_rev1.py NUEVO_REPO --donor TANDA_RTL_ORIGINAL`
verifica los hashes del donante, copia fuentes y aplica las dos sustituciones
y la extensión del banco. Después ejecutar:

```text
python -B NUEVO_REPO/tools/run_ntrpu_hbm_dma_rev1.py NUEVA_SALIDA --toolchain TOOLCHAIN_PRIVADO
```

El donante es `D:/AI/artifacts/ntrpu-hbm-dma-rtl/20261009T005545469986Z`.
Los manifiestos y recibos `BACKUP_*.json` registran cada tanda; los originales
y la cola de simulaciones del principal no se modifican.

## Límite de la revisión

Vivado 2026.1 build 6511674 completa la síntesis OOC VU47P en 236,102 s.
Desaparecen `Synth 8-7137`, `Synth 8-4767` y los dos `CDC-10` de parada.
La advertencia restante de síntesis recorta `incoming.bytes` de 64 a 13 bits:
el wrapper valida spans de hasta 4.096 bytes antes de admitirlos.

| Recurso | Donante | Revisión 1 |
| --- | ---: | ---: |
| LUT totales | 9.213 | 8.639 |
| LUT de lógica | 9.213 | 7.871 |
| LUT de RAM | 0 | 768 |
| FF | 13.216 | 7.944 |

El informe sin dispensas conserva 2.342 `CDC-1`, 14 `CDC-6`, 226 `CDC-15`
y cinco `CDC-3`; aparecen ocho `CDC-13`. Estos últimos llegan desde la
RAM de payload a los enables de escritura de la RAM de completions: necesitan
revisar estabilidad, validez y captura del consumidor, no dispensarlos por
haber inferido RAM. La reducción de avisos de reset/parada no cierra ese problema.

La síntesis OOC y el análisis de restricciones no sustituyen placement,
routing ni una integración real HBM/XDMA. El XDC de revisión 1 añade dos
máximos de 5 ns para parada al conjunto parcial anterior: 19 grupos, 45 bits
y 12 buses Gray. No excluye globalmente relojes ni dispensa advertencias.
Los datos FIFO y las interfaces externas requieren revisión independiente.
La revisión del DCP restaurado termina en 70,523 s y valida los 45 enlaces
de primera etapa y sus segundas etapas, sin excepciones ignoradas.
`tools/check_ntrpu_hbm_dma_rev1_review.py SYNTH_STAGE CONTROLS_STAGE` verifica
hashes, recursos, cobertura y hallazgos pendientes; rechaza cinco alteraciones
de evidencia. Su resultado termina en `NOT_QUALIFIED` deliberadamente.

El siguiente paso es revisar una captura de salida FIFO en su dominio receptor,
incluidos descriptor, AXIS y completions, conservar los contratos de estabilidad
y repetir pruebas antes de definir sus límites temporales. Siguen pendientes
presupuestos de I/O, `HD.CLK_SRC`, reset integrado, placement/routing y XDMA real.
