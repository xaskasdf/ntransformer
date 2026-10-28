# Ajuste de timing interno de memoria HBM

## Alcance y punto de partida

Esta prueba privada continúa [HBM_RANGE.md](HBM_RANGE.md) sobre VU47P/HBM.
El donante ruteado tiene SHA-256
`cb110a36e78b08860b041443d7ba85fb913d66ee0c5e4f896d064f0767170ce6`.
Quedan cinco destinos con setup negativo en `mem_clk`: WNS −0,007 ns y
TNS −0,025 ns. PCIe, lector y gestión ya tienen setup positivo.

La peor ruta atraviesa cinco LUT desde la cola de comandos del conversor AMD
hasta el reset síncrono de un registro de datos. Su demora es 2,042 ns:
0,522 ns de lógica y 1,520 ns de ruteo. La última red alimenta 512 registros.

## Experimento acotado

`refine_hbm_memory_timing.tcl` opera sobre una copia del checkpoint con
`phys_opt_design -critical_cell_opt -path_groups mem_clk`. Una estrategia
separada permite `-restruct_opt` si la primera no mejora el resultado.
Las opciones se documentan en
[AMD UG835, Vivado 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/phys_opt_design).

Tras dos pasadas sin mejora, se prueba una tercera estrategia:
`phys_opt_design -directive AggressiveExplore`. Esta directiva no se combina
con `-path_groups`; por ello se compara el resultado en los cuatro dominios.
Las tres estrategias parten del mismo checkpoint, sin encadenar resultados.

El primer intento incluyó `-fanout_opt`. Vivado lo rechazó explícitamente
para post-route (`Vivado_Tcl 4-265`); se conserva como intento fallido,
con su fuente original, log y respaldo. No produjo otro checkpoint.

El runner limita cada etapa a 600 segundos, dos CPU y prioridad reducida,
en `ntrpu-side/axi-bridge-hbm-memory-timing/`. Registra exclusivamente sus
propios grupos de procesos. El RTL, los perfiles de IP y las tandas previas
se conservan. Los períodos siguen siendo 4, 2,222, 5 y 10 ns.

## Criterios de selección

Otra ejecución de Vivado reabre el resultado y repite la auditoría física.
Se exige igualdad de las restricciones exportadas antes/después y tras
reapertura; cobertura de 1.501 rutas CDC, 88 conexiones entre etapas de
sincronización, 1.458 capturas RAM y 17 consumidores de reset canónico;
16 límites de skew válidos, ruteo completo y ausencia de nuevos errores.

Setup, hold y ancho de pulso se informan por separado. Una ejecución correcta
de la herramienta no significa cierre temporal. Las pruebas funcionales de
la variante RTL siguen siendo las de la tanda anterior; esta etapa no aporta
una nueva prueba de equivalencia secuencial del netlist optimizado.

## Resultado seleccionado

Las pasadas de replicación y reestructuración terminaron en 221,33 y 225,22 s,
respectivamente, incluyendo sus reaperturas. No modificaron celdas ni márgenes.
`explore-20261011T143922Z` completó optimización y reapertura en 244,93 s y
eliminó los cinco destinos fallidos. Su checkpoint tiene SHA-256
`3784d2b3b516b922917d66b7468d619d9677b4f7053989beb422d48aac69fe0b`.

| Dominio | Período | Setup mínimo | Hold mínimo | Fallos setup/hold/pulso |
| --- | ---: | ---: | ---: | ---: |
| Memoria | 2,222 ns | +0,001 ns | +0,015 ns | 0 / 0 / 0 |
| PCIe | 4 ns | +0,132 ns | +0,011 ns | 0 / 0 / 0 |
| Lector | 5 ns | +0,481 ns | +0,010 ns | 0 / 0 / 0 |
| Gestión | 10 ns | +8,776 ns | +0,037 ns | 0 / 0 / 0 |

Vivado informa cuatro redes optimizadas, sin celdas añadidas o eliminadas.
El log identifica `p_1_in[461]`, `p_1_in[37]`, `p_1_in[237]` y
`S_AXI_RDATA_II[0]` dentro del conversor del canal 0. La mejora total que imprime
PhysOpt es 0,007 ns; los márgenes de la tabla están redondeados por el informe.

La reapertura y revisión local confirman igualdad de restricciones y resultados.
Las 24.900 redes están ruteadas, sin errores ni gaps. CDC conserva margen
mínimo +1,613 ns; los 16 límites de skew pasan. Las auditorías de topología y
reset pasan, y 22 controles negativos rechazan evidencia alterada.
Se conservan las advertencias CDC, DRC y metodología previas, sin waivers.

**El timing interno OOC pasa con un margen de memoria de apenas 1 ps.** Esto
no proporciona margen de integración de placa ni certifica hardware. El siguiente
hito es sustituir las fronteras ideales por la integración nativa HBM/PCIe,
sus relojes y presupuestos externos, y repetir implementación y validación.

## Reproducción y evidencia

```bash
python -B run_hbm_memory_timing.py NUEVA_TANDA --prepared FUENTES \
  --strategy replicate --donor CHECKPOINT --donor-sha SHA256 \
  --settings VIVADO_SETTINGS
python -B review_hbm_memory_timing.py TANDA_RESTAURADA
```

Los artefactos se guardan en
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-memory-timing/`, con manifiestos SHA-256
y restauración verificada. El alcance sigue siendo una composición OOC con
árboles internos ruteados y relojes externos ideales. Faltan la integración
conjunta de los IP nativos HBM/PCIe y la calificación física de la tarjeta.

`20261011-hbm-memory-timing.zip` contiene las cuatro tandas, incluida la fallida,
los checkpoints, informes, fuentes ejecutadas, baseline y controles negativos.
`BUNDLE_BACKUP.json` registra el hash del ZIP y su restauración independiente;
`review_hbm_memory_timing_bundle.py` verifica la selección contra los informes.
Los cuatro runners y sus siete grupos de procesos propios finalizaron.
