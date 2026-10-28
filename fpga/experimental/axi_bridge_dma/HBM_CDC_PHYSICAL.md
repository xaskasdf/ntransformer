# Ruteo y restricciones físicas del cruce HBM–PCIe

## Resultado y alcance

Se implementó y reabrió el checkpoint del transporte VU47P/HBM descrito en
[HBM_CDC.md](HBM_CDC.md), conservando el RTL y sus pruebas funcionales.
Las rutas internas CDC cumplen los límites en este contexto **out of context
(OOC)**. El resultado no califica la tarjeta completa ni la composición con
HBM/PCIe nativos. La variante DDR4 de 512 GiB permanece separada.

La tanda `route-20261011T101007Z` tardó 327,5 s. La auditoría seleccionada,
`reopen-20261011T102115Z`, tardó 60,3 s y no aplicó restricciones ni modificó
el checkpoint. La comparación antes/después normaliza únicamente el comentario
de Vivado que contiene el nombre del archivo XDC exportado.

## Restricciones y cobertura

`constrain_hbm_cdc_physical.tcl` exige nombres, cardinalidades y relojes exactos:

| Cruce | Cobertura | Restricción |
| --- | --- | --- |
| Punteros Gray, ocho buses | 24 endpoints | demora ≤4 ns y skew ≤2 ns |
| RAM hacia registro de captura, cuatro bancos | 1.458 endpoints | demora ≤4 ns |
| Estado y vigilante, siete señales internas | 7 endpoints | demora ≤4 ns |

Son 19 restricciones de demora, todas con cobertura del 100 %, y ocho de
skew. El presupuesto de skew usa la mitad del período más rápido del perfil.
La captura RAM dispone de la espera impuesta por la sincronización del puntero;
su registro de salida precede al consumo externo de los datos.

Se sigue la separación entre demora y skew descrita en
[UG903, buses Gray](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/set_bus_skew-Example-Two).
No se añadieron waivers, multicycles ni cortes generales de reloj, que podrían
anular el límite de demora según
[UG903, señales asíncronas](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Constraining-Asynchronous-Signals).

## Mediciones del checkpoint ruteado

| Medición | Resultado |
| --- | --- |
| Redes ruteables / ruteadas / errores | 2.836 / 2.836 / 0 |
| Rutas CDC auditadas | 1.489 / 1.489 |
| Margen mínimo: punteros / captura / control | +3,513 / +3,131 / +3,617 ns |
| Skew máximo de punteros, límite 2 ns | 0,421 ns |
| Registros ASYNC_REG / conexiones entre etapas | 72 / 40 |
| Setup / hold mínimo entre etapas, 80 comprobaciones | +3,477 / +0,066 ns |
| WNS / WHS / WPWS globales del contexto | +1,364 / +0,004 / +1,468 ns |
| Endpoints de reset asociados a su dominio | 140 |
| Controles de evidencia alterada rechazados | 21/21 |

El hold global de **4 ps** corresponde al puntero de escritura del canal 0
de retorno hacia una dirección de RAM. Es un margen estrecho con relojes
estimados; debe revisarse tras implementar los relojes reales.

La auditoría prueba que cada bit de RAM alimenta exclusivamente su registro
de captura. Las etapas intermedias de sincronización no tienen consumidores
adicionales ni lógica insertada. Los 140 resets provienen del puerto frío
correspondiente; esto comprueba conectividad interna, no la liberación del
reset en el diseño padre.

## Límites que siguen abiertos

Los tres relojes tienen ubicaciones **supuestas** de buffers externos:
PCIe `BUFGCE_X0Y60`, lector `BUFGCE_X0Y61` y gestión `BUFGCE_X0Y62`.
`HD.CLK_SRC` modela ese contexto; no instancia los buffers, conforme a
[UG905](https://docs.amd.com/r/2022.1-English/ug905-vivado-hierarchical-design/Context-Constraints).
La auditoría identifica exactamente tres gaps: `p_clk`, `r_clk` y `mgmt_clk`,
sin drivers primitivos internos. Por ello algunos informes indican
`Partially Routed`, aunque las 2.836 redes ruteables internas estén completas.

Persisten 8 avisos CDC-6 y 1.458 CDC-15, documentados sin waivers. Las rutas y
su topología se revisaron explícitamente; estos avisos no desaparecen por
cumplir timing. Hay 1.602 bits de entrada y 1.602 de salida sin presupuesto
externo, 1.000 avisos TIMING-18 reportados y 7 XDCB-5. El único aviso DRC,
RTSTAT-10, comprende 1.469 redes sin cargas internas y con puertos de salida,
clasificadas individualmente.

El siguiente paso es integrar los relojes y el reset frío canónicos, conectar
el lector y el endpoint seleccionados y repetir implementación con sus rutas
reales. Después corresponde tráfico conjunto con HBM/PCIe nativos. La ruta
física del ACK, la autorización de regiones y el cierre de placa continúan
pendientes.

## Reproducción y respaldo

Los runners limitan cada trabajo a dos CPU, prioridad reducida y 900 s para
ruteo o 600 s para auditoría. Usan exclusivamente el directorio privado
`/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-cdc-physical/`.

```bash
python -B run_hbm_cdc_physical.py NUEVA_TANDA \
  --donor SYNTH_DCP --settings VIVADO_SETTINGS
python -B review_hbm_cdc_physical.py TANDA_RESTAURADA
python -B source/review_hbm_cdc_physical_bundle.py PAQUETE_RESTAURADO
```

Las fuentes Tcl acompañan al runner. El donor de síntesis está fijado por
SHA-256; la auditoría requiere también el hash del checkpoint ruteado.

`D:/AI/artifacts/ntrpu-axi-bridge-hbm-cdc-physical/20261011-hbm-cdc-physical.zip`
conserva fuentes, checkpoints, informes, revisores y el paquete funcional
anterior. `SELECTION.json`, `BUNDLE_BACKUP.json` y `BUNDLE_REVIEW.json` registran
selección, integridad y revisión de la restauración.

Se conserva una auditoría no seleccionada que falló al interpretar una
propiedad ASYNC_REG vacía en una LUT. Se corrigió el auditor y se repitió solo
la lectura del checkpoint; no se modificó ni repitió el ruteo.
`PROCESS_COMPLETION.json` registra el fin de los tres trabajos privados.
