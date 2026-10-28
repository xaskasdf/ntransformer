# Restricciones CDC de la composición nativa VU47P

## Alcance y entrada

Esta tanda continúa [HBM_NATIVE_FIX.md](HBM_NATIVE_FIX.md) sobre una copia del
checkpoint nativo corregido. Conserva el RTL, los IP, sus restricciones y los
relojes reales: PCIe a 250 MHz, lector a 200 MHz, gestión a 100 MHz y HBM a
450 MHz. No ejecuta síntesis, placement ni routing. El nodo DDR de 512 GiB
mantiene su diseño separado.

Donante: `fix-synth-20261011T214558Z/output/synth.dcp`, SHA-256
`4613dbb14df1eff881c3e5294825d6398a9af37e3f100f37ab52cd017ab2a89e`.

## Límites y comprobaciones

| Grupo | Cobertura | Restricción |
|---|---:|---|
| Punteros Gray de cuatro FIFO | 8 buses, 24 bits | Máximo 4 ns y skew 2 ns |
| Captura del contenido de las FIFO | 4 grupos, 1.458 bits | Máximo 4 ns |
| Estados y controles registrados | 21 señales | Máximo 4 ns; 2,222 ns hacia HBM |
| Controles asíncronos de reset | 18 arcos exactos | Excepción exclusiva hacia CLR/PRE |

Los límites de datos usan `set_max_delay -datapath_only`, acompañados de
`set_bus_skew` para los punteros. Se sigue la distinción entre límites de
retardo y skew de [AMD UG903](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Constraining-Asynchronous-Signals).
No se añaden grupos de relojes asíncronos que anulen esos límites.

La auditoría exige los 1.503 destinos, sus fuentes y relojes, una sola carga
por salida RAM capturada, 147 registros `ASYNC_REG` y 90 conexiones entre
etapas con sus comprobaciones de setup y hold conservadas. Reabrir el nuevo
DCP debe reproducir la misma evidencia sin aplicar restricciones nuevamente.
La comparación XDC exige conservar cada comando original.

Cada excepción de reset comprueba su cadena lógica exacta. El reset AXI
nativo atraviesa inversor, buffer LUT1 e inversor; dos inversiones conservan
la polaridad activa alta de `user_reset`. Se exportan nombres, funciones LUT,
fuente Q y número de inversiones. La primera tanda falló al esperar un solo
inversor y se conserva; el fallo corresponde al verificador, no demuestra
un defecto del reset.

## Ejecución y revisión

En un directorio preparado nuevo, copiar `constrain_hbm_native_constraints.tcl`,
`audit_hbm_native_constraints.tcl` y `run_hbm_native_join.py`. Copiar el entrypoint
`apply_hbm_native_constraints.tcl` como `inspect_hbm_native_join.tcl`.

```bash
python3 -B PREPARED/run_hbm_native_join.py NEW_OUTPUT \
  --prepared PREPARED --settings /path/to/Vivado/settings64.sh \
  --stage inspect --checkpoint DONOR.dcp
```

Repetir en otro directorio preparado usando `reopen_hbm_native_constraints.tcl`
y el nuevo `output/constrained.dcp` como donante. Revisar ambos respaldos:

```bash
python -B review_hbm_native_constraints.py APPLY_RESTORED REOPEN_RESTORED --self-test
```

Los jobs usan únicamente el namespace privado
`/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join`, dos CPU, prioridad
reducida y 1.200 segundos de límite por proceso. Los respaldos por manifiesto
están en `D:/AI/artifacts/ntrpu-axi-bridge-hbm-native-constraints/`.

## Resultado verificado

`constraints-apply-20261011T222937Z` terminó en 155,9 segundos y
`constraints-reopen-20261011T223258Z` en 148,3 segundos, conservando las fuentes
y sus checkpoints de entrada. El DCP nuevo tiene SHA-256
`629bae0ba8bddd48eec2fbdf78224a9f8ac0c149a32299d15bbb82c77f6bf313`.
El revisor devuelve `NATIVE_SCOPED_CONSTRAINTS_REOPENED_AND_REVIEWED` y rechaza
18 controles negativos, incluidos cambios de polaridad, alcance, presupuesto
de retardo y pérdida de restricciones durante la reapertura.

Las ocho tablas de evidencia y los recuentos CDC se reproducen exactamente.
Vivado serializa inicialmente ocho consultas de captura mediante variables y
las escribe en línea al reabrir. El revisor comprueba sus objetos exactos y
expande únicamente esas ocho variables al comparar; no evalúa Tcl ni ignora
diferencias en las restricciones originales.

El margen mínimo estimado de los 1.503 caminos seleccionados es +1,952 ns.
El timing global sigue incumplido: WNS −0,194 ns, WHS −0,842 ns y ancho de
pulso −0,092 ns, todos después de síntesis. Permanecen nueve advertencias
`CDC-6` y 1.746 `CDC-15`; se informan además 190 `CDC-3` y 36 `CDC-9`, de
severidad informativa, y ningún `CDC-10`. El informe conserva las excepciones
ignoradas del IP para su revisión; esta tanda no las elimina ni las califica.

`SELECTION.json` identifica las dos tandas aceptadas; `JOBS.json` conserva
también el intento fallido. `FINAL_PROCESS_AUDIT.json` registra el cierre de
los grupos de procesos propios y la integridad del donante. `FINAL_BACKUP.json`
identifica el ZIP consolidado y su SHA-256: el paquete incluye el donante,
fuentes, informes e intentos fallidos, y su restauración nueva debe superar
otra vez el mismo revisor con los 18 controles negativos.

## Límites pendientes

Esta prueba acredita la aplicación y persistencia de restricciones concretas,
no su cumplimiento después del routing. `MMCM/LOCKED` hacia la primera etapa
del sincronizador de gestión sigue explícitamente sin límite en esta tanda.
Los hallazgos CDC globales, presupuestos externos, timing de implementación,
pinout, alimentación y térmica requieren revisión independiente. Tampoco
acredita tráfico PCIe/HBM nativo completo, carga de pesos ni inferencia.
