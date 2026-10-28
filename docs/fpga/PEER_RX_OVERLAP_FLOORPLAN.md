# Distribución física de los controles del transporte compacto

## Hipótesis y alcance

Ensayo aislado del 6 de octubre de 2026 UTC, posterior a la
[optimización post-ruteo](PEER_RX_OVERLAP_PHYSOPT.md). La mejor variante
anterior tiene −0,244 ns de setup a 200 MHz; el cableado aporta el 76,46 %
del retardo de su peor ruta. Se evalúa acercar los registros de generación
al asignador, preservando el RTL `compact`, su intervalo N+4 y la reserva
atómica con oferta vigente.

Se reutiliza el checkpoint de síntesis validado
`20261006T022220067846Z-synth`, sin resintetizar ni incorporar cambios RTL.
La única intervención planificada es una guía de colocación antes del
`place_design` habitual. No parte del checkpoint ya ruteado: vuelve a
colocar y rutear para permitir otra distribución.

## Región propuesta

El Pblock `ntrpu_compact_control` cubre
`SLICE_X54Y260:SLICE_X71Y289`, dentro del entorno observado en la
implementación anterior. Contiene inicialmente:

- Los 128 FDRE de `active_generation_reg`, sin sus multiplexores de entrada.
- Las 1.961 primitivas de `receiver/core/allocator`, excluyendo GND y VCC.

Las 2.089 celdas suman 1.207 LUT, 658 registros y 224 CARRY8. La selección
no incluye BRAM ni buffers de reloj. `IS_SOFT=TRUE` permite que el placer
mueva celdas fuera de la región después de la colocación inicial.
`EXCLUDE_PLACEMENT=FALSE` y `CONTAIN_ROUTING=FALSE` dejan compartir recursos
y rutear fuera. El comportamiento se basa en las referencias de AMD sobre
[IS_SOFT](https://docs.amd.com/r/en-US/2026.1/ug912-vivado-properties/IS_SOFT)
y [Pblocks](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Using-Pblock-Based-Floorplanning).
La geometría es una hipótesis experimental, no una ubicación acreditada.

## Criterios de aceptación

Se conserva `xcvu5p-flva2104-2-i`, reloj de 5 ns, incertidumbres de
setup/hold de 0,100/0,050 ns y BUFGCE en `BUFGCE_X0Y60`. El revisor exige
los mismos presupuestos de timing, ausencia de excepciones, cobertura
interna completa, estructura válida y ruteo completo sin errores.

Los informes registran la selección inicial, miembros finales, recursos
y posiciones dentro/fuera de la región. Salir de un Pblock flexible no
es un fallo; debe comprobarse el resultado de timing completo. Para
acreditar 200 MHz deben pasar setup, hold y ancho de pulso, además de DRC.
Después se comparan 28 salidas con el RTL en 16 escenarios, sin añadir
latencia, `force` ni SDF. La simulación funcional no sustituye STA.

## Resultado de la tanda

**La distribución propuesta no se selecciona.** La colocación termina,
pero el ruteo alcanza el límite de 900 s y se detiene mediante SIGTERM.
El estado revisado es `ROUTING_TIMEOUT_DIAGNOSTIC`: no hay checkpoint
ruteado terminado, STA final ni netlist ruteada para comparar. La mejor
referencia `compact` conserva −0,244 ns; tampoco acredita 200 MHz.

| Etapa | Tanda | Duración | Resultado |
| --- | --- | ---: | --- |
| Colocación | `20261006T033355648529Z-place` | 265,594 s | Setup +0,567 ns; hold −0,190 ns, ambos previos al ruteo |
| Ruteo | `20261006T033840611659Z-route` | 900,019 s | Timeout; sin timing final |

La colocación usa 6.794 LUT, 16.024 registros y 15 BRAM: nueve LUT más que
la referencia anterior, con los mismos registros y BRAM. Conserva las
2.089 celdas seleccionadas y sus tipos; todas cambian de ubicación frente
al checkpoint anterior. Quedan 1.826 dentro de la región y 263 fuera,
incluidos 95 de los 128 registros de generación. La guía flexible permite
esas salidas. La cobertura interna y estructura pasan; `RTSTAT-13` refleja
conexiones aún pendientes en esta etapa.

El ruteador alcanza cero superposiciones en su primera iteración global,
pero informa **setup intermedio de −0,680 ns**, TNS de −1.142,457 ns y hold
de +0,005 ns. Inicia otra iteración para optimizar las rutas y llega al
timeout con 20 superposiciones en el último registro. Ninguna de esas
cifras constituye una evaluación final de esta variante. La congestión
aparece en `CLEM_X45Y273`, `CLEL_R_X45Y273`, `CLEM_X44Y274` y
`CLEL_R_X44Y274`, con dos redes señaladas:

- `core/core/receiver/core/return_generation[100]`.
- `core/core/receiver/core/allocator/bank_state[6].generation_reg[6]_12[36]`.

Agrupar todo el asignador de esta manera no aporta evidencia de mejora.
El siguiente ensayo se planteó más selectivo sobre los buses de generación
y los comparadores de cada banco, preservando la reserva atómica y la
validación completa de generaciones. Antes de repetir una implementación
conviene definir los grupos por conectividad y revisar su ocupación,
en lugar de ampliar sin más el tiempo de este mismo ruteo.

El [ensayo posterior por conectividad de bancos](PEER_RX_OVERLAP_BANKPLAN.md)
completa ruteo y los 16 escenarios funcionales, con setup +0,054 ns y
hold +0,006 ns a 200 MHz. Es la nueva referencia experimental `compact`
con cierre interno; este Pblock único sigue descartado.

## Reproducción y respaldo

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_overlap_floorplan.py /place/output \
  --stage place --prior /compact_synth/output
python3 -B tools/check_ntrpu_overlap_floorplan.py /place/output --review /place/REVIEW.json
python3 -B tools/run_ntrpu_overlap_floorplan.py /route/output \
  --stage route --prior /place/output
python3 -B tools/check_ntrpu_overlap_floorplan.py /route/output --review /route/REVIEW.json
# Sólo tras obtener y revisar una netlist ruteada completa:
python3 -B tools/run_ntrpu_overlap_floorplan.py /netlist/output \
  --stage netlist --prior /route/output
python3 -B tools/check_ntrpu_overlap_floorplan.py /netlist/output --review /netlist/REVIEW.json
```

Revisar y respaldar cada etapa antes de continuar. Cada salida debe ser
nueva. El ejecutor usa dos CPU, prioridad baja y un límite de 900 s por
etapa; verifica hashes de fuentes y checkpoints. El éxito del revisor
significa que generó un informe, no necesariamente cierre de timing:
se exige `ROUTED_INTERNAL_TIMING_PASS` para esa acreditación.

Para este timeout se conservó el informe inicial de fallo junto con la
tanda. `TIMEOUT_REVIEW.json`, separado y posterior, verifica sus hashes,
entradas inmutables, terminación por límite y ausencia de salida ruteada;
distingue los indicadores intermedios del resultado final inexistente.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-overlap-floorplan\20261006T033244656872Z`.
El respaldo por tanda usa el almacén incremental
`D:\AI\artifacts\ntrpu-content-store`, sin conservar ZIP de transferencia.
Manifiestos: `overlap-floorplan-place-20261006`,
`overlap-floorplan-route-20261006` y `overlap-floorplan-handoff-20261006`.
Incluyen el checkpoint de colocación, el diagnóstico de congestión,
fuentes, revisores, índices y documentación. La comparación funcional
prevista se omite porque no se produjo una netlist ruteada completa.
El alcance sigue siendo el bloque fuera de contexto: no acredita una
tarjeta PCIe/HBM, retardos externos, CDC ni reinicios independientes.
