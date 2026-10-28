# Optimización física del transporte compacto

## Resultado

Dos tandas del 6 de octubre de 2026 UTC mejoran el setup del
[ruteo original de `compact`](PEER_RX_OVERLAP_PHYSICAL.md) de −0,323 a
**−0,244 ns**, sin cambiar RTL, latencia ni recursos. La comparación
funcional de la netlist seleccionada pasa. **Todavía no acredita 200 MHz**;
su estado es `ROUTED_INTERNAL_NOT_QUALIFIED`. La referencia anterior con
intervalo N+7 conserva su cierre independiente; `compact` mantiene N+4.

Actualización posterior: la [distribución por conectividad de bancos](PEER_RX_OVERLAP_BANKPLAN.md)
cierra el mismo bloque `compact` a 200 MHz, con setup +0,054 ns y hold
+0,006 ns, y pasa los 16 escenarios de netlist. Sustituye al checkpoint
de −0,244 ns como referencia experimental; los resultados siguientes
describen únicamente las dos pasadas históricas de optimización.

| Resultado | Ruteo original | Pasada 1 | Pasada 2, seleccionada |
| --- | ---: | ---: | ---: |
| Setup | −0,323 ns | −0,250 ns | −0,244 ns |
| TNS | −180,530 ns | −81,202 ns | −80,429 ns |
| Endpoints con fallo de setup | 1.685 | 1.026 | 1.024 |
| Hold | +0,002 ns | +0,002 ns | +0,002 ns |
| LUT / registros / BRAM | 6.785 / 16.024 / 15 | iguales | iguales |

Las 17.840 redes ruteables están completamente conectadas, sin errores.
Pasan estructura, cobertura interna, DRC y ancho de pulso (+1,958 ns).
El aviso `RTSTAT-10` corresponde a 1.551 redes con conexiones externas y
sin cargas internas. No hay errores ni avisos críticos de consola.

## Intervención y controles

Se parte del checkpoint ruteado verificado por SHA-256, sobre
`xcvu5p-flva2104-2-i`. Se conservan el reloj de 5 ns, incertidumbres de
setup/hold de 0,100/0,050 ns, un BUFGCE en `BUFGCE_X0Y60` y las restricciones
originales, sin excepciones de timing.

La primera pasada ejecuta:

```tcl
phys_opt_design -critical_cell_opt -placement_opt -routing_opt -critical_pin_opt -hold_fix
```

La segunda añade `-restruct_opt`. Son opciones compatibles con esta etapa
según la [tabla de optimizaciones de AMD](https://docs.amd.com/r/en-US/ug904-vivado-implementation/Available-Physical-Optimizations).
Las opciones se seleccionan explícitamente; no se habilitan retiming,
optimización del reloj ni replicación forzada. No se vuelve a ejecutar
`route_design`; se revisa el ruteo que deja el optimizador.

El inventario anterior/posterior conserva 25.992 primitivas: ninguna
añadida, retirada o con cambio de tipo. En la primera pasada cambian las
marcas de modificación de dos LUT; en la segunda se mueve una LUT y cambia
su marca. Habilitar reestructuración no demuestra que haya ocurrido una
reestructuración o replicación concreta. Los TSV y las propiedades
`PHYS_OPT_MODIFIED` quedan conservados para inspección.

## Ruta pendiente

Tras optimizar, la peor ruta parte de `active_generation_reg[1][12]` y
termina en `bank_reg[8][2]/CE`. Atraviesa validación de devolución,
fallo, oferta, reserva y habilitación de metadatos. Tiene 11 niveles y
4,978 ns de retardo de datos: 1,172 ns de lógica y **3,806 ns de cableado
(76,46 %)**. El skew aporta −0,071 ns. Destacan 1,207 ns en
`return_generation[76]` y 0,802 ns en una habilitación de 141 destinos.

La segunda pasada sólo recupera 6 ps; no se programan repeticiones
idénticas. El siguiente ensayo debería evaluar otra distribución física
de estos controles, con nueva implementación y comparación funcional.
Cualquier modificación de la validación debe preservar rechazo de
generaciones obsoletas, prioridad del fallo y reserva atómica con oferta
vigente. Esta evidencia no demuestra que esa intervención cierre timing.

## Comparación funcional

La netlist seleccionada pasa los 13 escenarios originales y tres ventanas
de solapamiento: **5.247 ciclos, 28 salidas, 129.088 bytes**, 92 trabajos
completos y cinco cuarentenas esperadas. No hay divergencias respecto del
RTL en esos escenarios; el revisor rechaza 13 alteraciones de referencia y
siete supresiones de eventos ruteados. Se comprueban también las salidas
ofrecidas durante contrapresión, sin compensación de latencia.

La simulación es funcional, sin SDF ni `force`, con 150 ns de arranque
UNISIM. La definición exportada de `glbl` coincide con la estándar instalada.
Se conservan los avisos conocidos de declaración posterior de `tn_checks`,
redefinición de `glbl` y entorno del compilador XSIM. Esta comprobación
no sustituye STA ni constituye equivalencia formal.

## Reproducción y evidencia

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_overlap_physopt.py /pass1/output \
  --stage route --prior /compact_route/output
python3 -B tools/check_ntrpu_overlap_physopt.py /pass1/output --review /pass1/REVIEW.json
python3 -B tools/run_ntrpu_overlap_physopt.py /pass2/output \
  --stage route --prior /pass1/output --restructure
python3 -B tools/check_ntrpu_overlap_physopt.py /pass2/output --review /pass2/REVIEW.json
# Continuación exclusivamente diagnóstica si STA sigue sin pasar.
python3 -B tools/run_ntrpu_overlap_physopt.py /netlist/output \
  --stage netlist --prior /pass2/output
python3 -B tools/check_ntrpu_overlap_physopt.py /netlist/output --review /netlist/REVIEW.json
```

Usar salidas nuevas y respaldar cada tanda tras revisarla. El ejecutor
limita cada trabajo a dos CPU, prioridad baja y 900 s, y comprueba hashes
de fuentes, entradas y RTL generado. Un retorno correcto del revisor
indica que produjo el informe: para habilitar timing se exige además
`ROUTED_INTERNAL_TIMING_PASS`, ausente en ambas pasadas.

| Tanda | Duración |
| --- | ---: |
| `20261006T025457471818Z-route` | 177,766 s |
| `20261006T030021400903Z-route` | 193,889 s |
| `20261006T030416917647Z-netlist` | 153,552 s |

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-overlap-physopt\20261006T025306805735Z`.
Checkpoint seleccionado SHA-256:
`5ed2984373f96228c2b7093c992c2de4d91c28e76b76dfca93b7ba598c7c1140`.
Netlist SHA-256:
`aeaac906ebea53f92ca9e684f7154cff1684ab2682df003c609aa85efe95906f`.

El almacén incremental `D:\AI\artifacts\ntrpu-content-store` contiene los
manifiestos `overlap-physopt-route-20261006`,
`overlap-physopt-restruct-route-20261006` y
`overlap-physopt-netlist-20261006`, verificados por contenido.
`overlap-physopt-handoff-20261006` reúne documentación, revisores e índices.
Se reutilizan objetos por hash y no se retienen ZIP de transferencia.

## Ensayo posterior de distribución

Una [guía flexible de colocación](PEER_RX_OVERLAP_FLOORPLAN.md) para los
registros de generación y el asignador se evaluó desde la síntesis
original. La colocación termina, pero el ruteo alcanza su límite de 900 s
con congestión residual. No hay STA final ni comparación de netlist de
esa variante; en ese ensayo se conservó el checkpoint de −0,244 ns de
este documento. La distribución posterior por bancos, enlazada al inicio,
sí completa timing interno y comparación funcional.

Este bloque fuera de contexto todavía no caracteriza retardos externos,
PCIe/HBM, CDC, reinicios independientes ni comportamiento de placa.
