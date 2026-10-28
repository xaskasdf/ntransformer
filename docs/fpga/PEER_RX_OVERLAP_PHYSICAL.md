# Evaluación física del transporte con solapamiento

## Alcance y trazabilidad

Evaluación del 6 de octubre de 2026 UTC de la variante `compact` descrita en
[PEER_RX_TRANSPORT_OVERLAP.md](PEER_RX_TRANSPORT_OVERLAP.md). Conserva el
intervalo de fetch N+4, un solo llenado activo y ocho bancos de 4 KiB.
La referencia anterior permanece independiente.

El generador exige igualdad SHA-256 de los cuatro módulos RTL generados
respecto de la tanda funcional `20261006T011649068046Z-compact`. El ejecutor
comprueba también las fuentes directas y conserva ese vínculo en cada etapa.
Placement y ruteo reciben checkpoints verificados; la comparación recibe
la netlist funcional exportada por el ruteador.

## Condiciones de aceptación

Se reutiliza el flujo físico del estudio anterior: dispositivo
`xcvu5p-flva2104-2-i`, reloj de 5 ns, incertidumbre de setup/hold
0,100/0,050 ns y BUFGCE en `BUFGCE_X0Y60`. La envoltura física no añade
registros ni latencia a las interfaces. El nuevo Tcl sustituye el generador
y añade una comparación; no cambia restricciones, estrategias ni informes.

Para acreditar timing interno se requieren setup, hold y ancho de pulso
sin violaciones, cobertura completa de endpoints internos, ausencia de
latches y bloques sin resolver, todas las redes internas ruteadas y DRC
sin errores. Se conservan los avisos de conexiones externas del montaje
fuera de contexto, con revisión de sus cargas.

La comparación funcional contrasta los 28 puertos de salida del RTL y de
la netlist en los 13 escenarios originales y los tres casos nuevos de
solapamiento. Comprueba señales VALID y datos ofrecidos durante contrapresión.
Respeta 150 ns de arranque global UNISIM; no usa `force`, SDF ni compensación
de latencia. STA y equivalencia funcional son comprobaciones separadas.

## Estado de las etapas

**El ruteo original de `compact` no acredita 200 MHz.** Todas las redes
ruteables quedan conectadas y el hold pasa, pero el setup viola el objetivo
de 5 ns. Se conserva como diagnóstico; la referencia anterior sigue siendo
la implementación con timing interno acreditado.

| Etapa | Tanda | Duración | Setup / hold |
| --- | --- | ---: | ---: |
| Síntesis | `20261006T022220067846Z-synth` | 289,571 s | +0,882 / −0,126 ns |
| Placement | `20261006T022758316739Z-place` | 323,873 s | +0,509 / −0,161 ns |
| Ruteo | `20261006T023332707383Z-route` | 634,718 s | **−0,323 / +0,002 ns** |

| Resultado ruteado | Referencia | Compact |
| --- | ---: | ---: |
| LUT | 6.755 | 6.785 |
| Registros | 16.029 | 16.024 |
| Bloques BRAM | 15 | 15 |
| Setup | +0,144 ns | −0,323 ns |
| Hold | +0,006 ns | +0,002 ns |
| Redes completamente ruteadas | 17.806 | 17.840 |

`compact` añade 30 LUT y elimina cinco registros; conserva 14 RAMB36 y
dos RAMB18. No hay errores de ruteo, latches, blackboxes ni endpoints
internos sin restricciones. El ancho de pulso tiene +1,958 ns de margen.
Hay 1.685 endpoints con violación de setup y TNS de −180,530 ns.

DRC termina sin errores ni avisos críticos. `RTSTAT-10` abarca 1.551 redes
que tienen puertos externos y ninguna carga interna. Se conservan los
avisos de alcance fuera de contexto, puertos sin `HD.PARTPIN_LOCS`, reloj
sin `HD.CLK_SRC` y ubicación fija del BUFGCE. El aviso `RTSTAT-13` de
placement describía el ruteo todavía incompleto y desaparece tras rutear.

La fase COMMAND eliminada deja `cmd_ready` sin carga en el dispatcher;
Vivado retira un registro de estado no utilizado. La confirmación continúa
comprobándose en el adaptador que habilita el fetch. El informe de recursos
de síntesis marca `CLB LUTs*`; el lector acepta ese asterisco sin cambiar
los resultados ni repetir la síntesis.

## Diagnóstico del setup

La peor ruta sale de `active_generation_reg[0][20]`, atraviesa la validación
de devolución de bancos, la lógica de fallo y oferta, `reserve_take` y
`accept_cmd`, y llega al CE de registros de metadatos, por ejemplo
`length_reg[29][2]`. Tiene 11 niveles lógicos y 4,851 ns de retardo de datos:
1,093 ns de lógica y **3,758 ns de cableado (77,5 %)**. El skew de reloj
aporta −0,276 ns. Dos tramos destacan: `return_generation[20]` con 0,996 ns
y una habilitación de 141 destinos con 1,077 ns.

El ruteador informa congestión local cerca de BRAM y en los tiles
`CLEM_X44Y272` / `CLEL_R_X44Y272`. Finalmente resuelve todos los conflictos,
pero no el setup. Esta evidencia justifica un próximo ensayo de distribución
física o replicación de controles, preservando la comprobación de ofertas
vigentes y la reserva atómica. No demuestra que una estrategia concreta
vaya a cerrar timing; tampoco acredita una frecuencia alternativa.

## Comparación funcional de netlist

La tanda `20261006T024509255037Z-netlist` termina en 153,576 s y pasa:
4.935 ciclos de los 13 escenarios originales y 312 ciclos de los tres
casos nuevos. Se comparan 28 salidas durante **5.247 ciclos**, sin
divergencias. Se comprueban 129.088 bytes, 92 trabajos completos y cinco
cuarentenas esperadas, incluidos el fallo en el último beat y stop/barrera
con metadatos preparados durante otro llenado.

El revisor rechaza 13 alteraciones de la traza de referencia y siete
supresiones de eventos de las trazas ruteadas. La definición `glbl`
exportada coincide con la estándar de la instalación. Se conservan avisos
de declaración posterior del contador, redefinición de `glbl` y entorno
del compilador XSIM; no hay errores ni avisos críticos de simulación.

Esta equivalencia es funcional, sin SDF, y **no habilita 200 MHz**. El
resultado de STA sigue siendo `ROUTED_INTERNAL_NOT_QUALIFIED`.

Checkpoint ruteado SHA-256:
`98cfd4708252d005aaf2e6e46ce8648828c041cb4cbb498ad446f16273c70b11`.
Netlist funcional SHA-256:
`56d049018a80a5dfa6ca1ae88855e5357d993beb34cfa6c64f2ca0cae36bb5e6`.

## Reproducción

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_overlap_physical.py /synth/output --stage synth \
  --rtl-evidence /compact_validado/output
python3 -B tools/check_ntrpu_overlap_physical.py /synth/output --review /synth/REVIEW.json
python3 -B tools/run_ntrpu_overlap_physical.py /place/output --stage place --prior /synth/output
python3 -B tools/run_ntrpu_overlap_physical.py /route/output --stage route --prior /place/output
python3 -B tools/run_ntrpu_overlap_physical.py /netlist/output --stage netlist --prior /route/output
# Revisar y respaldar cada etapa antes de continuar.
```

Cada salida debe ser nueva. Las ejecuciones se limitan a dos CPU, prioridad
baja y 900 s. La evidencia local está en
`C:\Users\xasko\artifacts\ntrpu-overlap-physical\20261006T022200275151Z`.
El respaldo usa objetos por hash en `D:\AI\artifacts\ntrpu-content-store`;
las transferencias comprimidas se procesan en memoria.

Manifiestos verificados: `overlap-physical-synth-20261006`,
`overlap-physical-place-20261006`, `overlap-physical-route-20261006` y
`overlap-physical-netlist-20261006`. `overlap-physical-handoff-20261006`
reúne fuentes de revisión, documentación e índices. La continuación hacia
netlist tras el fallo de STA está registrada como diagnóstico funcional,
sin excepciones de timing ni promoción de la variante.

## Optimización posterior

Dos [pasadas de optimización física](PEER_RX_OVERLAP_PHYSOPT.md) reducen
el déficit a −0,244 ns, con los mismos recursos y hold de +0,002 ns.
La netlist seleccionada pasa los 16 escenarios. Aún no acredita 200 MHz;
las cifras anteriores describen el ruteo original conservado.

## Límites

Este montaje no tiene retardos de entrada/salida de placa: 1.069 entradas
y 1.641 salidas siguen sin caracterización externa. No acredita una tarjeta
PCIe/HBM, CDC, reinicios independientes ni recuperación de enlaces. La
frecuencia y los recursos se refieren solamente a este bloque y dispositivo.
