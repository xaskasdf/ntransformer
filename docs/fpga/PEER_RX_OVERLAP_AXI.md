# Transporte compacto con fronteras AXI registradas

## Alcance de esta integración

Ensayo aislado del 6 de octubre de 2026 UTC, posterior al
[cierre interno por bancos](PEER_RX_OVERLAP_BANKPLAN.md). Añade una IP
AMD `axi_register_slice:2.1`, revisión 37, a cada salida de escritura de
`compact`: AXI4 de 512 bits, direcciones de 64 bits, IDs de cuatro bits
y modos `REG_AW=REG_W=REG_B=1`. Se mantiene un reloj de 5 ns y reset frío
coordinado. El montaje usa la IP instalada, sin sustituirla por un modelo
de registros escrito para el ensayo. [Guía de AMD, PG373](https://docs.amd.com/r/en-US/pg373-axi-register-slice/Introduction).

La copia generada intercala las IP entre el núcleo y los puertos públicos;
no modifica el RTL fuente ni los cuatro módulos generados de `compact`.
Conserva el pool de ocho bancos y las ventanas AXI autorizadas de entrada.
AW, W y B avanzan independientemente, también durante STOP para trabajos
comprometidos. Los sidebands AWLOCK/AWPROT/AWQOS/AWREGION se fijan a cero.
No hay lectura, adaptación de ancho, CDC ni autorización de direcciones
adicional. Este paso no integra todavía los controladores PCIe/HBM.

## Pruebas y criterios

Los 13 escenarios originales y los tres adicionales comprueban bytes,
identidad, orden, W anterior a AW, contrapresión, STOP, errores, barreras y
drenaje. Los 15 perfiles miden bloques de 64 B, 512 B y 4 KiB, incluido
un destino bloqueado mientras el otro progresa. Las trazas públicas se
reconstruyen independientemente; las aserciones del banco no son el único
criterio. La simulación respeta los 150 ns de arranque global de UNISIM.

La tanda RTL `20261006T050227950777Z-rtl` pasa en 92,423 s. Compara
129.088 bytes en los 16 escenarios: 92 trabajos completos y cinco
cuarentenas esperadas. Rechaza 20 corrupciones de trazas entre protocolo,
casos adicionales y rendimiento.

La síntesis `20261006T050437781485Z-synth` termina en 165,466 s:
7.702 LUT, 18.529 registros y 15 bloques BRAM, sin latches ni cajas negras.
La estimación de setup es −2,216 ns antes de colocar; no acredita timing.
La lógica generada y la configuración de IP coinciden con la tanda RTL.

En los 15 perfiles se comparan otros 2.990.080 bytes y 1.920 trabajos.
Sin pausas, se conserva N+4 entre fetches y dos bancos ocupados como máximo:

| Bloque | Ciclos por lote de 128, antes | Con las dos IP | Intervalo fetch |
| --- | ---: | ---: | ---: |
| 64 B | 657 | 659 | 5 ciclos |
| 512 B | 1.560 | 1.562 | 12 ciclos |
| 4 KiB | 8.784 | 8.786 | 68 ciclos |

Son conteos de simulación, desde la primera publicación hasta la última
finalización incluidas. No representan rendimiento de una tarjeta física.

Cada etapa requiere revisión y respaldo antes de habilitar la siguiente.
La síntesis debe conservar el montaje simulado. El primer intento de
colocación `20261006T050740280901Z-place` rechaza el plano heredado:
la nueva síntesis ya no contiene la primitiva
`allocator/bank_state[2].used[2]_i_15` bajo la jerarquía original.
No se omiten miembros para forzar su aceptación. El intento queda
respaldado como fallo de correspondencia, no como fallo de timing.

La repetición parte del mismo checkpoint sintetizado y evalúa colocación
libre mediante un ejecutor separado. Verifica que las restricciones de
entrada/salida no contengan Pblocks. Conserva la propuesta anterior como
evidencia, sin aplicarla a celdas distintas. Este cambio de ejecutor no
regenera ni cambia la lógica ya simulada y sintetizada.
La colocación `20261006T051040863793Z-place` termina en 391,398 s con
7.698 LUT, 18.529 registros y 15 bloques BRAM. Su setup preliminar es
−0,707 ns y su hold −0,158 ns; requiere ruteo antes de evaluar cierre.
El ruteo mantiene los presupuestos de 5 ns, incertidumbres
de 0,100/0,050 ns y BUFGCE_X0Y60, sin excepciones añadidas. La comparación
final usa RTL integrado contra su propia netlist ruteada, ciclo a ciclo,
sin compensar latencias ni modificar el reset global.

## Resultado físico y siguiente ensayo

El ruteo `20261006T051741342229Z-route` alcanza el límite de 900,011 s
con 218 superposiciones de nodos pendientes en el último registro.
Se conserva `TIMEOUT_DIAGNOSTIC`, sin checkpoint ruteado final ni STA
final. La comparación de netlist ruteada no se ejecuta. **El conjunto
con IP AXI está validado funcionalmente en RTL, pero no a 200 MHz.**
La referencia anterior sin estas IP conserva su acreditación independiente.

El log señala congestión global/corta de nivel 5, con regiones próximas
a `INT_X40Y253:INT_X55Y276`. La peor ruta antes de rutear iba de
`active_generation_reg[1][7]` a `status_reg[30][0]/CE`: 25 niveles,
5,391 ns de datos, de los cuales 4,043 ns eran cableado estimado.
Estos datos sirven para orientar el siguiente ensayo; no son STA final.
Añadir las IP también incorpora rutas internas hacia AW/W y desde B que
antes cruzaban puertos externos sin retardos de interfaz definidos.

La lectura paralela `20261006T053023018152Z-connectivity` termina en
43,605 s sobre el checkpoint de colocación, sin modificarlo. El nuevo mapa
selecciona **928 celdas**: los mismos 528 registros de estado y 400 celdas
comparadoras/derivadas, 50 por banco. El recorrido independiente verifica
sus orígenes; los 128 registros entrantes compartidos quedan libres.
Comparada con la selección anterior, conserva 889 nombres, retira 65 y
añade 39. No debe interpretarse esa diferencia como una mera renumeración.

`candidate_plan/BANK_PLAN.json` y `groups.tsv`, dentro de esa tanda,
contienen ocho regiones candidatas disjuntas. **Son una propuesta sin
validación física**: faltan revisar sitios disponibles y normalización de
bordes, y comprobar los miembros en el punto exacto donde se aplicarán.
El siguiente ensayo debe partir de esa conectividad actual, evaluar el
coste de las IP y las rutas de generación/estado, y repetir ruteo y
comparación funcional completa. No se prolongó ni relanzó este ruteo.

## Reproducción y evidencia

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_overlap_axi.py /rtl/output --stage rtl \
  --reference /compact_bankplan_route/output
python3 -B tools/check_ntrpu_overlap_axi.py /rtl/output --review /rtl/REVIEW.json
# Tras revisar y respaldar:
python3 -B tools/run_ntrpu_overlap_axi.py /synth/output --stage synth \
  --reference /compact_bankplan_route/output --prior /rtl/output
python3 -B tools/run_ntrpu_overlap_axi_physical.py /place/output --stage place \
  --reference /compact_bankplan_route/output --prior /synth/output
# Continuar con el ejecutor físico para route -> netlist, revisando cada etapa.
```

Usar salidas nuevas; dos CPU, prioridad baja y hasta 900 s por etapa.
El revisor puede generar un diagnóstico sin acreditar timing. Para ello
se exige `ROUTED_INTERNAL_TIMING_PASS` y comparación funcional aprobada.
La sonda inicial generó la IP pero falló al copiar su envoltura desde
`sources_1`; el archivo estaba en el árbol `.gen` y se recuperó por lectura.
El diagnóstico original se conserva separado del ensayo aceptado.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-overlap-axi\20261006T045718113865Z`.
Respaldo incremental por contenido en
`D:\AI\artifacts\ntrpu-content-store`, sin conservar ZIP de transferencia.
Manifiestos: `overlap-axi-probe-20261006`, `overlap-axi-rtl-20261006`,
`overlap-axi-synth-20261006`, `overlap-axi-place-20261006` (plano rechazado),
`overlap-axi-free-place-20261006`, `overlap-axi-free-route-20261006`,
`overlap-axi-connectivity-20261006` y `overlap-axi-handoff-20261006`.
La referencia sin estas IP conserva su cierre de +0,054 ns: ese margen
no se transfiere automáticamente al montaje integrado.

## Continuación del plano físico

La revisión posterior a `opt_design`, el inventario real de sitios y el
ensayo con el mapa actualizado se documentan en
[Plano de bancos para el transporte con IP AXI](PEER_RX_OVERLAP_AXI_BANKPLAN.md).

La variante posterior con [devoluciones registradas](PEER_RX_RETURN_CUT.md)
conserva ambas IP AXI y obtiene cierre interno a 200 MHz (+0,251 ns setup,
+0,006 ns hold), con comparación RTL/netlist de 16 escenarios aprobada.
El resultado corresponde a esa variante y no modifica los resultados
históricos de este ensayo ni acredita las interfaces externas PCIe/HBM.
