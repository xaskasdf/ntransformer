# Observación temporal de comandos DDR por rank

Actualización: 2026-10-16 13:06 UTC. **R1 terminó y la integración del helper
quedó encolada en [una captura R2 separada](POWER_ACTIVITY_R2.md)**, después
del vocabulario. Todavía no hay un VCD nativo R2. El revisor se probó y se
aplicó al VCD histórico descrito abajo; ese resultado no califica R1/R2.

## Evidencia disponible

El [revisor temporal](../../../tools/review_ntrpu_ddr_rank_commands_r1.py)
analizó `20261002T122011Z-V3-reset-command-trace-r2`. Los 325 comandos
seleccionados de slot 0/rank 0 coinciden con el extractor archivado.
Entre 25 y 48 µs observó estas codificaciones en el dispositivo 0:

| Comando | Observaciones |
| --- | ---: |
| ACT | 91 |
| RD / WR | 86 / 53 |
| PRE / PREA | 64 / 3 |
| REF | 3 |

REF aparece en 30,128186; 38,248186 y 45,718186 µs. El mayor intervalo
entre esas observaciones es 8,120 µs; **no es una comprobación de tREFI**.
Hay 37 flancos ambiguos en ese intervalo, no contados como comandos.
Faltan los pines de comando de los otros tres ranks: observar sus relojes,
reset y CKE no permite completar la cobertura. El diagnóstico histórico
de reset caliente sigue fallando y no es la nueva prueba PRBS.

## Interpretación y límites

El decodificador sigue las tablas 23–24 de la ficha Micron
`16gb_ddr4_dram.pdf`, Rev. H, páginas 67–70
([documento del fabricante en espejo](https://device.report/m/92f06ed489b68537693753a3d634a7c26f183810e7ebb89a07b651d8b41e94da)).
Usa RESET, CKE anterior/actual, CS y ACT en el flanco positivo; separa
la codificación de entrada en self-refresh de REF. La dirección A10
identifica las variantes de precharge, auto-precharge y ZQ.

Sólo interpreta valores estables en el flanco diferencial `01→10`.
Cambios de pines en el mismo timestamp, X/Z y reloj ambiguo quedan
explícitos. Las ventanas son `[inicio, fin)`, dentro de la extensión VCD
observada. Un comando codificado no demuestra aceptación, cumplimiento
temporal, datos correctos ni funcionamiento de todos los chips del rank.
Todas las banderas de calificación de potencia y protocolo siguen falsas.

## Integración futura, después de revisar R1

El [helper Tcl](../../../tools/ntrpu_ddr_rank_trace_r1.tcl) selecciona 48
pines públicos: doce por dispositivo representativo de cada uno de los
cuatro ranks. Exige las cuatro interfaces completas, registra su inventario
y restaura el scope previo. No avanza tiempo ni fuerza señales.

Para una revisión nueva del capturador, abrir un VCD único antes de las dos
ventanas SAIF y cerrarlo después. Dejar al menos 10 ns simulados de margen
antes y después de las ventanas; comprobar luego que los timestamps del
VCD las contienen. Un archivo puede terminar en su último cambio, antes
del tiempo de cierre: la traza histórica acaba en 54,999561 µs aunque se
solicitó parar a 55 µs. No inventar esa fracción de cobertura.

Integrar llamadas `ntrpu_rank_trace_open <nuevo.vcd> <inventario.tsv>` y
`ntrpu_rank_trace_close` en una revisión nueva, tras inspeccionar R1.
Conservar las ventanas reales de `windows.tsv` y el mismo ámbito SAIF del
DUT. Verificar nativamente jerarquía, tiempos, errores vendor y correspondencia
con el DCP antes de usar actividad en potencia. Este helper **todavía no se
ha ejecutado en XSim**; R2 está encolada con revisión y respaldo automáticos.

## Validación y respaldo

Diez pruebas Python y diez comprobaciones Tcl con consultas XSim simuladas
pasaron en Windows y Linux. Incluyen cuatro ranks, REF/SRE/CKE, X/Z,
flancos ambiguos, límites temporales, formato corrupto y pines ausentes.
El reanálisis del VCD histórico produjo resultados idénticos en ambos
sistemas; Linux usó CPU 2–3 y límite de 120 segundos, sin Vivado.

Evidencia: `build/cloud-setup/20261002-board/ddr-rank-commands-r1-validation.json`.
Fuente ZIP: `f508f69d253d56d1676ffbd374dbe4c067a102964a96e41b13a5d07d654d82b8`.
VCD: `15667a5bab0988964af9190bec91c3addd3904b86e7d32a61940960b3ee70b99`.
El handoff `artifacts/ntrpu-board-vivado/ddr-rank-commands-handoff-20261016-r1/`
conserva fuentes, controles, resultados y referencia al TAR original,
con restauración comprobada en D: y NFS.
