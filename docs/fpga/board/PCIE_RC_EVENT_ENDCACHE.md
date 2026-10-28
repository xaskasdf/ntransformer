# Límite de payload retenido: candidato R7

## Cambio y criterio de aceptación

R7 registra el límite exclusivo de payload junto con el contexto del
paquete. Busca retirar la suma de la ruta crítica de
[R6](PCIE_RC_EVENT_COMBINED.md), cuyo resultado OOC fue WNS −0,093 ns,
6.161 LUT, 1.302 FF y 99 CARRY8, con reloj de 4 ns. R7 mantiene
la interfaz, dos ciclos de latencia y cuatro eventos RC por ciclo.

No se considera una mejora hasta verificar el banco completo de 22.339
filas y 68 escenarios, las 304.604 filas binarias, la síntesis con las
mismas restricciones y la reproducción de su propio netlist. La síntesis
OOC no demuestra cierre temporal después de colocar y rutear.

## Coherencia del registro

La igualdad requerida en cada estado es:
`context_payload_end = context_delivered + context_lower_address[1:0]`,
con suma sin signo de 14 bits, incluso con contexto inactivo o con fallos.

- La inicialización y el reset ponen los tres valores en cero.
- La retención conserva los tres valores conjuntamente.
- Cada cabecera actualiza bytes, dirección y límite en la misma rama y
  flanco de validación. Si hay varias, la última actualiza los tres.
- El fin de paquete desactiva el contexto y conserva los tres valores.

Los fragmentos de una cabecera del beat actual mantienen sus límites
combinacionales. Sólo el prefijo perteneciente al contexto anterior usa
el límite retenido. Las expresiones de máscaras y los anchos aritméticos
de R6 permanecen iguales. Este razonamiento sobre las transiciones no
sustituye una prueba formal de equivalencia.

## Validación

Un monitor exclusivo de simulación comprueba la igualdad después de
cada flanco. Registra cobertura de reset, retención, cabeceras y contexto
activo e inactivo. Un cuarto modo del banco fuerza `14'h3fff` después
de observar un contexto real; debe fallar específicamente por la igualdad
rota, sin emitir aprobación del banco.

Se conservan las mutaciones de RC omitido y metadatos ausentes, además
de los controles de corrupción de evidencia. Los tres bancos binarios
RTL también deben presentar sus contadores del monitor. El netlist
se comprueba por sus salidas, sin atribuirle cobertura del monitor interno
ni equivalencia general de valores X/Z.

## Resultados revisados

Las 22.339 filas/68 escenarios de cuatro estados conservan todas las
salidas de R6. Pasaron los tres bancos RTL binarios: 22.330 filas de
referencia, 370 de fronteras y 281.904 de matriz de fragmentos.
El monitor comprobó 22.340 estados en el banco completo y otros 304.754
en los bancos binarios, contando sus ciclos de arranque.

Las dos mutaciones originales fallaron en las filas 132 y 133. La nueva
mutación produjo `RC_END_CACHE_INVARIANT end=3fff delivered=0040 low=0`.
La revisión rechazó diez corrupciones del checker original, diez del
nuevo checker de invariantes y quince de los tres bancos binarios.

### Síntesis OOC a 4 ns

| Medida | R6 | R7 |
| --- | ---: | ---: |
| LUT | 6.161 | 6.020 |
| FF | 1.302 | 1.318 |
| CARRY8 | 99 | 99 |
| WNS | −0,093 ns | +0,222 ns |
| Peor camino | 4,075 ns | 3,760 ns |
| Niveles lógicos | 15 | 14 |

R7 mejora WNS en 0,315 ns y ahorra 141 LUT, a coste de 16 FF.
Setup pasa en los 3.309 endpoints. **Hold sigue fallando** en 539:
WHS −0,075 ns, THS −31,676 ns. No usa latches, BRAM, URAM ni DSP.
El peor camino va de `context_seen_reg[4]` a `packet_checked_reg[0]`,
con 0,911 ns de lógica y 2,849 ns de ruta estimada.

Se conservaron parte, reloj, retardos I/O y cero excepciones temporales.
El EDIF final contiene 8.298 primitivas y 11.290 redes, sin ciclos
combinacionales; pasaron ocho controles del auditor. El monitor interno
no aparece entre las primitivas sintetizadas. Los avisos conservados
incluyen conversión de igualdad de cuatro estados, padding y eliminación
del registro de datos usado sólo para detectar X; no hay avisos críticos
ni avisos de puertos internos constantes. XDCH-2 registra 1.334 casos.

## Tandas y respaldo

| ID, prefijo `20261008T` | Estado |
| --- | --- |
| `032247Z-V4-ffn-supervised-pcie-rc-event-endcache-rtl` | Revisada y respaldada; 78,536 s de herramienta |
| `032811Z-V4-ffn-supervised-pcie-rc-event-endcache-binary-rtl` | Revisada y respaldada; 182,061 s |
| `033528Z-V4-ffn-supervised-pcie-rc-event-endcache-synth` | Síntesis y EDIF revisados y respaldados; 247,625 s |
| `034104Z-V4-ffn-supervised-pcie-rc-event-endcache-binary-netlist` | Tres bancos sintetizados revisados y respaldados; 464,641 s |

Cada tanda conserva fuentes, informes y certificado en
`artifacts/ntrpu-board-vivado/<ID>/`, físicamente en D: y NFS.
Los tiempos corresponden a herramientas, no a inferencia en hardware.

La reproducción del netlist propio pasó las mismas **304.604 filas**,
idénticas a RTL, y los quince controles de corrupción del revisor. Los
tres bancos compilaron el circuito sintetizado, sin el RTL del DUT. Se
comprobaron las tres definiciones duplicadas de `glbl`: sus cuerpos son
idénticos. No se aplicó SDF ni el monitor interno de simulación; no se
afirma equivalencia general X/Z ni cierre físico. SHA-256 del netlist:
`f668203372bec8255a5a8d2a3f8e90ee2b1f6c6da0cab7f27bb72f874a7d94ca`.
Sigue integrar asignación, consultas y escrituras, con forwarding
ordenado y retiro correlacionado, sin metadatos activos suministrados por
la referencia monolítica. Se conservan todas las exigencias del
[contrato de etapas](PCIE_INVENTORY_PIPELINE_CONTRACT.md).
Las [diferencias de semántica en tráfico inválido](PCIE_INVENTORY_INTEGRATION_GAPS.md)
siguen abiertas: el shadow no prueba que los candidatos RC puedan
sustituir directamente las transiciones del inventario completo.

Integración completa, identidad host independiente, validaciones nativas
y CDC, memoria exacta de 512 GiB e implementación física siguen pendientes.
