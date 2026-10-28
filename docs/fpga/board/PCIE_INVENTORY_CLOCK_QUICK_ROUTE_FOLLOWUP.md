# Después del ruteo Quick: dependencia de tags RC

Decisión del 2026-10-08, condicionada al replay propio de Quick y al cierre
del respaldo R101. No se ha implementado ni medido la candidata siguiente.

## Evidencia que motiva el ensayo

El checkpoint Quick tiene reloj interno ruteado, pero falla setup y hold.
El análisis de veinte caminos registro→registro proviene del readback
`20261008T225600Z-V4-ffn-supervised-pcie-inventory-clock-route-audit`.
Todos parten de `producer/allocated_tags_reg[219]/C`; terminan en veinte
registros distintos de `remaining` (13) y `address_next` (7), con slack
−17,680…−15,956 ns. Son caminos correlacionados, no cobertura de todos los
cuellos de botella.

El peor llega a `remaining_reg[228][12]/CE`: 25 niveles, 21,640 ns de datos,
2,695 ns de lógica y 18,945 ns de rutas. Atraviesa ownership, liberación,
admisión RQ y habilitación del banco. Dos redes finales,
`producer/next_tags0518_out` y `producer/remaining[228][12]_i_1_n_0`,
aportan 5,064 y 6,262 ns. Todos los datos de esta muestra están ruteados.
Quick no optimiza timing: estas cifras no predicen una implementación
dirigida por timing ni prueban que una modificación concreta mejorará WNS.

Análisis local: `20261008T224646Z-V4-ffn-supervised-pcie-inventory-clock-quick-route-routed-path-sample-analysis.json`,
SHA-256 `c5d294bab018aa3bd7d5a66c13e0f878808a48869df88e889184e08789f4d670`.

## Candidata acotada

Preparar una revisión nueva del productor R12. Hoy
`rc_release_tag[i]=selected_release_tags[i*8+:8]` recibe un tag puesto a
cero cuando no se libera. Evaluar el uso directo de
`rc_data[i*128+64+:8]` en ese punto, conservando intactos el selector R3,
sus salidas, `rc_release_valid` y todas las guardas de los consumidores.
Así se intenta retirar la dependencia ownership→bits de tag→comparación;
el dato del tag y su autorización se calculan por caminos separados.

La sustitución solo puede aceptarse si cada consumidor está guardado por
el mismo `rc_release_valid[i]` y, cuando vale 1, ambos tags coinciden.
Conservar X/Z, SOP malformados, orden RC→RQ, reutilización, contexto
terminal con payload pendiente y estados posteriores a fallo. No asumir
que un tag inválido es un valor indiferente fuera de esa guarda.

## Puertas de aceptación

1. Inventariar todos los consumidores, demostrar el cambio exclusivo y
   preparar pruebas del productor real con tags válidos, inválidos y X/Z.
   Un mutante que altere un tag liberado debe ser rechazado.
2. Pasar los bancos RTL completos de R12 y su referencia R7 independiente,
   incluyendo los cinco bancos integrados y sus controles. Mantener
   256 tags, RQ2/RC4 por ciclo y la misma latencia y semántica de vacío.
3. Solo entonces sintetizar a 4 ns con el flujo comparable a R12. Medir
   recursos y todos los peores caminos; realizar el replay binario propio.
   Conservar R12 como preferida si el ensayo empeora o no satisface gates.
4. La [segmentación](PCIE_INVENTORY_PIPELINE_CONTRACT.md) sigue pendiente:
   este ensayo no corta toda la realimentación ni promete cerrar 250 MHz.
   No repetir el Default que agotó 3.600 s sin una entrada o decisión nueva.

El reloj nativo de XDMA proviene de BUFG_GT; no trasladar el BUFGCE de
diagnóstico ni su ubicación al árbol nativo. La integración requiere revisar
sus relaciones y emplazamiento reales. Continúan abiertos reset/CDC,
DIMM exacto, cuatro DDR/512 GiB, cómputo integrado y ECAD.
