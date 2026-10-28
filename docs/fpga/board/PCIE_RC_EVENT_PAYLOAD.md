# Predicados de payload RC antes de seleccionar el fragmento

Resultados de R5. La revisión posterior [R6](PCIE_RC_EVENT_COMBINED.md)
combina esta aritmética con framing registrado y pasa los mismos bancos
en RTL y netlist; es ahora la referencia temporal medida.

## Hipótesis R5

R5 parte de R3, la revisión con mejor estimación temporal medida. Calcula
la máscara de bytes para cada cabecera y para el contexto retenido antes
de seleccionar el fragmento. Así evita colocar la selección de longitud
y dirección delante de toda la aritmética de bytes. Conserva la latencia
de dos ciclos, cuatro eventos RC por ciclo y el manejo de estados inválidos.

El límite exclusivo `low + delivered` usa 14 bits; su cociente entre cuatro
usa 12 bits. Una longitud inválida puede alcanzar el DWORD 2048: truncarlo
a 11 bits cambiaría el comportamiento. El contador de DWORD conserva su
wrap de 11 bits. No se anticipa el contexto dinámico de otro ciclo.

## Validación aritmética local

`tools/check_ntrpu_pcie_rc_payload_bounds.py` compara ambas fórmulas para
todos los valores binarios de `seen` (0–2047), `low` (0–3) y `delivered`
(0–8191): **67.108.864 tuplas y 268.435.456 comparaciones de bytes**, sin
diferencias. El resultado vincula los hashes del verificador y ambos RTL.
Esta comprobación no demuestra equivalencia general del RTL ni de X/Z.
Requiere NumPy y una ruta de salida nueva:

```powershell
python tools/check_ntrpu_pcie_rc_payload_bounds.py build/payload-bounds-review.json
```

## Criterios de admisión

La revisión requiere el banco de cuatro estados de 22.339 filas y 68
escenarios, dos mutaciones reales y diez controles del verificador.
Después debe pasar los tres bancos binarios intactos: 22.330 filas de
referencia, 370 de límites y 281.904 de fragmentos; este último compara
simultáneamente con R1 renombrado. Todos se revisan con modelos independientes.

Solo entonces se permite síntesis con la misma parte, reloj de 4 ns,
retardos de E/S explícitos y ninguna excepción temporal. Se audita el EDIF
final y se reproducen las 304.604 filas binarias contra su propio netlist.
La comparación de recursos y timing se hace contra R3, no contra la
[regresión R4](PCIE_RC_EVENT_PREFIX.md).

## Estado

Validación aritmética, bancos RTL y síntesis completados y revisados.
Las 22.339 filas de cuatro estados coinciden con R3, incluidas las 17
filas con fallos. Las mutaciones se detectaron en los ciclos 132 y 133.
Pasaron diez controles del verificador de cuatro estados y quince de los
bancos binarios, tanto en RTL como en el netlist propio. Las 304.604 filas
binarias coinciden con los modelos independientes y con RTL. Las cuatro
tandas tienen revisión y respaldo verificados en D: y NFS.
La integración completa de inventario, identidad del host, reset/CDC,
DIMM exacto, cuatro DDR, inferencia y ensayos físicos permanecen abiertos.

## Tandas verificadas

| ID | Resultado |
| --- | --- |
| `20261008T015418Z-V4-ffn-supervised-pcie-rc-event-payload-rtl` | 22.339 filas / 68 escenarios, 69,245 s de ejecución; respaldo y revisión completos |
| `20261008T015816Z-V4-ffn-supervised-pcie-rc-event-payload-binary-rtl` | 304.604 filas en tres bancos, 184,251 s de ejecución; respaldo y revisión completos |
| `20261008T020534Z-V4-ffn-supervised-pcie-rc-event-payload-synth` | Síntesis y EDIF final revisados, 229,155 s; mejora temporal conservando los fallos restantes |
| `20261008T021049Z-V4-ffn-supervised-pcie-rc-event-payload-binary-netlist` | 304.604 filas binarias, 462,356 s; respaldo y revisión completos |

Son tiempos de la herramienta, no rendimiento de inferencia del hardware.

## Síntesis R5 revisada

Tanda `20261008T020534Z-V4-ffn-supervised-pcie-rc-event-payload-synth`,
229,155 s, parte `xcvu5p-flva2104-2-i`, reloj de 4 ns y E/S a 0 ns.
Los doce controles de cobertura temporal resultaron cero; no hay
excepciones temporales válidas ni ignoradas.

| Revisión | LUT | FF | CARRY8 | WNS (ns) | Peor camino (ns) / niveles |
| --- | ---: | ---: | ---: | ---: | ---: |
| R3 | 7.333 | 1.223 | 186 | −1,263 | 5,245 / 20 |
| R4 | 7.417 | 1.305 | 185 | −1,463 | 5,445 / 21 |
| R5 | 6.295 | 1.219 | 99 | −0,189 | 4,171 / 15 |

R5 ahorra 1.038 LUT y mejora el WNS en 1,074 ns frente a R3. Aún falla
setup en 8 de 2.971 endpoints (TNS −0,939 ns) y hold en 567
(WHS −0,075 ns; THS −31,844 ns). No usa latches, BRAM, URAM ni DSP.

El camino crítico va de `end_map_stage_reg[5]` a `packet_checked_reg[0]`,
atravesando el contador de DWORD y la acumulación de fallos. Comprende
1,167 ns de lógica y 3,004 ns de ruta **estimada, sin colocar ni rutear**.
El EDIF final tiene 8.359 primitivas y 11.170 redes; es acíclico y pasa
ocho controles del auditor. Vivado elimina el registro de payload usado
solo para detectar X: el netlist se valida únicamente con estímulos binarios.

## Reproducción y siguiente cambio

El netlist reproducido tiene SHA-256
`d2398ce16231fc1edc6c1d91ba9de4c6bb5f343e1ed510734fdbb82a0d046e0a`.
La revisión confirmó que los tres bancos compilaron ese circuito, sin
incluir el RTL del DUT. Los modelos R7 y R1 renombrado siguen siendo
referencias de prueba; no sustituyen la lógica sintetizada. No se usó SDF
ni se demostró equivalencia general de X/Z o integración completa.

R5 fue la referencia para evaluar una
revisión nueva que combine sus máscaras acotadas con los prefijos de
framing capturados en R4, ahora que el peor camino comienza en `end_map`.
R4 por sí sola empeoró el timing; esta combinación es una hipótesis.
El contexto retenido debe seguir leyéndose en el ciclo de validación,
sin adelantar `context_seen` ni `context_active`. Exigir los mismos bancos,
mutaciones, auditorías y reproducción, comparando recursos y timing con R5.
