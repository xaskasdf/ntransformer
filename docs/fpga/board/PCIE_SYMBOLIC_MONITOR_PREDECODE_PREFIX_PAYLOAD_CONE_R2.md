# Comparador de payload de la ruta por prefijo

## Atribución formal

Se analizó el segundo tramo CARRY8 de la ruta crítica de la tanda 966:
`delivery_stage/retirement_i/late_i/expected_enable2108_in`.
Su cono estructural tiene 390 nodos y 121 FF de frontera. El modelo usa
los INIT y conexiones de la netlist archivada y la semántica binaria
`SINGLE_CY8` de la biblioteca instalada.

La función coincide con:

```text
unsigned15((seen13 - 3) * 4) < unsigned15(offset13 + delivered13)
```

`seen13` es el contador de once bits al entrar al lane 13, reconstruido
con SOP antes del incremento y EOP después. `offset13` son los dos bits
bajos de la dirección seleccionada; `delivered13`, sus trece bits de
bytes entregados. El predicado corresponde al límite superior del primer
byte del lane. No incluye los demás guards ni el límite inferior.

Primero se obtuvo UNSAT bajo `partial13 && seen13 >= 3`; después se
repitió la desigualdad **sin ese guard y también fue UNSAT**. La fórmula
usa aritmética de quince bits explícita, incluso fuera del payload activo.
No se asumió tráfico válido ni alcanzabilidad de los 121 FF. Invertir la
salida y forzarla a cero o uno produjo tres contraejemplos bajo el guard.

## Dependencias que se cancelan

El cono estructural incluye `origin.queries[3][8]` y `[9]`, aunque no
pertenecen al offset bajo. Al invertir cada bit, la desigualdad de la
salida fue UNSAT tanto con guard como sin él. Por tanto, esos dos bits
no afectan funcionalmente la salida binaria del cono, aunque figuren en
el recorrido estructural. No se atribuyó esa presencia a alias ni a
restricciones de tráfico.

## Evidencia y decisión

`analyze_pcie_symbolic_monitor_predecode_prefix_payload_cone_r2.py` y su
resultado están en `build/cloud-setup/20261002-board/`. Análisis SHA-256:
`90fc172913d91ec7b6386e5e8856012b560ca488edc313d7dc0acafd987e02ef`.
El resultado conserva las ocho hipótesis, controles y comprobaciones de
dependencia; enlaza el cono R3 y la biblioteca del análisis anterior.

Esta atribución es de la netlist del padre 966. No concede equivalencia
RTL X/Z, cierre físico ni garantiza qué ruta dominará después del cambio
de bandera. Primero deben llegar sus medidas. Los experimentos de
factorización de enables y límites registrados permanecen separados.
