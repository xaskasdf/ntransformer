# Siguiente candidato: anchura del selector RC

Usar R8 sólo después de aprobar su propia netlist. La síntesis revisada
mejora WNS de −8,301 a **−7,442 ns**, pero eleva LUT de 42.362 a **51.937**
y CARRY8 de 306 a **1.890**. FF permanece en 10.729. Setup/hold siguen
incumpliendo y no hay implementación física.

## Evidencia y próximo cambio

La ruta peor parte de `rc_user[76]` y llega a `remaining_bytes_reg[20]/D`:
11,453 ns, 45 niveles; pasa por `selected_prefix_mask[5]` a 3,261 ns.
Los nombres son indicios para investigar, no demuestran equivalencia
semántica ni autorizan excepciones temporales.

R3 declaró `newest` como `integer` de 32 bits; R2 lo limitaba a cinco bits
con signo. Sólo recibe −1 o una posición EOP conocida entre 0 y 15.
La netlist contiene 556 CARRY8 con conexiones cuyo nombre incluye `newest`.
Esto no atribuye todo el crecimiento a esa variable.

El próximo experimento debe **limitar únicamente `newest` a cinco bits
con signo**, conservando el resto de R8. Verificar la promoción de signos,
el centinela −1, las comparaciones de prioridad y cuatro estados. Medir
LUT, CARRY8, WNS/TNS y rutas antes de seleccionar la variante.

## Experimentos posteriores separados

La máscara sólo puede incluir orígenes anteriores al encabezado consultado.
Las 155.168 filas binarias tienen unión de máscaras `0x7310`; la netlist aún
contiene salidas denominadas bits 5, 6 y 7. Investigar destinos constantes
por encabezado para reemplazar el índice dinámico, con equivalencia RTL
completa. No declarar esas rutas falsas basándose sólo en nombres o vectores.

Después, si continúa dominando la aritmética, evaluar prefijos constantes
calculados antes de seleccionar máscara/raíz. Mantener semántica de conteos
cero, desbordamientos, signos, X/Z y entradas no utilizadas. No combinar
estas hipótesis en una misma primera medición.

## Puertas y recuperación

Conservar todos los bancos de R8: 1.153.296 comparaciones heredadas,
32.768 adicionales, selector 89.632, RQ 615.536/5.712 X/Z y 8.850/240,
RC 45.056/6.624 X/Z y 544/60, propiedad integrada 46.356/3.872, principal
22.339/68 y siete mutantes. Exigir DCP acíclico, síntesis VU5P/4 ns y
replay propio de 22.330/67 con los tres suplementos binarios completos.

Mantener el inicio XSIM verificado, las pruebas y los logs. Preservar fuentes
ejecutadas, revisiones fallidas, la alternativa R7 de menor área y respaldos
por tanda. Diagnóstico local:
`build/cloud-setup/20261002-board/pcie-rc-event-width-mask-diagnostic.json`.

Siguen abiertos endpoint nativo/BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD. La
[cola general](VIVADO_QUEUE.md) conserva todas sus dependencias.
